"""
agent_tools.py
--------------
Wraps the ChEMBL pipeline and RDKit Lipinski calculator as LangChain
`@tool` functions, and assembles a tool-calling agent around them.

Requires: langchain, langchain-google-genai, pandas
    pip install langchain langchain-google-genai

The agent needs an LLM API key available as an environment variable:
    GEMINI_API_KEY   (or GOOGLE_API_KEY — used by ChatGoogleGenerativeAI, default below)
    OPENAI_API_KEY   (if you switch to ChatOpenAI instead)
"""

from __future__ import annotations

import json
import os
from typing import Optional

import streamlit as st
import pandas as pd
from langchain.agents import create_agent
from langchain_core.tools import tool
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_core.documents import Document

from chembl_pipeline import get_bioactivities, get_target_by_pref_name, search_target
from lipinski_rules import annotate_dataframe, calc_lipinski_batch
from chemotype_clustering import cluster_molecules
from sar_analysis import add_ligand_efficiency, calc_ligand_efficiency, rank_by_ligand_efficiency, summarize_sar
# ---------------------------------------------------------------------------
# In-memory store so tools can hand large DataFrames back to the Streamlit UI
# without stuffing raw tables through the LLM's context window. Tools return
# a short text summary to the agent, and the UI reads the full table from here.
# ---------------------------------------------------------------------------
LAST_RESULTS: dict[str, pd.DataFrame] = {}

# Local data overrides
_ACTIVE_DATA_SOURCE = "chembl"  # "chembl", "synthetic", or "custom"
_LOCAL_DF: Optional[pd.DataFrame] = None
_LOCAL_VECTORSTORE = None


def set_data_source(source: str, df: Optional[pd.DataFrame] = None) -> None:
    """Set the active data source to override ChEMBL fetching."""
    global _ACTIVE_DATA_SOURCE, _LOCAL_DF, _LOCAL_VECTORSTORE
    _ACTIVE_DATA_SOURCE = source
    _LOCAL_DF = df
    _LOCAL_VECTORSTORE = None
    
    if source in ("synthetic", "custom") and df is not None:
        try:
            from synthetic_data import standardize_dataframe
            std_df = standardize_dataframe(df)
            if "molecule_id" in std_df.columns:
                std_df = std_df.rename(columns={"molecule_id": "molecule_chembl_id"})
            if "assay_id" in std_df.columns:
                std_df = std_df.rename(columns={"assay_id": "assay_chembl_id"})
            LAST_RESULTS["bioactivities"] = std_df
        except Exception:
            pass


def set_local_vectorstore(vectorstore) -> None:
    """Set the active vector store for local searches."""
    global _LOCAL_VECTORSTORE
    _LOCAL_VECTORSTORE = vectorstore


def get_active_data_source() -> str:
    """Get the active data source name."""
    global _ACTIVE_DATA_SOURCE
    return _ACTIVE_DATA_SOURCE


def get_local_df() -> Optional[pd.DataFrame]:
    """Get the current loaded local DataFrame."""
    global _LOCAL_DF
    return _LOCAL_DF


def get_local_vectorstore():
    """Get the current local vector store."""
    global _LOCAL_VECTORSTORE
    return _LOCAL_VECTORSTORE


def get_session_dataset() -> Optional[pd.DataFrame]:
    df = get_local_df()
    if df is not None:
        return df
    try:
        if "dataset" in st.session_state:
            return st.session_state["dataset"]
    except Exception:
        pass
    return None


def get_session_vectorstore():
    vs = get_local_vectorstore()
    if vs is not None:
        return vs
    try:
        if "vector_store" in st.session_state:
            return st.session_state["vector_store"]
    except Exception:
        pass
    return None


def execute_python_code(python_code: str, df: pd.DataFrame):
    import ast
    import sys
    from io import StringIO
    
    # Capture stdout
    old_stdout = sys.stdout
    redirected_output = StringIO()
    sys.stdout = redirected_output
    
    try:
        tree = ast.parse(python_code)
        if not tree.body:
            sys.stdout = old_stdout
            return "Empty code", None
        
        # Check if the last statement is print(something)
        if isinstance(tree.body[-1], ast.Expr) and isinstance(tree.body[-1].value, ast.Call):
            func = tree.body[-1].value.func
            if isinstance(func, ast.Name) and func.id == "print" and len(tree.body[-1].value.args) == 1:
                # Replace the print call with its argument to evaluate it directly!
                tree.body[-1] = ast.Expr(value=tree.body[-1].value.args[0])
        
        from rdkit import Chem
        from rdkit.Chem import Descriptors, Lipinski
        context = {
            "df": df,
            "pd": pd,
            "Chem": Chem,
            "Descriptors": Descriptors,
            "Lipinski": Lipinski,
            "np": np if "np" in globals() else None
        }
        if "np" not in context or context["np"] is None:
            import numpy as np
            context["np"] = np
        
        res = None
        # If the last statement is an expression, evaluate it and return its value
        if isinstance(tree.body[-1], ast.Expr):
            # Compile and execute all but the last line
            if len(tree.body) > 1:
                exec_code = compile(ast.Module(body=tree.body[:-1], type_ignores=[]), filename="<ast>", mode="exec")
                exec(exec_code, context)
            
            # Evaluate the last line
            eval_code = compile(ast.Expression(body=tree.body[-1].value), filename="<ast>", mode="eval")
            res = eval(eval_code, context)
        else:
            # Otherwise exec the whole code
            exec_code = compile(tree, filename="<ast>", mode="exec")
            exec(exec_code, context)
            # Look for a dataframe in context or return "Executed successfully"
            df_vars = {k: v for k, v in context.items() if isinstance(v, pd.DataFrame) and k != 'df'}
            if df_vars:
                res = list(df_vars.values())[-1]
            else:
                res = "Executed successfully"
                
        # Restore stdout
        sys.stdout = old_stdout
        printed_val = redirected_output.getvalue()
        
        # If print was called, we should return the printed string or combine it.
        # If the return value is None but we have printed output, return the printed output!
        if (res is None or (isinstance(res, str) and res == "Executed successfully")) and printed_val.strip():
            return printed_val, context
            
        return res, context
    except Exception as e:
        sys.stdout = old_stdout
        return f"Execution Error: {e}", None


def build_temp_vectorstore(df: pd.DataFrame):
    """Builds a temporary, in-memory FAISS vector database from uploaded file rows."""
    id_col = None
    for col in ["compound_id", "molecule_id", "molecule name", "id", "compound", "molecule_chembl_id"]:
        for c in df.columns:
            if str(c).strip().lower() == col.lower():
                id_col = c
                break
        if id_col:
            break
    if not id_col:
        id_col = df.columns[0]

    # Find standard activity column
    ic50_col = None
    for col in ["ic50_um", "standard_value", "ic50", "value", "activity_value"]:
        for c in df.columns:
            if str(c).strip().lower() == col.lower():
                ic50_col = c
                break
        if ic50_col:
            break
            
    # Find text column
    text_col = None
    for col in ["comments", "notes", "description", "eln_reference", "eln", "remarks"]:
        for c in df.columns:
            if str(c).strip().lower() == col.lower():
                text_col = c
                break
        if text_col:
            break

    documents = []
    for idx, row in df.iterrows():
        # Build contents dynamically depending on what columns are present
        parts = [f"Compound: {row[id_col]}"]
        if ic50_col and ic50_col in row and pd.notna(row[ic50_col]):
            parts.append(f"IC50: {row[ic50_col]} uM")
        if text_col and text_col in row and pd.notna(row[text_col]):
            parts.append(f"Notes: {row[text_col]}")
        else:
            # If no comments/notes column, put other columns as context
            other_parts = []
            for col in df.columns:
                if col not in [id_col, ic50_col] and pd.notna(row[col]) and len(str(row[col])) < 100:
                    other_parts.append(f"{col}: {row[col]}")
            if other_parts:
                parts.append(" | ".join(other_parts))
                
        content = " | ".join(parts)
        doc = Document(page_content=content, metadata={"compound_id": str(row[id_col])})
        documents.append(doc)
    
    api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001", google_api_key=api_key)
    vectorstore = FAISS.from_documents(documents, embeddings)
    return vectorstore


@tool
def query_uploaded_lims(python_code: str) -> str:
    """
    Executes pandas/RDKit operations on the uploaded LIMS dataframe `df` stored in st.session_state['dataset'].
    You can write multi-line Python code to filter the dataframe, compute properties using RDKit, etc.
    Always refer to the dataframe as `df`.
    Example: df[df['ic50_um'] < 1.0]
    """
    df = get_session_dataset()
    if df is None:
        return "Error: No dataset uploaded."
    
    res, context = execute_python_code(python_code, df)
    
    df_to_save = None
    if isinstance(res, pd.DataFrame):
        df_to_save = res
    elif isinstance(res, pd.Series):
        df_to_save = pd.DataFrame([res])
        
    if df_to_save is None and context is not None:
        dfs = [v for k, v in context.items() if isinstance(v, pd.DataFrame) and k != 'df']
        if dfs:
            df_to_save = dfs[-1]
        else:
            series = [v for k, v in context.items() if isinstance(v, pd.Series)]
            if series:
                df_to_save = pd.DataFrame([series[-1]])
                
    if df_to_save is not None:
        LAST_RESULTS["uploaded_query_results"] = df_to_save
        print(f"DEBUG TOOLS: Saved to LAST_RESULTS. Keys: {list(LAST_RESULTS.keys())}, ID: {id(LAST_RESULTS)}")
        
    if isinstance(res, pd.DataFrame):
        return f"Successfully executed query. Returned DataFrame ({len(res)} rows):\n{res.to_string(max_rows=15, max_cols=10)}"
    elif isinstance(res, pd.Series):
        return f"Successfully executed query. Returned Series:\n{res.to_string()}"
    else:
        obs = str(res)
        if df_to_save is not None:
            obs += f"\n\n[DataFrame extracted from context: {len(df_to_save)} rows, columns: {list(df_to_save.columns)}]"
        return obs


@tool
def search_uploaded_notes(query: str) -> str:
    """
    Semantic search over unstructured notes/comments in the uploaded LIMS dataset using Vector RAG.
    Use this when the user asks questions about notes, comments, or references associated with compounds.
    """
    vector_store = get_session_vectorstore()
    if vector_store is None:
        # Try to build it on the fly if dataset exists
        df = get_session_dataset()
        if df is not None:
            try:
                vector_store = build_temp_vectorstore(df)
                try:
                    st.session_state["vector_store"] = vector_store
                except Exception:
                    pass
            except Exception as e:
                return f"Error building vector store: {e}"
        else:
            return "Error: No uploaded dataset or vector store available."
            
    if vector_store is not None:
        try:
            docs = vector_store.similarity_search(query, k=5)
            results = []
            for doc in docs:
                results.append(doc.page_content)
            return "\n\n".join(results)
        except Exception as e:
            return f"Error searching vector store: {e}"
    return "Error: No vector store available."



@tool
def search_chembl_target(query: str) -> str:
    """
    Search ChEMBL or local dataset for a target (protein/gene) by name, e.g. 'EGFR' or
    'acetylcholinesterase'. Returns a JSON list of candidate targets with
    their target_chembl_id, preferred name, organism, and target type.
    Use this first when you don't already know the exact target_chembl_id.
    """
    global _ACTIVE_DATA_SOURCE, _LOCAL_DF
    if _ACTIVE_DATA_SOURCE in ("synthetic", "custom"):
        if _LOCAL_DF is None:
            return f"Error: Active data source is set to '{_ACTIVE_DATA_SOURCE}', but no local dataset has been generated or uploaded in the UI. Please generate a synthetic dataset or upload a CSV in the sidebar before running queries."
            
        target_col = None
        for col in ["target", "target_name", "target_id", "Target", "Target_Name"]:
            if col in _LOCAL_DF.columns:
                target_col = col
                break
        if target_col is None:
            target_col = "target" if "target" in _LOCAL_DF.columns else _LOCAL_DF.columns[0]
            
        unique_targets = _LOCAL_DF[target_col].dropna().unique()
        matches = [str(t) for t in unique_targets if query.lower() in str(t).lower()]
        
        if not matches:
            return f"No targets found for '{query}' in local dataset."
            
        results = []
        for t in matches:
            results.append({
                "target_chembl_id": t,
                "pref_name": t,
                "organism": "Homo sapiens",
                "target_type": "SINGLE PROTEIN"
            })
        df_res = pd.DataFrame(results)
        LAST_RESULTS["target_search"] = df_res
        return df_res.to_json(orient="records")

    df = search_target(query)
    if df.empty:
        return f"No ChEMBL targets found for '{query}'."
    LAST_RESULTS["target_search"] = df
    return df.head(10).to_json(orient="records")


@tool
def lookup_target_by_exact_name(pref_name: str) -> str:
    """
    Resolve a target by its EXACT preferred name (case-insensitive).
    Faster and unambiguous compared to search_chembl_target, but only
    works if you already know the exact pref_name string. If this
    returns no match, fall back to search_chembl_target instead.
    """
    global _ACTIVE_DATA_SOURCE, _LOCAL_DF
    if _ACTIVE_DATA_SOURCE in ("synthetic", "custom"):
        if _LOCAL_DF is None:
            return f"Error: Active data source is set to '{_ACTIVE_DATA_SOURCE}', but no local dataset has been generated or uploaded in the UI. Please generate a synthetic dataset or upload a CSV in the sidebar before running queries."
            
        target_col = None
        for col in ["target", "target_name", "target_id", "Target", "Target_Name"]:
            if col in _LOCAL_DF.columns:
                target_col = col
                break
        if target_col is None:
            target_col = "target" if "target" in _LOCAL_DF.columns else _LOCAL_DF.columns[0]
            
        unique_targets = _LOCAL_DF[target_col].dropna().unique()
        for t in unique_targets:
            if str(t).lower() == pref_name.lower():
                return f"target_chembl_id: {t}"
        return f"No exact match for '{pref_name}' in local dataset. Try search_chembl_target instead."

    target_id = get_target_by_pref_name(pref_name)
    if target_id is None:
        return f"No exact pref_name match for '{pref_name}'. Try search_chembl_target instead."
    return f"target_chembl_id: {target_id}"


@tool
def fetch_bioactivities(target_chembl_id: str, standard_type: str = "IC50", limit: int = 50) -> str:
    """
    Fetch bioactivity records (default IC50) for a given target id
    (e.g. 'CHEMBL203' or 'EGFR_Kinase'). Returns a short text summary; the full table of
    molecule ID, smiles, standard_value, standard_units is stored
    for display in the UI, and can be used as input to
    `calculate_lipinski_for_dataset`. Default limit is kept modest (50) to
    keep the tool result — and therefore the token cost of the agent's next
    turn — small, which matters on Gemini's free tier request quotas.
    """
    global _ACTIVE_DATA_SOURCE, _LOCAL_DF
    if _ACTIVE_DATA_SOURCE in ("synthetic", "custom"):
        if _LOCAL_DF is None:
            return f"Error: Active data source is set to '{_ACTIVE_DATA_SOURCE}', but no local dataset has been generated or uploaded in the UI. Please generate a synthetic dataset or upload a CSV in the sidebar before running queries."
            
        from synthetic_data import standardize_dataframe
        std_df = standardize_dataframe(_LOCAL_DF)
        
        # Filter by target (case-insensitive)
        filtered = std_df[std_df["target"].astype(str).str.lower() == target_chembl_id.lower()]
        
        # Filter by standard_type if it is present and matches (case-insensitive)
        type_filtered = filtered[filtered["standard_type"].astype(str).str.lower() == standard_type.lower()]
        if not type_filtered.empty:
            filtered = type_filtered
            
        if filtered.empty:
            return f"No {standard_type} bioactivity data found for target '{target_chembl_id}' in local dataset."
            
        if limit:
            filtered = filtered.head(limit)
            
        LAST_RESULTS["bioactivities"] = filtered
        sample_smiles = filtered["smiles"].iloc[0] if not filtered.empty else "N/A"
        original_cols = list(_LOCAL_DF.columns)
        return (
            f"Fetched {len(filtered)} {standard_type} records for local target '{target_chembl_id}' from the local dataset.\n"
            f"Original local dataset columns: {original_cols}\n"
            f"Standardized columns: {list(filtered.columns)}\n"
            f"Sample smiles: {sample_smiles}"
        )

    df = get_bioactivities(target_chembl_id, standard_type=standard_type, limit=limit)
    if df.empty:
        return f"No {standard_type} bioactivity data found for target {target_chembl_id}."
    LAST_RESULTS["bioactivities"] = df
    return (
        f"Fetched {len(df)} {standard_type} records for {target_chembl_id}. "
        f"Columns: {list(df.columns)}. Sample smiles: {df['smiles'].iloc[0]}"
    )


@tool
def calculate_lipinski_for_smiles(smiles: str) -> str:
    """
    Compute Lipinski Rule-of-Five descriptors (MW, LogP, HBD, HBA) and
    violation count for a single SMILES string.
    """
    from lipinski_rules import calc_lipinski

    result = calc_lipinski(smiles)
    if result is None:
        return f"Could not parse SMILES: {smiles}"
    return json.dumps(result)


def _get_latest_dataset() -> tuple[Optional[pd.DataFrame], Optional[str]]:
    for key in ["sar_analysis", "clustering", "lipinski", "bioactivities"]:
        if key in LAST_RESULTS:
            return LAST_RESULTS[key], key
    return None, None


@tool
def calculate_lipinski_for_dataset(use_last_bioactivities: bool = True) -> str:
    """
    Compute Lipinski descriptors for every molecule in the most recently
    fetched bioactivity dataset (call `fetch_bioactivities` first). Adds
    MW, LogP, HBD, HBA, and RO5_violations columns, and stores the merged
    table for the UI to display and plot.
    """
    if use_last_bioactivities and "bioactivities" in LAST_RESULTS:
        df = LAST_RESULTS["bioactivities"]
    else:
        return "No bioactivity dataset available yet — call fetch_bioactivities first."

    merged = annotate_dataframe(df, smiles_col="smiles")
    if merged.empty:
        return "No valid molecules could be parsed by RDKit in this dataset."

    LAST_RESULTS["lipinski"] = merged
    n_pass = int(merged["RO5_pass"].sum())
    return (
        f"Computed Lipinski descriptors for {len(merged)} molecules. "
        f"{n_pass}/{len(merged)} pass Rule-of-Five (<=1 violation)."
    )


@tool
def cluster_dataset_by_chemotype(method: str = "scaffold", cutoff: float = 0.4) -> str:
    """
    Cluster the current dataset of molecules into chemotypes (scaffold families).
    Assumes a dataset has already been fetched (via fetch_bioactivities) or annotated (via calculate_lipinski_for_dataset).
    'method': 'scaffold' (Bemis-Murcko exact core scaffold grouping, default) or 'fingerprint'
              (Morgan fingerprint Butina Tanimoto similarity clustering).
    'cutoff': Butina Tanimoto distance threshold (lower is stricter/more clusters, range 0.2-0.5, default 0.4).
              Only applies to 'fingerprint' method.
    Saves the clustered dataset and returns a text summary of the clusters.
    """
    df, key = _get_latest_dataset()
    if df is None:
        return "No bioactivity or Lipinski dataset available yet — call fetch_bioactivities first."
    
    try:
        clustered = cluster_molecules(df, method=method, smiles_col="smiles", cutoff=cutoff)
        if clustered.empty:
            return "No valid molecules could be parsed/clustered."
            
        LAST_RESULTS["clustering"] = clustered
        
        # Create a nice summary table text to return to the agent
        from chemotype_clustering import cluster_summary
        summary_df = cluster_summary(clustered)
        num_clusters = len(summary_df)
        top_clusters = summary_df.head(5)
        
        summary_str = f"Successfully clustered {len(clustered)} molecules into {num_clusters} chemotypes using {method} clustering.\n"
        summary_str += "Top 5 largest clusters:\n"
        for _, row in top_clusters.iterrows():
            cluster_id = row["chemotype_cluster"]
            size = row["size"]
            scaffold = row.get("scaffold_smiles", "N/A")
            summary_str += f"  - Cluster {cluster_id}: {size} molecules"
            if method == "scaffold" and scaffold != "N/A":
                summary_str += f" (Scaffold: {scaffold})"
            summary_str += "\n"
        return summary_str
    except Exception as e:
        return f"Error running clustering: {str(e)}"


@tool
def perform_sar_analysis() -> str:
    """
    Perform Structure-Activity Relationship (SAR) analysis and calculate ligand efficiency
    for the current dataset. Computes pActivity, heavy atom count, and Ligand Efficiency (LE),
    and generates a compact summary.
    Assumes a dataset has already been fetched, and optionally clustered.
    Saves the annotated dataset and returns the SAR text digest.
    """
    df, key = _get_latest_dataset()
    if df is None:
        return "No dataset available — fetch bioactivities or calculate Lipinski descriptors first."
        
    try:
        annotated = add_ligand_efficiency(df, smiles_col="smiles", value_col="standard_value", units_col="standard_units")
        if annotated.empty:
            return "No molecules could be successfully evaluated for ligand efficiency."
            
        ranked = rank_by_ligand_efficiency(annotated)
        LAST_RESULTS["sar_analysis"] = ranked
        
        cluster_col = "chemotype_cluster" if "chemotype_cluster" in ranked.columns else None
        digest = summarize_sar(ranked, cluster_col=cluster_col)
        return digest
    except Exception as e:
        return f"Error running SAR analysis: {str(e)}"


TOOLS = [
    search_chembl_target,
    lookup_target_by_exact_name,
    fetch_bioactivities,
    calculate_lipinski_for_smiles,
    calculate_lipinski_for_dataset,
    cluster_dataset_by_chemotype,
    perform_sar_analysis,
    query_uploaded_lims,
    search_uploaded_notes,
]


SYSTEM_PROMPT = (
    "You are a drug-discovery research assistant. You have tools to search "
    "ChEMBL targets, fetch IC50/Ki/EC50 bioactivity data, compute Lipinski "
    "Rule-of-Five descriptors with RDKit, cluster molecules into chemotypes (scaffolds), "
    "and perform SAR analysis (Ligand Efficiency). When a user names a target, first try "
    "lookup_target_by_exact_name if they give (or you know) ChEMBL's exact "
    "preferred name; otherwise use search_chembl_target for a fuzzy lookup "
    "(e.g. 'EGFR'). Once you have a target_chembl_id, call fetch_bioactivities "
    "ONCE, then calculate_lipinski_for_dataset, cluster_dataset_by_chemotype, and "
    "perform_sar_analysis as appropriate or when requested.\n\n"
    "PREFER PRE-BUILT TOOLS FOR STANDARD WORKFLOWS: For standard operations (fetching target data, "
    "calculating Lipinski Rule-of-Five properties, clustering molecules, or running SAR analysis), "
    "you should ALWAYS prefer using the pre-built pipeline tools: `lookup_target_by_exact_name`, "
    "`fetch_bioactivities`, `calculate_lipinski_for_dataset`, `cluster_dataset_by_chemotype`, and "
    "`perform_sar_analysis`. These tools support both ChEMBL and local LIMS datasets transparently. "
    "For example, to check the drug likeness of a target's local LIMS dataset, first call `lookup_target_by_exact_name` "
    "to get the target, call `fetch_bioactivities` to load the dataset, and then call `calculate_lipinski_for_dataset`.\n\n"
    "USE QUERY_UPLOADED_LIMS ONLY FOR CUSTOM QUERYING: If the active data source is local/uploaded LIMS, "
    "you can use `query_uploaded_lims` to run custom Python/pandas/RDKit code on `df` only when the user "
    "asks custom questions that cannot be answered by the standard tools (such as finding the molecule with the "
    "smallest/largest IC50, custom column filtering, or custom mathematical calculations).\n\n"
    "If there are comments/notes in the local dataset, you can use `search_uploaded_notes` to perform vector search.\n\n"
    "Only report compound IDs and SMILES that are explicitly present in the tool outputs. "
    "To find which molecule corresponds to a specific value (like the minimum IC50), write Python code for "
    "`query_uploaded_lims` that returns the full row containing both the compound ID and the value (e.g. `df.nsmallest(1, 'ic50_um')`), "
    "rather than just the number.\n\n"
    "Be concise and always mention concrete numbers (counts, MW/LogP ranges, pass rates, clusters, and ligand efficiencies) in your final answer."
)

# Free-tier-friendly defaults:
#   - "gemini-2.5-flash-lite" instead of "gemini-2.5-pro": Pro's free-tier quota is
#     extremely small (a handful of requests/day, and on some accounts Pro is
#     paid-only entirely), while Flash gets a much higher RPM/RPD allowance.
#   - MAX_AGENT_STEPS caps how many tool-call round-trips (i.e. separate LLM
#     requests) a single query can make, so one runaway query can't burn
#     through the whole daily quota.
DEFAULT_MODEL = "gemini-3.1-flash-lite"
MAX_AGENT_STEPS = 15


def build_agent(model_name: str = DEFAULT_MODEL, temperature: float = 0.0):
    """
    Build a tool-calling agent bound to the tools above, using Gemini as the LLM.

    Uses langchain.agents.create_agent (LangChain 1.0+, LangGraph-based) — the
    legacy AgentExecutor/create_tool_calling_agent API was moved to the separate
    `langchain-classic` package and no longer ships in `langchain` by default.

    Tuned for Gemini's free API tier:
      - defaults to Flash rather than Pro (see DEFAULT_MODEL above)
      - `max_retries` gives native exponential-backoff retry on 429/5xx errors
        without wrapping the model (wrapping via `.with_retry()` would break
        `create_agent`'s internal `bind_tools()` call — that's the bug this
        replaces)
      - a rate limiter proactively spaces out requests to stay under Flash's
        free-tier RPM ceiling, rather than only reacting after a 429
      - the agent's LangGraph recursion limit is bounded (see run_agent)

    Swap `ChatGoogleGenerativeAI` for another LangChain chat model if you'd
    rather not use Gemini — the tools themselves are LLM-agnostic.
    """
    from langchain_google_genai import ChatGoogleGenerativeAI
    from langchain_core.rate_limiters import InMemoryRateLimiter

    if not (os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")):
        raise RuntimeError(
            "GEMINI_API_KEY is not set. Export it (e.g. `export GEMINI_API_KEY=...`), "
            "or switch build_agent() to a different LangChain chat model."
        )

    # Proactively space out requests to stay under Flash's free-tier RPM
    # (roughly 10-15 RPM as of mid-2026) rather than only reacting after a 429.
    # ~0.2 req/s = one request every 5s, comfortably under that ceiling.
    rate_limiter = InMemoryRateLimiter(requests_per_second=0.2, check_every_n_seconds=0.5, max_bucket_size=1)

    llm = ChatGoogleGenerativeAI(
        model=model_name,
        temperature=temperature,
        max_retries=5,  # native retry w/ backoff on 429/5xx — NOT .with_retry(), which
        rate_limiter=rate_limiter,  # would wrap the model and break create_agent's bind_tools()
    )

    return create_agent(llm, tools=TOOLS, system_prompt=SYSTEM_PROMPT)


def run_agent(agent, user_input: str) -> dict:
    """
    Invoke the agent with a plain user string and return the raw LangGraph
    result (a dict with a "messages" key containing the full message trace:
    HumanMessage -> AIMessage(tool_calls) -> ToolMessage -> ... -> final AIMessage).

    `recursion_limit` caps the number of internal steps (each of which is an
    LLM request against your Gemini quota) a single query can take.
    """
    return agent.invoke(
        {"messages": [{"role": "user", "content": user_input}]},
        config={"recursion_limit": MAX_AGENT_STEPS},
    )


def extract_final_answer(result: dict) -> str:
    """Pull the last AIMessage's text content out of a create_agent result."""
    messages = result.get("messages", [])
    for msg in reversed(messages):
        if getattr(msg, "type", None) == "ai" and getattr(msg, "content", None):
            content = msg.content
            if isinstance(content, str):
                return content
            elif isinstance(content, list):
                text_parts = []
                for part in content:
                    if isinstance(part, str):
                        text_parts.append(part)
                    elif isinstance(part, dict) and "text" in part:
                        text_parts.append(part["text"])
                return "".join(text_parts)
            return str(content)
    return ""


def extract_tool_trace(result: dict) -> list[dict]:
    """
    Turn the raw message list into a simple list of
    {"tool": name, "input": args, "output": observation} steps, matching
    each AIMessage tool_call to its corresponding ToolMessage by call id.
    """
    messages = result.get("messages", [])
    tool_calls_by_id = {}
    for msg in messages:
        if getattr(msg, "type", None) == "ai":
            for call in getattr(msg, "tool_calls", None) or []:
                tool_calls_by_id[call["id"]] = {"tool": call["name"], "input": call["args"]}

    steps = []
    for msg in messages:
        if getattr(msg, "type", None) == "tool":
            call_id = getattr(msg, "tool_call_id", None)
            meta = tool_calls_by_id.get(call_id, {"tool": getattr(msg, "name", "unknown"), "input": {}})
            steps.append({"tool": meta["tool"], "input": meta["input"], "output": msg.content})
    return steps


if __name__ == "__main__":
    agent = build_agent()
    result = run_agent(agent, "Find IC50 data for EGFR and check Lipinski rule compliance.")
    print(extract_final_answer(result))

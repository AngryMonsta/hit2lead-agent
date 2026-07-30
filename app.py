"""
app.py
------
One-page Streamlit dashboard for the ChEMBL + RDKit drug-discovery agent.

Shows:
    - A prompt input box
    - The agent's step-by-step tool calls ("thoughts")
    - A structured table of results (bioactivities and/or Lipinski descriptors)
    - 2D molecule images rendered with RDKit

Run with:
    streamlit run app.py

Requires GEMINI_API_KEY (or GOOGLE_API_KEY) to be set in your environment (see agent_tools.py).
"""

from __future__ import annotations

import pandas as pd
import streamlit as st
from rdkit import Chem
from rdkit.Chem import Draw

from agent_tools import LAST_RESULTS, MAX_AGENT_STEPS, build_agent, extract_final_answer, extract_tool_trace, run_agent, set_data_source, build_temp_vectorstore
from chemotype_clustering import cluster_summary
from sar_analysis import summarize_sar

st.set_page_config(page_title="Hit 2 Lead Agent", layout="wide")


@st.cache_resource(show_spinner=False)
def get_agent():
    return build_agent()


def mol_image(smiles: str, size=(220, 220)):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return Draw.MolToImage(mol, size=size)


def render_intermediate_steps(steps: list[dict]) -> None:
    """Render the agent's tool calls + observations as a readable trace."""
    if not steps:
        st.caption("No tool calls were made for this query.")
        return

    for i, step in enumerate(steps, start=1):
        tool_name = step['tool']
        if tool_name == "query_uploaded_lims":
            with st.status(f"Step {i}: Executing Python/pandas code on uploaded dataset...", state="complete") as status:
                st.markdown("**Generated Code:**")
                st.code(step['input'].get('python_code', ''), language='python')
                st.markdown("**Result Summary / Observation:**")
                st.write(step['output'])
        elif tool_name == "search_uploaded_notes":
            with st.status(f"Step {i}: Querying Vector RAG (unstructured notes)...", state="complete") as status:
                st.markdown("**Query:**")
                st.code(step['input'].get('query', ''), language='markdown')
                st.markdown("**Retrieved Notes / Observation:**")
                st.write(step['output'])
        else:
            with st.expander(f"Step {i}: called `{tool_name}`", expanded=False):
                st.markdown("**Tool input:**")
                st.json(step["input"])
                st.markdown("**Observation:**")
                st.write(step["output"])


def rename_id_column(df: pd.DataFrame) -> pd.DataFrame:
    """Rename standard molecule ID column back to original target name for display."""
    from agent_tools import get_active_data_source, get_local_df
    active_source = get_active_data_source()
    local_df = get_local_df()
    if active_source in ("synthetic", "custom") and local_df is not None:
        from synthetic_data import get_original_id_col
        orig_col = get_original_id_col(local_df)
        for std_col in ["molecule_id", "molecule_chembl_id"]:
            if orig_col != std_col and std_col in df.columns:
                return df.rename(columns={std_col: orig_col})
    return df


def render_results_tables() -> None:
    """Show whichever result tables the agent populated during this run."""
    print(f"DEBUG APP: Reading from LAST_RESULTS. Keys: {list(LAST_RESULTS.keys())}, ID: {id(LAST_RESULTS)}")
    # Find the most enriched table
    df = None
    table_type = None
    
    if "uploaded_query_results" in LAST_RESULTS:
        df = LAST_RESULTS["uploaded_query_results"]
        table_type = "uploaded_query_results"
    elif "sar_analysis" in LAST_RESULTS:
        df = LAST_RESULTS["sar_analysis"]
        table_type = "sar_analysis"
    elif "clustering" in LAST_RESULTS:
        df = LAST_RESULTS["clustering"]
        table_type = "clustering"
    elif "lipinski" in LAST_RESULTS:
        df = LAST_RESULTS["lipinski"]
        table_type = "lipinski"
    elif "bioactivities" in LAST_RESULTS:
        df = LAST_RESULTS["bioactivities"]
        table_type = "bioactivities"
    elif "target_search" in LAST_RESULTS:
        st.subheader("Target search results")
        st.dataframe(LAST_RESULTS["target_search"])
        return
    else:
        return

    # Check which title to show
    title_map = {
        "uploaded_query_results": "Query Results from Uploaded Dataset",
        "sar_analysis": "Structure-Activity Relationship (SAR) Analysis & Ligand Efficiency",
        "clustering": "Chemotype Clustering",
        "lipinski": "Bioactivities + Lipinski descriptors",
        "bioactivities": "Bioactivities"
    }
    st.subheader(title_map.get(table_type, "Molecules"))

    # Rename ID column back for display
    display_df = rename_id_column(df)

    # 1. SAR Analysis display (digest and metrics)
    if "LE" in display_df.columns:
        # Best LE Compound Metrics
        best_le_idx = display_df["LE"].idxmax() if not display_df.empty else None
        if best_le_idx is not None:
            best_comp = display_df.loc[best_le_idx]
            
            # Find a readable ID for the best compound
            best_id = "?"
            for id_col in ["compound_id", "molecule_id", "molecule name", "id", "compound", "molecule_chembl_id"]:
                if id_col in best_comp:
                    best_id = str(best_comp[id_col])
                    break
            
            st.markdown("### Top compound by ligand efficiency")
            flex = st.container(horizontal=True, border=True)
            flex.metric("Best Ligand Efficiency (LE)", f"{best_comp['LE']:.3f} kcal/mol/atom", help="Standard Hopkins et al. convention")
            flex.metric("Best Compound ID", best_id)
            p_act_val = best_comp.get("pActivity", None)
            p_act_str = f"{p_act_val:.2f}" if p_act_val is not None and pd.notna(p_act_val) else "N/A"
            flex.metric("Potency (pActivity)", p_act_str)
            
        # Display the SAR digest text
        cluster_col = "chemotype_cluster" if "chemotype_cluster" in display_df.columns else None
        try:
            sar_digest_text = summarize_sar(df, cluster_col=cluster_col)
            with st.expander("Show detailed SAR text digest", expanded=True):
                st.text(sar_digest_text)
        except Exception:
            pass

        # Render interactive scatter plot using st.scatter_chart
        st.markdown("### Activity vs. molecular properties")
        x_axis_col = "HeavyAtomCount" if "HeavyAtomCount" in display_df.columns else ("MW" if "MW" in display_df.columns else None)
        y_axis_col = "pActivity" if "pActivity" in display_df.columns else ("standard_value" if "standard_value" in display_df.columns else None)
        
        if x_axis_col and y_axis_col:
            color_col = "chemotype_cluster" if "chemotype_cluster" in display_df.columns else "LE"
            
            # Map cluster to categorical strings so we have distinct colors/legend
            plot_df = display_df.copy()
            if "chemotype_cluster" in plot_df.columns:
                plot_df["chemotype_cluster"] = plot_df["chemotype_cluster"].astype(str)
                
            st.scatter_chart(
                plot_df,
                x=x_axis_col,
                y=y_axis_col,
                color=color_col,
                size="LE" if "LE" in plot_df.columns else None,
                height=400,
            )

    # 2. Chemotype Clustering display (summary and filter)
    filtered_df = display_df
    if "chemotype_cluster" in display_df.columns:
        st.markdown("### Chemotype clusters summary")
        from chemotype_clustering import cluster_summary
        sum_df = cluster_summary(df)
        
        with st.container(border=True):
            col_tbl, col_filter = st.columns([2, 1])
            with col_tbl:
                st.dataframe(sum_df)
            with col_filter:
                st.markdown("**Filter view by cluster**")
                unique_clusters = sorted(df["chemotype_cluster"].unique())
                options = ["All clusters"] + [f"Cluster {int(c)}" for c in unique_clusters]
                
                selected_option = st.selectbox(
                    "Select a chemotype cluster to filter the molecule list and structure grid:",
                    options=options,
                    index=0,
                    label_visibility="collapsed"
                )
                
                if selected_option != "All clusters":
                    cluster_num = int(selected_option.split(" ")[1])
                    filtered_df = display_df[df["chemotype_cluster"] == cluster_num]
                    st.caption(f"Showing {len(filtered_df)} molecules from {selected_option}.")

    # Show the interactive DataFrame table
    st.markdown("### Evaluated compounds")
    st.dataframe(filtered_df)

    # Display 2D molecule structures for the filtered subset
    smiles_col = None
    for c in ["smiles", "canonical_smiles", "structure"]:
        if c in filtered_df.columns:
            smiles_col = c
            break
    if smiles_col is None:
        # Fallback to check if any column contains "smiles"
        for col in filtered_df.columns:
            if "smiles" in str(col).lower():
                smiles_col = col
                break
                
    if smiles_col:
        show_molecule_grid(filtered_df, smiles_col=smiles_col)


def show_molecule_grid(df: pd.DataFrame, smiles_col: str, max_mols: int = 12) -> None:
    st.subheader("2D structures")
    subset = df.head(max_mols)
    cols = st.columns(4)
    for i, (_, row) in enumerate(subset.iterrows()):
        img = mol_image(row[smiles_col])
        with cols[i % 4]:
            if img is not None:
                caption_bits = []
                
                # Check for any ID column to use in caption
                id_val = None
                for id_key in ["compound_id", "molecule_id", "molecule name", "id", "compound", "molecule_chembl_id"]:
                    for c in row.index:
                        if str(c).strip().lower() == id_key.lower():
                            id_val = str(row[c])
                            break
                    if id_val is not None:
                        break
                if id_val is not None:
                    caption_bits.append(id_val)
                    
                if "standard_value" in row and "standard_units" in row:
                    try:
                        val = float(row["standard_value"])
                        caption_bits.append(f"{val:.1f} {row['standard_units']}")
                    except Exception:
                        caption_bits.append(f"{row['standard_value']} {row['standard_units']}")
                elif "ic50_um" in row:
                    try:
                        val = float(row["ic50_um"])
                        caption_bits.append(f"{val:.3f} uM")
                    except Exception:
                        caption_bits.append(f"{row['ic50_um']}")
                        
                if "RO5_violations" in row:
                    caption_bits.append(f"RO5 violations: {row['RO5_violations']}")
                elif "lipinski_pass" in row:
                    caption_bits.append(f"Lipinski: {'Pass' if row['lipinski_pass'] else 'Fail'}")
                    
                if "chemotype_cluster" in row and pd.notna(row["chemotype_cluster"]):
                    caption_bits.append(f"Cluster: {int(row['chemotype_cluster'])}")
                if "LE" in row and pd.notna(row["LE"]):
                    try:
                        val = float(row["LE"])
                        caption_bits.append(f"LE: {val:.3f}")
                    except Exception:
                        caption_bits.append(f"LE: {row['LE']}")
                st.image(img, caption=" | ".join(caption_bits) or None)
            else:
                st.caption("Invalid SMILES")


# --- Sidebar Data Source Selection ---
st.sidebar.title("Data Source Setup")
data_source = st.sidebar.radio(
    "Data Source Selection",
    options=["ChEMBL Database (Live API)", "Generate Synthetic Dataset", "Upload Custom CSV/Excel"],
    index=0
)

local_df = None

if data_source == "ChEMBL Database (Live API)":
    st.sidebar.success("Connected to Live ChEMBL API.")
    set_data_source("chembl", None)
    st.session_state["dataset"] = None
    st.session_state["vector_store"] = None

elif data_source == "Generate Synthetic Dataset":
    import os
    import importlib
    import synthetic_data
    importlib.reload(synthetic_data)
    from synthetic_data import generate_synthetic_lims
    
    csv_path = "data/synthetic_data.csv"
    
    # Input target(s)
    targets_input = st.sidebar.text_input(
        "Assay Targets (comma-separated)",
        value="",
        placeholder="e.g. EGFR_Kinase, CDK4_Kinase, p38_MAPK",
        help="Specify the targets to distribute the 1,000 synthetic molecules across."
    )
    
    # Parse target names
    targets_list = [t.strip() for t in targets_input.split(",") if t.strip()]
    if not targets_list:
        targets_list = ["EGFR_Kinase", "CDK4_Kinase", "p38_MAPK"]
        
    generate_clicked = st.sidebar.button("Generate", type="primary")
    
    # Initialize state variable to store active synthetic data in memory for this session
    if "active_synthetic_df" not in st.session_state:
        st.session_state["active_synthetic_df"] = None
        
    if generate_clicked:
        with st.sidebar.spinner("Generating 100-molecule LIMS dataset..."):
            try:
                df_generated = generate_synthetic_lims(csv_path, targets=targets_list)
                st.session_state["active_synthetic_df"] = df_generated
                st.session_state["dataset"] = df_generated
                st.sidebar.success("Successfully generated new synthetic dataset!")
                
                # Build Vector RAG
                with st.sidebar.spinner("Building Vector RAG index..."):
                    try:
                        vectorstore = build_temp_vectorstore(df_generated)
                        st.session_state["vector_store"] = vectorstore
                        from agent_tools import set_local_vectorstore
                        set_local_vectorstore(vectorstore)
                        st.sidebar.success("Vector RAG index built successfully!")
                    except Exception as ve:
                        st.sidebar.warning(f"Could not build vector RAG: {ve}")
            except Exception as e:
                st.sidebar.error(f"Error generating data: {e}")
                
    # Load dataset if it has been generated in the current session
    local_df = st.session_state["active_synthetic_df"]
    if local_df is not None:
        st.session_state["dataset"] = local_df
        st.sidebar.success(f"Active synthetic dataset has {len(local_df)} records.")
        # Show targets in active dataset
        if "target_name" in local_df.columns:
            active_targets = list(local_df["target_name"].dropna().unique())
            st.sidebar.markdown(f"**Active Targets:** {', '.join(active_targets)}")
            
        # Preview in sidebar
        with st.sidebar.expander("Synthetic Data Preview", expanded=True):
            st.dataframe(local_df.head(5))
    else:
        st.sidebar.warning("No synthetic dataset active in this session. Click 'Generate' to build one.")
        
    set_data_source("synthetic", local_df)

elif data_source == "Upload Custom CSV/Excel":
    st.sidebar.info("Upload your custom dataset `.csv` or `.xlsx` files below:")
    uploaded_file = st.sidebar.file_uploader("Upload local CSV/Excel file", type=["csv", "xlsx"])
    if uploaded_file is not None:
        try:
            if uploaded_file.name.endswith(".csv"):
                local_df = pd.read_csv(uploaded_file)
            else:
                local_df = pd.read_excel(uploaded_file)
                
            st.sidebar.success(f"Loaded custom dataset: {len(local_df)} rows found.")
            st.session_state["dataset"] = local_df
            
            # Map columns / check target
            from synthetic_data import standardize_dataframe
            std_df = standardize_dataframe(local_df)
            
            st.sidebar.markdown("**Mapped Data Summary**:")
            st.sidebar.write({
                "Valid Molecules": len(std_df),
                "Targets Found": list(std_df["target"].unique())
            })
            
            # Build Vector RAG
            if "vector_store" not in st.session_state or st.session_state.get("vector_store_uploaded_name") != uploaded_file.name:
                with st.sidebar.spinner("Building Vector RAG index..."):
                    try:
                        vectorstore = build_temp_vectorstore(local_df)
                        st.session_state["vector_store"] = vectorstore
                        st.session_state["vector_store_uploaded_name"] = uploaded_file.name
                        from agent_tools import set_local_vectorstore
                        set_local_vectorstore(vectorstore)
                        st.sidebar.success("Vector RAG index built successfully!")
                    except Exception as ve:
                        st.sidebar.warning(f"Could not build vector RAG: {ve}")
            
            with st.sidebar.expander("Uploaded Data Preview", expanded=False):
                st.dataframe(local_df.head(5))
                
            set_data_source("custom", local_df)
        except Exception as e:
            st.sidebar.error(f"Error parsing uploaded file: {e}")
            set_data_source("custom", None)
            st.session_state["dataset"] = None
            st.session_state["vector_store"] = None
    else:
        set_data_source("custom", None)
        st.session_state["dataset"] = None
        st.session_state["vector_store"] = None


# --- Layout -------------------------------------------------------------

st.title("Hit-to-Lead Agent")

if data_source == "ChEMBL Database (Live API)":
    st.caption(
        "Ask about a target to fetch ChEMBL bioactivity data and check Lipinski "
        "Rule-of-Five compliance. Example: "
        "\"Get IC50 data for EGFR and flag Lipinski violations.\""
    )
    placeholder_text = "e.g. Find IC50 bioactivities for acetylcholinesterase and check drug-likeness"
else:
    available_targets = []
    if local_df is not None:
        target_col = None
        for col in ["target_name", "target", "target_id", "Target"]:
            if col in local_df.columns:
                target_col = col
                break
        if target_col is not None:
            available_targets = list(local_df[target_col].dropna().unique())
            
    targets_str = ", ".join(available_targets) if available_targets else "EGFR_Kinase, CDK4_Kinase, p38_MAPK"
    example_target = available_targets[0] if available_targets else "EGFR_Kinase"
    
    st.caption(
        f"Querying the local dataset. Available targets: {targets_str}. "
        f"Example: \"Find IC50 bioactivities for {example_target} and check Lipinski compliance.\""
    )
    placeholder_text = f"e.g. Find IC50 bioactivities for {example_target} and check drug-likeness"

prompt = st.text_area(
    "Prompt",
    placeholder=placeholder_text,
    height=80,
)

run_clicked = st.button("Run agent", type="primary")

if run_clicked:
    if not prompt.strip():
        st.warning("Enter a prompt first.")
    else:
        LAST_RESULTS.clear()
        try:
            agent = get_agent()
        except RuntimeError as e:
            st.error(str(e))
            st.stop()

        with st.spinner("Agent working ..."):
            try:
                from google.genai.errors import ClientError
                from langgraph.errors import GraphRecursionError

                from agent_tools import get_active_data_source, get_local_df
                active_source = get_active_data_source()
                local_df = get_local_df()
                
                context_prefix = ""
                if active_source in ("synthetic", "custom"):
                    if local_df is not None:
                        from synthetic_data import standardize_dataframe
                        try:
                            std_df = standardize_dataframe(local_df)
                            available_targets = list(std_df["target"].dropna().unique())
                            targets_str = ", ".join([str(t) for t in available_targets])
                        except Exception:
                            targets_str = "local targets"
                        context_prefix = (
                            f"[Context: You MUST query only the user's local dataset (source: {active_source}). "
                            f"The available target names in the local dataset are exactly: {targets_str}. "
                            "Do not search or fetch from the live ChEMBL database, only use the local dataset. "
                            "When describing your steps, refer to this as the 'local dataset'.] "
                        )
                        
                        # Inspect column names and datatypes so the agent knows where SMILES and metrics live
                        df_info = st.session_state.get("dataset")
                        if df_info is not None:
                            schema_info = []
                            for col in df_info.columns:
                                schema_info.append(f"'{col}' ({df_info[col].dtype})")
                            schema_str = ", ".join(schema_info)
                            context_prefix += (
                                f"[Dataset Schema: The uploaded LIMS dataset has columns: {schema_str}. "
                                "You MUST inspect these columns to understand where SMILES and IC50 metrics live. "
                                "For standard operations (such as target fetching, Lipinski Rule-of-Five checks, clustering, or SAR), "
                                "you MUST ALWAYS prefer calling the pre-built pipeline tools: `lookup_target_by_exact_name`, `fetch_bioactivities`, "
                                "`calculate_lipinski_for_dataset`, `cluster_dataset_by_chemotype`, and `perform_sar_analysis`. "
                                "Only use `query_uploaded_lims` for custom querying/filtering (like finding the molecule with the smallest/largest IC50). "
                                "Only report compound IDs and SMILES that are explicitly present in the tool outputs. "
                                "If you need to know which compound has a specific value (e.g. the smallest IC50), write code that returns the entire "
                                "row containing both the ID and the value (e.g. `df.nsmallest(1, 'ic50_um')`), rather than just the number.] "
                            )
                    else:
                        context_prefix = (
                            f"[Context: The active data source is '{active_source}', but no dataset has been generated or uploaded yet. "
                            "Politely inform the user that they must generate a synthetic dataset or upload a CSV first.] "
                        )
                else:
                    context_prefix = "[Context: You are querying the live ChEMBL database API.] "
                
                from agent_tools import set_data_source, set_local_vectorstore
                set_data_source(active_source, local_df)
                set_local_vectorstore(st.session_state.get("vector_store"))
                
                full_prompt = context_prefix + prompt
                result = run_agent(agent, full_prompt)
                
                # Main-thread synchronization for query_uploaded_lims to bypass caching/reloading issues
                try:
                    from agent_tools import extract_tool_trace
                    trace = extract_tool_trace(result)
                    for step in reversed(trace):
                        if step["tool"] == "query_uploaded_lims":
                            python_code = step["input"].get("python_code")
                            if python_code:
                                from agent_tools import execute_python_code
                                res, _ = execute_python_code(python_code, local_df)
                                df_to_save = None
                                if isinstance(res, pd.DataFrame):
                                    df_to_save = res
                                elif isinstance(res, pd.Series):
                                    df_to_save = pd.DataFrame([res])
                                
                                if df_to_save is None and _ is not None:
                                    dfs = [v for k, v in _.items() if isinstance(v, pd.DataFrame) and k != 'df']
                                    if dfs:
                                        df_to_save = dfs[-1]
                                    else:
                                        series = [v for k, v in _.items() if isinstance(v, pd.Series)]
                                        if series:
                                            df_to_save = pd.DataFrame([series[-1]])
                                
                                if df_to_save is not None:
                                    LAST_RESULTS["uploaded_query_results"] = df_to_save
                            break
                except Exception as se:
                    print(f"Sync error: {se}")
            except GraphRecursionError:
                st.error(
                    f"The agent took more than {MAX_AGENT_STEPS} steps without finishing — "
                    "this cap exists to protect your free-tier quota from a runaway query. "
                    "Try a more specific prompt (e.g. name the exact target and bioactivity type)."
                )
                st.stop()
            except ClientError as e:
                st.error(
                    "Gemini API rate limit or quota exceeded (free tier is capped at a low "
                    "requests-per-minute / requests-per-day allowance). The agent already "
                    "retried with backoff — try again in a minute, or in ~24h if it's the "
                    f"daily cap.\n\nDetails: {e}"
                )
                st.stop()

        st.subheader("Agent answer")
        ans = extract_final_answer(result)
        ans_str = ans if isinstance(ans, str) else str(ans)
        if not ans_str.strip():
            st.warning("Diagnostic: The extracted agent answer is empty. Here is the raw message trace:")
            for i, m in enumerate(result.get("messages", [])):
                st.code(f"Message {i} - Type: {getattr(m, 'type', None)}\nContent: {getattr(m, 'content', None)}\nTool calls: {getattr(m, 'tool_calls', None)}")
        st.write(ans_str)

        st.subheader("Agent reasoning trace")
        render_intermediate_steps(extract_tool_trace(result))

        render_results_tables()
else:
    st.info("Enter a prompt and click **Run agent** to get started.")
    render_results_tables()

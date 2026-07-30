# Hit to Lead Agent

A natural language assistant for medicinal chemists to run analysis on drug screen data, filter down compound collections, and select lead molecules. 

The agent connects structured compound records (from ChEMBL or custom/synthetic LIMS uploads) with chemical intelligence (RDKit Lipinski Rule-of-Five checks, scaffold clustering, and Ligand Efficiency SAR calculations) and semantic vector search (for unstructured comments/notes).

## Key Features

1. **Medicinal Chemistry Assistant**: Interacts with compound screens using natural language. For example: *"Which molecules targeting SMARCA2 have an IC50 < 1 uM and pass Lipinski rules?"* or *"Cluster the compounds and analyze the structure-activity relationship."*
2. **Unified Data Support**: Seamlessly queries the live **ChEMBL Database** or your own **Custom LIMS CSV/Excel** files. 
3. **In-Memory RAG & Pandas Execution**: Generates exact, secure pandas/RDKit operations on the fly for structured properties, and uses an in-memory **FAISS Vector RAG** to search free-text annotations/comments semantically.
4. **Interactive Dashboard**: Renders agent tool thought traces in real-time, displaying clean data tables and RDKit-drawn 2D molecular structures of the prioritized hits.
5. **Smart API Caching**: Automatically caches ChEMBL target searches and bioactivity fetches using Streamlit caching to speed up subsequent queries and save API quotas.

## Files

| File | Purpose |
|---|---|
| `app.py` | Main Streamlit application dashboard setting up page routing, sidebar uploads, main-thread synchronization, and molecular structures visualization. |
| `agent_tools.py` | Defines LangChain `@tool` interfaces (`fetch_bioactivities`, `query_uploaded_lims`, etc.), configures the system prompt, and builds the tool-calling agent. |
| `chembl_pipeline.py` | Fetches target metadata and bioactivity tables from the ChEMBL API, supported by in-memory query caching. |
| `synthetic_data.py` | Generates dummy LIMS datasets and standardizes custom CSV/Excel uploads (flexible mapping of column headers and unit conversion). |
| `lipinski_rules.py` | RDKit-based calculations for Molecular Weight, LogP, HBD, HBA, and Rule-of-Five violations. |
| `chemotype_clustering.py` | Groups molecules into chemotypes/families using Bemis-Murcko scaffolds or Butina similarity clustering. |
| `sar_analysis.py` | Computes Ligand Efficiency (LE) and ranks molecules/clusters to yield structural activity relationship summaries. |
| `admet_pred.py` | Runs ADMET-AI model predictions and generates polar radar plot visualizations of ADMET profile percentiles. |

## Setup

1. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```
2. Set your Gemini API key in your environment:
   ```bash
   export GEMINI_API_KEY="YOUR_API_KEY"
   ```

## Run the dashboard

Start the Streamlit application:
```bash
streamlit run app.py
```

## How the Agent Works

The agent runs on a LangGraph-based tool-calling engine, bound to specialized tools:
- **`lookup_target_by_exact_name` & `search_chembl_target`**: Resolves target names against ChEMBL or local LIMS targets.
- **`fetch_bioactivities`**: Loads structured compound activity data (IC50, Ki, EC50) from ChEMBL or the active local dataset.
- **`calculate_lipinski_for_dataset`**: Calculates drug-likeness parameters and flags Lipinski violations.
- **`cluster_dataset_by_chemotype`**: Clusters molecules by core scaffolds (Bemis-Murcko) or fingerprint similarity (Butina).
- **`perform_sar_analysis`**: Calculates Ligand Efficiency and outputs a summary group ranking.
- **`query_uploaded_lims`**: Executes Python code on local datasets for custom queries (e.g. finding minimum values, column filtering).
- **`search_uploaded_notes`**: Semantically searches text comments/notes in local uploads using FAISS embeddings.
- **`predict_compound_admet`**: Evaluates ADMET properties using admet_ai, caching the predictions and matplotlib radar chart for Streamlit UI rendering.

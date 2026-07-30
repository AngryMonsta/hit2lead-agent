"""
chembl_pipeline.py
------------------
Search ChEMBL for a target and pull its bioactivity data (default: IC50)
into a clean pandas DataFrame of [SMILES, molecule id, IC50 value, units].

Requires: chembl_webresource_client, pandas
    pip install chembl_webresource_client pandas
"""

from __future__ import annotations

import logging
from typing import Optional

import pandas as pd
from chembl_webresource_client.new_client import new_client

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

try:
    import streamlit as st
    cache_data = st.cache_data
except ImportError:
    def cache_data(*args, **kwargs):
        def decorator(func):
            return func
        return decorator


@cache_data(show_spinner=False)
def get_target_by_pref_name(pref_name: str) -> Optional[str]:
    """
    Resolve a target by its exact ChEMBL preferred name, e.g.
    'Voltage-gated inwardly rectifying potassium channel KCNH2' (hERG).

    Case-insensitive exact match on pref_name — more precise than
    `search_target()`'s free-text search when you already know the exact
    name ChEMBL uses. Returns the target_chembl_id, or None if no exact
    match exists (caller should fall back to `search_target()`).
    """
    target_client = new_client.target
    matches = target_client.filter(pref_name__iexact=pref_name).only("target_chembl_id")
    matches = list(matches)
    if not matches:
        return None
    return matches[0]["target_chembl_id"]


@cache_data(show_spinner=False)
def search_target(query: str, organism: Optional[str] = "Homo sapiens") -> pd.DataFrame:
    """
    Search ChEMBL for targets matching a free-text query (e.g. gene name,
    protein name, or synonym like 'EGFR', 'acetylcholinesterase').

    Returns a DataFrame of candidate targets so the caller/agent can pick
    the correct ChEMBL target id (target_chembl_id) when there are multiple
    matches (e.g. different species or target types).
    """
    target_client = new_client.target
    results = target_client.search(query)
    # Slice to top 20 hits to prevent endless pagination loops
    results_list = list(results[:20])
    df = pd.DataFrame(results_list)
    if df.empty:
        logger.warning("No targets found for query: %s", query)
        return df

    if organism is not None and "organism" in df.columns:
        filtered = df[df["organism"] == organism]
        if not filtered.empty:
            df = filtered

    keep_cols = [c for c in ["target_chembl_id", "pref_name", "organism", "target_type"] if c in df.columns]
    return df[keep_cols].reset_index(drop=True)


@cache_data(show_spinner=False)
def get_bioactivities(
    target_chembl_id: str,
    standard_type: str = "IC50",
    limit: Optional[int] = None,
) -> pd.DataFrame:
    """
    Pull bioactivity records for a given ChEMBL target id and standard_type
    (e.g. 'IC50', 'Ki', 'EC50'), and return a tidy DataFrame with:
        molecule_chembl_id, smiles, standard_value, standard_units, standard_type, assay_chembl_id

    Only records with a numeric standard_value and a resolvable SMILES are kept.
    """
    activity_client = new_client.activity
    query = activity_client.filter(
        target_chembl_id=target_chembl_id,
        standard_type=standard_type,
    ).only(
        "molecule_chembl_id",
        "canonical_smiles",
        "standard_value",
        "standard_units",
        "standard_type",
        "assay_chembl_id",
    )

    records = list(query[:limit]) if limit else list(query)
    df = pd.DataFrame(records)

    if df.empty:
        logger.warning("No %s bioactivities found for target %s", standard_type, target_chembl_id)
        return df

    df = df.rename(columns={"canonical_smiles": "smiles"})
    df["standard_value"] = pd.to_numeric(df["standard_value"], errors="coerce")
    df = df.dropna(subset=["smiles", "standard_value"])
    df = df.drop_duplicates(subset=["molecule_chembl_id", "smiles"]).reset_index(drop=True)

    return df[
        ["molecule_chembl_id", "smiles", "standard_value", "standard_units", "standard_type", "assay_chembl_id"]
    ]


def search_and_fetch(
    query: str,
    standard_type: str = "IC50",
    organism: Optional[str] = "Homo sapiens",
    limit: Optional[int] = None,
) -> pd.DataFrame:
    """
    Convenience one-shot: resolve a target name to a ChEMBL target id, then
    fetch its bioactivities. Good default path for an agent tool that just
    wants "give me data for EGFR" without a two-step lookup.

    Resolution order:
        1. Exact preferred-name match (fast, unambiguous — e.g. the full
           ChEMBL pref_name for hERG/KCNH2).
        2. Fuzzy free-text search, filtered to `organism` and taking the
           top hit, for everyday queries like 'EGFR' or 'hERG'.
    """
    target_id = get_target_by_pref_name(query)
    if target_id is not None:
        logger.info("Resolved '%s' -> %s via exact pref_name match", query, target_id)
        return get_bioactivities(target_id, standard_type=standard_type, limit=limit)

    targets = search_target(query, organism=organism)
    if targets.empty:
        return pd.DataFrame()

    target_id = targets.iloc[0]["target_chembl_id"]
    logger.info("Resolved '%s' -> %s (%s) via fuzzy search", query, target_id, targets.iloc[0].get("pref_name"))
    return get_bioactivities(target_id, standard_type=standard_type, limit=limit)


if __name__ == "__main__":
    # Quick manual smoke test
    df = search_and_fetch("EGFR", standard_type="IC50", limit=25)
    print(df.head())
    print(f"\nFetched {len(df)} bioactivity records.")

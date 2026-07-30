"""
chemotype_clustering.py
------------------------
Groups a set of molecules (SMILES) into chemotypes ("scaffold families") two
ways:

    1. Bemis-Murcko scaffold clustering — exact grouping by each molecule's
       core ring/linker scaffold (side chains stripped off). Fast, simple,
       and matches how medicinal chemists usually talk about "chemical
       series".
    2. Morgan fingerprint clustering (Butina algorithm) — groups molecules
       by overall structural (Tanimoto) similarity, which can pick up
       related chemotypes that don't share an identical scaffold.

Requires: rdkit, pandas
    pip install rdkit pandas
"""

from __future__ import annotations

from typing import Optional

import pandas as pd
from rdkit import Chem, DataStructs
from rdkit.Chem import rdFingerprintGenerator
from rdkit.Chem.Scaffolds import MurckoScaffold
from rdkit.ML.Cluster import Butina

_FP_GENERATOR = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)


def get_murcko_scaffold(smiles: str) -> Optional[str]:
    """Return the canonical SMILES of a molecule's Bemis-Murcko scaffold, or None if unparseable."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    scaffold = MurckoScaffold.GetScaffoldForMol(mol)
    return Chem.MolToSmiles(scaffold)


def get_morgan_fingerprint(smiles: str):
    """Return an RDKit ExplicitBitVect Morgan fingerprint (radius 2, 2048 bits) for a SMILES string, or None."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return _FP_GENERATOR.GetFingerprint(mol)


def cluster_by_scaffold(df: pd.DataFrame, smiles_col: str = "smiles") -> pd.DataFrame:
    """
    Add `scaffold_smiles` and `chemotype_cluster` columns by grouping
    molecules that share an identical Bemis-Murcko scaffold.

    `chemotype_cluster` is an integer id, 0 being the largest cluster
    (i.e. the most common scaffold in the dataset), so downstream
    summaries can talk about "cluster 0" as the dominant chemotype.
    Rows with an unparseable SMILES are dropped.
    """
    if smiles_col not in df.columns:
        raise ValueError(f"Column '{smiles_col}' not found in DataFrame")

    out = df.copy()
    out["scaffold_smiles"] = out[smiles_col].apply(get_murcko_scaffold)
    out = out.dropna(subset=["scaffold_smiles"]).reset_index(drop=True)

    # Rank scaffolds by cluster size (largest = cluster 0) for a stable, readable ordering.
    cluster_order = out["scaffold_smiles"].value_counts().index.tolist()
    scaffold_to_id = {scaffold: i for i, scaffold in enumerate(cluster_order)}
    out["chemotype_cluster"] = out["scaffold_smiles"].map(scaffold_to_id)
    return out


def cluster_by_fingerprint(df: pd.DataFrame, smiles_col: str = "smiles", cutoff: float = 0.4) -> pd.DataFrame:
    """
    Add a `chemotype_cluster` column via Butina clustering on Morgan
    fingerprints (Tanimoto distance). `cutoff` is the Butina distance
    threshold (lower = stricter/more clusters, typical range 0.2-0.5).

    `chemotype_cluster` is an integer id, 0 being the largest cluster.
    Rows with an unparseable SMILES are dropped.
    """
    if smiles_col not in df.columns:
        raise ValueError(f"Column '{smiles_col}' not found in DataFrame")

    out = df.copy()
    fps = out[smiles_col].apply(get_morgan_fingerprint)
    out = out[fps.notna()].reset_index(drop=True)
    fps = [fp for fp in fps if fp is not None]

    if not fps:
        return out.iloc[0:0]

    n = len(fps)
    distances = []
    for i in range(1, n):
        sims = DataStructs.BulkTanimotoSimilarity(fps[i], fps[:i])
        distances.extend(1 - s for s in sims)

    raw_clusters = Butina.ClusterData(distances, n, cutoff, isDistData=True)
    # raw_clusters: tuple of tuples of row-indices, already sorted largest-first by RDKit
    cluster_id = [None] * n
    for cid, member_indices in enumerate(raw_clusters):
        for idx in member_indices:
            cluster_id[idx] = cid

    out["chemotype_cluster"] = cluster_id
    return out


def cluster_molecules(df: pd.DataFrame, method: str = "scaffold", smiles_col: str = "smiles", cutoff: float = 0.4) -> pd.DataFrame:
    """
    Dispatch to `cluster_by_scaffold` or `cluster_by_fingerprint`.
    `method`: "scaffold" (Bemis-Murcko, exact grouping) or "fingerprint"
    (Morgan/Butina, similarity-based grouping). `cutoff` only applies to
    the fingerprint method.
    """
    if method == "scaffold":
        return cluster_by_scaffold(df, smiles_col=smiles_col)
    elif method == "fingerprint":
        return cluster_by_fingerprint(df, smiles_col=smiles_col, cutoff=cutoff)
    raise ValueError(f"Unknown clustering method '{method}'. Use 'scaffold' or 'fingerprint'.")


def cluster_summary(df: pd.DataFrame, cluster_col: str = "chemotype_cluster") -> pd.DataFrame:
    """
    Return a one-row-per-cluster summary DataFrame: cluster id, member
    count, and (if present) a representative scaffold_smiles.
    """
    if cluster_col not in df.columns:
        raise ValueError(f"Column '{cluster_col}' not found in DataFrame")

    agg = {"size": (cluster_col, "count")}
    grouped = df.groupby(cluster_col).size().reset_index(name="size")
    if "scaffold_smiles" in df.columns:
        rep = df.groupby(cluster_col)["scaffold_smiles"].first().reset_index()
        grouped = grouped.merge(rep, on=cluster_col)
    return grouped.sort_values("size", ascending=False).reset_index(drop=True)


if __name__ == "__main__":
    test_smiles = [
        "CC(=O)OC1=CC=CC=C1C(=O)O",  # aspirin
        "CC(=O)OC1=CC=CC=C1C(=O)OC",  # aspirin methyl ester analog (same scaffold)
        "CN1CCC[C@H]1c1cccnc1",  # nicotine
        "c1ccc2c(c1)ccc(=O)o2",  # coumarin-like
    ]
    df = pd.DataFrame({"smiles": test_smiles})
    print("--- scaffold clustering ---")
    print(cluster_by_scaffold(df)[["smiles", "scaffold_smiles", "chemotype_cluster"]])
    print("\n--- fingerprint clustering ---")
    print(cluster_by_fingerprint(df, cutoff=0.6)[["smiles", "chemotype_cluster"]])

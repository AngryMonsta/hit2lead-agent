"""
sar_analysis.py
----------------
Ligand efficiency (LE) calculation, ranking, and structure-activity
relationship (SAR) summary statistics.

Ligand efficiency (Hopkins et al. 2004 convention):
    LE = (1.37 * pActivity) / HeavyAtomCount
where pActivity = -log10(activity in molar units), e.g. pIC50 for IC50 data.
LE estimates binding energy per heavy atom (kcal/mol per atom) — useful for
comparing potency across molecules of different size, since raw IC50 favors
bigger/greasier molecules.

This module only computes numbers; it deliberately does NOT call an LLM.
`summarize_sar()` returns a compact, fact-grounded digest (counts, ranges,
medians per cluster) that an agent's LLM can turn into prose — this keeps
the actual figures accurate and avoids burning extra API requests on a
free-tier quota.

Requires: rdkit, pandas
    pip install rdkit pandas
"""

from __future__ import annotations

import math
from typing import Optional

import pandas as pd
from rdkit import Chem

# Multiplier to convert a given standard_units string to molar (M).
_UNITS_TO_MOLAR = {
    "M": 1.0,
    "mM": 1e-3,
    "uM": 1e-6,
    "µM": 1e-6,
    "nM": 1e-9,
    "pM": 1e-12,
}

LE_CONSTANT = 1.37  # kcal/mol, standard Hopkins et al. ligand efficiency constant


def _to_molar(value: float, units: str) -> Optional[float]:
    factor = _UNITS_TO_MOLAR.get(units)
    if factor is None or value is None or value <= 0:
        return None
    return value * factor


def calc_p_activity(value: float, units: str) -> Optional[float]:
    """Convert a standard_value/standard_units pair (e.g. 25, 'nM') to a pActivity (e.g. pIC50)."""
    molar = _to_molar(value, units)
    if molar is None:
        return None
    return -math.log10(molar)


def calc_ligand_efficiency(smiles: str, value: float, units: str) -> Optional[dict]:
    """
    Compute pActivity, heavy atom count, and ligand efficiency for one
    molecule. Returns None if the SMILES can't be parsed or the activity
    value/units can't be converted to molar.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    p_activity = calc_p_activity(value, units)
    if p_activity is None:
        return None

    heavy_atoms = mol.GetNumHeavyAtoms()
    if heavy_atoms == 0:
        return None

    le = (LE_CONSTANT * p_activity) / heavy_atoms
    return {
        "pActivity": round(p_activity, 3),
        "HeavyAtomCount": heavy_atoms,
        "LE": round(le, 4),
    }


def add_ligand_efficiency(
    df: pd.DataFrame,
    smiles_col: str = "smiles",
    value_col: str = "standard_value",
    units_col: str = "standard_units",
) -> pd.DataFrame:
    """
    Add pActivity, HeavyAtomCount, and LE columns to a bioactivity
    DataFrame. Rows that can't be converted (bad SMILES, unrecognized
    units, non-positive values) are dropped.
    """
    for col in (smiles_col, value_col, units_col):
        if col not in df.columns:
            raise ValueError(f"Column '{col}' not found in DataFrame")

    records = []
    for _, row in df.iterrows():
        result = calc_ligand_efficiency(row[smiles_col], row[value_col], row[units_col])
        if result is not None:
            records.append(result)
        else:
            records.append({"pActivity": None, "HeavyAtomCount": None, "LE": None})

    result_df = pd.DataFrame(records, index=df.index)
    out = pd.concat([df, result_df], axis=1)
    return out.dropna(subset=["LE"]).reset_index(drop=True)


def rank_by_ligand_efficiency(df: pd.DataFrame, ascending: bool = False) -> pd.DataFrame:
    """Sort by LE (descending by default, i.e. best/most efficient binders first)."""
    if "LE" not in df.columns:
        raise ValueError("DataFrame has no 'LE' column — call add_ligand_efficiency() first.")
    return df.sort_values("LE", ascending=ascending).reset_index(drop=True)


def summarize_sar(df: pd.DataFrame, cluster_col: Optional[str] = "chemotype_cluster") -> str:
    """
    Build a compact, numeric SAR digest: per-cluster (or whole-dataset, if
    no cluster column) potency range, LE range, MW/LogP range, and the
    single best compound by LE. Returns plain text meant to be handed to
    an LLM to turn into a narrative SAR summary — the LLM should not need
    to invent or recompute any numbers, just describe the trends in them.
    """
    required = {"LE", "pActivity"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"DataFrame is missing required columns for SAR summary: {missing}")

    def _group_digest(group: pd.DataFrame, label: str) -> str:
        lines = [f"{label}: n={len(group)}"]
        lines.append(
            f"  pActivity range {group['pActivity'].min():.2f}-{group['pActivity'].max():.2f} "
            f"(median {group['pActivity'].median():.2f})"
        )
        lines.append(
            f"  LE range {group['LE'].min():.3f}-{group['LE'].max():.3f} (median {group['LE'].median():.3f})"
        )
        if "MW" in group.columns:
            lines.append(f"  MW range {group['MW'].min():.1f}-{group['MW'].max():.1f}")
        if "LogP" in group.columns:
            lines.append(f"  LogP range {group['LogP'].min():.2f}-{group['LogP'].max():.2f}")
        best = group.sort_values("LE", ascending=False).iloc[0]
        best_id = best.get("molecule_id", best.get("molecule_chembl_id", best.get("smiles", "?")))
        lines.append(f"  Best LE compound: {best_id} (LE={best['LE']:.3f}, pActivity={best['pActivity']:.2f})")
        if "scaffold_smiles" in group.columns:
            lines.append(f"  Scaffold: {group['scaffold_smiles'].iloc[0]}")
        return "\n".join(lines)

    sections = []
    if cluster_col and cluster_col in df.columns:
        cluster_sizes = df[cluster_col].value_counts()
        for cluster_id in cluster_sizes.index:
            group = df[df[cluster_col] == cluster_id]
            sections.append(_group_digest(group, f"Cluster {cluster_id}"))
    else:
        sections.append(_group_digest(df, "Whole dataset (no clustering applied)"))

    header = f"SAR digest across {len(df)} molecules" + (
        f" in {df[cluster_col].nunique()} chemotype clusters:" if cluster_col in df.columns else ":"
    )
    return header + "\n\n" + "\n\n".join(sections)


if __name__ == "__main__":
    test_df = pd.DataFrame(
        {
            "molecule_chembl_id": ["CHEMBL1", "CHEMBL2", "CHEMBL3"],
            "smiles": [
                "CC(=O)OC1=CC=CC=C1C(=O)O",  # aspirin
                "CN1CCC[C@H]1c1cccnc1",  # nicotine
                "c1ccc2c(c1)ccc(=O)o2",  # coumarin
            ],
            "standard_value": [500, 20, 1200],
            "standard_units": ["nM", "nM", "nM"],
        }
    )
    annotated = add_ligand_efficiency(test_df)
    ranked = rank_by_ligand_efficiency(annotated)
    print(ranked[["molecule_chembl_id", "pActivity", "HeavyAtomCount", "LE"]])
    print()
    print(summarize_sar(ranked, cluster_col=None))

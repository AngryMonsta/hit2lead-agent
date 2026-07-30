"""
lipinski_rules.py
------------------
RDKit-based wrapper that computes Lipinski "Rule of Five" descriptors
(Molecular Weight, LogP, H-Bond Donors, H-Bond Acceptors) for one SMILES
string or a whole DataFrame column of them, and flags rule violations.

Rule of Five (violated if 2+ of these are broken):
    MW  <= 500
    LogP <= 5
    HBD <= 5
    HBA <= 10

HBD/HBA use RDKit's Lipinski.NumHDonors/NumHAcceptors, a SMARTS-based
H-bonding definition (distinct from the raw NH/OH and N/O atom counts used
in Lipinski's original 1997 paper, via Descriptors.NHOHCount/NOCount).

Requires: rdkit
    pip install rdkit
"""

from __future__ import annotations

from typing import Optional

import pandas as pd
from rdkit import Chem
from rdkit.Chem import Crippen, Descriptors, Lipinski


def calc_lipinski(smiles: str) -> Optional[dict]:
    """
    Compute Lipinski descriptors for a single SMILES string.
    Returns None if the SMILES cannot be parsed by RDKit.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    mw = Descriptors.MolWt(mol)
    logp = Crippen.MolLogP(mol)
    hbd = Lipinski.NumHDonors(mol)
    hba = Lipinski.NumHAcceptors(mol)

    violations = sum([mw > 500, logp > 5, hbd > 5, hba > 10])

    return {
        "smiles": smiles,
        "MW": round(mw, 2),
        "LogP": round(logp, 2),
        "HBD": hbd,
        "HBA": hba,
        "RO5_violations": violations,
        "RO5_pass": violations <= 1,  # standard convention: <=1 violation is acceptable
    }


def calc_lipinski_batch(smiles_list: list[str]) -> pd.DataFrame:
    """
    Compute Lipinski descriptors for a list of SMILES strings.
    Invalid SMILES are dropped and logged via a `valid` column set to False
    before being filtered out (kept simple: they're just skipped here).
    """
    rows = []
    for smi in smiles_list:
        result = calc_lipinski(smi)
        if result is not None:
            rows.append(result)
    return pd.DataFrame(rows)


def annotate_dataframe(df: pd.DataFrame, smiles_col: str = "smiles") -> pd.DataFrame:
    """
    Take an existing DataFrame (e.g. output of chembl_pipeline.get_bioactivities)
    and join in Lipinski descriptor columns computed from its SMILES column.
    Rows with unparseable SMILES are dropped from the merged result.
    """
    if smiles_col not in df.columns:
        raise ValueError(f"Column '{smiles_col}' not found in DataFrame")

    unique_smiles = list(set(df[smiles_col].tolist()))
    lipinski_df = calc_lipinski_batch(unique_smiles)
    if lipinski_df.empty:
        return df.iloc[0:0]  # nothing valid, return empty frame with original shape hint

    merged = df.merge(lipinski_df, left_on=smiles_col, right_on="smiles", how="inner", suffixes=("", "_calc"))
    if smiles_col != "smiles":
        merged = merged.drop(columns=["smiles"])
    return merged


if __name__ == "__main__":
    test_smiles = [
        "CC(=O)OC1=CC=CC=C1C(=O)O",  # aspirin
        "CN1CCC[C@H]1c1cccnc1",  # nicotine
        "not_a_real_smiles",
    ]
    print(calc_lipinski_batch(test_smiles))

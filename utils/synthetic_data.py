"""
synthetic_data.py
-----------------
Generates a realistic 1,000-molecule synthetic LIMS export using provided templates,
and provides utility functions to load, generate, and standardize local datasets for the agent.
"""

from __future__ import annotations

import os
import random
from datetime import datetime, timedelta
from typing import Optional

import numpy as np
import pandas as pd

# Set seed for reproducibility
np.random.seed(42)
random.seed(42)

# --- 1. Representative Molecular Scaffolds (SMILES) ---
SMILES_TEMPLATES = [
    "Cc1ccc(nc1)Nc2cc(nn2C)c3ccc(cc3)F",              # Kinase-like pyrazole
    "c1ccc(cc1)C(=O)Nc2ccc(cc2)S(=O)(=O)N",           # Sulfonamide core
    "COc1cc2c(nc(nc2cc1OC)N3CCN(CC3)C(=O)C4CCCO4)N",   # Quinazoline scaffold
    "CC1CCN(CC1)c2ccc(cc2)Nc3ncc(c(n3)C4=CCCCCC4)F",  # Aminopyrimidine
    "c1ccc2c(c1)nc(s2)NC(=O)c3ccc(cc3)Cl",             # Benzothiazole amide
    "CN1CCN(CC1)Cc2ccc(cc2)C(=O)Nc3cccc(c3)C(F)(F)F",  # Piperazine benzamide
    "COc1ccc2c(c1)c(c(n2)C)CC(=O)O",                  # Indole acetic acid core
    "c1ccc(cc1)CN2CCN(CC2)c3cccc(c3)Cl",              # Phenylpiperazine
]


def generate_synthetic_smiles() -> str:
    """Generates slight variations of SMILES templates to simulate analogs, ensuring RDKit validity."""
    from rdkit import Chem
    for _ in range(50):
        base = random.choice(SMILES_TEMPLATES)
        replacements = [("Cl", "F"), ("c1ccc", "c1ccn"), ("OC", "OCC"), ("C(=O)O", "C(=O)N")]
        if random.random() > 0.5:
            orig, sub = random.choice(replacements)
            if orig in base:
                modified = base.replace(orig, sub, 1)
                # Ensure the modified SMILES can be successfully parsed by RDKit
                if Chem.MolFromSmiles(modified) is not None:
                    return modified
        if Chem.MolFromSmiles(base) is not None:
            return base
    return SMILES_TEMPLATES[0]


def generate_synthetic_lims(output_path: Optional[str] = None, targets: Optional[list[str]] = None) -> pd.DataFrame:
    """Generates 100 LIMS Compound Records based on the reference template."""
    if not targets:
        targets = ["EGFR_Kinase", "CDK4_Kinase", "p38_MAPK"]
        
    # Ensure reproducibility for each generation call if needed
    np.random.seed(42)
    random.seed(42)

    n_samples = 100

    # Compound Identifiers
    cmpd_ids = [f"CMPD-2026-{i+10000:05d}" for i in range(n_samples)]
    batch_ids = [f"B-{random.randint(1, 3):02d}" for _ in range(n_samples)]

    # Assay Targets and Dates
    assigned_targets = [random.choice(targets) for _ in range(n_samples)]

    start_date = datetime(2026, 1, 15)
    assay_dates = [(start_date + timedelta(days=random.randint(0, 180))).strftime("%Y-%m-%d") for _ in range(n_samples)]

    # Bioactivity Data (Log-normal distribution simulating HTS/Hit Expansion screen)
    # IC50 values range from 0.001 uM (1 nM) to 100 uM
    log_ic50 = np.random.normal(loc=0.5, scale=1.2, size=n_samples)  # Mean ~3.16 uM
    ic50_um = np.round(10**log_ic50, 4)
    ic50_um = np.clip(ic50_um, 0.0008, 100.0)  # Clip between 0.8 nM and 100 uM

    # Calculate pIC50 (-log10(IC50 in Molar))
    pic50 = np.round(-np.log10(ic50_um * 1e-6), 2)

    # Curve Fitting & QC Metrics
    # High potency hits usually have better curve fits in quality screens
    r2_fit = np.round(np.clip(np.random.beta(a=8, b=1.5, size=n_samples) * (pic50 / 8.0) + 0.1, 0.35, 0.99), 3)
    hill_slope = np.round(np.random.normal(loc=1.05, scale=0.3, size=n_samples), 2)
    max_inhibition_pct = np.round(np.clip(np.random.normal(loc=88, scale=12, size=n_samples), 15.0, 100.0), 1)

    # Flag potential PAINS/Aggregators based on steep hill slopes or poor curve fits
    curve_qc_pass = (r2_fit >= 0.85) & (hill_slope >= 0.5) & (hill_slope <= 2.0)

    # Generate SMILES & Molecular Descriptors
    smiles = [generate_synthetic_smiles() for _ in range(n_samples)]
    mw = np.round(np.random.normal(loc=380, scale=65, size=n_samples), 1)
    logp = np.round(np.random.normal(loc=3.2, scale=1.1, size=n_samples), 2)

    # Construct DataFrame
    lims_df = pd.DataFrame({
        "compound_id": cmpd_ids,
        "batch_id": batch_ids,
        "target_name": assigned_targets,
        "canonical_smiles": smiles,
        "ic50_um": ic50_um,
        "pic50": pic50,
        "max_inhibition_pct": max_inhibition_pct,
        "r2_fit": r2_fit,
        "hill_slope": hill_slope,
        "curve_qc_pass": curve_qc_pass,
        "mw_da": mw,
        "clogp": logp,
        "assay_date": assay_dates
    })

    if output_path:
        dir_name = os.path.dirname(output_path)
        if dir_name:
            os.makedirs(dir_name, exist_ok=True)
        lims_df.to_csv(output_path, index=False)

    return lims_df


def get_original_id_col(df: pd.DataFrame) -> str:
    """Find the column in df that matches our molecule_id / compound_id criteria."""
    mappings = ["molecule_id", "molecule_chembl_id", "compound_id", "molecule name", "id", "compound"]
    for option in mappings:
        for col in df.columns:
            if str(col).strip().lower() == option.lower():
                return col
    return "molecule_id"


def standardize_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Standardize a local dataset (synthetic or custom) to match the schema
    expected by downstream pipeline steps (like RDKit Lipinski calculations and molecule drawing).

    Expected final columns:
        molecule_id, smiles, standard_value, standard_units, standard_type, assay_id, target

    We perform flexible case-insensitive mapping for common column names.
    """
    std_df = df.copy()

    # Column mapping configurations (case-insensitive keys mapped to standard target name)
    mappings = {
        "molecule_id": ["molecule_id", "compound_id", "molecule_chembl_id", "molecule name", "id", "compound"],
        "smiles": ["smiles", "canonical_smiles", "structure"],
        "standard_value": ["standard_value", "ic50_um", "value", "ic50", "activity_value", "activity", "potency"],
        "standard_units": ["standard_units", "units", "activity_units"],
        "standard_type": ["standard_type", "type", "activity_type", "assay_type"],
        "assay_id": ["assay_id", "assay_chembl_id", "eln_reference", "eln", "assay"],
        "target": ["target", "target_name", "target_id", "gene", "protein"]
    }

    # Find matches and rename columns
    rename_dict = {}
    for std_col, options in mappings.items():
        found = False
        for option in options:
            for col in std_df.columns:
                if str(col).strip().lower() == option.lower():
                    rename_dict[col] = std_col
                    found = True
                    break
            if found:
                break

    std_df = std_df.rename(columns=rename_dict)

    # Ensure smiles exists
    if "smiles" not in std_df.columns:
        # Fallback: find any column containing "smiles"
        for col in std_df.columns:
            if "smiles" in str(col).lower():
                std_df = std_df.rename(columns={col: "smiles"})
                break

    # If it is standard_value, check if it was originally 'ic50_um' or if values are in uM.
    # If the renamed column was originally 'ic50_um', or if the values are generally small (< 100)
    # and units are micromolar, convert to nM.
    is_um = False
    original_value_col = None
    for k, v in rename_dict.items():
        if v == "standard_value":
            original_value_col = k
            break

    if original_value_col and "um" in str(original_value_col).lower():
        is_um = True
    elif "standard_units" in std_df.columns:
        first_unit = str(std_df["standard_units"].iloc[0]).lower() if not std_df.empty else ""
        if "um" in first_unit or "micromolar" in first_unit:
            is_um = True

    # Coerce to numeric
    if "standard_value" in std_df.columns:
        std_df["standard_value"] = pd.to_numeric(std_df["standard_value"], errors="coerce")
        if is_um:
            # Convert micromolar (uM) to nanomolar (nM)
            std_df["standard_value"] = std_df["standard_value"] * 1000.0
            std_df["standard_units"] = "nM"
        elif "standard_units" not in std_df.columns:
            std_df["standard_units"] = "nM"
    else:
        # Create dummy standard_value if missing
        std_df["standard_value"] = 0.0
        std_df["standard_units"] = "N/A"

    # Default missing standard columns
    if "molecule_id" not in std_df.columns:
        std_df["molecule_id"] = [f"COMP-{i+1}" for i in range(len(std_df))]
    if "standard_type" not in std_df.columns:
        std_df["standard_type"] = "IC50"
    if "assay_id" not in std_df.columns:
        std_df["assay_id"] = "LOCAL-ASSAY"
    if "target" not in std_df.columns:
        std_df["target"] = "CustomTarget"

    # Clean up and ensure we have at least these columns
    cols_to_keep = ["molecule_id", "smiles", "standard_value", "standard_units", "standard_type", "assay_id", "target"]
    for c in cols_to_keep:
        if c not in std_df.columns:
            std_df[c] = None

    # Keep all other original columns as well, putting standard columns first
    other_cols = [c for c in std_df.columns if c not in cols_to_keep]
    final_cols = cols_to_keep + other_cols
    return std_df[final_cols].dropna(subset=["smiles"]).reset_index(drop=True)


if __name__ == "__main__":
    df = generate_synthetic_lims("data/synthetic_data.csv")
    print(f"Generated synthetic dataset with shape {df.shape}")
    std_df = standardize_dataframe(df)
    print("Standardized schema snippet:")
    print(std_df.head(2))

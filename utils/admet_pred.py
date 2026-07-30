import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from admet_ai import ADMETModel
from rdkit import Chem
from rdkit.Chem import Draw

# Initialize the model once
ADMET_MODEL = ADMETModel()


def generate_admet_card_native(
    smiles: str,
    molecule_title: str = "Molecule 1",
) -> tuple[plt.Figure, pd.DataFrame]:
    """Generates the ADMET radar chart alongside an RDKit 2D structure.

    Returns the matplotlib figure and predictions dataframe.
    """
    # 1. Run ADMET-AI prediction
    preds_dict = ADMET_MODEL.predict(smiles=smiles)
    
    # Standardize predictions to a DataFrame for return and displaying in st.dataframe
    preds_df = pd.DataFrame([preds_dict])

    # 2. Setup a dual-panel figure (Radar Plot + Molecule Structure)
    fig = plt.figure(figsize=(10, 5), facecolor="#e6f4f8")

    # --- Left Panel: Radar Plot ---
    ax1 = fig.add_subplot(121, polar=True)
    
    # Implement the radial plot summary logic
    max_percentile = 100
    percentile_suffix = "drugbank_approved_percentile"
    
    properties = {
        "Blood-Brain Barrier Safe": {
            "percentile": max_percentile - preds_dict[f"BBB_Martins_{percentile_suffix}"],
        },
        "Non-\nToxic": {
            "percentile": max_percentile - preds_dict[f"ClinTox_{percentile_suffix}"],
            "vertical_alignment": "bottom",
        },
        "Soluble": {
            "percentile": preds_dict[f"Solubility_AqSolDB_{percentile_suffix}"],
            "vertical_alignment": "top",
        },
        "Bioavailable": {
            "percentile": preds_dict[f"Bioavailability_Ma_{percentile_suffix}"],
            "vertical_alignment": "top",
        },
        "hERG\nSafe": {
            "percentile": max_percentile - preds_dict[f"hERG_{percentile_suffix}"],
            "vertical_alignment": "bottom",
        },
    }
    
    property_names = list(properties.keys())
    percentiles = [properties[name]["percentile"] for name in property_names]
    
    # Angles start at pi / 2 and go counter-clockwise
    angles = ((np.linspace(0, 2 * np.pi, len(properties), endpoint=False) + np.pi / 2) % (2 * np.pi)).tolist()
    
    percentiles += percentiles[:1]
    angles += angles[:1]
    
    # Plot the radar data
    ax1.fill(angles, percentiles, color="red", alpha=0.25)
    ax1.plot(angles, percentiles, color="red", linewidth=2)
    ax1.set_ylim(0, 100)
    
    yticks = [0, 25, 50, 75, 100]
    yticklabels = [str(ytick) for ytick in yticks]
    ax1.set_yticks(yticks)
    ax1.set_yticklabels(yticklabels)
    ax1.set_rlabel_position(335)
    
    ax1.set_xticks(angles[:-1])
    ax1.set_xticklabels(property_names)
    
    for label, property_name in zip(ax1.get_xticklabels(), property_names):
        if "vertical_alignment" in properties[property_name]:
            label.set_verticalalignment(properties[property_name]["vertical_alignment"])
            
    ax1.set_aspect("equal", "box")

    # --- Right Panel: RDKit 2D Molecule Drawing ---
    ax2 = fig.add_subplot(122)
    ax2.axis("off")

    mol = Chem.MolFromSmiles(smiles)
    if mol:
        img = Draw.MolToImage(mol, size=(400, 400))
        ax2.imshow(img)

    # Set overall layout title
    plt.suptitle(
        f"{molecule_title}: {smiles[:35]}...",
        fontsize=14,
        weight="bold",
        y=0.98,
    )
    plt.tight_layout()

    return fig, preds_df

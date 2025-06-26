# diffcrysgen/analyze_generated_structures.py

import numpy as np
import pandas as pd
import spglib
from diffcrysgen.invert_pcr import InvertPCR
from ase import Atoms
import ase.io
import os

def analyze_ircr_data(generated_ircr_path="generated_ircr.npy", output_csv="generated_material_data.csv", min_valid_distance=0.5, save_cifs=False, cif_dir="cif-files"):
    """
    Analyzes generated IRCR data and extracts crystallographic info.

    Args:
        generated_ircr_path (str): Path to the generated IRCR .npy file
        output_csv (str): Path to save the resulting CSV
        min_valid_distance (float): Distance threshold for validity check in Angstrom
        save_cifs (bool): If True, saves CIF files into subfolders
        cif_dir (str): Base directory to store valid/invalid CIFs
    """
    data = np.load(generated_ircr_path)
    Nmat = data.shape[0]
    print(f"Loaded {Nmat} materials from {generated_ircr_path}")

    results = {
        "ID": [], "Formula": [], "type": [], "Natoms": [], "min-pairwise-distance": [],
        "Zmax": [], "SPG-symbol": [], "SPG-number": [], "validity": [],
        "a": [], "b": [], "c": [], "alpha": [], "beta": [], "gamma": []
    }

    if save_cifs:
        os.makedirs(os.path.join(cif_dir, "valid"), exist_ok=True)
        os.makedirs(os.path.join(cif_dir, "invalid"), exist_ok=True)

    for i in range(Nmat):
        mat = data[i, :]
        ipcr = InvertPCR(mat, 94, 20)

        try:
            atoms = ipcr.get_atoms_object()
            formula = atoms.get_chemical_formula()
            atomic_numbers = atoms.get_atomic_numbers()
            unique_elements = len(set(atomic_numbers))

            mat_type = {1: "elemental", 2: "binary", 3: "ternary"}.get(unique_elements, "complex")
            a, b, c = atoms.cell.lengths()
            alpha, beta, gamma = atoms.cell.angles()

            structure = (atoms.get_cell(), atoms.get_scaled_positions(), atomic_numbers)
            spg_symbol, spg_number = spglib.get_spacegroup(structure, symprec=0.1).split(" ")
            spg_number = int(spg_number.strip("()"))

            distances = ipcr.get_distances()
            min_dist = min(distances) if distances else 3.0
            is_valid = min_dist >= min_valid_distance
            mat_id = f"smps-{i+1}"

            if save_cifs:
                cif_path = os.path.join(cif_dir, "valid" if is_valid else "invalid", f"{mat_id}.cif")
                ase.io.write(cif_path, atoms)

            # Append data
            results["ID"].append(mat_id)
            results["Formula"].append(formula)
            results["type"].append(mat_type)
            results["Natoms"].append(len(atomic_numbers))
            results["min-pairwise-distance"].append(min_dist)
            results["Zmax"].append(max(atomic_numbers))
            results["SPG-symbol"].append(spg_symbol)
            results["SPG-number"].append(spg_number)
            results["validity"].append("valid" if is_valid else "invalid")
            results["a"].append(a)
            results["b"].append(b)
            results["c"].append(c)
            results["alpha"].append(alpha)
            results["beta"].append(beta)
            results["gamma"].append(gamma)

        except Exception as e:
            #print(f"[Material {i}] Error: {e}")
            print("spglib failed to assign spg")

    df = pd.DataFrame(results)
    df.to_csv(output_csv, index=False)
    print(f"Saved extracted data to: {output_csv}")
    return df


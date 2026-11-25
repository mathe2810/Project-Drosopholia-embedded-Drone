#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
explore_connectome.py - Liste les types cellulaires dans le connectome MaleCNS
pour aider à configurer les groupes d'entrée/sortie.
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
from scipy import sparse

FLYDRONES_SRC = Path(__file__).resolve().parent / "src"
if str(FLYDRONES_SRC) not in sys.path:
    sys.path.insert(0, str(FLYDRONES_SRC))

from flydrones.brain import Connectome  # noqa: E402

def main():
    npz_path = Path(__file__).resolve().parent / "data" / "malecns_brain.npz"
    
    if not npz_path.exists():
        print(f"ERREUR : {npz_path} n'existe pas")
        print("Exécutez d'abord prepare_brain.py")
        return 1
    
    print(f"Chargement du connectome : {npz_path}")
    c = Connectome.load(str(npz_path))
    print(f"\n=== Connectome : {c.name} ===")
    print(f"Neurones : {c.n:,}")
    print(f"Connexions : {c.n_connections:,}")
    
    # Types cellulaires uniques
    types = c.types.astype(str)
    unique_types = np.unique(types)
    print(f"\n=== Types cellulaires uniques ({len(unique_types)}) ===")
    
    # Trier et afficher par groupes
    for t in sorted(unique_types):
        count = int((types == t).sum())
        print(f"  {t:30s} : {count:6d} neurones")
    
    # Chercher les types qui pourraient correspondre à nos groupes
    print("\n=== Recherche de correspondances pour les groupes FlyDrones ===")
    
    search_terms = {
        "Photorecepteurs (R1-R6)": ["R1-R6", "R1", "R6", "photoreceptor"],
        "T4/T5 (motion)": ["T4", "T5"],
        "LPLC2 (looming)": ["LPLC2"],
        "LC4 (looming speed)": ["LC4"],
        "Haltere": ["haltere", "Haltere"],
        "HS (horizontal)": ["HS"],
        "VS (vertical)": ["VS"],
        "DNg02 (wing stroke)": ["DNg02"],
        "DNp03 (saccade)": ["DNp03"],
        "DNp01 (giant fiber)": ["DNp01", "GF"],
    }
    
    for group_name, terms in search_terms.items():
        print(f"\n  {group_name}:")
        found = False
        for term in terms:
            matches = [t for t in unique_types if term.lower() in t.lower()]
            if matches:
                found = True
                for m in matches[:10]:  # max 10 par terme
                    count = int((types == m).sum())
                    print(f"    -> {m:30s} : {count:6d} neurones")
        if not found:
            print("    (aucun match trouvé)")
    
    # Côtés (L/R)
    sides = c.sides.astype(str)
    unique_sides = np.unique(sides)
    print(f"\n=== Côtés uniques : {unique_sides} ===")
    
    # Superclass
    if c.superclass is not None:
        sc = c.superclass.astype(str)
        unique_sc = np.unique(sc)
        print(f"\n=== Superclasses uniques ({len(unique_sc)}) ===")
        for s in sorted(unique_sc):
            count = int((sc == s).sum())
            print(f"  {s:30s} : {count:6d} neurones")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())

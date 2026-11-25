#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
reduce_malecns.py - Réduit le connectome MaleCNS avec les bons noms de types cellulaires.

Ce script :
1. Charge le connectome MaleCNS complet (166k neurones)
2. Résout manuellement les groupes d'entrée/sortie avec les noms réels de MaleCNS
3. Définit meta['roles'] pour permettre sensorimotor_core
4. Applique la réduction (hops=3, max 5000 neurones)
5. Sauvegarde le connectome réduit pour utilisation rapide
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np

FLYDRONES_SRC = Path(__file__).resolve().parent / "src"
if str(FLYDRONES_SRC) not in sys.path:
    sys.path.insert(0, str(FLYDRONES_SRC))

from flydrones.brain import Connectome, Brain  # noqa: E402
from flydrones.config import load_config  # noqa: E402

def main():
    npz_path = Path(__file__).resolve().parent / "data" / "malecns_brain.npz"
    out_path = Path(__file__).resolve().parent / "data" / "malecns_brain_reduced.npz"
    
    if not npz_path.exists():
        print(f"ERREUR : {npz_path} n'existe pas")
        print("Exécutez d'abord prepare_brain.py")
        return 1
    
    print(f"Chargement du connectome : {npz_path}")
    connectome = Connectome.load(str(npz_path))
    print(f"Connectome : {connectome.name} ({connectome.n:,} neurones, {connectome.n_connections:,} connexions)")
    
    # Définir manuellement les groupes d'entrée avec les VRAIS noms de MaleCNS
    # Basé sur l'exploration : T4a, T4b, T4c, T4d, LPLC2, LC4, R1-R6
    input_groups = {
        "R16_L":   {"types": ["^R1-R6$"], "side": None, "feature": "brightness", "eye": "L", "max_hz": 40},
        "R16_R":   {"types": ["^R1-R6$"], "side": None, "feature": "brightness", "eye": "R", "max_hz": 40},
        "T4a_L":   {"types": ["^T4a$"],   "side": None, "feature": "ftb",        "eye": "L", "max_hz": 120},
        "T4a_R":   {"types": ["^T4a$"],   "side": None, "feature": "ftb",        "eye": "R", "max_hz": 120},
        "T4b_L":   {"types": ["^T4b$"],   "side": None, "feature": "btf",        "eye": "L", "max_hz": 120},
        "T4b_R":   {"types": ["^T4b$"],   "side": None, "feature": "btf",        "eye": "R", "max_hz": 120},
        "T4c_L":   {"types": ["^T4c$"],   "side": None, "feature": "up",         "eye": "L", "max_hz": 120},
        "T4c_R":   {"types": ["^T4c$"],   "side": None, "feature": "up",         "eye": "R", "max_hz": 120},
        "T4d_L":   {"types": ["^T4d$"],   "side": None, "feature": "down",       "eye": "L", "max_hz": 120},
        "T4d_R":   {"types": ["^T4d$"],   "side": None, "feature": "down",       "eye": "R", "max_hz": 120},
        "LPLC2_L": {"types": ["^LPLC2$"], "side": None, "feature": "loom",       "eye": "L", "max_hz": 200},
        "LPLC2_R": {"types": ["^LPLC2$"], "side": None, "feature": "loom",       "eye": "R", "max_hz": 200},
        "LC4_L":   {"types": ["^LC4$"],   "side": None, "feature": "loom_speed", "eye": "L", "max_hz": 150},
        "LC4_R":   {"types": ["^LC4$"],   "side": None, "feature": "loom_speed", "eye": "R", "max_hz": 150},
    }
    
    # Groupes de sortie
    output_groups = {
        "DNg02_L": {"types": ["^DNg02"], "side": None, "note": "wing-stroke amplitude"},
        "DNg02_R": {"types": ["^DNg02"], "side": None, "note": "wing-stroke amplitude"},
        "DNp03_L": {"types": ["^DNp03$"], "side": None, "note": "flight saccades"},
        "DNp03_R": {"types": ["^DNp03$"], "side": None, "note": "flight saccades"},
        "DNp01_L": {"types": ["^DNp01$", "^GFC"], "side": None, "note": "giant fiber escape"},
        "DNp01_R": {"types": ["^DNp01$", "^GFC"], "side": None, "note": "giant fiber escape"},
    }
    
    # Résoudre les groupes manuellement
    print("\nRésolution des groupes d'entrée...")
    for name, spec in input_groups.items():
        idx = connectome.select(spec["types"], side=spec["side"])
        print(f"  {name:15s}: {idx.size:6d} neurones (types={spec['types']})")
    
    print("\nRésolution des groupes de sortie...")
    for name, spec in output_groups.items():
        idx = connectome.select(spec["types"], side=spec["side"])
        print(f"  {name:15s}: {idx.size:6d} neurones (types={spec['types']})")
    
    # Définir meta['roles'] pour sensorimotor_core
    print("\nDéfinition de meta['roles']...")
    connectome.meta.setdefault("roles", {})
    for name in input_groups:
        connectome.meta["roles"][name] = "input"
    for name in output_groups:
        connectome.meta["roles"][name] = "output"
    
    # Résoudre les groupes dans le connectome
    all_specs = {**input_groups, **output_groups}
    group_specs = {}
    for name, spec in all_specs.items():
        from flydrones.brain.connectome import GroupSpec
        group_specs[name] = GroupSpec.from_dict(name, spec)
        group_specs[name].role = "input" if name in input_groups else "output"
    
    connectome.resolve_groups(group_specs)
    
    # Vérifier que les groupes ont des neurones
    total_input = sum(connectome.group(name).size for name in input_groups)
    total_output = sum(connectome.group(name).size for name in output_groups)
    print(f"\nTotal neurones d'entrée : {total_input}")
    print(f"Total neurones de sortie : {total_output}")
    
    if total_input == 0 or total_output == 0:
        print("ERREUR : Certains groupes sont vides, impossible de réduire")
        return 1
    
    # Appliquer la réduction sensorimotor_core
    print("\nApplication de sensorimotor_core (hops=3, max_neurons=5000)...")
    try:
        core = connectome.sensorimotor_core(hops=3, max_neurons=5000)
        print(f"Connectome réduit : {core.summary()}")
    except Exception as e:
        print(f"Erreur lors de la réduction : {e}")
        print("Sauvegarde du connectome complet...")
        core = connectome
    
    # Sauvegarder
    print(f"\nSauvegarde vers {out_path}...")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    core.save(str(out_path))
    print(f"✓ Sauvegarde terminée : {out_path}")
    print(f"  Neurones : {core.n:,}")
    print(f"  Connexions : {core.n_connections:,}")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())

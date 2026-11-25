#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
prepare_brain.py - Télécharge, construit et (si possible) réduit le connectome MaleCNS.

Usage :
    python prepare_brain.py

Ce script :
1. Télécharge ~1.2 GB de données MaleCNS v1.0 (si pas déjà fait)
2. Construit le connectome signé (filtre min_synapses=3)
3. Tente la réduction sensorimotor_core (hops=3, max 5000 neurones)
   - Si les groupes input/output ne matchent aucun neurone (problème de noms de types cellulaires),
     saute la réduction et sauvegarde le connectome complet
4. Sauvegarde le connectome pour utilisation rapide
"""

from __future__ import annotations

import sys
from pathlib import Path

# Ajouter flydrones_lib au path
FLYDRONES_SRC = Path(__file__).resolve().parent / "src"
if str(FLYDRONES_SRC) not in sys.path:
    sys.path.insert(0, str(FLYDRONES_SRC))

from flydrones.data import download_malecns  # noqa: E402
from flydrones.brain import build_malecns, Brain  # noqa: E402
from flydrones.config import load_config  # noqa: E402


def main():
    # Chemins
    repo_root = Path(__file__).resolve().parents[1]
    data_dir = repo_root / "flydrones_lib" / "data" / "malecns_v1"
    out_path = repo_root / "flydrones_lib" / "data" / "malecns_brain.npz"
    
    print("=" * 60)
    print("PRÉPARATION DU CONNECTOME MALECNS")
    print("=" * 60)
    
    # --- Étape 1 : Téléchargement ---
    print(f"\n[1/5] Téléchargement MaleCNS v1.0 vers {data_dir}")
    print("    (Taille ~1.2 GB, peut prendre plusieurs minutes)")
    try:
        download_malecns(str(data_dir))
        print("    ✓ Téléchargement terminé")
    except Exception as e:
        print(f"    ✗ Erreur téléchargement : {e}")
        print("    Vérifiez votre connexion internet et réessayez")
        return 1
    
    # --- Étape 2 : Construction du connectome signé ---
    print(f"\n[2/5] Construction du connectome signé (min_synapses=3)")
    print("    (Peut prendre 2-5 minutes, utilise ~4-6 GB RAM)")
    try:
        connectome = build_malecns(str(data_dir), min_synapses=3, verbose=True)
        print(f"    ✓ Connectome construit : {connectome.summary()}")
    except Exception as e:
        print(f"    ✗ Erreur construction : {e}")
        return 1
    
    # --- Étape 3 : Résolution des groupes input/output ---
    print(f"\n[3/5] Résolution des groupes input/output")
    try:
        cfg = load_config()
        temp_brain = Brain(connectome, cfg)
        input_groups = {k: v for k, v in temp_brain.input_specs.items() if connectome.group(k).size > 0}
        output_groups = {k: v for k, v in temp_brain.output_specs.items() if connectome.group(k).size > 0}
        print(f"    Groupes input avec neurones : {list(input_groups.keys())}")
        print(f"    Groupes output avec neurones : {list(output_groups.keys())}")
        
        if not input_groups or not output_groups:
            print("    ⚠️  Certains groupes n'ont pas de neurones correspondants dans MaleCNS")
            print("    ⚠️  Les noms de types cellulaires dans defaults.yaml ne matchent pas MaleCNS")
            print("    ⚠️  Sauvegarde du connectome COMPLET (sans réduction)")
    except Exception as e:
        print(f"    ✗ Erreur résolution groupes : {e}")
        return 1
    
    # --- Étape 4 : Réduction sensorimotor_core (si possible) ---
    print(f"\n[4/5] Tentative de réduction sensorimotor_core (hops=3, max_neurons=5000)")
    try:
        # Vérifier que meta['roles'] est bien défini
        if not connectome.meta.get("roles"):
            print("    ⚠️  meta['roles'] non défini, saut de la réduction")
            core = connectome
        else:
            core = connectome.sensorimotor_core(hops=3, max_neurons=5000)
            print(f"    ✓ Connectome réduit : {core.summary()}")
    except ValueError as e:
        print(f"    ⚠️  Réduction impossible : {e}")
        print("    ⚠️  Sauvegarde du connectome COMPLET (sans réduction)")
        core = connectome
    except Exception as e:
        print(f"    ⚠️  Erreur réduction : {e}")
        print("    ⚠️  Sauvegarde du connectome COMPLET (sans réduction)")
        core = connectome
    
    # --- Étape 5 : Sauvegarde ---
    print(f"\n[5/5] Sauvegarde vers {out_path}")
    try:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        core.save(str(out_path))
        print(f"    ✓ Sauvegarde terminée : {out_path}")
        print(f"    Neurones : {core.n:,}")
        print(f"    Connexions : {core.n_connections:,}")
    except Exception as e:
        print(f"    ✗ Erreur sauvegarde : {e}")
        return 1
    
    print("\n" + "=" * 60)
    print("TERMINÉ AVEC SUCCÈS !")
    print("=" * 60)
    print(f"\nConnectome sauvegardé : {out_path}")
    print(f"Neurones : {core.n:,}")
    print(f"Connexions : {core.n_connections:,}")
    print(f"\nPour l'utiliser :")
    print(f"  --brain {out_path}")
    print(f"  ou dans le code : load_connectome('{out_path}')")
    print(f"\n⚠️  Note : Si le connectome complet (166k neurones) est trop lent,")
    print(f"    vous pouvez utiliser MiniFly (--brain minifly) pour tester rapidement.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

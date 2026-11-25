#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
analyze_logs.py - Analyse les logs de simulation FlyDrones

Usage :
    python analyze_logs.py [fichier_log.npz]
    
Si aucun fichier n'est spécifié, prend le plus récent dans flydrones_lib/data/logs/
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Backend non-interactif pour éviter les crashs Wayland
import matplotlib.pyplot as plt

def find_latest_log(log_dir: Path = None) -> Path:
    """Trouve le fichier de log le plus récent."""
    if log_dir is None:
        log_dir = Path(__file__).resolve().parent / "data" / "logs"
    if not log_dir.exists():
        return None
    logs = sorted(log_dir.glob("neural_log_*.npz"), key=lambda p: p.stat().st_mtime, reverse=True)
    return logs[0] if logs else None

def analyze_log(npz_path: Path):
    """Analyse un fichier de log neuronal."""
    print(f"Chargement : {npz_path}")
    data = np.load(npz_path)
    
    print(f"\nVariables disponibles : {list(data.files)}")
    
    # Variables principales
    times = data['times']
    alts = data['alts']
    ticks = data['ticks']
    
    print(f"\n=== Résumé de la simulation ===")
    print(f"Durée : {times[-1] - times[0]:.2f} secondes")
    print(f"Nombre de ticks : {len(ticks)}")
    print(f"Altitude min : {alts.min():.2f} m")
    print(f"Altitude max : {alts.max():.2f} m")
    print(f"Altitude finale : {alts[-1]:.2f} m")
    
    # Trouver les groupes de rates (toutes les variables qui ne sont pas des métadonnées)
    metadata = {'times', 'alts', 'ticks', 'cmd_throttle', 'cmd_yaw', 'cmd_forward', 'cmd_lateral', 'cmd_escape',
                'motors_m1', 'motors_m2', 'motors_m3', 'motors_m4', 'retina_exp', 'retina_rot', 'retina_vert'}
    rate_names = [f for f in data.files if f not in metadata and not f.startswith('input_')]
    print(f"\nGroupes de neurones trouvés : {len(rate_names)}")
    for name in sorted(rate_names):
        values = data[name]
        print(f"  {name:25s}: min={values.min():6.1f}  max={values.max():6.1f}  mean={values.mean():6.1f}")
    
    # Commandes
    print(f"\n=== Commandes ===")
    print(f"Throttle : min={data['cmd_throttle'].min():+.2f}  max={data['cmd_throttle'].max():+.2f}  mean={data['cmd_throttle'].mean():+.2f}")
    print(f"Yaw      : min={data['cmd_yaw'].min():+.2f}  max={data['cmd_yaw'].max():+.2f}  mean={data['cmd_yaw'].mean():+.2f}")
    print(f"Forward  : min={data['cmd_forward'].min():+.2f}  max={data['cmd_forward'].max():+.2f}  mean={data['cmd_forward'].mean():+.2f}")
    print(f"Lateral  : min={data['cmd_lateral'].min():+.2f}  max={data['cmd_lateral'].max():+.2f}  mean={data['cmd_lateral'].mean():+.2f}")
    
    # Moteurs
    print(f"\n=== Moteurs ===")
    for m in ['m1', 'm2', 'm3', 'm4']:
        v = data[f'motors_{m}']
        print(f"  {m}: min={v.min():.0f}  max={v.max():.0f}  mean={v.mean():.0f}")
    
    # Rétine
    print(f"\n=== Rétine ===")
    print(f"Expansion : min={data['retina_exp'].min():+.2f}  max={data['retina_exp'].max():+.2f}  mean={data['retina_exp'].mean():+.2f}")
    print(f"Rotation  : min={data['retina_rot'].min():+.2f}  max={data['retina_rot'].max():+.2f}  mean={data['retina_rot'].mean():+.2f}")
    print(f"Vertical  : min={data['retina_vert'].min():+.2f}  max={data['retina_vert'].max():+.2f}  mean={data['retina_vert'].mean():+.2f}")
    
    # Graphiques
    fig, axes = plt.subplots(4, 1, figsize=(12, 12), sharex=True)
    
    # 1. Altitude
    axes[0].plot(times, alts, 'b-', linewidth=2)
    axes[0].axhline(y=1.0, color='r', linestyle='--', label='Cible 1.0m')
    axes[0].axhline(y=0.3, color='orange', linestyle='--', label='Plancher 0.3m')
    axes[0].axhline(y=1.5, color='orange', linestyle='--', label='Plafond 1.5m')
    axes[0].set_ylabel('Altitude (m)')
    axes[0].set_title('Altitude du drone')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # 2. Taux de feu des neurones descendants (DN)
    ax = axes[1]
    for name in sorted(rate_names):
        if 'DN' in name:
            ax.plot(times, data[name], label=name, alpha=0.7)
    ax.set_ylabel('Taux (Hz)')
    ax.set_title('Neurones descendants (DN)')
    ax.legend(loc='upper right', fontsize=8)
    ax.grid(True, alpha=0.3)
    
    # 3. Commandes
    ax = axes[2]
    ax.plot(times, data['cmd_throttle'], label='Throttle', alpha=0.7)
    ax.plot(times, data['cmd_yaw'], label='Yaw', alpha=0.7)
    ax.plot(times, data['cmd_forward'], label='Forward', alpha=0.7)
    ax.plot(times, data['cmd_lateral'], label='Lateral', alpha=0.7)
    ax.set_ylabel('Commande (-1..1)')
    ax.set_title('Commandes de vol')
    ax.legend(loc='upper right', fontsize=8)
    ax.grid(True, alpha=0.3)
    
    # 4. Rétine
    ax = axes[3]
    ax.plot(times, data['retina_exp'], label='Expansion (looming)', alpha=0.7)
    ax.plot(times, data['retina_rot'], label='Rotation', alpha=0.7)
    ax.plot(times, data['retina_vert'], label='Vertical', alpha=0.7)
    ax.set_ylabel('Valeur (0..1)')
    ax.set_xlabel('Temps (s)')
    ax.set_title('Features rétiniennes')
    ax.legend(loc='upper right', fontsize=8)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Sauvegarder le graphique
    output_path = npz_path.with_suffix('.png')
    plt.savefig(output_path, dpi=150)
    print(f"\nGraphique sauvegardé : {output_path}")
    # plt.show()  # désactivé pour backend Agg

def main():
    log_path = None
    if len(sys.argv) > 1:
        log_path = Path(sys.argv[1])
    
    if log_path is None or not log_path.exists():
        log_path = find_latest_log()
    
    if log_path is None:
        print("Aucun fichier de log trouvé !")
        print("Cherché dans : flydrones_lib/data/logs/")
        print("Utilisation : python analyze_logs.py <fichier.npz>")
        return 1
    
    analyze_log(log_path)
    return 0

if __name__ == "__main__":
    sys.exit(main())

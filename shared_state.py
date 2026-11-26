#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
État partagé entre le contrôleur RL et le GUI du connectome.
Permet la communication bidirectionnelle sans dépendance lourde (pas de ROS, pas de socket).
"""

from __future__ import annotations
import json
import os
import threading
from pathlib import Path
from typing import Any, Dict

STATE_PATH = Path(__file__).resolve().parent / "connectome_state.json"
_lock = threading.Lock()


def _default_state() -> Dict[str, Any]:
    return {
        "t": 0.0,
        "alt_m": 0.0,
        "rates": {},           # nom_neuron -> taux de décharge (Hz)
        "inputs": {},          # nom_input -> valeur
        "cmd": {"throttle": 0.0, "yaw": 0.0, "forward": 0.0, "lateral": 0.0, "escape": 0.0},
        "motors": [0.0, 0.0, 0.0, 0.0],
        "retina": {"expansion": 0.0, "rotation": 0.0, "vertical": 0.0},
        "manual": {"forward": 0.0, "lateral": 0.0, "yaw": 0.0, "throttle": 0.0, "active": False},
        "synapses_active": [],  # liste de (pre_idx, post_idx, weight) pour le GUI
        "crash": False,
        "brain_silent": False,
        "smooth_cmd": {"throttle": 0.0, "yaw": 0.0, "forward": 0.0, "lateral": 0.0},
    }


def read_state() -> Dict[str, Any]:
    with _lock:
        if not STATE_PATH.exists():
            return _default_state()
        try:
            with open(STATE_PATH, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return _default_state()


def write_state(state: Dict[str, Any]) -> None:
    with _lock:
        tmp = str(STATE_PATH) + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(state, f)
        os.replace(tmp, STATE_PATH)


def get_manual_cmd() -> Dict[str, float]:
    """Lit la commande manuelle depuis le GUI (flèches)."""
    state = read_state()
    m = state.get("manual", {})
    return {
        "forward": float(m.get("forward", 0.0)),
        "lateral": float(m.get("lateral", 0.0)),
        "yaw": float(m.get("yaw", 0.0)),
        "throttle": float(m.get("throttle", 0.0)),
        "active": bool(m.get("active", False)),
    }
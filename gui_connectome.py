#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GUI du connectome Crazyflie - Visualisation légère des synapses et contrôle manuel.

Lit l'état partagé (connectome_state.json) écrit par le contrôleur RL.
Affiche :
  - Réseau de neurones avec synapses actives (top N par poids)
  - Taux de décharge des neurones DN (sorties motrices)
  - Commandes (throttle, yaw, forward, lateral) et moteurs
  - Altitude et rétine (expansion, rotation, vertical)
Contrôle manuel au clavier (écrit dans l'état partagé).

Touches manuelles :
  T / G : avant / arrière        F / H : gauche / droite
  W / S : monter / descendre     A / D : yaw gauche / droite
  ESPACE : arrêt d'urgence

Correctifs de cette version :
  - touches T/G/F/H/W/S/A/D réellement déclarées (avant : KeyError / touches ignorées) ;
  - majuscules et minuscules gérées de la même façon (normalisation en minuscule) ;
  - les touches sont ignorées quand on tape dans le champ « Neurones » ;
  - toutes les touches sont relâchées si la fenêtre perd le focus ;
  - le nombre de neurones s'adapte automatiquement aux indices des synapses ;
  - synapses tracées avec une LineCollection, neurones avec un seul scatter.
"""

from __future__ import annotations

import json
import os
import threading
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import tkinter as tk
from tkinter import ttk

import matplotlib
matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.collections import LineCollection
from matplotlib.figure import Figure
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  (nécessaire pour projection='3d' sur anciennes versions)

# ---------------------------------------------------------------------------
# État partagé
# ---------------------------------------------------------------------------
STATE_PATH = Path(__file__).resolve().parent / "connectome_state.json"
_lock = threading.Lock()

DEFAULT_N_NEURONS = 10536
DN_NAMES = ["DNg02_L", "DNg02_R", "DNp03_L", "DNp03_R", "DNp01_L", "DNp01_R"]

# Touches de contrôle manuel (toujours en minuscule)
MANUAL_KEYS = ("t", "g", "f", "h", "w", "s", "a", "d")


def read_state() -> Dict[str, Any]:
    with _lock:
        if not STATE_PATH.exists():
            return {}
        try:
            with open(STATE_PATH, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {}


def write_state(state: Dict[str, Any]) -> None:
    with _lock:
        tmp = str(STATE_PATH) + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(state, f)
        os.replace(tmp, STATE_PATH)


# ---------------------------------------------------------------------------
# Positions des neurones (cercle stable)
# ---------------------------------------------------------------------------
def load_connectome_positions(n_neurons: int = DEFAULT_N_NEURONS):
    angles = np.linspace(0, 2 * np.pi, n_neurons, endpoint=False)
    return np.cos(angles), np.sin(angles)


# ---------------------------------------------------------------------------
# GUI
# ---------------------------------------------------------------------------
class ConnectomeGUI:
    def __init__(self, root: tk.Tk, refresh_ms: int = 100):
        self.root = root
        self.root.title("Crazyflie Connectome - GUI Synapses & Contrôle Manuel")
        self.root.geometry("1200x800")
        self.refresh_ms = refresh_ms

        # Données du connectome
        self.n_neurons = DEFAULT_N_NEURONS
        self.pos_x, self.pos_y = load_connectome_positions(self.n_neurons)

        # État
        self.state: Dict[str, Any] = {}
        self.running = True

        # Touches (défini AVANT la construction de l'UI et la liaison clavier)
        self.manual_keys: Dict[str, bool] = {k: False for k in MANUAL_KEYS}

        # Artistes matplotlib persistants (connectome)
        self._neuron_scatter = None
        self._dn_scatter = None
        self._syn_collection = None

        # Estimation de position 3D
        self._est_x = 0.0
        self._est_y = 0.0
        self._est_z = 0.0

        self._build_ui()

        # Liaison clavier
        self.root.focus_set()
        self.root.bind("<KeyPress>", self._on_key_press)
        self.root.bind("<KeyRelease>", self._on_key_release)
        self.root.bind("<FocusOut>", self._on_focus_out)
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

        self._update_loop()

    # ------------------------------------------------------------------
    # Construction UI
    # ------------------------------------------------------------------
    def _build_ui(self):
        main_paned = ttk.PanedWindow(self.root, orient=tk.HORIZONTAL)
        main_paned.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # === Gauche : visualisation connectome ===
        left_frame = ttk.Frame(main_paned)
        main_paned.add(left_frame, weight=2)

        ttk.Label(left_frame, text="Connectome - Neurones & Synapses actives",
                  font=("Arial", 12, "bold")).pack(pady=(0, 5))

        self.fig = Figure(figsize=(6, 6), dpi=100)
        self.ax = self.fig.add_subplot(111)
        self._setup_connectome_axes()

        self.canvas = FigureCanvasTkAgg(self.fig, master=left_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        ctrl_frame = ttk.Frame(left_frame)
        ctrl_frame.pack(fill=tk.X, pady=5)
        ttk.Label(ctrl_frame, text="Synapses affichées:").pack(side=tk.LEFT, padx=5)
        self.syn_scale = ttk.Scale(ctrl_frame, from_=50, to=500, orient=tk.HORIZONTAL,
                                   value=200, command=self._on_syn_scale, takefocus=0)
        self.syn_scale.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
        self.syn_label = ttk.Label(ctrl_frame, text="200")
        self.syn_label.pack(side=tk.LEFT, padx=5)

        # === Droite : informations & contrôle ===
        right_frame = ttk.Frame(main_paned)
        main_paned.add(right_frame, weight=1)

        notebook = ttk.Notebook(right_frame, takefocus=0)
        notebook.pack(fill=tk.BOTH, expand=True)

        tab_flight = ttk.Frame(notebook)
        notebook.add(tab_flight, text="État de vol")
        self._build_flight_tab(tab_flight)

        tab_dn = ttk.Frame(notebook)
        notebook.add(tab_dn, text="Neurones DN")
        self._build_dn_tab(tab_dn)

        tab_manual = ttk.Frame(notebook)
        notebook.add(tab_manual, text="Contrôle manuel")
        self._build_manual_tab(tab_manual)

        tab_activity = ttk.Frame(notebook)
        notebook.add(tab_activity, text="Activité neuronale")
        self._build_activity_tab(tab_activity)

        tab_3d = ttk.Frame(notebook)
        notebook.add(tab_3d, text="Vue 3D Drone")
        self._build_3d_tab(tab_3d)

    def _setup_connectome_axes(self):
        self.ax.set_aspect("equal")
        self.ax.set_title("Connectome - Synapses actives (épaisseur = |poids|)")
        self.ax.set_xlim(-1.3, 1.3)
        self.ax.set_ylim(-1.3, 1.3)
        self.ax.set_xticks([])
        self.ax.set_yticks([])
        self.ax.grid(True, alpha=0.3)

    def _build_flight_tab(self, parent):
        self.lbl_alt = ttk.Label(parent, text="Altitude: -- m", font=("Arial", 11, "bold"))
        self.lbl_alt.pack(pady=5)

        frame_retina = ttk.LabelFrame(parent, text="Rétine (flux optique)")
        frame_retina.pack(fill=tk.X, pady=5, padx=5)
        self.lbl_exp = ttk.Label(frame_retina, text="Expansion: --")
        self.lbl_exp.pack(anchor=tk.W, padx=5)
        self.lbl_rot = ttk.Label(frame_retina, text="Rotation: --")
        self.lbl_rot.pack(anchor=tk.W, padx=5)
        self.lbl_vert = ttk.Label(frame_retina, text="Vertical: --")
        self.lbl_vert.pack(anchor=tk.W, padx=5)

        frame_cmd = ttk.LabelFrame(parent, text="Commandes (après safety)")
        frame_cmd.pack(fill=tk.X, pady=5, padx=5)
        self.lbl_thr = ttk.Label(frame_cmd, text="Throttle: --")
        self.lbl_thr.pack(anchor=tk.W, padx=5)
        self.lbl_yaw = ttk.Label(frame_cmd, text="Yaw: --")
        self.lbl_yaw.pack(anchor=tk.W, padx=5)
        self.lbl_fwd = ttk.Label(frame_cmd, text="Forward: --")
        self.lbl_fwd.pack(anchor=tk.W, padx=5)
        self.lbl_lat = ttk.Label(frame_cmd, text="Lateral: --")
        self.lbl_lat.pack(anchor=tk.W, padx=5)

        frame_mot = ttk.LabelFrame(parent, text="Moteurs")
        frame_mot.pack(fill=tk.X, pady=5, padx=5)
        self.lbl_motors: List[ttk.Label] = []
        for i in range(4):
            lbl = ttk.Label(frame_mot, text=f"M{i + 1}: --")
            lbl.pack(anchor=tk.W, padx=5)
            self.lbl_motors.append(lbl)

        self.lbl_status = ttk.Label(parent, text="Statut: --", font=("Arial", 10, "italic"))
        self.lbl_status.pack(pady=10)

        self.lbl_brain_status = ttk.Label(parent, text="Cerveau: --", font=("Arial", 10, "bold"))
        self.lbl_brain_status.pack(pady=2)

        self.lbl_smooth = ttk.Label(parent, text="Lissage EMA: --", font=("Arial", 10, "italic"))
        self.lbl_smooth.pack(pady=2)

    def _build_dn_tab(self, parent):
        self.dn_labels = {}
        self.dn_bars = {}

        frame = ttk.Frame(parent)
        frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        for name in DN_NAMES:
            row = ttk.Frame(frame)
            row.pack(fill=tk.X, pady=2)
            ttk.Label(row, text=name, width=12).pack(side=tk.LEFT)
            bar = ttk.Progressbar(row, orient=tk.HORIZONTAL, length=150, mode="determinate")
            bar.pack(side=tk.LEFT, padx=5, fill=tk.X, expand=True)
            val_lbl = ttk.Label(row, text="0.0 Hz", width=8)
            val_lbl.pack(side=tk.LEFT, padx=5)
            self.dn_labels[name] = val_lbl
            self.dn_bars[name] = bar

        ttk.Label(parent, text="Taux de décharge des neurones de sortie (DN)",
                  font=("Arial", 9, "italic")).pack(pady=5)

    def _build_manual_tab(self, parent):
        ttk.Label(parent, text="Contrôle manuel du drone",
                  font=("Arial", 12, "bold")).pack(pady=10)

        info = ("Utilisez le clavier :\n"
                "  T / G : Avant / Arrière\n"
                "  F / H : Gauche / Droite\n"
                "  W / S : Monter / Descendre\n"
                "  A / D : Yaw gauche / droite\n"
                "  ESPACE : Arrêt d'urgence\n\n"
                "Le connectome corrige pour éviter les obstacles.")
        ttk.Label(parent, text=info, justify=tk.LEFT).pack(pady=10, padx=10)

        self.lbl_manual_status = ttk.Label(parent, text="Manuel: INACTIF",
                                           font=("Arial", 11, "bold"), foreground="red")
        self.lbl_manual_status.pack(pady=10)

        # Indicateurs visuels : symbole affiché -> touche (minuscule)
        self.key_indicators: Dict[str, tk.Label] = {}
        keys_frame = ttk.Frame(parent)
        keys_frame.pack(pady=10)
        for key in MANUAL_KEYS:
            lbl = tk.Label(keys_frame, text=key.upper(), width=3, relief="raised",
                           bg="lightgray", font=("Arial", 10, "bold"))
            lbl.pack(side=tk.LEFT, padx=2)
            self.key_indicators[key] = lbl

    # ------------------------------------------------------------------
    # Clavier - contrôle manuel
    # ------------------------------------------------------------------
    @staticmethod
    def _normalize_key(event) -> str:
        """Retourne le keysym, en minuscule pour les lettres (T == t)."""
        key = event.keysym
        return key.lower() if len(key) == 1 else key

    def _typing_in_entry(self) -> bool:
        """True si le focus est dans un champ de saisie (on ne pilote pas le drone)."""
        try:
            w = self.root.focus_get()
        except Exception:
            return False
        return isinstance(w, (tk.Entry, ttk.Entry))

    def _on_key_press(self, event):
        if self._typing_in_entry():
            return
        key = self._normalize_key(event)
        if key in self.manual_keys:
            if not self.manual_keys[key]:  # ignore l'auto-repeat
                self.manual_keys[key] = True
                self._update_manual_cmd()
                self._refresh_key_indicators()
        elif key == "space":
            self._send_emergency_stop()

    def _on_key_release(self, event):
        key = self._normalize_key(event)
        if key in self.manual_keys and self.manual_keys[key]:
            self.manual_keys[key] = False
            self._update_manual_cmd()
            self._refresh_key_indicators()

    def _on_focus_out(self, _event=None):
        """Relâche toutes les touches si la fenêtre perd le focus (évite les touches « coincées »)."""
        if any(self.manual_keys.values()):
            for key in self.manual_keys:
                self.manual_keys[key] = False
            self._update_manual_cmd()
            self._refresh_key_indicators()

    def _update_manual_cmd(self):
        """Calcule la commande manuelle depuis les touches pressées."""
        k = self.manual_keys
        forward = 0.5 * (int(k["t"]) - int(k["g"]))
        lateral = 0.5 * (int(k["h"]) - int(k["f"]))
        throttle = 0.3 * (int(k["w"]) - int(k["s"]))
        yaw = 0.4 * (int(k["d"]) - int(k["a"]))

        active = any(k.values())

        state = read_state()
        state["manual"] = {
            "forward": forward,
            "lateral": lateral,
            "yaw": yaw,
            "throttle": throttle,
            "active": active,
        }
        write_state(state)

        status = "ACTIF" if active else "INACTIF"
        color = "green" if active else "red"
        self.lbl_manual_status.config(text=f"Manuel: {status}", foreground=color)

    def _refresh_key_indicators(self):
        for key, lbl in self.key_indicators.items():
            lbl.config(bg="yellow" if self.manual_keys.get(key, False) else "lightgray")

    def _send_emergency_stop(self):
        state = read_state()
        state["manual"] = {"forward": 0.0, "lateral": 0.0, "yaw": 0.0,
                           "throttle": 0.0, "active": True}
        state["crash"] = True
        write_state(state)
        self.lbl_manual_status.config(text="Manuel: ARRÊT D'URGENCE", foreground="red")
        for key in self.manual_keys:
            self.manual_keys[key] = False
        self._refresh_key_indicators()

    # ------------------------------------------------------------------
    # Onglet activité
    # ------------------------------------------------------------------
    def _build_activity_tab(self, parent):
        ttk.Label(parent, text="Taux de décharge en temps réel (Hz)",
                  font=("Arial", 11, "bold")).pack(pady=5)

        self.fig_rates = Figure(figsize=(8, 4), dpi=100)
        self.ax_rates = self.fig_rates.add_subplot(111)

        self.canvas_rates = FigureCanvasTkAgg(self.fig_rates, master=parent)
        self.canvas_rates.get_tk_widget().pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        self.rate_history: Dict[str, list] = {}
        self.max_history_points = 200

        ctrl = ttk.Frame(parent)
        ctrl.pack(fill=tk.X, padx=5)
        ttk.Label(ctrl, text="Neurones:").pack(side=tk.LEFT)
        self.activity_neurons = tk.StringVar(value=",".join(DN_NAMES))
        ttk.Entry(ctrl, textvariable=self.activity_neurons, width=50).pack(side=tk.LEFT, padx=5)
        ttk.Button(ctrl, text="Appliquer", command=self._on_activity_neurons_change).pack(side=tk.LEFT)
        self.selected_neurons = [n.strip() for n in self.activity_neurons.get().split(",") if n.strip()]

    def _on_activity_neurons_change(self):
        self.selected_neurons = [n.strip() for n in self.activity_neurons.get().split(",") if n.strip()]
        for n in self.selected_neurons:
            self.rate_history.setdefault(n, [])
        self.root.focus_set()  # rend le focus à la fenêtre pour les touches de pilotage

    # ------------------------------------------------------------------
    # Onglet 3D
    # ------------------------------------------------------------------
    def _build_3d_tab(self, parent):
        ttk.Label(parent, text="Position du drone en 3D (X, Y, Z)",
                  font=("Arial", 11, "bold")).pack(pady=5)

        self.fig_3d = Figure(figsize=(8, 6), dpi=100)
        self.ax_3d = self.fig_3d.add_subplot(111, projection="3d")
        self._setup_3d_axes()

        self.canvas_3d = FigureCanvasTkAgg(self.fig_3d, master=parent)
        self.canvas_3d.get_tk_widget().pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        self.pos_history: List[tuple] = []
        self.max_pos_history = 500

        ctrl = ttk.Frame(parent)
        ctrl.pack(fill=tk.X, padx=5)
        ttk.Button(ctrl, text="Reset trajectoire", command=self._reset_3d_traj).pack(side=tk.LEFT, padx=5)
        self.lbl_3d_pos = ttk.Label(ctrl, text="Pos: --, --, --")
        self.lbl_3d_pos.pack(side=tk.LEFT, padx=10)

    def _setup_3d_axes(self):
        self.ax_3d.set_title("Trajectoire 3D du Crazyflie")
        self.ax_3d.set_xlabel("X (m)")
        self.ax_3d.set_ylabel("Y (m)")
        self.ax_3d.set_zlabel("Z (m) / Altitude")
        self.ax_3d.set_xlim(-3, 3)
        self.ax_3d.set_ylim(-3, 3)
        self.ax_3d.set_zlim(0, 2)
        self.ax_3d.grid(True, alpha=0.3)

    def _reset_3d_traj(self):
        self.pos_history = []
        self._est_x = 0.0
        self._est_y = 0.0
        self.ax_3d.clear()
        self._setup_3d_axes()
        self.canvas_3d.draw_idle()
        self.root.focus_set()

    def _update_3d(self):
        if not self.state:
            return

        alt = float(self.state.get("alt_m", 0.0))
        cmd = self.state.get("cmd", {})
        forward = float(cmd.get("forward", 0.0))
        lateral = float(cmd.get("lateral", 0.0))

        # Intégration très approximative des commandes (pour la visualisation uniquement)
        dt = self.refresh_ms / 1000.0
        self._est_x += lateral * 0.1 * dt
        self._est_y += forward * 0.1 * dt
        self._est_z = alt

        self.pos_history.append((self._est_x, self._est_y, self._est_z))
        if len(self.pos_history) > self.max_pos_history:
            self.pos_history = self.pos_history[-self.max_pos_history:]

        self.lbl_3d_pos.config(
            text=f"Pos: {self._est_x:+.2f}, {self._est_y:+.2f}, {self._est_z:+.2f} m")

        self.ax_3d.clear()
        self._setup_3d_axes()

        if len(self.pos_history) > 1:
            xs, ys, zs = zip(*self.pos_history)
            self.ax_3d.plot(xs, ys, zs, color="#3498db", linewidth=1.5, alpha=0.7, label="Trajectoire")

        dx, dy, dz = self._est_x, self._est_y, self._est_z
        arm = 0.15
        self.ax_3d.plot([dx - arm, dx + arm], [dy, dy], [dz, dz], color="black", linewidth=3)
        self.ax_3d.plot([dx, dx], [dy - arm, dy + arm], [dz, dz], color="black", linewidth=3)
        self.ax_3d.scatter([dx], [dy], [dz], color="#2ecc71", s=100, marker="s", label="Drone")

        self.ax_3d.legend(loc="upper left", fontsize=7)
        self.canvas_3d.draw_idle()

    # ------------------------------------------------------------------
    # Boucle de mise à jour
    # ------------------------------------------------------------------
    def _update_loop(self):
        if not self.running:
            return
        try:
            self.state = read_state()
            self._update_flight_info()
            self._update_dn_bars()
            self._draw_connectome()
            self._update_activity_plot()
            self._update_3d()
        except Exception as exc:  # une erreur d'affichage ne doit pas tuer la GUI
            print(f"[GUI] Erreur de mise à jour : {exc!r}")
        self.root.after(self.refresh_ms, self._update_loop)

    def _update_flight_info(self):
        if not self.state:
            return
        alt = self.state.get("alt_m", 0.0)
        self.lbl_alt.config(text=f"Altitude: {alt:.2f} m")

        retina = self.state.get("retina", {})
        self.lbl_exp.config(text=f"Expansion: {retina.get('expansion', 0.0):+.2f}")
        self.lbl_rot.config(text=f"Rotation: {retina.get('rotation', 0.0):+.2f}")
        self.lbl_vert.config(text=f"Vertical: {retina.get('vertical', 0.0):+.2f}")

        cmd = self.state.get("cmd", {})
        self.lbl_thr.config(text=f"Throttle: {cmd.get('throttle', 0.0):+.3f}")
        self.lbl_yaw.config(text=f"Yaw: {cmd.get('yaw', 0.0):+.3f}")
        self.lbl_fwd.config(text=f"Forward: {cmd.get('forward', 0.0):+.3f}")
        self.lbl_lat.config(text=f"Lateral: {cmd.get('lateral', 0.0):+.3f}")

        motors = list(self.state.get("motors", [0, 0, 0, 0])) + [0, 0, 0, 0]
        for i, lbl in enumerate(self.lbl_motors):
            lbl.config(text=f"M{i + 1}: {float(motors[i]):.0f}")

        crash = self.state.get("crash", False)
        manual_active = self.state.get("manual", {}).get("active", False)
        if crash:
            status, color = "CRASH / URGENCE", "red"
        elif manual_active:
            status, color = "MANUEL", "green"
        else:
            status, color = "CONNECTOME", "blue"
        self.lbl_status.config(text=f"Statut: {status}", foreground=color)

        brain_silent = self.state.get("brain_silent", False)
        self.lbl_brain_status.config(
            text="Cerveau: " + ("SILENCE/HOVER" if brain_silent else "ACTIF"),
            foreground="orange" if brain_silent else "green",
        )
        smooth = self.state.get("smooth_cmd", {})
        self.lbl_smooth.config(
            text=(f"Lissage EMA: thr={smooth.get('throttle', 0):+.3f} "
                  f"yaw={smooth.get('yaw', 0):+.3f} "
                  f"fwd={smooth.get('forward', 0):+.3f} "
                  f"lat={smooth.get('lateral', 0):+.3f}"))

    def _update_dn_bars(self):
        rates = self.state.get("rates", {})
        for name in DN_NAMES:
            rate = float(rates.get(name, 0.0))
            self.dn_bars[name]["value"] = min(rate, 100.0)
            self.dn_labels[name].config(text=f"{rate:.1f} Hz")

        t = float(self.state.get("t", 0.0))
        for n in self.selected_neurons:
            if n in rates:
                hist = self.rate_history.setdefault(n, [])
                hist.append((t, float(rates[n])))
                if len(hist) > self.max_history_points:
                    del hist[:-self.max_history_points]

    def _update_activity_plot(self):
        if not self.selected_neurons:
            return

        self.ax_rates.clear()
        self.ax_rates.set_title("Taux de décharge (dernières ~10 s)")
        self.ax_rates.set_xlabel("Temps relatif (s)")
        self.ax_rates.set_ylabel("Taux (Hz)")
        self.ax_rates.grid(True, alpha=0.3)
        self.ax_rates.set_ylim(0, 150)

        t_now = float(self.state.get("t", 0.0))
        plotted = False
        for n in self.selected_neurons:
            hist = self.rate_history.get(n, [])
            if len(hist) > 1:
                times, vals = zip(*hist)
                self.ax_rates.plot([t - t_now for t in times], vals, label=n, linewidth=1.0)
                plotted = True

        if plotted:
            self.ax_rates.legend(loc="upper right", fontsize=7)
        self.canvas_rates.draw_idle()

    # ------------------------------------------------------------------
    # Connectome
    # ------------------------------------------------------------------
    def _ensure_positions(self, synapses) -> bool:
        """Agrandit le cercle de positions si une synapse référence un indice hors limites.

        Retourne True si le nombre de neurones a changé.
        """
        max_idx = -1
        for s in synapses:
            try:
                max_idx = max(max_idx, int(s[0]), int(s[1]))
            except (TypeError, ValueError, IndexError):
                continue
        if max_idx >= self.n_neurons:
            self.n_neurons = max_idx + 1
            self.pos_x, self.pos_y = load_connectome_positions(self.n_neurons)
            return True
        return False

    def _init_connectome_artists(self):
        """(Re)crée les artistes persistants : neurones, neurones DN, synapses."""
        self.ax.clear()
        self._setup_connectome_axes()

        self._syn_collection = LineCollection([], zorder=1)
        self.ax.add_collection(self._syn_collection)

        self._neuron_scatter = self.ax.scatter(
            self.pos_x, self.pos_y, s=3, color="#4a90e2", linewidths=0, zorder=2, alpha=0.6)

        dn_indices = [i for i in (50, 100, 150, 200, 250, 300) if i < self.n_neurons]
        self._dn_scatter = self.ax.scatter(
            self.pos_x[dn_indices], self.pos_y[dn_indices], s=80,
            color="#e67e22", edgecolor="black", linewidth=1.0, zorder=3)

    def _draw_connectome(self):
        synapses = self.state.get("synapses_active", [])
        max_syn = int(self.syn_scale.get())
        synapses = synapses[:max_syn]

        changed = self._ensure_positions(synapses)
        if self._neuron_scatter is None or changed:
            self._init_connectome_artists()

        # Filtrage de sécurité sur les indices et la forme des entrées
        valid = []
        for s in synapses:
            try:
                pre, post, w = int(s[0]), int(s[1]), float(s[2])
            except (TypeError, ValueError, IndexError):
                continue
            if 0 <= pre < self.n_neurons and 0 <= post < self.n_neurons:
                valid.append((pre, post, w))

        if valid:
            pre_idx = np.array([v[0] for v in valid])
            post_idx = np.array([v[1] for v in valid])
            w = np.array([v[2] for v in valid])
            absw = np.abs(w)
            w_min, w_max = absw.min(), absw.max()
            norm = (absw - w_min) / (w_max - w_min) if w_max > w_min else np.zeros_like(absw)

            segments = np.stack([
                np.column_stack([self.pos_x[pre_idx], self.pos_y[pre_idx]]),
                np.column_stack([self.pos_x[post_idx], self.pos_y[post_idx]]),
            ], axis=1)

            colors = np.zeros((len(valid), 4))
            colors[w > 0] = (0xE7 / 255, 0x4C / 255, 0x3C / 255, 1.0)   # excitateur : rouge
            colors[w <= 0] = (0x34 / 255, 0x98 / 255, 0xDB / 255, 1.0)  # inhibiteur : bleu
            colors[:, 3] = 0.3 + 0.7 * norm

            self._syn_collection.set_segments(segments)
            self._syn_collection.set_color(colors)
            self._syn_collection.set_linewidth(0.5 + 3.0 * norm)
        else:
            self._syn_collection.set_segments([])

        self.canvas.draw_idle()

    def _on_syn_scale(self, val):
        self.syn_label.config(text=str(int(float(val))))

    # ------------------------------------------------------------------
    def _on_close(self):
        self.stop()
        self.root.destroy()

    def stop(self):
        self.running = False


def main():
    root = tk.Tk()
    app = ConnectomeGUI(root)
    try:
        root.mainloop()
    except KeyboardInterrupt:
        app.stop()


if __name__ == "__main__":
    main()
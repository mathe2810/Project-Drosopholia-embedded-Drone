"""
connectome_brain.py
-------------------
Cerveau de décision basé sur le connectome de Drosophila.

PRINCIPE FONDAMENTAL : le connectome ne pilote JAMAIS les moteurs directement.
Il produit une décision de haut niveau (GAUCHE / DROITE / HAUT / RALENTIR /
STATIONNER / AVANCER) qui est traduite en consignes de vitesse pour le PID
du contrôleur Webots. Le PID garde la stabilité du drone.

Deux backends :
  1. "manuel"  -> flydrones_lib : rétine -> encodeur -> cerveau LIF (MaleCNS)
                 -> décodeur moteur linéaire -> décision
  2. "entraine" -> FlyDrone-SNN : topologie connectome + poids entraînés
                 (e-prop) -> forward SNN -> décision

Les deux exposent la même méthode : decide(frame, ranges, telemetry) -> str
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional
import numpy as np


# ---------------------------------------------------------------------------
# Décisions de haut niveau (ce que le connectome peut demander)
# ---------------------------------------------------------------------------
class Decision(Enum):
    HOVER = "HOVER"          # stationner (maintenir l'altitude)
    FORWARD = "FORWARD"      # avancer
    BACKWARD = "BACKWARD"    # reculer
    LEFT = "LEFT"            # aller à gauche
    RIGHT = "RIGHT"          # aller à droite
    UP = "UP"                # monter
    DOWN = "DOWN"            # descendre
    SLOW = "SLOW"            # ralentir (mur proche / obstacle)


DECISION_SPEEDS = {
    Decision.HOVER:   (0.0, 0.0, 0.0, 0.0),   # (vx, vy, yaw_rate, d_alt)
    Decision.FORWARD: (0.3, 0.0, 0.0, 0.0),
    Decision.BACKWARD:(-0.3, 0.0, 0.0, 0.0),
    Decision.LEFT:    (0.0, 0.3, 0.0, 0.0),   # vy > 0 = gauche (corps)
    Decision.RIGHT:   (0.0, -0.3, 0.0, 0.0),
    Decision.UP:      (0.0, 0.0, 0.0, +0.15),
    Decision.DOWN:    (0.0, 0.0, 0.0, -0.15),
    Decision.SLOW:    (0.05, 0.0, 0.0, 0.0),  # quasi-stationnaire
}


@dataclass
class BrainConfig:
    method: str = "manuel"            # "manuel" | "entraine"
    flydrones_lib_path: str = "/home/matheo/Desktop/crazyflie/flydrones_lib/build/lib"
    flydrone_snn_path: str = "/home/matheo/Desktop/crazyflie/FlyDrone-SNN"
    connectome_file: str = "/home/matheo/Desktop/crazyflie/flydrones_lib/data/malecns_brain.npz"
    snn_checkpoint: str = "/home/matheo/Desktop/crazyflie/FlyDrone-SNN/flydrone/ckpt/connectome_spatial_1d"
    # seuils de décision (sur les axes du décodeur / SNN)
    axis_threshold: float = 0.2
    # obstacle : si loom > ce seuil, on force SLOW (sécurité connectome)
    loom_slow_threshold: float = 0.5
    # clé de chiffrement vidéo (partagée avec le contrôleur Webots)
    crypto_key: bytes = b"crazyflie-connectome-cam-key-2026"


class ConnectomeBrain:
    """
    Wrapper unifié autour des deux librairies connectome.

    Exemple :
        brain = ConnectomeBrain(BrainConfig(method="manuel"))
        decision = brain.decide(frame_rgb, {"front": 1.2, "left": 0.8, ...}, telemetry)
        # decision == Decision.LEFT
    """

    def __init__(self, cfg: BrainConfig | None = None):
        self.cfg = cfg or BrainConfig()
        self.method = self.cfg.method
        self._brain = None          # instance flydrones_lib Brain (manuel)
        self._retina = None         # Retina (manuel)
        self._encoder = None        # InputEncoder (manuel)
        self._decoder = None        # MotorDecoder (manuel)
        self._snn_runner = None     # GPUForwardRunner / JAX forward (entraîné)
        self._snn_params = None
        self._snn_cfg = None
        self._snn_dg = None
        self._prev_loom = 0.0
        self._initialized = False

    # ------------------------------------------------------------------ init
    def initialize(self) -> None:
        """Charge le connectome / les poids selon la méthode choisie."""
        if self.method == "manuel":
            self._init_manuel()
        elif self.method == "entraine":
            self._init_entraine()
        else:
            raise ValueError(f"méthode inconnue: {self.method}")
        self._initialized = True

    def _init_manuel(self) -> None:
        """Backend flydrones_lib (connectome MaleCNS + LIF + décodeur linéaire)."""
        import sys
        if self.cfg.flydrones_lib_path not in sys.path:
            sys.path.insert(0, self.cfg.flydrones_lib_path)

        from flydrones.brain.connectome import Connectome
        from flydrones.brain.brain import Brain
        from flydrones.senses.retina import Retina
        from flydrones.senses.encoder import InputEncoder
        from flydrones.motor.decoder import MotorDecoder

        # --- connectome ---
        connectome = Connectome.load(self.cfg.connectome_file)

        # --- config minimale pour le cerveau ---
        # On sélectionne des groupes sensoriels (rétine) et moteurs (descendants)
        # de manière approximative : on utilise les groupes disponibles.
        cfg = {
            "brain": {"lif": {}, "seed": 0},
            "inputs": {
                "retina_L": {"types": [".*"], "side": "L", "feature": "brightness", "eye": "L", "max_hz": 100.0},
                "retina_R": {"types": [".*"], "side": "R", "feature": "brightness", "eye": "R", "max_hz": 100.0},
            },
            "outputs": {
                "DN_fwd": {"types": [".*"], "side": None, "max_hz": 100.0},
                "DN_yaw": {"types": [".*"], "side": None, "max_hz": 100.0},
            },
            "decoder": {
                "axes": {
                    "forward": {"gain": 1.0, "terms": {"DN_fwd": 1.0}},
                    "lateral": {"gain": 1.0, "terms": {"DN_yaw": 1.0}},
                    "throttle": {"gain": 0.0, "terms": {}},
                    "yaw": {"gain": 0.0, "terms": {}},
                },
                "settle_s": 0.5,
            },
        }

        self._brain = Brain(connectome, cfg)
        self._retina = Retina(width=96, height=72, grid=(6, 8))
        self._encoder = InputEncoder(connectome, cfg)
        self._decoder = MotorDecoder(cfg)

        # warm-up du cerveau (mesure les taux de repos)
        for _ in range(20):
            self._brain.tick({}, ms=50.0)
        self._decoder.update(self._brain.last_rates, dt=0.05)
        print(f"[connectome_brain] manuel initialisé: {connectome.n:,} neurones")

    def _init_entraine(self) -> None:
        """Backend FlyDrone-SNN (topologie connectome + poids entraînés e-prop)."""
        import sys
        if self.cfg.flydrone_snn_path not in sys.path:
            sys.path.insert(0, self.cfg.flydrone_snn_path)

        from flydrone import snn
        from flydrone.graph import load_connectome, make_variant
        from flydrone.envs import encode_obs, N_INPUTS
        import pickle, json
        from pathlib import Path
        import jax
        import jax.numpy as jnp

        # On charge le connectome FlyWire (topologie uniquement)
        data_dir = Path(self.cfg.flydrone_snn_path) / "fly-brain"
        ann = Path(self.cfg.flydrone_snn_path) / "data_ext" / "flywire_neuron_annotations.tsv"
        g = load_connectome(
            data_dir / "2025_Completeness_783.csv",
            data_dir / "2025_Connectivity_783.parquet",
            ann,
        )
        g = make_variant(g, "connectome", seed=0)

        n_actions = 4  # throttle, yaw, forward, lateral
        cfg = snn.SNNConfig(
            n=g.n, n_inputs=N_INPUTS, n_actions=n_actions, dt=1.0,
            learning_signal="spatial", w_scale=0.275 / 7.0, in_scale=0.1,
        )

        # Charger les poids entraînés s'ils existent
        ckpt = Path(self.cfg.snn_checkpoint)
        if (ckpt / "ckpt.pkl").exists():
            with open(ckpt / "ckpt.pkl", "rb") as f:
                raw_params = pickle.load(f)["params"]
            params = jax.tree_util.tree_map(jnp.asarray, raw_params)
        else:
            params = snn.init_params(g, cfg, seed=0)
            print(f"[connectome_brain] aucun checkpoint trouvé, poids aléatoires")

        dg = snn.to_device(g, cfg.n_actions, seed=0)
        st = snn.init_state(params, cfg, batch=1)

        self._snn_runner = snn
        self._snn_params = params
        self._snn_cfg = cfg
        self._snn_dg = dg
        self._snn_st = st
        self._snn_n_sub = max(1, int(round(1000.0 / 50.0 / cfg.dt)))  # 50 Hz ctrl
        print(f"[connectome_brain] entraîné initialisé: {g.n} neurones, checkpoint={ckpt.exists()}")

    # ------------------------------------------------------------------ décision
    def decide(self, frame_rgb: Optional[np.ndarray],
               ranges: dict,
               telemetry: dict) -> Decision:
        """
        Produit une décision de haut niveau à partir de la caméra et des
        capteurs de distance.

        Paramètres
        ----------
        frame_rgb : (H, W, 3) uint8 ou None
            Image de la caméra Webots (déjà déchiffrée par le serveur).
        ranges : dict
            {"front": m, "left": m, "right": m, "back": m} en mètres.
        telemetry : dict
            {"alt": m, "vx": m/s, "vy": m/s, "yaw_rate": deg/s}

        Retour
        ------
        Decision
        """
        if not self._initialized:
            self.initialize()

        # --- Sécurité obstacle : capteurs de distance (priorité absolue) ---
        front = ranges.get("front", float("inf"))
        left = ranges.get("left", float("inf"))
        right = ranges.get("right", float("inf"))
        min_range = min(front, left, right)

        if min_range < 0.35:
            # Trop près : forcer SLOW / arrêt
            return Decision.SLOW

        # --- Backend manuel (flydrones_lib) ---
        if self.method == "manuel":
            return self._decide_manuel(frame_rgb, ranges, telemetry)

        # --- Backend entraîné (FlyDrone-SNN) ---
        return self._decide_entraine(frame_rgb, ranges, telemetry)

    # ------------------------------------------------------------------ manuel
    def _decide_manuel(self, frame_rgb, ranges, telemetry) -> Decision:
        """Pipeline : rétine -> encodeur -> LIF -> décodeur -> décision."""
        # Rétine : features visuelles (optic flow, looming, luminosité)
        visual = self._retina.encode(frame_rgb)

        # Encodage en taux de Poisson pour les groupes d'entrée
        yaw_rate = telemetry.get("yaw_rate", 0.0)
        input_rates = self._encoder.encode(visual, yaw_rate_dps=yaw_rate)

        # Tick du cerveau LIF (50 ms par tick)
        rates = self._brain.tick(input_rates, ms=50.0)

        # Décodeur : taux des DN -> commande de vol normalisée
        cmd = self._decoder.update(rates, dt=0.05)

        # Détection de looming (obstacle qui approche) depuis la rétine
        loom = visual.expansion
        self._prev_loom = 0.7 * self._prev_loom + 0.3 * loom

        # Sécurité connectome : looming fort -> ralentir
        if self._prev_loom > self.cfg.loom_slow_threshold:
            return Decision.SLOW

        # Traduction des axes du décodeur en décision discrète
        return self._axes_to_decision(cmd.forward, cmd.lateral, cmd.throttle, cmd.yaw)

    # ------------------------------------------------------------------ entraîné
    def _decide_entraine(self, frame_rgb, ranges, telemetry) -> Decision:
        """Forward SNN sur topologie connectome avec poids entraînés."""
        snn = self._snn_runner
        cfg = self._snn_cfg
        params = self._snn_params
        dg = self._snn_dg
        st = self._snn_st

        # Construire une observation 25-dim (comme à l'entraînement) à partir
        # des données disponibles : caméra (features réduites) + télémétrie.
        obs = self._obs_from_frame_and_telemetry(frame_rgb, telemetry)
        x = np.concatenate([np.ones((1, 1)), np.maximum(obs, 0), np.maximum(-obs, 0)], axis=1).T.astype(np.float32)

        # Forward du SNN (n_sub pas pour 1 pas de contrôle)
        steps_arr = __import__("jax").numpy.zeros((self._snn_n_sub,), __import__("jax").numpy.int32)
        st, _ = snn.forward_chunk(params, dg, cfg, st, __import__("jax").numpy.asarray(x), steps_arr)
        self._snn_st = st
        action = np.clip(np.asarray(st["y"]).T, -1, 1)[0]  # (A,)

        # action : [throttle, yaw, forward, lateral] (selon l'entraînement)
        throttle, yaw, forward, lateral = action[0], action[1], action[2], action[3]

        # Sécurité : looming estimé depuis la télémétrie (proximité)
        front = ranges.get("front", float("inf"))
        if front < 0.6:
            return Decision.SLOW

        return self._axes_to_decision(forward, lateral, throttle, yaw)

    # ------------------------------------------------------------------ helpers
    def _obs_from_frame_and_telemetry(self, frame_rgb, telemetry) -> np.ndarray:
        """Construit un obs 12-dim simplifié à partir de la caméra + télémétrie."""
        # On utilise la luminosité moyenne et le contraste comme proxy visuel
        if frame_rgb is not None:
            gray = frame_rgb[..., :3].astype(np.float32) @ np.array([0.299, 0.587, 0.114])
            gray = gray / 255.0 if gray.max() > 1.5 else gray
            brightness = float(gray.mean())
            contrast = float(gray.std())
        else:
            brightness, contrast = 0.5, 0.0

        alt = telemetry.get("alt", 1.0)
        vx = telemetry.get("vx", 0.0)
        vy = telemetry.get("vy", 0.0)
        yaw_rate = telemetry.get("yaw_rate", 0.0)

        # Obs 12-dim : [pos_err(3), rpy(3), vel(3), angv(3)] -> on remplit avec ce qu'on a
        obs = np.array([
            0.0, 0.0, alt - 1.0,          # position (z vers cible 1.0m)
            0.0, 0.0, 0.0,                # roll, pitch, yaw (inconnus -> 0)
            vx, vy, 0.0,                  # vitesse linéaire
            0.0, 0.0, yaw_rate,           # vitesse angulaire
        ], dtype=np.float32)
        # injecter un peu de vision dans les erreurs de position x/y
        obs[0] = (brightness - 0.5) * 2.0   # proxy x
        obs[1] = contrast * 2.0             # proxy y
        return obs

    def _axes_to_decision(self, forward: float, lateral: float,
                          throttle: float, yaw: float) -> Decision:
        """Traduit les 4 axes normalisés [-1,1] en une décision discrète."""
        # Priorité verticale si forte
        if abs(throttle) > self.cfg.axis_threshold:
            return Decision.UP if throttle > 0 else Decision.DOWN
        # Priorité yaw si forte
        if abs(yaw) > self.cfg.axis_threshold:
            return Decision.RIGHT if yaw > 0 else Decision.LEFT
        # Priorité latérale
        if abs(lateral) > self.cfg.axis_threshold:
            return Decision.LEFT if lateral > 0 else Decision.RIGHT
        # Priorité avant/arrière
        if abs(forward) > self.cfg.axis_threshold:
            return Decision.FORWARD if forward > 0 else Decision.BACKWARD
        return Decision.HOVER

    def decision_to_setpoints(self, decision: Decision,
                              current_alt: float) -> dict:
        """
        Convertit une décision en consignes de vitesse pour le PID.

        Retourne : {"vx": m/s, "vy": m/s, "yaw_rate": rad/s, "altitude": m}
        (convention : vx > 0 = avant, vy > 0 = gauche, yaw_rate > 0 = gauche)
        """
        vx, vy, yaw_rate, d_alt = DECISION_SPEEDS[decision]
        return {
            "vx": vx,
            "vy": vy,
            "yaw_rate": yaw_rate,
            "altitude": current_alt + d_alt,
        }

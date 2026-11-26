#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
crazyflie_flydrones_rl.py - Contrôleur avec connectome MaleCNS + Apprentissage Synaptique RL

Architecture :
- PID d'altitude : contrôle principal (maintient 1.0m, stabilité)
- Connectome MaleCNS : reçoit la caméra, produit des corrections de mouvement
- Apprentissage synaptique : modifie les poids W du connectome selon la récompense
- Système de récompense : +survie, -crash, +progression dans le labyrinthe

Sauvegarde : poids synaptiques + logs neuronaux à la fin de la simulation
"""
from __future__ import annotations
import sys
from pathlib import Path
import time
import atexit
import numpy as np
import scipy.sparse as sp
import threading
import queue
import tkinter as tk
from tkinter import ttk
import matplotlib.pyplot as plt
import threading
import queue
import tkinter as tk
from tkinter import ttk
import matplotlib.pyplot as plt

FLYDRONES_SRC = Path(__file__).resolve().parents[2] / "flydrones_lib" / "src"
if str(FLYDRONES_SRC) not in sys.path:
    sys.path.insert(0, str(FLYDRONES_SRC))

# État partagé avec le GUI du connectome
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from shared_state import write_state, get_manual_cmd  # noqa: E402

from controller import Robot
from flydrones.brain import Brain, load_connectome
from flydrones.config import load_config
from flydrones.motor.command import FlightCommand
from flydrones.motor.decoder import MotorDecoder
from flydrones.safety import SafetyGovernor, Telemetry
from flydrones.senses.encoder import InputEncoder
from flydrones.senses.retina import Retina


# ======================================================================
# Logger neuronal (défini AVANT son utilisation)
# ======================================================================
class NeuralLogger:
    """Sauvegarde l'activité neuronale à chaque tick pour analyse ultérieure."""

    def __init__(self, log_path: str):
        self.log_path = log_path
        self.records = []
        self.tick_count = 0

    def log_tick(self, t, dt, alt_m, rates, inputs, cmd, motors, retina_features,
                 pid_thrust, brain_thrust):
        record = {
            "t": t, "dt": dt, "alt_m": alt_m,
            "rates": dict(rates),
            "inputs": {k: v.tolist() if hasattr(v, "tolist") else v for k, v in inputs.items()},
            "cmd": {"throttle": cmd.throttle, "yaw": cmd.yaw, "forward": cmd.forward,
                    "lateral": cmd.lateral, "escape": cmd.escape},
            "motors": list(motors),
            "retina": retina_features,
            "tick": self.tick_count,
            "pid_thrust": pid_thrust,
            "brain_thrust": brain_thrust,
        }
        self.records.append(record)
        self.tick_count += 1

    def save(self, path: str = None):
        if not self.records:
            return
        if path is None:
            path = self.log_path
        times = [r["t"] for r in self.records]
        alts = [r["alt_m"] for r in self.records]
        ticks = [r["tick"] for r in self.records]

        all_rate_names = set()
        for r in self.records:
            all_rate_names.update(r["rates"].keys())
        rate_arrays = {}
        for name in sorted(all_rate_names):
            rate_arrays[name] = np.array([r["rates"].get(name, 0.0) for r in self.records])

        input_arrays = {}
        for name in sorted(self.records[0]["inputs"].keys()):
            input_arrays[name] = np.array([r["inputs"].get(name, []) for r in self.records])

        np.savez(
            path,
            times=np.array(times), alts=np.array(alts), ticks=np.array(ticks),
            **rate_arrays,
            **{f"input_{k}": v for k, v in input_arrays.items()},
            cmd_throttle=np.array([r["cmd"]["throttle"] for r in self.records]),
            cmd_yaw=np.array([r["cmd"]["yaw"] for r in self.records]),
            cmd_forward=np.array([r["cmd"]["forward"] for r in self.records]),
            cmd_lateral=np.array([r["cmd"]["lateral"] for r in self.records]),
            cmd_escape=np.array([r["cmd"]["escape"] for r in self.records]),
            motors_m1=np.array([r["motors"][0] for r in self.records]),
            motors_m2=np.array([r["motors"][1] for r in self.records]),
            motors_m3=np.array([r["motors"][2] for r in self.records]),
            motors_m4=np.array([r["motors"][3] for r in self.records]),
            pid_thrust=np.array([r["pid_thrust"] for r in self.records]),
            brain_thrust=np.array([r["brain_thrust"] for r in self.records]),
            retina_exp=np.array([r["retina"].get("expansion", 0.0) for r in self.records]),
            retina_rot=np.array([r["retina"].get("rotation", 0.0) for r in self.records]),
            retina_vert=np.array([r["retina"].get("vertical", 0.0) for r in self.records]),
        )
        print(f"\n📊 Logs sauvegardés : {path} ({self.tick_count} ticks)")

    def print_summary(self):
        if not self.records:
            print("Aucune donnée enregistrée")
            return
        print("\n" + "=" * 60)
        print("RÉSUMÉ DE L'ACTIVITÉ NEURONALE")
        print("=" * 60)
        all_rate_names = set()
        for r in self.records:
            all_rate_names.update(r["rates"].keys())
        for name in sorted(all_rate_names):
            values = [r["rates"].get(name, 0.0) for r in self.records]
            print(f"  {name:20s}: min={min(values):6.1f}  max={max(values):6.1f}  "
                  f"mean={np.mean(values):6.1f}  std={np.std(values):6.1f}")


# ======================================================================
# Apprentissage synaptique
# ======================================================================
class SynapticLearner:
    """
    Apprentissage synaptique sur le connectome.

    Règle : si récompense positive, renforcer les synapses sortantes des neurones
            éligibles ; si récompense négative (crash), les affaiblir.

    Les poids restent bornés : le signe d'une synapse ne change jamais et son
    amplitude reste entre 0 et `max_scale` fois sa valeur d'origine.
    """

    def __init__(self, connectome, learning_rate=0.001, decay=0.95,
                 max_scale=3.0, eligibility_threshold=0.1, max_weight_change=0.05):
        self.connectome = connectome
        self.lr = learning_rate
        self.decay = decay
        self.max_scale = max_scale
        self.elig_thr = eligibility_threshold
        self.max_weight_change = max_weight_change
        self.n = connectome.n

        # On travaille en CSC : colonne = neurone présynaptique, lignes = postsynaptiques
        W = connectome.weights
        if not sp.isspmatrix_csc(W):
            print(f"⚠️  Matrice de poids en {type(W).__name__}, conversion en CSC")
            W = W.tocsc()
            connectome.weights = W
        self.W = W

        # Colonne (pré) de chaque entrée de W.data -> permet une mise à jour vectorisée
        self.col_of_entry = np.repeat(np.arange(W.shape[1]), np.diff(W.indptr))

        # Poids d'origine (pour les bornes)
        self.W_original = W.data.copy()
        self.w_min = np.minimum(0.0, self.W_original * self.max_scale)
        self.w_max = np.maximum(0.0, self.W_original * self.max_scale)

        # Trace d'éligibilité
        self.eligibility = np.zeros(self.n, dtype=np.float32)

        self.total_updates = 0
        self.total_reward = 0.0

    def update_eligibility(self, rates: dict):
        """Met à jour la trace d'éligibilité selon l'activité des neurones DN."""
        self.eligibility *= self.decay
        for name, rate in rates.items():
            if name.startswith("DN"):
                idx = self.connectome.group(name)
                if idx.size > 0:
                    activity = np.clip(rate / 50.0, 0, 1)
                    self.eligibility[idx] += activity

    def apply_reward(self, reward: float):
        """Renforce (reward > 0) ou affaiblit (reward < 0) les synapses des neurones éligibles."""
        if abs(reward) < 0.01:
            return

        eligible = self.eligibility > self.elig_thr
        if not np.any(eligible):
            return

        # Seules les synapses dont le neurone présynaptique est éligible sont modifiées
        entry_mask = eligible[self.col_of_entry]
        updated = int(entry_mask.sum())
        if updated == 0:
            return

        delta = self.lr * reward * self.eligibility[self.col_of_entry[entry_mask]]
        new_data = self.W.data.copy()
        new_data[entry_mask] += delta
        # Bornes : pas d'inversion de signe, amplitude limitée
        new_data = np.clip(new_data, self.w_min, self.w_max)
        self.W.data[:] = new_data

        self.total_updates += 1
        self.total_reward += reward

        if abs(reward) > 0.5:
            print(f"  🧠 Mise à jour synaptique : reward={reward:+.2f}, {updated} synapses modifiées")

    def save(self, path: str):
        W = self.W
        np.savez(path,
                 weights_data=W.data,
                 weights_indices=W.indices,
                 weights_indptr=W.indptr,
                 weights_shape=W.shape,
                 total_updates=self.total_updates,
                 total_reward=self.total_reward)
        print(f"💾 Poids synaptiques sauvegardés : {path}")

    def load(self, path: str):
        data = np.load(path)
        W = self.W
        if data["weights_data"].shape != W.data.shape:
            print("❌ Fichier de poids incompatible (structure différente), chargement annulé")
            return
        W.data[:] = data["weights_data"]
        self.total_updates = int(data["total_updates"]) if "total_updates" in data else 0
        self.total_reward = float(data["total_reward"]) if "total_reward" in data else 0.0
        print(f"📂 Poids synaptiques chargés : {path} ({self.total_updates} mises à jour)")


# ======================================================================
# Système de récompense
# ======================================================================
class RewardSystem:
    def __init__(self):
        self.total_reward = 0.0
        self.episode_reward = 0.0
        self.steps = 0
        self.crashes = 0
        self.stable_steps = 0
        self.start_pos = None
        self.best_dist = 0.0

    def compute_reward(self, alt_m, crashed, pos_x, pos_y, start_pos, has_obstacle_ahead):
        """
        - +0.01 survie par tick
        - +0.05 * gain de distance maximale depuis le départ (exploration)
        - -10.0 si crash (événement unique)
        - -0.5 si obstacle trop proche devant
        - +0.02 si altitude stable entre 0.8 et 1.2 m
        """
        reward = 0.01

        if crashed:
            reward = -10.0
            self.crashes += 1

        if start_pos is not None:
            dist = np.sqrt((pos_x - start_pos[0]) ** 2 + (pos_y - start_pos[1]) ** 2)
            if dist > self.best_dist:
                reward += 0.05 * (dist - self.best_dist)
                self.best_dist = dist

        if has_obstacle_ahead:
            reward -= 0.5

        if 0.8 <= alt_m <= 1.2:
            reward += 0.02
            self.stable_steps += 1

        self.episode_reward += reward
        self.total_reward += reward
        self.steps += 1
        return reward

    def reset_episode(self, start_pos):
        self.episode_reward = 0.0
        self.steps = 0
        self.start_pos = start_pos
        self.best_dist = 0.0


# ======================================================================
# Contrôleur principal
# ======================================================================
class CrazyflieFlyDronesRLController:
    # Limites d'altitude pour la détection de crash
    ALT_MIN_CRASH = 0.15
    ALT_MAX_CRASH = 1.8       # marge au-dessus de la cible (1.0 m) pour éviter les faux crashs
    RESPAWN_DELAY_S = 2.0     # temps moteurs coupés avant de relancer un épisode

    # ---- Paramètres de stabilité (anti-instabilité) ----
    SMOOTH_ALPHA = 0.25       # lissage des commandes cerveau (0 = gel, 1 = brut)
    MAX_CMD_RATE = 0.8        # commandes du cerveau bornées (throttle/yaw/fwd/lat)
    BRAIN_SILENCE_TIMEOUT = 1.0  # si le cerveau ne tire plus -> hover
    DECODER_SMOOTHING = 0.35  # EMA de lissage des axes du décodeur

    def __init__(self):
        self.robot = Robot()
        self.timestep = int(self.robot.getBasicTimeStep())

        # Config MaleCNS
        config_path = Path(__file__).resolve().parents[2] / "flydrones_lib" / "configs" / "malecns.yaml"
        self.cfg = load_config(str(config_path) if config_path.exists() else None)

        # Connectome réduit
        brain_path = Path(__file__).resolve().parents[2] / "flydrones_lib" / "data" / "malecns_brain_reduced.npz"
        self.cfg["brain"]["source"] = str(brain_path)
        print(f"🧠 Chargement du connectome : {brain_path}")
        connectome = load_connectome(self.cfg["brain"]["source"])
        self.brain = Brain(connectome, self.cfg)
        self.retina = Retina.from_config(self.cfg)
        self.encoder = InputEncoder(self.brain.connectome, self.cfg)
        self.decoder = MotorDecoder(self.cfg)
        self.safety = SafetyGovernor(self.cfg)

        # Apprentissage (on passe le connectome du Brain pour être sûr de modifier le même objet)
        self.learner = SynapticLearner(self.brain.connectome, learning_rate=0.0005)
        self.reward = RewardSystem()

        # Logger
        log_dir = Path(__file__).resolve().parents[2] / "flydrones_lib" / "data" / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        self.log_path = str(log_dir / f"rl_log_{int(time.time())}.npz")
        self.logger = NeuralLogger(self.log_path)

        # Sauvegarde automatique à la sortie (une seule fois)
        atexit.register(self._save_all)

        # Devices Webots
        self.camera = self.robot.getDevice("camera")
        self.camera.enable(self.timestep)
        self.imu = self.robot.getDevice("inertial_unit")
        self.imu.enable(self.timestep)
        self.gyro = self.robot.getDevice("gyro")
        self.gyro.enable(self.timestep)
        self.gps = self.robot.getDevice("gps")
        self.gps.enable(self.timestep)
        self.range_front = self.robot.getDevice("range_front")
        self.range_front.enable(self.timestep)
        self.range_left = self.robot.getDevice("range_left")
        self.range_left.enable(self.timestep)
        self.range_right = self.robot.getDevice("range_right")
        self.range_right.enable(self.timestep)
        self.range_back = self.robot.getDevice("range_back")
        self.range_back.enable(self.timestep)

        self.m1 = self.robot.getDevice("m1_motor")
        self.m2 = self.robot.getDevice("m2_motor")
        self.m3 = self.robot.getDevice("m3_motor")
        self.m4 = self.robot.getDevice("m4_motor")
        for m in (self.m1, self.m2, self.m3, self.m4):
            m.setPosition(float("inf"))
            m.setVelocity(0.0)

        # État
        self.past_x = 0.0
        self.past_y = 0.0
        self.past_time = self.robot.getTime()
        self.first_time = True
        self.t = 0.0
        self.past_altitude_error = 0.0
        self.start_pos = None

        # PID altitude
        self.FLYING_ALTITUDE = 1.0
        self.TRIM_BASE = 45.8
        self.KP_Z = 15.0
        self.KD_Z = 8.0
        self.MAX_THRUST = 200.0
        self.MAX_THRUST_STEP = 10.0
        self.last_thrust = 90.0

        # PID attitude
        self.past_roll_error = 0.0
        self.past_pitch_error = 0.0
        self.KP_ATT = 0.5
        self.KD_ATT = 0.1
        self.MAX_ATT = 15.0

        # Lissage des commandes du cerveau (anti-oscillation)
        self._cmd_smooth = {"throttle": 0.0, "yaw": 0.0, "forward": 0.0, "lateral": 0.0}
        self._cmd_prev = dict(self._cmd_smooth)
        self._last_brain_spike_t = 0.0
        self._brain_silent = False

        # Sécurité / épisodes
        self.emergency_landed = False
        self.crash_time = 0.0
        self.debug_counter = 0
        self.has_taken_off = False
        self.episode_count = 0

        print("=== Crazyflie FlyDrones RL (Apprentissage Synaptique) ===")
        print(f"Brain: {self.brain.connectome.name} ({self.brain.n_neurons} neurones)")
        print("Apprentissage synaptique: ACTIF")
        print(f"Logging: {self.log_path}")
        print("=======================================")

    # ------------------------------------------------------------------
    def _save_all(self):
        """Sauvegarde finale (appelée une seule fois via atexit)."""
        try:
            if self.logger.tick_count > 0:
                self.logger.save()
                self.logger.print_summary()
            self.learner.save(str(Path(self.log_path).with_suffix(".weights.npz")))
        except Exception as e:
            print(f"Erreur sauvegarde : {e}")

    def _get_camera_frame(self):
        image = self.camera.getImage()
        if image is None:
            return None
        width = self.camera.getWidth()
        height = self.camera.getHeight()
        buf = np.frombuffer(image, dtype=np.uint8)
        return buf.reshape((height, width, 4))[:, :, :3]

    def _get_telemetry(self):
        roll, pitch, yaw = self.imu.getRollPitchYaw()
        yaw_rate = self.gyro.getValues()[2]
        x, y, z = self.gps.getValues()
        tel = Telemetry(t=self.t, alt_m=float(z), vz_mps=0.0,
                        yaw_deg=float(np.degrees(yaw) % 360),
                        yaw_rate_dps=float(np.degrees(yaw_rate)),
                        x_m=float(x), y_m=float(y), battery_pct=100.0, flying=True)
        return tel, roll, pitch, yaw

    def _compute_body_velocities(self, x, y, yaw, dt):
        if self.first_time:
            self.past_x = x
            self.past_y = y
            self.past_time = self.robot.getTime()
            self.first_time = False
            return 0.0, 0.0
        vx_global = (x - self.past_x) / dt
        vy_global = (y - self.past_y) / dt
        cos_yaw = np.cos(yaw)
        sin_yaw = np.sin(yaw)
        vx = vx_global * cos_yaw + vy_global * sin_yaw
        vy = -vx_global * sin_yaw + vy_global * cos_yaw
        self.past_x = x
        self.past_y = y
        return float(vx), float(vy)

    def _pid_altitude(self, alt_m, dt):
        altitude_error = self.FLYING_ALTITUDE - alt_m
        altitude_derivative = (altitude_error - self.past_altitude_error) / dt if dt > 0 else 0.0
        thrust = (self.TRIM_BASE
                  + self.KP_Z * max(-1.0, min(1.0, altitude_error))
                  + self.KD_Z * altitude_derivative)
        max_step = self.MAX_THRUST_STEP * max(dt, 0.001)
        thrust = max(self.last_thrust - max_step, min(self.last_thrust + max_step, thrust))
        thrust = max(0.0, min(self.MAX_THRUST, thrust))
        self.last_thrust = thrust
        self.past_altitude_error = altitude_error
        return thrust

    def _pid_attitude(self, roll_desired, pitch_desired, roll_actual, pitch_actual, dt):
        roll_error = roll_desired - roll_actual
        pitch_error = pitch_desired - pitch_actual
        roll_derivative = (roll_error - self.past_roll_error) / dt if dt > 0 else 0.0
        pitch_derivative = (pitch_error - self.past_pitch_error) / dt if dt > 0 else 0.0
        roll_cmd = self.KP_ATT * roll_error + self.KD_ATT * roll_derivative
        pitch_cmd = self.KP_ATT * pitch_error + self.KD_ATT * pitch_derivative
        roll_cmd = max(-self.MAX_ATT, min(self.MAX_ATT, roll_cmd))
        pitch_cmd = max(-self.MAX_ATT, min(self.MAX_ATT, pitch_cmd))
        self.past_roll_error = roll_error
        self.past_pitch_error = pitch_error
        return roll_cmd, pitch_cmd

    def _apply_motor_commands(self, cmd, alt_m, roll_actual, pitch_actual, dt):
        max_vel = 600.0
        base_thrust = self._pid_altitude(alt_m, dt)
        # Correction du cerveau (petite influence sur la poussée)
        brain_thrust = cmd.throttle * 20.0
        base = base_thrust + brain_thrust

        roll_desired = cmd.lateral * 0.2
        pitch_desired = cmd.forward * 0.2
        roll_cmd, pitch_cmd = self._pid_attitude(roll_desired, pitch_desired,
                                                 roll_actual, pitch_actual, dt)
        yaw = cmd.yaw

        m1 = base - roll_cmd + pitch_cmd + yaw * 20
        m2 = base - roll_cmd - pitch_cmd - yaw * 20
        m3 = base + roll_cmd - pitch_cmd + yaw * 20
        m4 = base + roll_cmd + pitch_cmd - yaw * 20
        m1 = max(0.0, min(max_vel, m1))
        m2 = max(0.0, min(max_vel, m2))
        m3 = max(0.0, min(max_vel, m3))
        m4 = max(0.0, min(max_vel, m4))
        self.m1.setVelocity(-m1)
        self.m2.setVelocity(m2)
        self.m3.setVelocity(-m3)
        self.m4.setVelocity(m4)
        return m1, m2, m3, m4, roll_cmd, pitch_cmd, base_thrust, brain_thrust

    def _stop_motors(self):
        for m in (self.m1, self.m2, self.m3, self.m4):
            m.setVelocity(0.0)

    def _reset_flight_state(self):
        """Remet les PID à zéro pour un nouvel épisode."""
        self.last_thrust = self.TRIM_BASE
        self.past_altitude_error = 0.0
        self.past_roll_error = 0.0
        self.past_pitch_error = 0.0

    # ------------------------------------------------------------------
    def run(self):
        while self.robot.step(self.timestep) != -1:
            self.t = self.robot.getTime()
            dt = self.t - self.past_time if self.past_time > 0 else self.timestep / 1000.0
            if dt <= 0:
                dt = self.timestep / 1000.0

            # 1. Caméra
            frame = self._get_camera_frame()
            vision = self.retina.encode(frame)

            # 2. IMU
            roll, pitch, yaw = self.imu.getRollPitchYaw()
            yaw_rate = self.gyro.getValues()[2]
            yaw_rate_dps = float(np.degrees(yaw_rate))

            # 3. GPS
            x, y, z = self.gps.getValues()
            vx, vy = self._compute_body_velocities(float(x), float(y), float(yaw), dt)

            # 4. Télémétrie
            tel, roll_actual, pitch_actual, yaw_actual = self._get_telemetry()

            # 5. Obstacle devant (range_front)
            obstacle_ahead = self.range_front.getValue() < 0.5  # moins de 50 cm

            # 6. Encoder
            inputs = self.encoder.encode(vision, yaw_rate_dps)

            # 7. Cerveau (LIF)
            rates = self.brain.tick(inputs, ms=dt * 1000.0)

            # 8. Commande manuelle depuis le GUI (flèches) - priorité sur le connectome
            manual = get_manual_cmd()
            if manual["active"]:
                # Override manuel : on force les axes demandés
                raw_cmd = FlightCommand(
                    throttle=manual["throttle"],
                    yaw=manual["yaw"],
                    forward=manual["forward"],
                    lateral=manual["lateral"],
                    escape=0.0,
                )
            else:
                # Watchdog du cerveau : si aucun DN n'a tiré depuis > BRAIN_SILENCE_TIMEOUT
                any_dn = any(k.startswith("DN") and v > 0.5 for k, v in rates.items())
                if any_dn:
                    self._last_brain_spike_t = self.t
                    self._brain_silent = False
                elif self.t - self._last_brain_spike_t > self.BRAIN_SILENCE_TIMEOUT:
                    self._brain_silent = True

                raw_cmd = self.decoder.update(rates, dt)

                # Lissage EMA des 4 axes du cerveau (anti-oscillation / instabilité)
                if not self._brain_silent:
                    axes = ["throttle", "yaw", "forward", "lateral"]
                    for ax in axes:
                        v = max(-self.MAX_CMD_RATE, min(self.MAX_CMD_RATE, getattr(raw_cmd, ax)))
                        self._cmd_smooth[ax] = (1.0 - self.SMOOTH_ALPHA) * self._cmd_smooth[ax] + self.SMOOTH_ALPHA * v
                    raw_cmd = FlightCommand(
                        throttle=self._cmd_smooth["throttle"],
                        yaw=self._cmd_smooth["yaw"],
                        forward=self._cmd_smooth["forward"],
                        lateral=self._cmd_smooth["lateral"],
                        escape=getattr(raw_cmd, "escape", 0.0),
                    )
                else:
                    # Cerveau silencieux : hover sécurisé
                    raw_cmd = FlightCommand(throttle=0.0, yaw=0.0, forward=0.0, lateral=0.0, escape=0.0)

            # --- Évitement proactif d'obstacles (RL) ---
            # Si obstacle devant : freiner, monter, dévier latéralement
            obstacle_ahead = self.range_front.getValue() < 0.5
            obstacle_left = self.range_left.getValue() < 0.4
            obstacle_right = self.range_right.getValue() < 0.4
            avoid_forward = 0.0
            avoid_lateral = 0.0
            avoid_throttle = 0.0
            if obstacle_ahead:
                avoid_forward = -0.4
                avoid_throttle = 0.15
                if obstacle_left and not obstacle_right:
                    avoid_lateral = 0.4
                elif obstacle_right and not obstacle_left:
                    avoid_lateral = -0.4
            elif obstacle_left:
                avoid_lateral = 0.3
            elif obstacle_right:
                avoid_lateral = -0.3

            # Mélanger commande cerveau + évitement (évitement prioritaire sur forward/lateral)
            blended_cmd = FlightCommand(
                throttle=max(-1.0, min(1.0, raw_cmd.throttle + avoid_throttle)),
                yaw=raw_cmd.yaw,
                forward=max(-1.0, min(1.0, raw_cmd.forward + avoid_forward)),
                lateral=max(-1.0, min(1.0, raw_cmd.lateral + avoid_lateral)),
                escape=getattr(raw_cmd, "escape", 0.0),
            )
            safe_cmd = self.safety.filter(blended_cmd, tel, dt)

            # 9. Délai de démarrage
            if self.t < 3.0:
                self._stop_motors()
                self.past_time = self.t
                continue

            # Position de départ du premier épisode
            if self.start_pos is None:
                self.start_pos = (float(x), float(y))
                self.reward.reset_episode(self.start_pos)

            # 10. Détection de crash (événement unique, pas à chaque tick)
            crashed_now = (self.has_taken_off
                           and not self.emergency_landed
                           and (tel.alt_m > self.ALT_MAX_CRASH or tel.alt_m < self.ALT_MIN_CRASH))
            if crashed_now:
                self.emergency_landed = True
                self.crash_time = self.t
                print(f"!!! CRASH : altitude {tel.alt_m:.2f}m !!!")

            # 11. Récompense
            reward = self.reward.compute_reward(tel.alt_m, crashed_now, float(x), float(y),
                                                self.start_pos, obstacle_ahead)

            # 12. Apprentissage synaptique
            self.learner.update_eligibility(rates)
            if abs(reward) > 0.5:  # seulement pour les événements significatifs
                self.learner.apply_reward(reward)

            # 13. Redémarrage après crash : moteurs coupés RESPAWN_DELAY_S puis nouvel épisode
            if self.emergency_landed and (self.t - self.crash_time) >= self.RESPAWN_DELAY_S:
                self.emergency_landed = False
                self.has_taken_off = False
                self.episode_count += 1
                self._reset_flight_state()
                self.start_pos = (float(x), float(y))
                self.reward.reset_episode(self.start_pos)
                # Reset du lissage et watchdog
                self._cmd_smooth = {"throttle": 0.0, "yaw": 0.0, "forward": 0.0, "lateral": 0.0}
                self._cmd_prev = dict(self._cmd_smooth)
                self._last_brain_spike_t = self.t
                self._brain_silent = False
                print(f"!!! Nouvel épisode {self.episode_count + 1} (alt={tel.alt_m:.2f}m) !!!")

            if tel.alt_m > 0.3:
                self.has_taken_off = True

            # 14. Moteurs
            if self.emergency_landed:
                self._stop_motors()
                m1 = m2 = m3 = m4 = 0.0
                roll_cmd = pitch_cmd = 0.0
                base_thrust = 0.0
                brain_thrust = 0.0
            else:
                (m1, m2, m3, m4, roll_cmd, pitch_cmd,
                 base_thrust, brain_thrust) = self._apply_motor_commands(
                    safe_cmd, tel.alt_m, roll_actual, pitch_actual, dt)

            # 15. Log neuronal
            retina_features = vision.scalars() if vision else {"expansion": 0, "rotation": 0, "vertical": 0}
            self.logger.log_tick(self.t, dt, tel.alt_m, rates, inputs, safe_cmd,
                                 (m1, m2, m3, m4), retina_features, base_thrust, brain_thrust)

            # 16. Log périodique
            self.debug_counter += 1
            if self.debug_counter % 20 == 0:
                dn_rates = {k: round(v, 1) for k, v in rates.items() if k.startswith("DN")}
                print(f"t={self.t:5.1f}s alt={tel.alt_m:5.2f}m reward={reward:+.2f} "
                      f"cmd(thr={safe_cmd.throttle:+.2f} fwd={safe_cmd.forward:+.2f} lat={safe_cmd.lateral:+.2f}) "
                      f"retina(exp={retina_features['expansion']:+.2f}) DN={dn_rates} "
                      f"PID={base_thrust:.0f} Brain={brain_thrust:+.0f} "
                      f"motors(m1={m1:.0f} m2={m2:.0f} m3={m3:.0f} m4={m4:.0f})")

            # 17. Écriture de l'état partagé pour le GUI du connectome
            try:
                # Synapses actives : top 200 synapses par poids absolu pour le GUI
                W = self.brain.connectome.weights
                if sp.isspmatrix(W):
                    W_coo = W.tocoo()
                    abs_w = np.abs(W_coo.data)
                    top_idx = np.argsort(abs_w)[-200:][::-1] if len(abs_w) > 0 else []
                    synapses_active = [
                        (int(W_coo.row[i]), int(W_coo.col[i]), float(W_coo.data[i]))
                        for i in top_idx
                    ]
                else:
                    synapses_active = []

                gui_state = {
                    "t": self.t,
                    "alt_m": float(tel.alt_m),
                    "rates": {k: float(v) for k, v in rates.items()},
                    "inputs": {k: float(v) if np.isscalar(v) else v.tolist() for k, v in inputs.items()},
                    "cmd": {
                        "throttle": float(safe_cmd.throttle),
                        "yaw": float(safe_cmd.yaw),
                        "forward": float(safe_cmd.forward),
                        "lateral": float(safe_cmd.lateral),
                        "escape": float(safe_cmd.escape),
                    },
                    "motors": [float(m1), float(m2), float(m3), float(m4)],
                    "retina": retina_features,
                    "manual": manual,
                    "synapses_active": synapses_active,
                    "crash": bool(self.emergency_landed),
                    "brain_silent": self._brain_silent,
                    "smooth_cmd": {
                        "throttle": float(self._cmd_smooth["throttle"]),
                        "yaw": float(self._cmd_smooth["yaw"]),
                        "forward": float(self._cmd_smooth["forward"]),
                        "lateral": float(self._cmd_smooth["lateral"]),
                    },
                }
                write_state(gui_state)
            except Exception as e:
                # Ne pas faire planter le contrôleur si l'écriture GUI échoue
                pass

            self.past_time = self.t

        # La sauvegarde finale est faite par _save_all (atexit)


def main():
    controller = CrazyflieFlyDronesRLController()
    controller.run()


if __name__ == "__main__":
    main()
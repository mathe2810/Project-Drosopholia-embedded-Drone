#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
crazyflie_flydrones_minifly.py - Contrôleur Crazyflie avec le cerveau Minifly (synthétique).

Minifly est un circuit de ~1000 neurones câblé à la main (pas de téléchargement 1.1 GB MaleCNS).
Il suit la même pipeline FlyDrones :
  caméra -> rétine (flux optique, looming) -> encodeur Poisson -> LIF -> décodeur DN -> moteurs

Usage dans Webots : mettre ce fichier comme contrôleur du robot Crazyflie.
"""

from __future__ import annotations

import sys
from pathlib import Path
import time
import atexit
import numpy as np
import scipy.sparse as sp

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
# Contrôleur principal (Minifly)
# ======================================================================
class CrazyflieFlyDronesMiniflyController:
    # Limites d'altitude pour la détection de crash
    ALT_MIN_CRASH = 0.15
    ALT_MAX_CRASH = 1.8
    RESPAWN_DELAY_S = 2.0

    # ---- Paramètres de stabilité ----
    SMOOTH_ALPHA = 0.25
    MAX_CMD_RATE = 0.8
    BRAIN_SILENCE_TIMEOUT = 1.0

    def __init__(self):
        self.robot = Robot()
        self.timestep = int(self.robot.getBasicTimeStep())

        # Config Minifly (source: minifly = cerveau synthétique intégré)
        config_path = Path(__file__).resolve().parents[2] / "flydrones_lib" / "configs" / "crazyflie.yaml"
        self.cfg = load_config(str(config_path) if config_path.exists() else None)
        self.cfg["brain"]["source"] = "minifly"
        print("🧠 Cerveau Minifly (synthétique, ~1000 neurones)")

        connectome = load_connectome(self.cfg["brain"]["source"])
        self.brain = Brain(connectome, self.cfg)
        self.retina = Retina.from_config(self.cfg)
        self.encoder = InputEncoder(self.brain.connectome, self.cfg)
        self.decoder = MotorDecoder(self.cfg)
        self.safety = SafetyGovernor(self.cfg)

        # Logger
        log_dir = Path(__file__).resolve().parents[2] / "flydrones_lib" / "data" / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        self.log_path = str(log_dir / f"minifly_log_{int(time.time())}.npz")
        self.records = []
        self.tick_count = 0
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

        # Lissage des commandes du cerveau
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

        print("=== Crazyflie FlyDrones Minifly ===")
        print(f"Brain: {self.brain.connectome.name} ({self.brain.n_neurons} neurones)")
        print("=======================================")

    # ------------------------------------------------------------------
    def _save_all(self):
        if self.records:
            import numpy as np
            times = [r["t"] for r in self.records]
            alts = [r["alt_m"] for r in self.records]
            np.savez(self.log_path,
                     times=np.array(times), alts=np.array(alts),
                     cmd_throttle=np.array([r["cmd"]["throttle"] for r in self.records]),
                     cmd_yaw=np.array([r["cmd"]["yaw"] for r in self.records]),
                     cmd_forward=np.array([r["cmd"]["forward"] for r in self.records]),
                     cmd_lateral=np.array([r["cmd"]["lateral"] for r in self.records]),
                     motors_m1=np.array([r["motors"][0] for r in self.records]),
                     motors_m2=np.array([r["motors"][1] for r in self.records]),
                     motors_m3=np.array([r["motors"][2] for r in self.records]),
                     motors_m4=np.array([r["motors"][3] for r in self.records]))
            print(f"\n📊 Logs sauvegardés : {self.log_path} ({self.tick_count} ticks)")

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

            # 5. Obstacles (range sensors)
            obstacle_ahead = self.range_front.getValue() < 0.5
            obstacle_left = self.range_left.getValue() < 0.4
            obstacle_right = self.range_right.getValue() < 0.4

            # 6. Encoder
            inputs = self.encoder.encode(vision, yaw_rate_dps)

            # 7. Cerveau (LIF)
            rates = self.brain.tick(inputs, ms=dt * 1000.0)

            # 8. Commande manuelle depuis le GUI (priorité)
            manual = get_manual_cmd()
            if manual["active"]:
                raw_cmd = FlightCommand(
                    throttle=manual["throttle"],
                    yaw=manual["yaw"],
                    forward=manual["forward"],
                    lateral=manual["lateral"],
                    escape=0.0,
                )
            else:
                # Watchdog cerveau
                any_dn = any(k.startswith("DN") and v > 0.5 for k, v in rates.items())
                if any_dn:
                    self._last_brain_spike_t = self.t
                    self._brain_silent = False
                elif self.t - self._last_brain_spike_t > self.BRAIN_SILENCE_TIMEOUT:
                    self._brain_silent = True

                raw_cmd = self.decoder.update(rates, dt)

                # Lissage EMA
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
                    raw_cmd = FlightCommand(throttle=0.0, yaw=0.0, forward=0.0, lateral=0.0, escape=0.0)

            # --- Évitement d'obstacles proactif (Minifly) ---
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

            blended_cmd = FlightCommand(
                throttle=max(-1.0, min(1.0, raw_cmd.throttle + avoid_throttle)),
                yaw=raw_cmd.yaw,
                forward=max(-1.0, min(1.0, raw_cmd.forward + avoid_forward)),
                lateral=max(-1.0, min(1.0, raw_cmd.lateral + avoid_lateral)),
                escape=raw_cmd.escape,
            )

            safe_cmd = self.safety.filter(blended_cmd, tel, dt)

            # 9. Délai de démarrage
            if self.t < 3.0:
                self._stop_motors()
                self.past_time = self.t
                continue

            # Position de départ
            if self.start_pos is None:
                self.start_pos = (float(x), float(y))

            # 10. Détection de crash
            crashed_now = (self.has_taken_off
                           and not self.emergency_landed
                           and (tel.alt_m > self.ALT_MAX_CRASH or tel.alt_m < self.ALT_MIN_CRASH))
            if crashed_now:
                self.emergency_landed = True
                self.crash_time = self.t
                print(f"!!! CRASH : altitude {tel.alt_m:.2f}m !!!")

            # 11. Redémarrage après crash
            if self.emergency_landed and (self.t - self.crash_time) >= self.RESPAWN_DELAY_S:
                self.emergency_landed = False
                self.has_taken_off = False
                self.episode_count += 1
                self._reset_flight_state()
                self.start_pos = (float(x), float(y))
                self._cmd_smooth = {"throttle": 0.0, "yaw": 0.0, "forward": 0.0, "lateral": 0.0}
                self._cmd_prev = dict(self._cmd_smooth)
                self._last_brain_spike_t = self.t
                self._brain_silent = False
                print(f"!!! Nouvel épisode {self.episode_count + 1} (alt={tel.alt_m:.2f}m) !!!")

            if tel.alt_m > 0.3:
                self.has_taken_off = True

            # 12. Moteurs
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

            # 13. Log neuronal
            retina_features = vision.scalars() if vision else {"expansion": 0, "rotation": 0, "vertical": 0}
            self.records.append({
                "t": self.t, "alt_m": tel.alt_m,
                "cmd": {"throttle": safe_cmd.throttle, "yaw": safe_cmd.yaw,
                        "forward": safe_cmd.forward, "lateral": safe_cmd.lateral},
                "motors": [m1, m2, m3, m4],
            })
            self.tick_count += 1

            # 14. Log périodique
            self.debug_counter += 1
            if self.debug_counter % 20 == 0:
                dn_rates = {k: round(v, 1) for k, v in rates.items() if k.startswith("DN")}
                print(f"t={self.t:5.1f}s alt={tel.alt_m:5.2f}m "
                      f"cmd(thr={safe_cmd.throttle:+.2f} fwd={safe_cmd.forward:+.2f} lat={safe_cmd.lateral:+.2f}) "
                      f"retina(exp={retina_features['expansion']:+.2f}) DN={dn_rates} "
                      f"PID={base_thrust:.0f} Brain={brain_thrust:+.0f} "
                      f"motors(m1={m1:.0f} m2={m2:.0f} m3={m3:.0f} m4={m4:.0f})")

            # 15. État partagé pour le GUI
            try:
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
            except Exception:
                pass

            self.past_time = self.t


def main():
    controller = CrazyflieFlyDronesMiniflyController()
    controller.run()


if __name__ == "__main__":
    main()
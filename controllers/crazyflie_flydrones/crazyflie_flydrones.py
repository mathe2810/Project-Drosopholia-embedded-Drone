#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Controleur Python Webots pour Crazyflie + cerveau FlyDrones (Drosophila SNN).

Ce controleur tourne EN PARALLELE du controleur C existant (crazyflie.c).
Il ne le remplace pas.

Pipeline par tick :
    camera Webots -> Retina.encode() -> grilles oeil de mouche
    IMU Webots    -> taux de lacet   -> InputEncoder
    GPS Webots    -> position/altitude -> Telemetry
    grilles + IMU -> InputEncoder.encode() -> {groupe: Hz}
    {groupe: Hz}  -> Brain.tick()      -> {groupe: rate Hz}
    taux          -> MotorDecoder.update() -> FlightCommand
    cmd + telemetrie -> SafetyGovernor.filter() -> safe FlightCommand
    safe cmd      -> PID altitude + PID attitude + melange moteur -> wb_motor_set_velocity()
"""

from __future__ import annotations

import sys
from pathlib import Path

# --- Ajouter flydrones_lib au path Python ---
FLYDRONES_SRC = Path(__file__).resolve().parents[2] / "flydrones_lib" / "src"
if str(FLYDRONES_SRC) not in sys.path:
    sys.path.insert(0, str(FLYDRONES_SRC))

from controller import Robot  # noqa: E402
import numpy as np  # noqa: E402

from flydrones.brain import Brain, load_connectome  # noqa: E402
from flydrones.config import load_config  # noqa: E402
from flydrones.motor.command import FlightCommand  # noqa: E402
from flydrones.motor.decoder import MotorDecoder  # noqa: E402
from flydrones.safety import SafetyGovernor, Telemetry  # noqa: E402
from flydrones.senses.encoder import InputEncoder  # noqa: E402
from flydrones.senses.retina import Retina  # noqa: E402


class CrazyflieFlyDronesController:
    """Controleur Webots qui pilote le Crazyflie avec le cerveau FlyDrones."""

    def __init__(self):
        self.robot = Robot()
        self.timestep = int(self.robot.getBasicTimeStep())

        # --- Charger la config FlyDrones (adaptee Crazyflie) ---
        config_path = Path(__file__).resolve().parents[2] / "flydrones_lib" / "configs" / "crazyflie.yaml"
        self.cfg = load_config(str(config_path) if config_path.exists() else None)
        self.cfg["brain"]["source"] = "minifly"

        # --- Charger le cerveau ---
        connectome = load_connectome(self.cfg["brain"]["source"])
        self.brain = Brain(connectome, self.cfg)
        self.retina = Retina.from_config(self.cfg)
        self.encoder = InputEncoder(self.brain.connectome, self.cfg)
        self.decoder = MotorDecoder(self.cfg)
        self.safety = SafetyGovernor(self.cfg)

        # --- Devices Webots ---
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
        self.range_back = self.robot.getDevice("range_back")
        self.range_back.enable(self.timestep)
        self.range_right = self.robot.getDevice("range_right")
        self.range_right.enable(self.timestep)

        self.m1 = self.robot.getDevice("m1_motor")
        self.m2 = self.robot.getDevice("m2_motor")
        self.m3 = self.robot.getDevice("m3_motor")
        self.m4 = self.robot.getDevice("m4_motor")
        for m in (self.m1, self.m2, self.m3, self.m4):
            m.setPosition(float("inf"))
            m.setVelocity(0.0)

        # --- Etat ---
        self.past_x = 0.0
        self.past_y = 0.0
        self.past_time = self.robot.getTime()
        self.first_time = True
        self.t = 0.0

        # --- PID d'altitude (P+D seulement, valeurs proches du C original) ---
        self.past_altitude_error = 0.0
        self.FLYING_ALTITUDE = 1.0
        self.TRIM_BASE = 45.8      # trim de base (comme le C original: +35)
        self.KP_Z = 10.0           # gain proportionnel altitude (comme le C)
        self.KD_Z = 5.0            # gain derive altitude (comme le C)
        self.MAX_THRUST = 100.0    # poussée max absolue
        self.MAX_THRUST_STEP = 5.0 # variation max de poussée par tick (slew rate)

        # --- PID d'attitude (roll/pitch) ---
        self.past_roll_error = 0.0
        self.past_pitch_error = 0.0
        self.KP_ATT = 0.5          # gain proportionnel attitude (comme le C: kp_att_rp=0.5)
        self.KD_ATT = 0.1          # gain derive attitude (comme le C: kd_att_rp=0.1)
        self.MAX_ATT = 15.0        # correction max d'attitude (rad/s)

        # --- Securite d'urgence ---
        self.emergency_landed = False
        self.last_thrust = 48.0    # pour slew rate

        # --- Compteur de debug ---
        self.debug_counter = 0

        print("=== Crazyflie FlyDrones Controller ===")
        print(f"Brain: {self.brain.connectome.name} ({self.brain.n_neurons} neurones)")
        print(f"Controle a {self.cfg['control']['hz']} Hz")
        print(f"Camera Webots activee (324x324)")
        print(f"Altitude cible: {self.FLYING_ALTITUDE} m")
        print(f"Trim de base: {self.TRIM_BASE} rad/s")
        print("=======================================")

    def _get_camera_frame(self):
        """Recupere l'image de la camera Webots comme tableau numpy BGR."""
        image = self.camera.getImage()
        if image is None:
            return None
        width = self.camera.getWidth()
        height = self.camera.getHeight()
        buf = np.frombuffer(image, dtype=np.uint8)
        img = buf.reshape((height, width, 4))[:, :, :3]  # BGRA -> BGR
        return img

    def _get_telemetry(self):
        """Construit la telemetrie depuis les capteurs Webots."""
        roll, pitch, yaw = self.imu.getRollPitchYaw()
        yaw_rate = self.gyro.getValues()[2]  # rad/s
        x, y, z = self.gps.getValues()
        return Telemetry(
            t=self.t,
            alt_m=float(z),
            vz_mps=0.0,
            yaw_deg=float(np.degrees(yaw) % 360),
            yaw_rate_dps=float(np.degrees(yaw_rate)),
            x_m=float(x),
            y_m=float(y),
            battery_pct=100.0,
            flying=True,
        ), roll, pitch, yaw

    def _compute_body_velocities(self, x, y, yaw, dt):
        """Calcule les vitesses dans le repere du corps."""
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
        """PID d'altitude P+D (valeurs proches du C original)."""
        altitude_error = self.FLYING_ALTITUDE - alt_m
        altitude_derivative = (altitude_error - self.past_altitude_error) / dt if dt > 0 else 0.0

        thrust = (
            self.TRIM_BASE
            + self.KP_Z * max(-1.0, min(1.0, altitude_error))
            + self.KD_Z * altitude_derivative
        )

        # Slew rate sur la poussée pour éviter les à-coups
        max_step = self.MAX_THRUST_STEP * max(dt, 0.001)
        thrust = max(self.last_thrust - max_step, min(self.last_thrust + max_step, thrust))
        thrust = max(0.0, min(self.MAX_THRUST, thrust))

        self.last_thrust = thrust
        self.past_altitude_error = altitude_error
        return thrust

    def _pid_attitude(self, roll_desired, pitch_desired, roll_actual, pitch_actual, dt):
        """PID d'attitude pour stabiliser roll et pitch (valeurs proches du C)."""
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
        """Applique la commande de vol aux 4 moteurs.

        Melange identique au motor_mixing du PID C :
          m1 = altitude - roll + pitch + yaw
          m2 = altitude - roll - pitch - yaw
          m3 = altitude + roll - pitch + yaw
          m4 = altitude + roll + pitch - yaw
        """
        max_vel = 600.0

        # --- PID d'altitude ---
        base_thrust = self._pid_altitude(alt_m, dt)

        # Correction throttle du decodeur FlyDrones (-1..1 -> -20..+20)
        throttle_correction = cmd.throttle * 20.0
        base = base_thrust + throttle_correction

        # --- PID d'attitude : forward/lateral du decodeur -> roll/pitch desire ---
        roll_desired = cmd.lateral * 0.2   # max 0.2 rad (~11 deg)
        pitch_desired = cmd.forward * 0.2  # max 0.2 rad (~11 deg)

        roll_cmd, pitch_cmd = self._pid_attitude(
            roll_desired, pitch_desired, roll_actual, pitch_actual, dt
        )

        # --- Commande yaw du decodeur ---
        yaw = cmd.yaw  # -1..1

        # --- Melange moteur identique au motor_mixing du PID C ---
        m1 = base - roll_cmd + pitch_cmd + yaw * 20
        m2 = base - roll_cmd - pitch_cmd - yaw * 20
        m3 = base + roll_cmd - pitch_cmd + yaw * 20
        m4 = base + roll_cmd + pitch_cmd - yaw * 20

        # Clamp
        m1 = max(0.0, min(max_vel, m1))
        m2 = max(0.0, min(max_vel, m2))
        m3 = max(0.0, min(max_vel, m3))
        m4 = max(0.0, min(max_vel, m4))

        self.m1.setVelocity(-m1)
        self.m2.setVelocity(m2)
        self.m3.setVelocity(-m3)
        self.m4.setVelocity(m4)

        return m1, m2, m3, m4, roll_cmd, pitch_cmd

    def run(self):
        """Boucle principale Webots."""
        while self.robot.step(self.timestep) != -1:
            self.t = self.robot.getTime()
            dt = self.t - self.past_time if self.past_time > 0 else self.timestep / 1000.0
            if dt <= 0:
                dt = self.timestep / 1000.0

            # --- 1. Camera -> Retine ---
            frame = self._get_camera_frame()
            vision = self.retina.encode(frame)

            # --- 2. IMU / Gyro -> taux de lacet + attitude ---
            roll, pitch, yaw = self.imu.getRollPitchYaw()
            yaw_rate = self.gyro.getValues()[2]
            yaw_rate_dps = float(np.degrees(yaw_rate))

            # --- 3. GPS -> position ---
            x, y, z = self.gps.getValues()
            vx, vy = self._compute_body_velocities(float(x), float(y), float(yaw), dt)

            # --- 4. Telemetrie ---
            tel, roll_actual, pitch_actual, yaw_actual = self._get_telemetry()

            # --- 5. Encoder -> taux de Poisson ---
            inputs = self.encoder.encode(vision, yaw_rate_dps)

            # --- 6. Cerveau (LIF) ---
            rates = self.brain.tick(inputs, ms=dt * 1000.0)

            # --- 7. Decodeur -> commande de vol ---
            raw_cmd = self.decoder.update(rates, dt)

            # --- 8. Securite ---
            safe_cmd = self.safety.filter(raw_cmd, tel, dt)

            # --- 9. Attente 3s pour stabilisation du decodeur ---
            if self.t < 3.0:
                self.m1.setVelocity(0.0)
                self.m2.setVelocity(0.0)
                self.m3.setVelocity(0.0)
                self.m4.setVelocity(0.0)
                self.past_time = self.t
                continue

            # --- 10. SECURITE D'URGENCE : couper si altitude hors [0.2, 1.2] ---
            if tel.alt_m > 1.2 or tel.alt_m < 0.01  :
                self.emergency_landed = True
                print(f"!!! SECURITE : altitude {tel.alt_m:.2f}m hors [0.2, 1.2], arret des moteurs !!!")

            # --- 11. Redemarrage autorise seulement si altitude dans [0.3, 1.1] ---
            if self.emergency_landed and 0.3 <= tel.alt_m <= 1.1:
                self.emergency_landed = False
                print(f"!!! SECURITE : altitude redescendue a {tel.alt_m:.2f}m, redemarrage autorise !!!")

            # --- 12. Appliquer aux moteurs (ou couper si urgence) ---
            if self.emergency_landed:
                self.m1.setVelocity(0.0)
                self.m2.setVelocity(0.0)
                self.m3.setVelocity(0.0)
                self.m4.setVelocity(0.0)
                m1 = m2 = m3 = m4 = 0
                roll_cmd = pitch_cmd = 0
            else:
                m1, m2, m3, m4, roll_cmd, pitch_cmd = self._apply_motor_commands(
                    safe_cmd, tel.alt_m, roll_actual, pitch_actual, dt
                )

            # --- 13. Log periodique (toutes les 0.5s) ---
            self.debug_counter += 1
            if self.debug_counter % 12 == 0:  # ~0.5s a 25 Hz
                features = vision.scalars() if vision else {"expansion": 0, "rotation": 0, "vertical": 0}
                dn_rates = {k: round(v, 1) for k, v in rates.items() if k.startswith("DN")}
                print(
                    f"t={self.t:5.1f}s alt={tel.alt_m:5.2f}m err={self.FLYING_ALTITUDE-tel.alt_m:+.2f}m "
                    f"att(r={roll_actual:+.2f} p={pitch_actual:+.2f} y={yaw_actual:+.2f}) "
                    f"cmd(thr={safe_cmd.throttle:+.2f} yaw={safe_cmd.yaw:+.2f} "
                    f"fwd={safe_cmd.forward:+.2f} lat={safe_cmd.lateral:+.2f}) "
                    f"retina(exp={features['expansion']:+.2f} rot={features['rotation']:+.2f} vert={features['vertical']:+.2f}) "
                    f"DN={dn_rates} "
                    f"PID(roll={roll_cmd:+.0f} pitch={pitch_cmd:+.0f}) "
                    f"motors(m1={m1:.0f} m2={m2:.0f} m3={m3:.0f} m4={m4:.0f})"
                )

            self.past_time = self.t


def main():
    controller = CrazyflieFlyDronesController()
    controller.run()


if __name__ == "__main__":
    main()

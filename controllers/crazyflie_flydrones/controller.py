#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Controleur Python Webots pour Crazyflie + cerveau FlyDrones (Drosophila SNN).

Ce controleur tourne EN PARALLELE du controleur C existant (crazyflie.c).
Il ne le remplace pas. Il lit les capteurs Webots, fait tourner le cerveau
drosophile (retine -> encodeur -> LIF -> decodeur -> securite) et applique
les commandes aux moteurs.

Pipeline par tick (20 Hz par defaut) :
    camera Webots -> Retina.encode() -> grilles oeil de mouche
    IMU Webots    -> taux de lacet   -> InputEncoder
    GPS Webots    -> position/altitude -> Telemetry
    grilles + IMU -> InputEncoder.encode() -> {groupe: Hz}
    {groupe: Hz}  -> Brain.tick()      -> {groupe: rate Hz}
    taux          -> MotorDecoder.update() -> FlightCommand
    cmd + telemetrie -> SafetyGovernor.filter() -> safe FlightCommand
    safe cmd      -> wb_motor_set_velocity() sur m1..m4

Usage dans Webots :
    1. Ouvrir worlds/crazyflie.wbt
    2. Selectionner le robot Crazyflie
    3. Dans le champ "controller", mettre : crazyflie_flydrones
    4. Lancer la simulation
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
        self.cfg["brain"]["source"] = "minifly"  # MiniFly par defaut (pas de telechargement 1.2 GB)

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

        print("=== Crazyflie FlyDrones Controller ===")
        print(f"Brain: {self.brain.connectome.name} ({self.brain.n_neurons} neurones)")
        print(f"Controle a {self.cfg['control']['hz']} Hz")
        print("Camera Webots activee (324x324)")
        print("=======================================")

    def _get_camera_frame(self):
        """Recupere l'image de la camera Webots comme tableau numpy BGR."""
        image = self.camera.getImage()
        if image is None:
            return None
        width = self.camera.getWidth()
        height = self.camera.getHeight()
        # Webots retourne BGRA (4 canaux par pixel)
        buf = np.frombuffer(image, dtype=np.uint8)
        img = buf.reshape((height, width, 4))[:, :, :3]  # drop alpha -> BGR
        return img

    def _get_telemetry(self):
        """Construit la telemetrie depuis les capteurs Webots."""
        roll, pitch, yaw = self.imu.getRollPitchYaw()
        yaw_rate = self.gyro.getValues()[2]  # rad/s
        x, y, z = self.gps.getValues()
        return Telemetry(
            t=self.t,
            alt_m=float(z),
            vz_mps=0.0,  # pas de vz direct, on pourrait le deriver
            yaw_deg=float(np.degrees(yaw) % 360),
            yaw_rate_dps=float(np.degrees(yaw_rate)),
            x_m=float(x),
            y_m=float(y),
            battery_pct=100.0,
            flying=True,
        )

    def _compute_body_velocities(self, x, y, yaw, dt):
        """Calcule les vitesses dans le repere du corps (comme dans crazyflie.c)."""
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

    def _apply_motor_commands(self, cmd):
        """Applique la commande de vol aux 4 moteurs (melange simplifie)."""
        # Le Crazyflie attend des vitesses de moteur. On mappe la commande
        # normalisee (-1..1) sur une plage de vitesse moteur (0..600 rad/s).
        max_vel = 600.0
        base = 150.0  # vitesse de base pour maintenir l'altitude

        # Melange simplifie inspire du motor_mixing du PID C
        throttle = cmd.throttle  # -1..1 (vertical)
        yaw = cmd.yaw            # -1..1
        forward = cmd.forward    # -1..1
        lateral = cmd.lateral    # -1..1

        # Commandes moteur brutes (a affiner selon la geometrie du Crazyflie)
        m1 = base + throttle * 200 - yaw * 100 + forward * 100 + lateral * 100
        m2 = base + throttle * 200 - yaw * 100 - forward * 100 - lateral * 100
        m3 = base + throttle * 200 + yaw * 100 - forward * 100 + lateral * 100
        m4 = base + throttle * 200 + yaw * 100 + forward * 100 - lateral * 100

        # Clamp
        m1 = max(0.0, min(max_vel, m1))
        m2 = max(0.0, min(max_vel, m2))
        m3 = max(0.0, min(max_vel, m3))
        m4 = max(0.0, min(max_vel, m4))

        self.m1.setVelocity(-m1)
        self.m2.setVelocity(m2)
        self.m3.setVelocity(-m3)
        self.m4.setVelocity(m4)

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

            # --- 2. IMU / Gyro -> taux de lacet ---
            roll, pitch, yaw = self.imu.getRollPitchYaw()
            yaw_rate = self.gyro.getValues()[2]
            yaw_rate_dps = float(np.degrees(yaw_rate))

            # --- 3. GPS -> position ---
            x, y, z = self.gps.getValues()
            vx, vy = self._compute_body_velocities(float(x), float(y), float(yaw), dt)

            # --- 4. Telemetrie ---
            tel = Telemetry(
                t=self.t,
                alt_m=float(z),
                vz_mps=0.0,
                yaw_deg=float(np.degrees(yaw) % 360),
                yaw_rate_dps=yaw_rate_dps,
                x_m=float(x),
                y_m=float(y),
                battery_pct=100.0,
                flying=True,
            )

            # --- 5. Encoder -> taux de Poisson ---
            inputs = self.encoder.encode(vision, yaw_rate_dps)

            # --- 6. Cerveau (LIF) ---
            rates = self.brain.tick(inputs, ms=dt * 1000.0)

            # --- 7. Decodeur -> commande de vol ---
            raw_cmd = self.decoder.update(rates, dt)

            # --- 8. Securite ---
            safe_cmd = self.safety.filter(raw_cmd, tel, dt)

            # --- 9. Appliquer aux moteurs ---
            self._apply_motor_commands(safe_cmd)

            # --- 10. Log periodique ---
            if int(self.t * 10) % 10 == 0:
                dn_rates = {k: round(v, 1) for k, v in rates.items() if k.startswith("DN")}
                print(
                    f"t={self.t:5.1f}s alt={tel.alt_m:4.2f}m "
                    f"cmd(thr={safe_cmd.throttle:+.2f} yaw={safe_cmd.yaw:+.2f} "
                    f"fwd={safe_cmd.forward:+.2f} lat={safe_cmd.lateral:+.2f}) "
                    f"DN={dn_rates}"
                )

            self.past_time = self.t


def main():
    controller = CrazyflieFlyDronesController()
    controller.run()


if __name__ == "__main__":
    main()
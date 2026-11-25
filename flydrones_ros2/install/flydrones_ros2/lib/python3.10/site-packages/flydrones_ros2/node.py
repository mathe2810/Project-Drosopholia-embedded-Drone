#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
flydrones_node — Node ROS 2 pour Webots + cerveau Drosophila (FlyDrones).

Pipeline (un tick de controle, 20 Hz par defaut) :
    /drone/camera/image_raw  -> Retina.encode()        -> grilles oeil de mouche
    /drone/imu               -> taux de lacet          -> InputEncoder
    /drone/odom              -> position/altitude      -> Telemetry
    grilles + IMU            -> InputEncoder.encode()  -> {groupe: Hz}
    {groupe: Hz}             -> Brain.tick()           -> {groupe: rate Hz}
    taux                     -> MotorDecoder.update()  -> FlightCommand
    cmd + telemetrie         -> SafetyGovernor.filter() -> safe FlightCommand
    safe cmd                 -> /drone/cmd_vel (Twist)

Abonnements :
    /drone/camera/image_raw  (sensor_msgs/Image)
    /drone/imu               (sensor_msgs/Imu)
    /drone/odom              (nav_msgs/Odometry)

Publications :
    /drone/cmd_vel           (geometry_msgs/Twist)
    /flydrones/brain_state   (std_msgs/String)  — JSON debug

Ce node est adapte a Webots (pas Gazebo). Les noms de topics sont identiques
a ceux utilises par le pont Webots-ROS2 (webots_ros2) pour faciliter
l'interoperabilite.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Image, Imu
from std_msgs.msg import String

# --- Ajouter flydrones_lib au path Python ---
FLYDRONES_SRC = Path(__file__).resolve().parents[3] / "flydrones_lib" / "src"
if str(FLYDRONES_SRC) not in sys.path:
    sys.path.insert(0, str(FLYDRONES_SRC))

import numpy as np  # noqa: E402

from flydrones.brain import Brain, load_connectome  # noqa: E402
from flydrones.config import load_config  # noqa: E402
from flydrones.motor.command import FlightCommand  # noqa: E402
from flydrones.motor.decoder import MotorDecoder  # noqa: E402
from flydrones.safety import SafetyGovernor, Telemetry  # noqa: E402
from flydrones.senses.encoder import InputEncoder  # noqa: E402
from flydrones.senses.retina import Retina  # noqa: E402


class FlyDronesNode(Node):
    """Node ROS 2 executant le pipeline cerveau Drosophila de FlyDrones."""

    def __init__(self):
        super().__init__("flydrones_node")

        # --- Parametres ---
        self.declare_parameter("config", "")
        self.declare_parameter("brain", "minifly")
        self.declare_parameter("control_hz", 20.0)

        config_path = self.get_parameter("config").get_parameter_value().string_value
        brain_source = self.get_parameter("brain").get_parameter_value().string_value
        control_hz = self.get_parameter("control_hz").get_parameter_value().double_value

        # --- Charger la config FlyDrones + cerveau ---
        cfg = load_config(config_path if config_path else None)
        cfg["brain"]["source"] = brain_source

        self.get_logger().info(f"Chargement du cerveau FlyDrones: {brain_source}")
        connectome = load_connectome(brain_source)
        self.get_logger().info(connectome.summary())

        self.brain = Brain(connectome, cfg)
        self.retina = Retina.from_config(cfg)
        self.encoder = InputEncoder(self.brain.connectome, cfg)
        self.decoder = MotorDecoder(cfg)
        self.safety = SafetyGovernor(cfg)

        # --- Etat ---
        self.last_image = None
        self.last_imu = None
        self.last_odom = None
        self.dt = 1.0 / control_hz
        self.t = 0.0

        # --- Interfaces ROS 2 ---
        self.cmd_pub = self.create_publisher(Twist, "/drone/cmd_vel", 10)
        self.state_pub = self.create_publisher(String, "/flydrones/brain_state", 10)

        self.camera_sub = self.create_subscription(
            Image, "/drone/camera/image_raw", self._camera_cb, 10
        )
        self.imu_sub = self.create_subscription(
            Imu, "/drone/imu", self._imu_cb, 10
        )
        self.odom_sub = self.create_subscription(
            Odometry, "/drone/odom", self._odom_cb, 10
        )

        # Boucle de controle par timer
        self.timer = self.create_timer(self.dt, self._control_loop)

        self.get_logger().info(
            f"FlyDrones node demarre: brain={brain_source}, control_hz={control_hz}"
        )
        self.get_logger().info(
            "Pipeline: camera -> retine -> encodeur -> LIF -> decodeur -> securite -> cmd_vel"
        )

    # ------------------------------------------------------------------ callbacks
    def _camera_cb(self, msg: Image):
        self.last_image = msg

    def _imu_cb(self, msg: Imu):
        self.last_imu = msg

    def _odom_cb(self, msg: Odometry):
        self.last_odom = msg

    # ------------------------------------------------------------------ boucle de controle
    def _control_loop(self):
        """Un tick de la boucle fermee FlyDrones."""
        self.t += self.dt

        # --- 1. Camera -> Retine ---
        vision = None
        if self.last_image is not None:
            frame = self._image_to_numpy(self.last_image)
            vision = self.retina.encode(frame)

        # --- 2. IMU -> taux de lacet ---
        yaw_rate_dps = 0.0
        if self.last_imu is not None:
            yaw_rate_dps = float(self.last_imu.angular_velocity.z) * 180.0 / 3.14159

        # --- 3. Telemetrie depuis odometrie ---
        tel = Telemetry(t=self.t, flying=True)
        if self.last_odom is not None:
            p = self.last_odom.pose.pose.position
            tel.alt_m = float(p.z)
            tel.x_m = float(p.x)
            tel.y_m = float(p.y)
            tel.vz_mps = float(self.last_odom.twist.twist.linear.z)

        # --- 4. Encoder -> taux de Poisson ---
        inputs = self.encoder.encode(vision, yaw_rate_dps)

        # --- 5. Cerveau (LIF) ---
        rates = self.brain.tick(inputs, ms=self.dt * 1000.0)

        # --- 6. Decodeur: taux DN -> FlightCommand ---
        raw_cmd = self.decoder.update(rates, self.dt)

        # --- 7. Securite ---
        safe_cmd = self.safety.filter(raw_cmd, tel, self.dt)

        # --- 8. Publier Twist ---
        twist = Twist()
        twist.linear.x = safe_cmd.forward
        twist.linear.y = safe_cmd.lateral
        twist.linear.z = -safe_cmd.throttle
        twist.angular.z = safe_cmd.yaw
        self.cmd_pub.publish(twist)

        # --- 9. Etat debug ---
        state = {
            "t": round(self.t, 3),
            "alt_m": round(tel.alt_m or 0, 3),
            "rates": {k: round(v, 1) for k, v in rates.items() if k.startswith("DN")},
            "cmd": {
                "throttle": round(safe_cmd.throttle, 3),
                "yaw": round(safe_cmd.yaw, 3),
                "forward": round(safe_cmd.forward, 3),
                "lateral": round(safe_cmd.lateral, 3),
                "escape": safe_cmd.escape,
            },
            "note": safe_cmd.note,
        }
        self.state_pub.publish(String(data=json.dumps(state)))

    # ------------------------------------------------------------------ helpers
    @staticmethod
    def _image_to_numpy(msg: Image) -> np.ndarray:
        """Convertit sensor_msgs/Image en tableau numpy BGR."""
        if msg.encoding in ("rgb8", "bgr8"):
            dtype = np.uint8
            channels = 3
        elif msg.encoding == "mono8":
            dtype = np.uint8
            channels = 1
        else:
            dtype = np.uint8
            channels = 3

        buf = np.frombuffer(msg.data, dtype=dtype)
        if channels == 3:
            img = buf.reshape((msg.height, msg.width, 3))
            if msg.encoding == "rgb8":
                img = img[..., ::-1]
        else:
            img = buf.reshape((msg.height, msg.width))
        return img


def main(args=None):
    rclpy.init(args=args)
    node = FlyDronesNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
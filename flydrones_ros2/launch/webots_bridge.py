#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Pont Webots -> ROS 2 pour le Crazyflie.

Publie les donnees des capteurs Webots sur des topics ROS 2
compatibles avec flydrones_node :
  /drone/camera/image_raw  (sensor_msgs/Image)
  /drone/imu               (sensor_msgs/Imu)
  /drone/odom              (nav_msgs/Odometry)

S'abonne a /drone/cmd_vel (geometry_msgs/Twist) pour appliquer
les commandes aux moteurs Webots.

Usage :
    ros2 run flydrones_ros2 webots_bridge.py
"""

from __future__ import annotations

import sys
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, Imu
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Twist
from std_msgs.msg import Header
import numpy as np

try:
    from controller import Robot, Camera, InertialUnit, Gyro, GPS, Motor
except ImportError:
    print("ERREUR: L'API Webots n'est pas disponible. Ce script doit tourner dans Webots.")
    sys.exit(1)
    def _publish_camera(self):
        image = self.camera.getImage()
        if image is None:
            return
        width = self.camera.getWidth()
        height = self.camera.getHeight()
        buf = np.frombuffer(image, dtype=np.uint8)
        img = buf.reshape((height, width, 4))[:, :, :3]

        msg = Image()
        msg.header = Header()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "camera_link"
        msg.width = width
        msg.height = height
        msg.encoding = "bgr8"
        msg.is_bigendian = False
        msg.step = width * 3
        msg.data = img.tobytes()
        self.camera_pub.publish(msg)

    def _publish_imu(self):
        roll, pitch, yaw = self.imu.getRollPitchYaw()
        gyro_vals = self.gyro.getValues()

        msg = Imu()
        msg.header = Header()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "imu_link"

        msg.orientation.x = 0.0
        msg.orientation.y = 0.0
        msg.orientation.z = 0.0
        msg.orientation.w = 1.0

        msg.angular_velocity.x = float(gyro_vals[0])
        msg.angular_velocity.y = float(gyro_vals[1])
        msg.angular_velocity.z = float(gyro_vals[2])

        msg.linear_acceleration.x = 0.0
        msg.linear_acceleration.y = 0.0
        msg.linear_acceleration.z = 0.0

        self.imu_pub.publish(msg)

    def _publish_odom(self):
        x, y, z = self.gps.getValues()

        msg = Odometry()
        msg.header = Header()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "odom"
        msg.child_frame_id = "base_link"

        msg.pose.pose.position.x = float(x)
        msg.pose.pose.position.y = float(y)
        msg.pose.pose.position.z = float(z)
        msg.pose.pose.orientation.w = 1.0

        self.odom_pub.publish(msg)

    def _apply_cmd(self):
        max_vel = 600.0
        base = 150.0

        forward = self.latest_cmd.linear.x
        lateral = self.latest_cmd.linear.y
        throttle = -self.latest_cmd.linear.z
        yaw = self.latest_cmd.angular.z

        m1 = base + throttle * 200 - yaw * 100 + forward * 100 + lateral * 100
        m2 = base + throttle * 200 - yaw * 100 - forward * 100 - lateral * 100
        m3 = base + throttle * 200 + yaw * 100 - forward * 100 + lateral * 100
        m4 = base + throttle * 200 + yaw * 100 + forward * 100 - lateral * 100

        m1 = max(0.0, min(max_vel, m1))
        m2 = max(0.0, min(max_vel, m2))
        m3 = max(0.0, min(max_vel, m3))
        m4 = max(0.0, min(max_vel, m4))

        self.m1.setVelocity(-m1)
        self.m2.setVelocity(m2)
        self.m3.setVelocity(-m3)
        self.m4.setVelocity(m4)

    def run(self):
        while self.robot.step(self.timestep) != -1:
            rclpy.spin_once(self, timeout_sec=0)

            self._publish_camera()
            self._publish_imu()
            self._publish_odom()
            self._apply_cmd()


def main(args=None):
    rclpy.init(args=args)
    bridge = WebotsBridge()
    try:
        bridge.run()
    except KeyboardInterrupt:
        pass
    finally:
        bridge.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()

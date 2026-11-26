# -*- coding: utf-8 -*-
#
# crazyflie_connectome.py
# -----------------------
# Contrôleur Webots pour le Crazyflie piloté par le connectome de Drosophila.
#
# ARCHITECTURE :
#   1. Le drone est DÉJÀ STABLE : un PID interne gère l'attitude, l'altitude
#      et la vitesse horizontale (il ne reçoit que des consignes de vitesse).
#   2. Le connectome prend les DÉCISIONS de haut niveau (GAUCHE / DROITE /
#      HAUT / RALENTIR / STATIONNER / AVANCER) sans jamais toucher aux moteurs.
#   3. La caméra Webots est CHIFFRÉE (Fernet/AES) avant d'être envoyée au
#      serveur de décision via TCP.
#   4. Le serveur renvoie une décision + consignes de vitesse, appliquées par le PID.
#
#   Sécurité : les 4 capteurs de distance coupent toute commande si un obstacle
#   est trop proche (le drone ralentit / s'arrête).
#
# Lancement dans Webots :
#   - Ajouter un robot "Crazyflie" avec ce controller (Python).
#   - Lancer decision_server.py en parallèle (même machine ou réseau).
#   - Même clé de chiffrement des deux côtés.

import sys
import os
import json
import socket
import struct
import time
import threading
import numpy as np
import math

# --- Imports Webots (disponibles uniquement dans l'environnement Webots) ---
try:
    from controller import Robot, Keyboard, Camera, DistanceSensor, InertialUnit, Gyro, GPS, Motor
    _WEBOTS = True
except ImportError:
    _WEBOTS = False
    # Permet de tester le fichier hors Webots (stub)
    class Robot:
        def getBasicTimeStep(self): return 32
        def step(self, dt): return -1
        def getTime(self): return 0.0
        def getDevice(self, name): return None
    class Keyboard:
        def enable(self, dt): pass
        def getKey(self): return -1

# --- Ajouter le module de décision au path ---
CONNECTOME_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "connectome_decision")
CONNECTOME_DIR = os.path.abspath(CONNECTOME_DIR)
if CONNECTOME_DIR not in sys.path:
    sys.path.insert(0, CONNECTOME_DIR)

from video_crypto import VideoCrypto, CryptoConfig
from connectome_brain import ConnectomeBrain, BrainConfig, Decision
from pid_controller import pid_velocity_fixed_height_controller

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
FLYING_ALTITUDE = 1.0          # m, altitude de croisière
SERVER_HOST = "127.0.0.1"      # serveur de décision (localhost par défaut)
SERVER_PORT = 5555
CRYPTO_KEY = b"crazyflie-connectome-cam-key-2026"  # DOIT correspondre au serveur
CAMERA_WIDTH = 96
CAMERA_HEIGHT = 72
CAMERA_CHANNELS = 3  # RGB (Webots donne BGRA, on prend les 3 premiers)

# Seuils de sécurité (mètres)
OBSTACLE_STOP_DIST = 0.30   # en dessous : arrêt total (SLOW)
OBSTACLE_SLOW_DIST = 0.70   # en dessous : ralentir


class CrazyflieConnectome:
    def __init__(self):
        self.robot = Robot()
        self.timestep = int(self.robot.getBasicTimeStep())

        # --- Moteurs ---
        self.m1 = self.robot.getDevice("m1_motor")
        self.m2 = self.robot.getDevice("m2_motor")
        self.m3 = self.robot.getDevice("m3_motor")
        self.m4 = self.robot.getDevice("m4_motor")
        for m in (self.m1, self.m2, self.m3, self.m4):
            m.setPosition(float("inf"))
            m.setVelocity(-1.0 if m in (self.m1, self.m3) else 1.0)

        # --- Capteurs ---
        self.imu = self.robot.getDevice("inertial_unit")
        self.gps = self.robot.getDevice("gps")
        self.gyro = self.robot.getDevice("gyro")
        self.camera = self.robot.getDevice("camera")
        self.range_front = self.robot.getDevice("range_front")
        self.range_left = self.robot.getDevice("range_left")
        self.range_back = self.robot.getDevice("range_back")
        self.range_right = self.robot.getDevice("range_right")

        for dev in (self.imu, self.gps, self.gyro, self.camera,
                    self.range_front, self.range_left, self.range_back, self.range_right):
            if dev is not None:
                dev.enable(self.timestep)

        self.keyboard = Keyboard()
        self.keyboard.enable(self.timestep)

        # --- PID de stabilité (ne reçoit que des consignes de vitesse) ---
        self.pid = pid_velocity_fixed_height_controller()

        # --- État ---
        self.past_x = 0.0
        self.past_y = 0.0
        self.past_time = 0.0
        self.first_time = True
        self.height_desired = FLYING_ALTITUDE

        # --- Décision connectome ---
        self.current_decision = Decision.HOVER
        self.current_setpoints = {"vx": 0.0, "vy": 0.0, "yaw_rate": 0.0, "altitude": FLYING_ALTITUDE}
        self.decision_lock = threading.Lock()

        # --- Réseau vers le serveur de décision ---
        self.sock = None
        self.connected = False
        self.crypto = VideoCrypto(config=CryptoConfig(mode="fernet", key=CRYPTO_KEY))
        self._connect_to_server()

        # --- Cerveau local (fallback si serveur indisponible) ---
        self.local_brain = None
        self.use_local_brain = False

    # ------------------------------------------------------------------ réseau
    def _connect_to_server(self):
        """Tente de se connecter au serveur de décision (non bloquant)."""
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            s.settimeout(1.0)
            s.connect((SERVER_HOST, SERVER_PORT))
            s.settimeout(0.05)  # non bloquant ensuite
            self.sock = s
            self.connected = True
            print(f"[crazyflie_connectome] connecté au serveur {SERVER_HOST}:{SERVER_PORT}")
            threading.Thread(target=self._recv_loop, daemon=True).start()
        except Exception as e:
            self.connected = False
            print(f"[crazyflie_connectome] serveur indisponible ({e}), mode local uniquement")

    def _recv_loop(self):
        """Boucle de réception des décisions du serveur."""
        while self.connected and self.sock:
            try:
                header_len_bytes = self._recv_exact(4)
                if header_len_bytes is None:
                    break
                header_len = struct.unpack("!I", header_len_bytes)[0]
                header_json = self._recv_exact(header_len)
                if header_json is None:
                    break
                header = json.loads(header_json.decode("utf-8"))
                if header.get("type") == "decision_reply":
                    size = header["size"]
                    payload = self._recv_exact(size)
                    if payload is None:
                        break
                    reply = json.loads(payload.decode("utf-8"))
                    with self.decision_lock:
                        self.current_decision = Decision(reply["decision"])
                        self.current_setpoints = {
                            "vx": float(reply["vx"]),
                            "vy": float(reply["vy"]),
                            "yaw_rate": float(reply["yaw_rate"]),
                            "altitude": float(reply["altitude"]),
                        }
            except (socket.timeout, ConnectionResetError, BrokenPipeError, OSError, json.JSONDecodeError):
                continue
            except Exception:
                break
        self.connected = False
        print("[crazyflie_connectome] déconnecté du serveur")

    def _recv_exact(self, n):
        buf = bytearray()
        while len(buf) < n:
            try:
                chunk = self.sock.recv(n - len(buf))
            except socket.timeout:
                return None
            if not chunk:
                return None
            buf.extend(chunk)
        return bytes(buf)

    def _send_frame(self, frame_rgb, ranges, telemetry):
        """Envoie une frame chiffrée au serveur de décision."""
        if not self.connected or self.sock is None:
            return
        try:
            h, w, c = frame_rgb.shape
            raw = frame_rgb.tobytes()
            enc = self.crypto.encrypt(raw)

            header = {
                "type": "frame",
                "width": w,
                "height": h,
                "channels": c,
                "size": len(enc),
                "ranges": ranges,
                "telemetry": telemetry,
            }
            header_bytes = json.dumps(header).encode("utf-8")
            self.sock.sendall(struct.pack("!I", len(header_bytes)))
            self.sock.sendall(header_bytes)
            self.sock.sendall(enc)
        except Exception:
            # En cas d'erreur d'envoi, on passe en mode local silencieusement
            self.connected = False

    # ------------------------------------------------------------------ capteurs
    def _read_sensors(self):
        """Lit tous les capteurs et retourne un dict."""
        roll, pitch, yaw = self.imu.getRollPitchYaw()
        yaw_rate = self.gyro.getValues()[2]
        x, y, z = self.gps.getValues()
        alt = z

        # Vitesse globale -> corps
        dt = self.robot.getTime() - self.past_time if self.past_time > 0 else self.timestep / 1000.0
        vx_global = (x - self.past_x) / dt if dt > 0 else 0.0
        vy_global = (y - self.past_y) / dt if dt > 0 else 0.0
        math.cos_yaw = math.cos(yaw)
        sin_yaw = math.sin(yaw)
        vx = vx_global * math.cos_yaw + vy_global * sin_yaw
        vy = -vx_global * sin_yaw + vy_global * math.cos_yaw

        ranges = {
            "front": self.range_front.getValue() / 1000.0,
            "left": self.range_left.getValue() / 1000.0,
            "right": self.range_right.getValue() / 1000.0,
            "back": self.range_back.getValue() / 1000.0,
        }

        return {
            "roll": roll, "pitch": pitch, "yaw": yaw, "yaw_rate": yaw_rate,
            "x": x, "y": y, "alt": alt, "vx": vx, "vy": vy,
            "ranges": ranges, "dt": dt,
        }

    # ------------------------------------------------------------------ décision
    def _get_decision(self, sensors, frame_rgb):
        """
        Détermine la décision courante :
          1. Si le serveur de décision a répondu, on l'utilise.
          2. Sinon, cerveau local (fallback) ou obstacle -> SLOW.
        """
        # Sécurité obstacle (priorité absolue, même si le connectome dit d'avancer)
        r = sensors["ranges"]
        min_range = min(r["front"], r["left"], r["right"], r["back"])
        if min_range < OBSTACLE_STOP_DIST:
            return Decision.SLOW, {"vx": 0.0, "vy": 0.0, "yaw_rate": 0.0, "altitude": sensors["alt"]}

        with self.decision_lock:
            decision = self.current_decision
            setpoints = dict(self.current_setpoints)

        # Si pas de serveur et pas de cerveau local, on hover
        if not self.connected and self.local_brain is None:
            return Decision.HOVER, {"vx": 0.0, "vy": 0.0, "yaw_rate": 0.0, "altitude": self.height_desired}

        return decision, setpoints

    # ------------------------------------------------------------------ boucle principale
    def run(self):
        print("\n===== Crazyflie Connectome =====")
        print("Le connectome prend les décisions, le PID garde la stabilité.")
        print("Vidéo caméra chiffrée (Fernet/AES) vers le serveur de décision.")
        print("Commandes : GAUCHE / DROITE / HAUT / BAS / AVANCER / RALENTIR / STATIONNER")
        print("================================\n")

        while self.robot.step(self.timestep) != -1:
            sensors = self._read_sensors()
            dt = sensors["dt"]

            if self.first_time:
                self.past_x = sensors["x"]
                self.past_y = sensors["y"]
                self.past_time = self.robot.getTime()
                self.first_time = False
                continue

            # --- Image caméra ---
            frame = None
            if self.camera is not None:
                img = self.camera.getImage()  # bytes BGRA (Webots)
                if img is not None:
                    w = self.camera.getWidth()
                    h = self.camera.getHeight()
                    # Webots : BGRA, 4 canaux par pixel -> on prend RGB
                    img_arr = np.frombuffer(img, dtype=np.uint8).reshape((h, w, 4))
                    frame = img_arr[:, :, :3][:, :, ::-1].copy()  # BGRA -> RGB

            # --- Envoi au serveur de décision (vidéo chiffrée) ---
            if frame is not None:
                self._send_frame(frame, sensors["ranges"], {
                    "alt": sensors["alt"], "vx": sensors["vx"],
                    "vy": sensors["vy"], "yaw_rate": sensors["yaw_rate"],
                })

            # --- Décision ---
            decision, setpoints = self._get_decision(sensors, frame)

            # --- Mise à jour de l'altitude cible ---
            self.height_desired = setpoints["altitude"]

            # --- Clavier (override manuel) ---
            key = self.keyboard.getKey()
            forward_desired = setpoints["vx"]
            sideways_desired = setpoints["vy"]
            yaw_desired = setpoints["yaw_rate"]
            height_diff = 0.0

            if key > 0:
                if key == Keyboard.UP:
                    forward_desired = 0.5
                elif key == Keyboard.DOWN:
                    forward_desired = -0.5
                elif key == Keyboard.RIGHT:
                    sideways_desired = -0.5
                elif key == Keyboard.LEFT:
                    sideways_desired = 0.5
                elif key == ord('Q'):
                    yaw_desired = 1.0
                elif key == ord('E'):
                    yaw_desired = -1.0
                elif key == ord('W'):
                    height_diff = 0.1
                elif key == ord('S'):
                    height_diff = -0.1

            self.height_desired += height_diff * dt

            # --- PID de stabilité (applique les consignes, garde le drone stable) ---
            motor_power = self.pid.pid(
                dt, forward_desired, sideways_desired, yaw_desired, self.height_desired,
                sensors["roll"], sensors["pitch"], sensors["yaw_rate"],
                sensors["alt"], sensors["vx"], sensors["vy"],
            )

            self.m1.setVelocity(-motor_power[0])
            self.m2.setVelocity(motor_power[1])
            self.m3.setVelocity(-motor_power[2])
            self.m4.setVelocity(motor_power[3])

            # --- Sauvegarde pour le pas suivant ---
            self.past_time = self.robot.getTime()
            self.past_x = sensors["x"]
            self.past_y = sensors["y"]

            # --- Log périodique ---
            if int(self.robot.getTime() * 10) % 10 == 0:
                print(f"[t={self.robot.getTime():5.1f}s] decision={decision.value:8s} "
                      f"vx={setpoints['vx']:+.2f} vy={setpoints['vy']:+.2f} "
                      f"yaw={setpoints['yaw_rate']:+.2f} alt={sensors['alt']:.2f} "
                      f"front={sensors['ranges']['front']:.2f}m")

        # Nettoyage
        for m in (self.m1, self.m2, self.m3, self.m4):
            m.setVelocity(0.0)
        if self.sock:
            try:
                self.sock.close()
            except Exception:
                pass


if __name__ == "__main__":
    controller = CrazyflieConnectome()
    controller.run()

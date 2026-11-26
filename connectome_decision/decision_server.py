"""
decision_server.py
------------------
Serveur TCP qui fait le pont entre la caméra Webots (chiffrée) et le
cerveau connectome.

Protocole (TCP, JSON + binaire) :
  Client (contrôleur Webots) -> Serveur :
      {"type": "frame", "width": W, "height": H, "channels": C, "size": N}
      suivi de N bytes de frame CHIFFRÉS
  Serveur -> Client :
      {"type": "decision", "decision": "LEFT|RIGHT|...|HOVER",
       "vx": ..., "vy": ..., "yaw_rate": ..., "altitude": ...,
       "loom": ..., "ts": ...}

Le serveur :
  1. Attend une connexion du contrôleur Webots
  2. Reçoit les frames chiffrées, les déchiffre (clé partagée)
  3. Passe l'image à la rétine + cerveau connectome
  4. Renvoie la décision et les consignes de vitesse

Usage :
    python decision_server.py --method manuel --host 0.0.0.0 --port 5555
"""

from __future__ import annotations

import argparse
import json
import socket
import struct
import sys
import time
import threading
from typing import Optional

import numpy as np

# Importer le cerveau de décision (même répertoire)
from connectome_brain import ConnectomeBrain, BrainConfig, Decision
from video_crypto import VideoCrypto, CryptoConfig


class DecisionServer:
    def __init__(self, host: str = "0.0.0.0", port: int = 5555,
                 method: str = "manuel", key: bytes = b"crazyflie-connectome-cam-key-2026"):
        self.host = host
        self.port = port
        self.crypto = VideoCrypto(config=CryptoConfig(mode="fernet", key=key))
        self.brain = ConnectomeBrain(BrainConfig(method=method, crypto_key=key))
        self._sock: Optional[socket.socket] = None
        self._running = False
        self._conn: Optional[socket.socket] = None
        self._addr = None
        self._lock = threading.Lock()
        self.last_decision = Decision.HOVER
        self.last_setpoints = {"vx": 0.0, "vy": 0.0, "yaw_rate": 0.0, "altitude": 1.0}

    def start(self) -> None:
        self.brain.initialize()
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._sock.bind((self.host, self.port))
        self._sock.listen(1)
        self._sock.settimeout(0.5)
        self._running = True
        print(f"[decision_server] écoute sur {self.host}:{self.port} (méthode={self.brain.method})")
        threading.Thread(target=self._accept_loop, daemon=True).start()

    def stop(self) -> None:
        self._running = False
        if self._conn:
            try:
                self._conn.close()
            except Exception:
                pass
        if self._sock:
            try:
                self._sock.close()
            except Exception:
                pass

    # ------------------------------------------------------------------ accept
    def _accept_loop(self) -> None:
        while self._running:
            try:
                conn, addr = self._sock.accept()
                with self._lock:
                    if self._conn:
                        try:
                            self._conn.close()
                        except Exception:
                            pass
                    self._conn = conn
                    self._addr = addr
                print(f"[decision_server] client connecté: {addr}")
                threading.Thread(target=self._client_loop, args=(conn,), daemon=True).start()
            except socket.timeout:
                continue
            except OSError:
                break

    # ------------------------------------------------------------------ client loop
    def _client_loop(self, conn: socket.socket) -> None:
        conn.settimeout(1.0)
        try:
            while self._running:
                # Lire l'en-tête JSON (longueur 4 octets + JSON)
                header_len_bytes = self._recv_exact(conn, 4)
                if header_len_bytes is None:
                    break
                header_len = struct.unpack("!I", header_len_bytes)[0]
                header_json = self._recv_exact(conn, header_len)
                if header_json is None:
                    break
                header = json.loads(header_json.decode("utf-8"))

                if header.get("type") != "frame":
                    continue

                width = header["width"]
                height = header["height"]
                channels = header["channels"]
                size = header["size"]

                # Lire les bytes de la frame chiffrée
                enc_frame = self._recv_exact(conn, size)
                if enc_frame is None:
                    break

                # Déchiffrer
                raw_frame = self.crypto.decrypt(enc_frame)

                # Reconstituer l'image numpy
                frame = np.frombuffer(raw_frame, dtype=np.uint8).reshape((height, width, channels))

                # Télémétrie (optionnelle, envoyée par le client)
                ranges = header.get("ranges", {"front": float("inf"), "left": float("inf"),
                                               "right": float("inf"), "back": float("inf")})
                telemetry = header.get("telemetry", {"alt": 1.0, "vx": 0.0, "vy": 0.0, "yaw_rate": 0.0})

                # Décision du connectome
                decision = self.brain.decide(frame, ranges, telemetry)
                setpoints = self.brain.decision_to_setpoints(decision, telemetry.get("alt", 1.0))

                with self._lock:
                    self.last_decision = decision
                    self.last_setpoints = setpoints

                # Réponse JSON
                reply = {
                    "type": "decision",
                    "decision": decision.value,
                    "vx": setpoints["vx"],
                    "vy": setpoints["vy"],
                    "yaw_rate": setpoints["yaw_rate"],
                    "altitude": setpoints["altitude"],
                    "ts": time.time(),
                }
                reply_bytes = json.dumps(reply).encode("utf-8")
                self._send_message(conn, {"type": "decision_reply", "size": len(reply_bytes)}, reply_bytes)
        except (ConnectionResetError, BrokenPipeError, OSError, socket.timeout):
            pass
        finally:
            with self._lock:
                if self._conn is conn:
                    self._conn = None
                    self._addr = None
            try:
                conn.close()
            except Exception:
                pass
            print("[decision_server] client déconnecté")

    # ------------------------------------------------------------------ helpers réseau
    @staticmethod
    def _recv_exact(conn: socket.socket, n: int) -> Optional[bytes]:
        buf = bytearray()
        while len(buf) < n:
            chunk = conn.recv(n - len(buf))
            if not chunk:
                return None
            buf.extend(chunk)
        return bytes(buf)

    @staticmethod
    def _send_message(conn: socket.socket, header: dict, payload: bytes) -> None:
        header_bytes = json.dumps(header).encode("utf-8")
        conn.sendall(struct.pack("!I", len(header_bytes)))
        conn.sendall(header_bytes)
        if payload:
            conn.sendall(payload)

    # ------------------------------------------------------------------ API pour test / usage direct
    def get_last_decision(self) -> dict:
        with self._lock:
            return {"decision": self.last_decision.value, **self.last_setpoints}


def main() -> None:
    ap = argparse.ArgumentParser(description="Serveur de décision connectome (vidéo chiffrée)")
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=5555)
    ap.add_argument("--method", default="manuel", choices=["manuel", "entraine"])
    ap.add_argument("--key", default="crazyflie-connectome-cam-key-2026",
                    help="clé de chiffrement partagée (chaîne ou base64)")
    a = ap.parse_args()

    key = a.key.encode()
    server = DecisionServer(host=a.host, port=a.port, method=a.method, key=key)
    try:
        server.start()
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nArrêt du serveur")
    finally:
        server.stop()


if __name__ == "__main__":
    main()

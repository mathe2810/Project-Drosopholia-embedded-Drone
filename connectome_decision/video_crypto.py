"""
video_crypto.py
---------------
Chiffrement du flux caméra Webots avant envoi vers le module de décision
(connectome). Le but est de simuler un lien vidéo sécurisé entre le drone
et le cerveau : la caméra de Webots chiffre chaque frame, le serveur de
décision le déchiffre avant de le passer à la rétine du connectome.

Deux modes disponibles :
  * FERNET (AES-128 en mode CBC + HMAC) via le paquet `cryptography` (défaut,
    déjà installé dans l'environnement).
  * XOR simple avec une clé pré-partagée (fallback sans dépendance).

Usage :
    from video_crypto import VideoCrypto
    crypto = VideoCrypto(key=b"ma-cle-secrete-de-32-octets!!")
    enc = crypto.encrypt(frame_bytes)   # bytes -> bytes
    dec = crypto.decrypt(enc)           # bytes -> bytes (image originale)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Union

try:
    from cryptography.fernet import Fernet
    from cryptography.hazmat.primitives import hashes
    from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
    import base64

    _FERNET_AVAILABLE = True
except Exception:  # pragma: no cover - cryptography peut être absent
    _FERNET_AVAILABLE = False


@dataclass
class CryptoConfig:
    mode: str = "fernet"          # "fernet" | "xor"
    key: bytes = b""              # clé brute (sera dérivée si fernet)
    salt: bytes = b"flydrone-cam-salt"


class VideoCrypto:
    """Chiffre / déchiffre des frames d'image (bytes) pour le lien vidéo."""

    def __init__(self, config: CryptoConfig | None = None,
                 key: Union[str, bytes, None] = None):
        self.config = config or CryptoConfig()
        if key is not None:
            self.config.key = key.encode() if isinstance(key, str) else key
        if not self.config.key:
            # clé par défaut (à changer en production !)
            self.config.key = b"crazyflie-connectome-cam-key-2026"

        self._fernet = None
        if self.config.mode == "fernet":
            if not _FERNET_AVAILABLE:
                raise RuntimeError(
                    "mode 'fernet' demande le paquet cryptography: pip install cryptography"
                )
            kdf = PBKDF2HMAC(
                algorithm=hashes.SHA256(),
                length=32,
                salt=self.config.salt,
                iterations=100_000,
            )
            derived = base64.urlsafe_b64encode(kdf.derive(self.config.key))
            self._fernet = Fernet(derived)

    # ------------------------------------------------------------------ API
    def encrypt(self, data: Union[bytes, bytearray, memoryview]) -> bytes:
        """Chiffre des bytes (frame JPEG/raw) et retourne des bytes chiffrés."""
        raw = bytes(data)
        if self.config.mode == "fernet" and self._fernet is not None:
            return self._fernet.encrypt(raw)
        # fallback XOR
        key = self.config.key
        out = bytearray(len(raw))
        klen = len(key)
        for i, b in enumerate(raw):
            out[i] = b ^ key[i % klen]
        return bytes(out)

    def decrypt(self, data: Union[bytes, bytearray, memoryview]) -> bytes:
        """Déchiffre des bytes et retourne les bytes d'origine."""
        raw = bytes(data)
        if self.config.mode == "fernet" and self._fernet is not None:
            return self._fernet.decrypt(raw)
        # XOR est symétrique
        return self.encrypt(raw)

    def generate_key(self) -> bytes:
        """Génère une nouvelle clé Fernet utilisable directement."""
        if _FERNET_AVAILABLE:
            return Fernet.generate_key()
        return bytes(range(256))  # placeholder pour XOR

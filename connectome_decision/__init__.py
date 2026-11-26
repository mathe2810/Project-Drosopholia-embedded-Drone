"""
connectome_decision
===================
Module de décision basé sur le connectome de Drosophila.

Architecture :
  1. La caméra Webots capture une frame
  2. La frame est chiffrée (video_crypto) et envoyée via TCP
  3. Le serveur de décision (decision_server) la déchiffre
  4. La rétine connectome extrait les features (optic flow, looming)
  5. Le cerveau (manuel ou entraîné) produit une décision
  6. La décision est renvoyée au contrôleur Webots
  7. Le PID applique la décision en maintenant la stabilité

Décisions possibles :
  - FORWARD / BACKWARD : avancer / reculer
  - LEFT / RIGHT : tourner à gauche / droite
  - UP / DOWN : monter / descendre
  - HOVER : stationner
  - SLOW : ralentir (obstacle détecté)
"""

from .video_crypto import VideoCrypto, CryptoConfig
from .decision_server import DecisionServer

__all__ = ["VideoCrypto", "CryptoConfig", "DecisionServer"]

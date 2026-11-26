# 🚁 Crazyflie + Connectome de Drosophila — Évitement d'obstacles

Ce projet ajoute une **couche de décision basée sur le connectome de la drosophile** au-dessus du contrôleur Webots existant du Crazyflie.

## Principe

| Couche | Rôle | Qui commande les moteurs ? |
|--------|------|---------------------------|
| **PID de stabilité** | Maintient l'attitude, l'altitude et la vitesse horizontale | ✅ Oui (seul autorisé) |
| **Connectome (décision)** | Donne une décision de haut niveau : GAUCHE / DROITE / HAUT / BAS / AVANCER / RALENTIR / STATIONNER | ❌ Non — jamais les moteurs |
| **Vidéo chiffrée** | Caméra Webots → Fernet/AES → TCP → serveur de décision | — |

Le drone est **déjà stable** grâce au PID. Le connectome est un **accompagnateur** qui oriente le drone et l'aide à éviter les obstacles (ralentir devant un mur, tourner, monter).

## Arborescence

```
crazyflie/
├── worlds/
│   ├── crazyflie.wbt              # monde d'origine (vide)
│   └── crazyflie_maze.wbt         # NOUVEAU : labyrinthe avec obstacles
├── controllers/
│   ├── crazyflie/                 # contrôleur C d'origine (PID + clavier)
│   ├── crazyflie_py_wallfollowing/# contrôleur Python d'origine (suivi de mur)
│   └── crazyflie_connectome/      # NOUVEAU : contrôleur connectome
│       ├── crazyflie_connectome.py# contrôleur Webots principal
│       └── pid_controller.py      # PID de stabilité (Python)
├── connectome_decision/           # NOUVEAU : module de décision
│   ├── video_crypto.py            # chiffrement/déchiffrement des frames (Fernet/AES)
│   ├── connectome_brain.py        # cerveau unifié (manuel + entraîné)
│   ├── decision_server.py         # serveur TCP décision (déchiffre + connectome)
│   └── __init__.py
├── flydrones_lib/                 # connectome "manuel" (MaleCNS + LIF)
├── FlyDrone-SNN/                  # connectome "entraîné" (e-prop + poids)
├── requirements.txt
└── README_CONNECTOME.md
```

## Comment ça marche

1. **Caméra Webots** capture une frame (RGB).
2. La frame est **chiffrée** (Fernet/AES-128 + HMAC) par `video_crypto.py`.
3. Elle est envoyée via **TCP** au `decision_server.py` (même machine ou réseau).
4. Le serveur **déchiffre** la frame, la passe à la **rétine connectome** (optic flow, looming, luminosité), puis au **cerveau** (méthode manuelle ou entraînée).
5. Le connectome retourne une **décision** (ex. `LEFT`, `SLOW`, `UP`).
6. Le serveur renvoie la décision + les consignes de vitesse (`vx`, `vy`, `yaw_rate`, `altitude`).
7. Le contrôleur Webots `crazyflie_connectome.py` applique ces consignes via le **PID de stabilité**, qui commande les 4 moteurs.
8. **Sécurité** : si un capteur de distance détecte un obstacle < 0,35 m, le drone force `SLOW` (arrêt), quelle que soit la décision du connectome.

## Décisions possibles

| Décision | Action du drone |
|----------|-----------------|
| `HOVER` | Stationner (maintenir l'altitude) |
| `FORWARD` | Avancer |
| `BACKWARD` | Reculer |
| `LEFT` | Aller à gauche |
| `RIGHT` | Aller à droite |
| `UP` | Monter |
| `DOWN` | Descendre |
| `SLOW` | Ralentir / s'arrêter (obstacle proche) |

## Lancement

### 1. Installer les dépendances

```bash
cd /home/matheo/Desktop/crazyflie
pip install -r requirements.txt
```

### 2. Lancer le serveur de décision

Dans un terminal :

```bash
cd /home/matheo/Desktop/crazyflie/connectome_decision
python decision_server.py --method manuel --host 0.0.0.0 --port 5555
```

Options :
- `--method manuel` : utilise `flydrones_lib` (connectome MaleCNS + LIF, pas d'entraînement requis)
- `--method entraine` : utilise `FlyDrone-SNN` (topologie + poids entraînés e-prop, nécessite un checkpoint)

La clé de chiffrement par défaut est `crazyflie-connectome-cam-key-2026` (doit être identique des deux côtés).

### 3. Lancer Webots avec le monde labyrinthe

Ouvrir Webots et charger :
```
/home/matheo/Desktop/crazyflie/worlds/crazyflie_maze.wbt
```

Dans les propriétés du robot Crazyflie, définir le **controller** :
```
crazyflie_connectome
```

Puis lancer la simulation.

### 4. Contrôle clavier (override)

Le clavier Webots peut toujours prendre la main :
- Flèches : avancer / reculer / gauche / droite
- Q / E : rotation yaw
- W / S : monter / descendre

## Deux méthodes de connectome

### Méthode "manuel" (flydrones_lib)
Utilise le connectome MaleCNS v1.0 avec un réseau LIF et un décodeur moteur linéaire. Aucun entraînement nécessaire, fonctionne immédiatement.

### Méthode "entraîné" (FlyDrone-SNN)
Utilise la topologie du connectome FlyWire v783 avec des poids entraînés par e-prop (imitation d'un PID). Nécessite un checkpoint dans `FlyDrone-SNN/flydrone/ckpt/`.

## Sécurité

- Le connectome **ne commande jamais les moteurs directement** : il retourne une décision discrète traduite en consignes de vitesse.
- Le PID de stabilité est la **seule couche autorisée à piloter les moteurs**.
- Un **override obstacle** basé sur les 4 capteurs de distance coupe toute commande si le drone est trop près d'un mur (< 0,35 m).
- Le lien vidéo est **chiffré** (Fernet/AES) entre la caméra Webots et le serveur de décision.

## Prochaines étapes (labyrinthe)

Une fois l'évitement d'obstacles simple validé, le connectome pourra être étendu pour :
- Naviguer dans un labyrinthe complet (états : explorer / tourner / avancer / sortir).
- Utiliser la mémoire du réseau récurrent pour se repérer.
- Combiner les deux méthodes (manuel pour la réactivité, entraîné pour la stratégie).

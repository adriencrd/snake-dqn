# 🐍 Snake + Deep Q-Network (DQN)

Projet d'apprentissage par renforcement où un agent apprend à jouer au Snake en utilisant l'algorithme DQN.

## 📋 Table des matières

- [Installation](#-installation)
- [Structure du projet](#-structure-du-projet)
- [Commandes principales](#-commandes-principales)
- [Configuration](#-configuration)
- [Comment ça marche](#-comment-ça-marche)
- [Résultats attendus](#-résultats-attendus)
- [Troubleshooting](#-troubleshooting)

---

## 🔧 Installation

### Prérequis
- Python 3.8 ou supérieur
- pip

### Étape 1 : Cloner ou télécharger le projet

```bash
cd snake-dqn
```

### Étape 2 : Installer les dépendances

```bash
pip install torch numpy pygame
```

**Ou avec un fichier requirements.txt :**

```bash
pip install -r requirements.txt
```

**Contenu du requirements.txt :**
```
torch>=2.0.0
numpy>=1.24.0
pygame>=2.5.0
```

---

## 📁 Structure du projet

```
snake-dqn/
│
├── snake_game.py          # Environnement du jeu Snake
├── dqn_agent.py           # Agent DQN avec réseau de neurones
├── main.py                # Script d'entraînement et de test
├── jouer_manuel.py        # Mode jeu manuel
├── README.md              # Cette documentation
│
└── (après entraînement)
    ├── meilleur_modele.pth    # Meilleur modèle sauvegardé
    └── modele_final.pth       # Modèle final
```

---

## 🎮 Commandes principales

### 1. Jouer manuellement

Pour tester le jeu et comprendre la mécanique :

```bash
python jouer_manuel.py
```

**Contrôles :**
- `←` Flèche gauche : Tourner à gauche
- `→` Flèche droite : Tourner à droite
- `↑` ou `↓` : Continuer tout droit
- `ESPACE` : Redémarrer la partie
- `ESC` : Quitter

---

### 2. Entraîner l'agent DQN

#### Mode standard (sans affichage, rapide)

```bash
python main.py
```

**Avantages :**
- Entraînement rapide
- Pas de ralentissement graphique
- Idéal pour les longues sessions

#### Mode avec affichage (voir l'agent apprendre)

```bash
python main.py --display
```

**Avantages :**
- Visualiser l'apprentissage en temps réel
- Comprendre les stratégies de l'agent
- Plus lent mais éducatif

**Paramètres par défaut :**
- 1000 épisodes d'entraînement
- Sauvegarde automatique du meilleur modèle
- Affichage des statistiques tous les 10 épisodes

---

### 3. Tester l'agent entraîné

```bash
python main.py test
```

**Ce qui se passe :**
- Charge le modèle `meilleur_modele.pth`
- Joue 10 parties en mode greedy (sans exploration)
- Affiche les statistiques finales

---

## ⚙️ Configuration

### Modifier les paramètres du jeu

Dans `main.py`, modifiez `ConfigJeu` :

```python
config_jeu = ConfigJeu(
    taille_grille=20,      # Taille de la grille (20x20)
    taille_case=32,        # Taille d'une case en pixels
    fps=15,                # Images par seconde
    max_etapes=500,        # Limite d'étapes par épisode
    recompense_nourriture=1.0,    # Récompense pour manger
    recompense_mort=-1.0,          # Pénalité pour mourir
    recompense_deplacement=-0.01   # Pénalité par mouvement
)
```

### Modifier les paramètres DQN

Dans `main.py`, modifiez `ConfigDQN` :

```python
config_dqn = ConfigDQN(
    gamma=0.99,              # Facteur de discount
    lr=1e-3,                 # Taux d'apprentissage
    batch_size=64,           # Taille du batch
    buffer_size=100_000,     # Taille du replay buffer
    debut_apprentissage=1_000,    # Étapes avant d'apprendre
    freq_entrainement=1,     # Fréquence d'entraînement
    sync_target=1000,        # Sync target network
    epsilon_debut=1.0,       # Exploration initiale (100%)
    epsilon_fin=0.05,        # Exploration finale (5%)
    epsilon_decay=20_000     # Decay sur 20k étapes
)
```

### Modifier le nombre d'épisodes

```python
entrainer(nb_episodes=2000, affichage=False)  # 2000 épisodes
```

---

## 🧠 Comment ça marche

### Architecture

#### 1. **Environnement (snake_game.py)**
- Grille NxN
- 3 actions : avancer, tourner gauche, tourner droite
- État : vecteur de 11 dimensions
  - 3 dangers (devant, gauche, droite)
  - 4 directions (haut, bas, gauche, droite)
  - 4 positions nourriture (relative)

#### 2. **Agent DQN (dqn_agent.py)**
- Réseau de neurones : 11 → 128 → 128 → 3
- Replay buffer pour stocker les expériences
- Target network pour stabiliser l'apprentissage
- Epsilon-greedy pour l'exploration

#### 3. **Boucle d'entraînement**
```
Pour chaque épisode :
    1. Reset l'environnement
    2. Tant que non terminé :
        - Choisir action (epsilon-greedy)
        - Exécuter action
        - Mémoriser transition
        - Apprendre du batch
        - Synchroniser target network
    3. Sauvegarder si meilleur score
```

---

## 📊 Résultats attendus

### Progression typique

| Épisodes | Taille serpent | Epsilon | Comportement |
|----------|----------------|---------|--------------|
| 0-100    | 2-3            | 1.0-0.8 | Exploration aléatoire |
| 100-500  | 3-5            | 0.8-0.3 | Commence à apprendre |
| 500-1000 | 5-10           | 0.3-0.1 | Stratégies émergentes |
| 1000+    | 10-20+         | 0.05    | Bon joueur |

### Fichiers générés

```
meilleur_modele.pth    # Meilleur score pendant l'entraînement
modele_final.pth       # État final après tous les épisodes
```

### Statistiques affichées

```
Épisode 100/1000
  Score moyen (10 derniers): 4.2
  Meilleur score: 8
  Epsilon: 0.368
  Perte: 0.0234
  Buffer: 15420
```

---

## 🔍 Troubleshooting

### Erreur : "ModuleNotFoundError: No module named 'pygame'"

**Solution :**
```bash
pip install pygame
```

### Erreur : "ModuleNotFoundError: No module named 'torch'"

**Solution :**
```bash
pip install torch
```

**Ou pour CPU uniquement :**
```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

### L'agent n'apprend pas / stagne

**Solutions possibles :**

1. **Augmenter les épisodes :**
```python
entrainer(nb_episodes=2000)
```

2. **Réduire epsilon_decay :**
```python
config_dqn = ConfigDQN(epsilon_decay=10_000)  # Plus lent
```

3. **Ajuster le learning rate :**
```python
config_dqn = ConfigDQN(lr=5e-4)  # Plus petit
```

### Pygame ne s'affiche pas

**Sur Linux :**
```bash
sudo apt-get install python3-pygame
```

**Sur macOS :**
```bash
brew install sdl sdl_image sdl_mixer sdl_ttf portmidi
pip install pygame
```

### CUDA out of memory

**Solution :** Le code utilise automatiquement CPU si CUDA n'est pas disponible.

Pour forcer le CPU :
```python
self.device = torch.device("cpu")  # Dans dqn_agent.py
```

---

## 🚀 Commandes avancées

### Entraînement personnalisé

Créez votre propre script :

```python
from snake_game import SnakeEnv, ConfigJeu
from dqn_agent import AgentDQN, ConfigDQN

# Configuration personnalisée
config_jeu = ConfigJeu(taille_grille=30)
config_dqn = ConfigDQN(lr=5e-4, epsilon_decay=50_000)

env = SnakeEnv(config_jeu)
agent = AgentDQN(dim_etat=11, nb_actions=3, config=config_dqn)

# Votre boucle d'entraînement...
```

### Charger et continuer l'entraînement

```python
agent.charger('meilleur_modele.pth')
# Continue l'entraînement...
```

### Évaluer sans affichage

Modifiez la fonction `tester()` dans main.py :

```python
env = SnakeEnv(config_jeu, affichage=False)  # Pas d'affichage
```

---

## 📈 Amélioration du modèle

### Idées pour améliorer l'agent

1. **Double DQN** : Réduire la surestimation des Q-values
2. **Dueling DQN** : Séparer V(s) et A(s,a)
3. **Prioritized Experience Replay** : Échantillonner les transitions importantes
4. **Reward shaping** : Ajouter des récompenses intermédiaires
5. **Curriculum learning** : Commencer avec une petite grille

---

## 📝 Licence

Projet éducatif - Libre d'utilisation et de modification

---

## 🤝 Contribution

N'hésitez pas à :
- Expérimenter avec les hyperparamètres
- Améliorer l'architecture du réseau
- Ajouter de nouvelles fonctionnalités
- Partager vos résultats !

---

**Bon entraînement ! 🐍🎮🤖**
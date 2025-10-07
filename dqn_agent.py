"""
Agent DQN (Deep Q-Network) pour Snake
Implémentation simple avec replay buffer et target network
"""
import random
import math
import numpy as np
from collections import deque
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class ConfigDQN:
    """Configuration de l'agent DQN"""
    gamma: float = 0.99              # Facteur de discount
    lr: float = 1e-3                 # Taux d'apprentissage
    batch_size: int = 256            # Taille du batch
    buffer_size: int = 100_000       # Taille du replay buffer
    debut_apprentissage: int = 1_000 # Étapes avant d'apprendre
    freq_entrainement: int = 1       # Fréquence d'entraînement
    sync_target: int = 1000          # Sync target network
    epsilon_debut: float = 1.0       # Exploration initiale
    epsilon_fin: float = 0.05        # Exploration finale
    epsilon_decay: int = 50_000      # Decay sur N étapes


class ReseauQ(nn.Module):
    """Réseau de neurones pour approximer Q(s,a)"""
    
    def __init__(self, dim_entree: int, nb_actions: int):
        super().__init__()
        self.reseau = nn.Sequential(
            nn.Linear(dim_entree, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, nb_actions)
        )
    
    def forward(self, x):
        return self.reseau(x)


class ReplayBuffer:
    """Buffer pour stocker les expériences (s, a, r, s', done)"""
    
    def __init__(self, capacite):
        self.buffer = deque(maxlen=capacite)
    
    def ajouter(self, etat, action, recompense, etat_suivant, termine):
        """Ajoute une transition"""
        self.buffer.append((etat, action, recompense, etat_suivant, termine))
    
    def sample(self, batch_size):
        """Samples a random batch"""
        batch = random.sample(self.buffer, batch_size)
        etats, actions, recompenses, etats_suivants, termines = zip(*batch)
        
        return (
            np.array(etats, dtype=np.float32),
            np.array(actions, dtype=np.int64),
            np.array(recompenses, dtype=np.float32),
            np.array(etats_suivants, dtype=np.float32),
            np.array(termines, dtype=np.float32)
        )
    
    def __len__(self):
        return len(self.buffer)


class AgentDQN:
    """Agent DQN avec target network et epsilon-greedy"""
    
    def __init__(self, dim_etat, nb_actions, config: ConfigDQN):
        self.nb_actions = nb_actions
        self.cfg = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Réseau Q et réseau cible
        self.q_reseau = ReseauQ(dim_etat, nb_actions).to(self.device)
        self.target_reseau = ReseauQ(dim_etat, nb_actions).to(self.device)
        self.target_reseau.load_state_dict(self.q_reseau.state_dict())
        
        # Optimiseur
        self.optimiseur = torch.optim.Adam(self.q_reseau.parameters(), lr=config.lr)
        
        # Replay buffer
        self.buffer = ReplayBuffer(config.buffer_size)
        
        # Compteur d'étapes
        self.etapes = 0
    
    def choisir_action(self, etat, epsilon=0.0):
        """Choisit une action avec politique epsilon-greedy"""
        if random.random() < epsilon:
            return random.randrange(self.nb_actions)
        
        # Mode greedy: choisit la meilleure action
        with torch.no_grad():
            etat_t = torch.from_numpy(etat).float().to(self.device).unsqueeze(0)
            q_values = self.q_reseau(etat_t)
            return int(q_values.argmax().item())
    
    def memoriser(self, etat, action, recompense, etat_suivant, termine):
        """Stocke une transition dans le buffer"""
        self.buffer.ajouter(etat, action, recompense, etat_suivant, float(termine))
    
    def apprendre(self):
        """Effectue une étape d'apprentissage"""
        if len(self.buffer) < self.cfg.batch_size:
            return None
        
        # Sample a batch
        etats, actions, recompenses, etats_suivants, termines = \
            self.buffer.sample(self.cfg.batch_size)
        
        # Convertit en tensors
        etats = torch.from_numpy(etats).to(self.device)
        actions = torch.from_numpy(actions).to(self.device)
        recompenses = torch.from_numpy(recompenses).to(self.device)
        etats_suivants = torch.from_numpy(etats_suivants).to(self.device)
        termines = torch.from_numpy(termines).to(self.device)
        
        # Calcule Q(s,a) actuel
        q_values = self.q_reseau(etats).gather(1, actions.unsqueeze(1)).squeeze(1)
        
        # Calcule la cible: r + gamma * max_a' Q_target(s', a')
        with torch.no_grad():
            max_q_suivant = self.target_reseau(etats_suivants).max(1)[0]
            cible = recompenses + (1.0 - termines) * self.cfg.gamma * max_q_suivant
        
        # Calcule la perte (Huber loss)
        perte = F.smooth_l1_loss(q_values, cible)
        
        # Backpropagation
        self.optimiseur.zero_grad()
        perte.backward()
        torch.nn.utils.clip_grad_norm_(self.q_reseau.parameters(), 10.0)
        self.optimiseur.step()
        
        return float(perte.item())
    
    def sync_target_network(self):
        """Copie les poids du réseau Q vers le réseau cible"""
        if self.etapes % self.cfg.sync_target == 0:
            self.target_reseau.load_state_dict(self.q_reseau.state_dict())
    
    def obtenir_epsilon(self):
        """Calcule epsilon selon un decay exponentiel"""
        eps = self.cfg.epsilon_fin + (self.cfg.epsilon_debut - self.cfg.epsilon_fin) * \
              math.exp(-1.0 * self.etapes / self.cfg.epsilon_decay)
        return eps
    
    def sauvegarder(self, chemin):
        """Sauvegarde le modèle"""
        torch.save(self.q_reseau.state_dict(), chemin)
    
    def charger(self, chemin):
        """Charge le modèle"""
        self.q_reseau.load_state_dict(torch.load(chemin, map_location=self.device))
        self.target_reseau.load_state_dict(self.q_reseau.state_dict())
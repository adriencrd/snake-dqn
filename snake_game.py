"""
Jeu Snake pour l'apprentissage par renforcement
Environnement simplifié avec observation vectorielle
"""
import random
import numpy as np
from collections import deque
from dataclasses import dataclass

try:
    import pygame
    PYGAME_DISPONIBLE = True
except ImportError:
    PYGAME_DISPONIBLE = False


@dataclass
class ConfigJeu:
    """Configuration du jeu"""
    taille_grille: int = 40        # Grille NxN
    taille_case: int = 32          # Pixels par case
    fps: int = 15                  # Vitesse d'affichage
    max_etapes: int = 1000         # Limite par épisode
    recompense_nourriture: float = 1.0
    recompense_mort: float = -1.0
    recompense_deplacement: float = -0.01


class SnakeEnv:
    """Environnement Snake avec 3 actions: avancer, gauche, droite"""
    
    # Directions: (dx, dy)
    HAUT = (0, -1)
    BAS = (0, 1)
    GAUCHE = (-1, 0)
    DROITE = (1, 0)
    
    def __init__(self, config: ConfigJeu, affichage=False):
        self.cfg = config
        self.grille = config.taille_grille
        self.affichage = affichage and PYGAME_DISPONIBLE
        self.random = random.Random(42)
        
        # Initialisation pygame si nécessaire
        self.ecran = None
        if self.affichage:
            self._init_pygame()
        
        self.reset()
    
    def _init_pygame(self):
        """Initialise l'affichage pygame"""
        if not PYGAME_DISPONIBLE:
            self.affichage = False
            return
        pygame.init()
        taille = self.grille * self.cfg.taille_case
        self.ecran = pygame.display.set_mode((taille, taille))
        pygame.display.set_caption("Snake RL")
        self.horloge = pygame.time.Clock()
    
    def reset(self):
        """Réinitialise l'environnement"""
        self.direction = self.DROITE
        milieu = self.grille // 2
        self.serpent = deque([(milieu-1, milieu), (milieu, milieu)])
        self._placer_nourriture()
        self.etapes = 0
        self.vivant = True
        return self._obtenir_etat()
    
    def _placer_nourriture(self):
        """Place la nourriture aléatoirement"""
        while True:
            x = self.random.randrange(self.grille)
            y = self.random.randrange(self.grille)
            if (x, y) not in self.serpent:
                self.nourriture = (x, y)
                break
    
    def _tourner(self, action):
        """Applique l'action: 0=avancer, 1=gauche, 2=droite"""
        if action == 0:
            return  # Continue tout droit
        
        # Rotation à gauche
        if action == 1:
            rotations = {
                self.HAUT: self.GAUCHE,
                self.GAUCHE: self.BAS,
                self.BAS: self.DROITE,
                self.DROITE: self.HAUT
            }
        # Rotation à droite
        else:
            rotations = {
                self.HAUT: self.DROITE,
                self.DROITE: self.BAS,
                self.BAS: self.GAUCHE,
                self.GAUCHE: self.HAUT
            }
        self.direction = rotations[self.direction]
    
    def step(self, action):
        """Effectue une action et retourne (etat, recompense, termine, info)"""
        if not self.vivant:
            return self._obtenir_etat(), 0.0, True, {}
        
        self.etapes += 1
        recompense = self.cfg.recompense_deplacement
        
        # Tourne et avance
        self._tourner(action)
        tete_x, tete_y = self.serpent[-1]
        dx, dy = self.direction
        nouvelle_tete = (tete_x + dx, tete_y + dy)
        
        # Vérifie collision (murs ou corps)
        if self._est_collision(nouvelle_tete):
            self.vivant = False
            return self._obtenir_etat(), self.cfg.recompense_mort, True, {}
        
        # Avance
        self.serpent.append(nouvelle_tete)
        
        # Mange la nourriture ?
        if nouvelle_tete == self.nourriture:
            recompense += self.cfg.recompense_nourriture
            self._placer_nourriture()
        else:
            self.serpent.popleft()  # Enlève la queue
        
        # Vérifie fin d'épisode
        termine = self.etapes >= self.cfg.max_etapes
        
        return self._obtenir_etat(), recompense, termine, {}
    
    def _est_collision(self, position):
        """Vérifie si une position est en collision"""
        x, y = position
        # Collision avec les murs
        if x < 0 or x >= self.grille or y < 0 or y >= self.grille:
            return True
        # Collision avec le corps
        if position in self.serpent and position != self.serpent[-1]:
            return True
        return False
    
    def _obtenir_etat(self):
        """
        Retourne un vecteur d'état de 11 dimensions:
        - 3 dangers (devant, gauche, droite)
        - 4 directions (haut, bas, gauche, droite)
        - 4 positions nourriture (haut, bas, gauche, droite)
        """
        tete_x, tete_y = self.serpent[-1]
        dx, dy = self.direction
        
        # Calcule les directions relatives
        if self.direction == self.HAUT:
            gauche, droite = self.GAUCHE, self.DROITE
        elif self.direction == self.BAS:
            gauche, droite = self.DROITE, self.GAUCHE
        elif self.direction == self.GAUCHE:
            gauche, droite = self.BAS, self.HAUT
        else:  # DROITE
            gauche, droite = self.HAUT, self.BAS
        
        # Détecte les dangers
        danger_devant = float(self._est_collision((tete_x + dx, tete_y + dy)))
        danger_gauche = float(self._est_collision((tete_x + gauche[0], tete_y + gauche[1])))
        danger_droite = float(self._est_collision((tete_x + droite[0], tete_y + droite[1])))
        
        # Direction actuelle (one-hot)
        dir_haut = float(self.direction == self.HAUT)
        dir_bas = float(self.direction == self.BAS)
        dir_gauche = float(self.direction == self.GAUCHE)
        dir_droite = float(self.direction == self.DROITE)
        
        # Position relative de la nourriture
        fx, fy = self.nourriture
        nourriture_haut = float(fy < tete_y)
        nourriture_bas = float(fy > tete_y)
        nourriture_gauche = float(fx < tete_x)
        nourriture_droite = float(fx > tete_x)
        
        return np.array([
            danger_devant, danger_gauche, danger_droite,
            dir_haut, dir_bas, dir_gauche, dir_droite,
            nourriture_haut, nourriture_bas, nourriture_gauche, nourriture_droite
        ], dtype=np.float32)
    
    def render(self):
        """Affiche le jeu avec pygame"""
        if not self.affichage:
            return
        
        # Gère les événements pygame
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                return
        
        # Fond noir
        self.ecran.fill((40, 40, 40))
        
        # Dessine le serpent (corps en vert clair)
        for x, y in self.serpent:
            rect = (x * self.cfg.taille_case, y * self.cfg.taille_case,
                   self.cfg.taille_case, self.cfg.taille_case)
            pygame.draw.rect(self.ecran, (60, 200, 60), rect)
        
        # Dessine la tête (vert foncé)
        tete_x, tete_y = self.serpent[-1]
        rect = (tete_x * self.cfg.taille_case, tete_y * self.cfg.taille_case,
               self.cfg.taille_case, self.cfg.taille_case)
        pygame.draw.rect(self.ecran, (40, 160, 40), rect)
        
        # Dessine la nourriture (rouge)
        fx, fy = self.nourriture
        rect = (fx * self.cfg.taille_case, fy * self.cfg.taille_case,
               self.cfg.taille_case, self.cfg.taille_case)
        pygame.draw.rect(self.ecran, (220, 80, 80), rect)
        
        pygame.display.flip()
        self.horloge.tick(self.cfg.fps)
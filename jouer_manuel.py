"""
Jouer au Snake manuellement avec les flèches du clavier
"""
import pygame
from snake_game import SnakeEnv, ConfigJeu


def jouer_manuellement():
    """Permet de jouer manuellement au Snake"""
    
    # Configuration
    config = ConfigJeu(
        taille_grille=20,
        taille_case=25,
        fps=10,
        max_etapes=10000
    )
    
    env = SnakeEnv(config, affichage=True)
    
    if not env.affichage:
        print("❌ Pygame n'est pas disponible!")
        print("Installez-le avec: pip install pygame")
        return
    
    print("🐍 Snake - Mode Manuel")
    print("Utilisez les flèches du clavier:")
    print("  ↑ = Tourner à gauche")
    print("  ↓ = Continuer tout droit")
    print("  ← ↓ → = Contrôles alternatifs")
    print("  ESPACE = Redémarrer")
    print("  ESC = Quitter")
    print("-" * 50)
    
    etat = env.reset()
    action = 0  # Par défaut: tout droit
    score = 0
    partie = 1
    
    en_cours = True
    
    while en_cours:
        # Gère les événements
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                en_cours = False
            
            elif event.type == pygame.KEYDOWN:
                # Contrôles simplifiés (basés sur les 3 actions)
                if event.key == pygame.K_LEFT:
                    action = 1  # Tourner à gauche
                elif event.key == pygame.K_RIGHT:
                    action = 2  # Tourner à droite
                elif event.key == pygame.K_UP or event.key == pygame.K_DOWN:
                    action = 0  # Tout droit
                
                # Redémarre
                elif event.key == pygame.K_SPACE:
                    print(f"\n🔄 Partie {partie} terminée - Score: {len(env.serpent)}")
                    partie += 1
                    etat = env.reset()
                    score = 0
                    action = 0
                    print(f"🎮 Nouvelle partie {partie}")
                
                # Quitte
                elif event.key == pygame.K_ESCAPE:
                    en_cours = False
        
        # Effectue l'action
        etat, recompense, termine, _ = env.step(action)
        score += recompense
        
        # Affiche
        env.render()
        
        # Réinitialise l'action à "tout droit"
        action = 0
        
        # Fin de partie
        if termine:
            print(f"\n💀 Game Over!")
            print(f"   Taille finale: {len(env.serpent)}")
            print(f"   Appuyez sur ESPACE pour rejouer")
            
            # Attend que le joueur appuie sur ESPACE ou quitte
            attente = True
            while attente and en_cours:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        en_cours = False
                        attente = False
                    elif event.type == pygame.KEYDOWN:
                        if event.key == pygame.K_SPACE:
                            partie += 1
                            etat = env.reset()
                            score = 0
                            action = 0
                            attente = False
                            print(f"\n🎮 Nouvelle partie {partie}")
                        elif event.key == pygame.K_ESCAPE:
                            en_cours = False
                            attente = False
                
                env.horloge.tick(config.fps)
    
    pygame.quit()
    print("\n👋 À bientôt!")


if __name__ == "__main__":
    jouer_manuellement()
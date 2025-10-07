"""
Script principal pour entraîner l'agent DQN sur Snake
"""
import numpy as np
from snake_game import SnakeEnv, ConfigJeu
from dqn_agent import AgentDQN, ConfigDQN


def entrainer(nb_episodes=1000, affichage=False):
    """Entraîne l'agent DQN"""
    
    # Configuration
    config_jeu = ConfigJeu(
        taille_grille=20,
        max_etapes=500,
        fps=15
    )
    config_dqn = ConfigDQN(
        lr=1e-3,
        batch_size=64,
        epsilon_decay=20_000
    )
    
    # Initialise l'environnement et l'agent
    env = SnakeEnv(config_jeu, affichage=affichage)
    agent = AgentDQN(dim_etat=11, nb_actions=3, config=config_dqn)
    
    # Statistiques
    scores = []
    pertes = []
    meilleur_score = 0
    
    print("🐍 Début de l'entraînement...")
    print(f"Device: {agent.device}")
    print("-" * 60)
    
    for episode in range(nb_episodes):
        etat = env.reset()
        score = 0
        perte_episode = []
        
        while True:
            # Choisit une action
            epsilon = agent.obtenir_epsilon()
            action = agent.choisir_action(etat, epsilon)
            
            # Effectue l'action
            etat_suivant, recompense, termine, _ = env.step(action)
            score += recompense
            
            # Mémorise la transition
            agent.memoriser(etat, action, recompense, etat_suivant, termine)
            
            # Apprend
            agent.etapes += 1
            if agent.etapes >= agent.cfg.debut_apprentissage:
                if agent.etapes % agent.cfg.freq_entrainement == 0:
                    perte = agent.apprendre()
                    if perte is not None:
                        perte_episode.append(perte)
                
                # Synchronise le target network
                agent.sync_target_network()
            
            # Affiche le jeu
            if affichage:
                env.render()
            
            etat = etat_suivant
            
            if termine:
                break
        
        # Statistiques
        taille_serpent = len(env.serpent)
        scores.append(score)
        
        if score > meilleur_score:
            meilleur_score = score
            agent.sauvegarder('meilleur_modele.pth')
        
        # Affiche les progrès
        if (episode + 1) % 10 == 0:
            score_moyen = np.mean(scores[-10:])
            perte_moyenne = np.mean(perte_episode) if perte_episode else 0
            epsilon = agent.obtenir_epsilon()
            
            print(f"Épisode {episode + 1}/{nb_episodes}")
            print(f"  Score moyen (10 derniers): {score_moyen:.1f}")
            print(f"  Meilleur score: {meilleur_score}")
            print(f"  Epsilon: {epsilon:.3f}")
            print(f"  Perte: {perte_moyenne:.4f}")
            print(f"  Buffer: {len(agent.buffer)}")
            print("-" * 60)
    
    # Sauvegarde finale
    agent.sauvegarder('modele_final.pth')
    print("\n✅ Entraînement terminé!")
    print(f"Meilleur score: {meilleur_score}")
    print(f"Modèle sauvegardé: meilleur_modele.pth")


def tester(chemin_modele='meilleur_modele.pth', nb_parties=10):
    """Teste l'agent entraîné"""
    
    config_jeu = ConfigJeu(
        taille_grille=20,
        max_etapes=500,
        fps=10
    )
    config_dqn = ConfigDQN()
    
    env = SnakeEnv(config_jeu, affichage=True)
    agent = AgentDQN(dim_etat=11, nb_actions=3, config=config_dqn)
    
    # Charge le modèle
    try:
        agent.charger(chemin_modele)
        print(f"✅ Modèle chargé: {chemin_modele}")
    except Exception:
        print(f"❌ Impossible de charger {chemin_modele}")
        return
    
    scores = []
    
    for partie in range(nb_parties):
        etat = env.reset()
        score = 0
        
        print(f"\n🎮 Partie {partie + 1}/{nb_parties}")
        
        while True:
            # Mode greedy (epsilon=0)
            action = agent.choisir_action(etat, epsilon=0.0)
            etat, recompense, termine, _ = env.step(action)
            score += recompense
            
            env.render()
            
            if termine:
                break
        
        taille = len(env.serpent)
        scores.append(taille)
        print(f"  Taille finale: {taille}")
    
    print(f"\n📊 Score moyen: {np.mean(scores):.1f}")
    print(f"📊 Meilleur score: {max(scores)}")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "test":
        # Mode test
        tester()
    else:
        # Mode entraînement
        affichage = "--display" in sys.argv
        entrainer(nb_episodes=1000, affichage=affichage)
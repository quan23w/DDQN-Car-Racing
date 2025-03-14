from game import GameEnv
import pygame
import numpy as np

# Change from DDQN to TabularTD agent
from model.tabular_td_agent import TabularTDAgent

from collections import deque
import random, math

TOTAL_GAMETIME = 10000
N_EPISODES = 100  # Reduced for testing
REPLACE_TARGET = 10

game = GameEnv.RacingEnv()
game.fps = 60

GameTime = 0 
GameHistory = []
renderFlag = True

# Initialize the tabular TD agent with testing parameters (low epsilon for exploitation)
td_agent = TabularTDAgent(alpha=0.1, gamma=0.99, n_actions=9, epsilon=0.05,
                            epsilon_end=0.05, epsilon_dec=1.0, state_bins=5)

# Load the trained model
td_agent.load_model('td_agent_qtable.pkl')
print(f"Model loaded with {len(td_agent.q_table)} states in Q-table")

scores = []
eps_history = []


def run():
    for e in range(N_EPISODES):
        # Reset env
        game.reset()

        done = False
        score = 0
        counter = 0

        gtime = 0

        # First step
        observation_, reward, done = game.step(0)
        observation = np.array(observation_)

        print(f"Episode {e+1}/{N_EPISODES}")
        
        while not done:
            
            for event in pygame.event.get():
                if event.type == pygame.QUIT: 
                    run = False
                    return

            # Choose action using the loaded policy
            action = td_agent.choose_action(observation)
            observation_, reward, done = game.step(action)
            observation_ = np.array(observation_)

            if reward == 0:
                counter += 1
                if counter > 100:
                    done = True
            else:
                counter = 0

            score += reward
            observation = observation_

            gtime += 1

            if gtime >= TOTAL_GAMETIME:
                done = True

            if renderFlag:
                game.render(action)

        # Episode complete
        scores.append(score)
        avg_score = np.mean(scores[-min(100, len(scores)):])
        print(f"Score: {score:.2f}, Average Score: {avg_score:.2f}")

    # Testing complete
    print(f"\nTesting completed for {N_EPISODES} episodes")
    print(f"Average Score: {np.mean(scores):.2f}")
    print(f"Max Score: {np.max(scores):.2f}")


if __name__ == "__main__":
    run()

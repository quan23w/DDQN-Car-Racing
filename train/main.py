import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from game import GameEnv
import pygame
import numpy as np
from model.tabular_td_agent import TabularTDAgent

TOTAL_GAMETIME = 1000  # Max game time for one episode
N_EPISODES = 10000

game = GameEnv.RacingEnv()
game.fps = 60

# Initialize the tabular TD agent with convergence parameters
td_agent = TabularTDAgent(alpha=0.1, gamma=0.99, n_actions=9, epsilon=1.00, 
                        epsilon_end=0.10, epsilon_dec=0.9995, state_bins=5,
                        convergence_threshold=1e-5, convergence_window=100)

scores = []
eps_history = []

def run():
    converged_count = 0  # Counter for consecutive convergence detections
    required_convergences = 5  # Number of consecutive convergence detections required
    
    for e in range(N_EPISODES):
        game.reset()
        
        done = False
        score = 0
        counter = 0

        observation_, reward, done = game.step(0)
        observation = np.array(observation_)
        gtime = 0  # game time counter
        
        # Render every 10 episodes
        renderFlag = False
        if e % 10 == 0 and e > 0:
            renderFlag = True

        while not done:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    return

            action = td_agent.choose_action(observation)
            observation_, reward, done = game.step(action)
            observation_ = np.array(observation_)

            # Countdown: if no reward is collected within 100 ticks, finish the episode
            if reward == 0:
                counter += 1
                if counter > 100:
                    done = True
            else:
                counter = 0

            score += reward
            # Direct learning without storing transitions in memory
            td_agent.learn(observation, action, reward, observation_, done)
            observation = observation_

            gtime += 1
            if gtime >= TOTAL_GAMETIME:
                done = True

            if renderFlag:
                game.render(action)

        eps_history.append(td_agent.epsilon)
        scores.append(score)
        avg_score = np.mean(scores[max(0, e-100):(e+1)])

        # Check for convergence
        if td_agent.check_convergence():
            converged_count += 1
            print("Convergence detected! Count:", converged_count)
            if converged_count >= required_convergences:
                print("Training converged after", e+1, "episodes. Stopping...")
                td_agent.save_model()
                print("Final model saved.")
                return
        else:
            converged_count = 0  # Reset counter if not converged

        if e % 10 == 0 and e > 10:
            td_agent.save_model()
            print("Model saved.")

        # Print Q-table size instead of memory size
        print('Episode:', e, 'Score: %.2f' % score,
              'Average Score: %.2f' % avg_score,
              'Epsilon:', td_agent.epsilon,
              'Q-table Size:', len(td_agent.q_table))

run()

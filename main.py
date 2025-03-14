import GameEnv
import pygame
import numpy as np
from ddqn_keras import DDQNAgent
import random, math

TOTAL_GAMETIME = 1000  # Max game time for one episode
N_EPISODES = 10000
REPLACE_TARGET = 50

game = GameEnv.RacingEnv()
game.fps = 60

# Initialize the DDQN agent.
# (Make sure the input dimension remains correct—if you modify the state space with new sensors, update input_dims accordingly)
ddqn_agent = DDQNAgent(alpha=0.0005, gamma=0.99, n_actions=5, epsilon=1.00, epsilon_end=0.10,
                         epsilon_dec=0.9995, replace_target=REPLACE_TARGET, batch_size=512, input_dims=19)

ddqn_scores = []
eps_history = []

def run():
    for e in range(N_EPISODES):
        game.reset()  # Reset the environment
        
        # If obstacles and speed signs aren't created automatically in reset(),
        # you can manually add them here.
        # For example:
        # from GameEnv import Obstacle, SpeedSign
        # game.obstacles = [Obstacle(300, 200, 50, 50), Obstacle(500, 400, 60, 60)]
        # game.speed_signs = [SpeedSign(400, 250, 40, 40, speed_limit=5)]
        
        done = False
        score = 0
        counter = 0

        observation_, reward, done = game.step(0)
        observation = np.array(observation_)
        gtime = 0  # game time counter
        
        # Render every 10 episodes (set renderFlag to True for visualization)
        renderFlag = False
        if e % 10 == 0 and e > 0:
            renderFlag = True

        while not done:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    return

            action = ddqn_agent.choose_action(observation)
            observation_, reward, done = game.step(action)
            observation_ = np.array(observation_)

            # Countdown: if no reward is collected within 100 ticks, finish the episode.
            if reward == 0:
                counter += 1
                if counter > 100:
                    done = True
            else:
                counter = 0

            score += reward
            ddqn_agent.remember(observation, action, reward, observation_, int(done))
            observation = observation_
            ddqn_agent.learn()

            gtime += 1
            if gtime >= TOTAL_GAMETIME:
                done = True

            if renderFlag:
                game.render(action)

        eps_history.append(ddqn_agent.epsilon)
        ddqn_scores.append(score)
        avg_score = np.mean(ddqn_scores[max(0, e-100):(e+1)])

        if e % REPLACE_TARGET == 0 and e > REPLACE_TARGET:
            ddqn_agent.update_network_parameters()

        if e % 10 == 0 and e > 10:
            ddqn_agent.save_model()
            print("Model saved.")

        print('Episode:', e, 'Score: %.2f' % score,
              'Average Score: %.2f' % avg_score,
              'Epsilon:', ddqn_agent.epsilon,
              'Memory Size:', ddqn_agent.memory.mem_cntr % ddqn_agent.memory.mem_size)

run()

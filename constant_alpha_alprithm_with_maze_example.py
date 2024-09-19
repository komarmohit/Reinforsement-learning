import numpy as np
import pygame
import itertools
import time
from collections import defaultdict
from maze import Maze  

def main():
    env = Maze()  
    env.reset()  

    Q = defaultdict(float)  
    all_actions = [0, 1, 2, 3]  
    alpha = 0.1  
    epsilon = 0.1 
    gamma = 0.9 
    num_episodes = 500  

    def behavior_policy(state, epsilon=0.1):
        """
        Epsilon-greedy behavior policy based on current Q-values.
        With probability epsilon, choose a random action.
        With probability (1 - epsilon), choose the best action based on Q-values.
        """
        if np.random.rand() < epsilon:  
            return np.random.choice(all_actions)
        else:  
            q_values = [Q[(state, a)] for a in all_actions]
            max_q = max(q_values)
            max_actions = [a for a in all_actions if Q[(state, a)] == max_q]
            return np.random.choice(max_actions)

    def normalize_Q(Q_render):
        """
        Normalize Q-values for rendering between -1 and 0,
        as expected by the value_to_color function in the Maze environment.
        """
        max_q = max(Q_render.values())
        min_q = min(Q_render.values())
        range_q = max_q - min_q if max_q - min_q != 0 else 1
        Q_normalized = {}
        for key, value in Q_render.items():
            Q_normalized[key] = (value - max_q) / range_q
        return Q_normalized

    for episode_num in range(1, num_episodes + 1):
        episode = []  
        state = env.reset()  
        done = False

        while not done:
            action = behavior_policy(state, epsilon)  
            next_state, reward, done, _ = env.step(action)  
            episode.append((state, action, reward))  
            state = next_state  

        G = 0

        
        for t in reversed(range(len(episode))):
            state_t, action_t, reward_t = episode[t]  
            G = gamma * G + reward_t  
            Q[(state_t, action_t)] += alpha * (G - Q[(state_t, action_t)])

        epsilon = max(0.01, epsilon * 0.995)

        if episode_num % 100 == 0:
            print(f"Episode {episode_num}/{num_episodes}")
            Q_render = {}
            for (state_key, action), value in Q.items():
                Q_render[(state_key[0], state_key[1], action)] = value
            Q_render_normalized = normalize_Q(Q_render)  
            env.render(Q=Q_render_normalized) 
            time.sleep(1) 

    print("Training completed. Testing the learned policy...")

    state = env.reset() 
    done = False
    while not done:
        action = behavior_policy(state, epsilon=0)  
        next_state, reward, done, _ = env.step(action) 
        env.render()  
        time.sleep(0.5)  
        state = next_state  

    env.close()  

if __name__ == '__main__':
    main()

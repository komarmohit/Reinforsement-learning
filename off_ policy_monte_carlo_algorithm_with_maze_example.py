import numpy as np
import pygame
import itertools
import time
from collections import defaultdict
from maze import Maze  # Import the custom Maze environment

def main():
    env = Maze()
    env.reset()

    Q = defaultdict(float)             # Q-values for state-action pairs
    C = defaultdict(float)             # Cumulative weights for importance sampling
    target_policy = defaultdict(lambda: 0)  # Greedy target policy (initialized to always take action 0)

    all_actions = [0, 1, 2, 3]  # List of all possible actions (UP, RIGHT, DOWN, LEFT)

    epsilon = 0.1  # Epsilon for exploration in behavior policy
    gamma = 0.9    # Discount factor for future rewards
    num_episodes = 500  # Number of episodes for training

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

    # Main loop: iterate over episodes
    for episode_num in range(1, num_episodes + 1):
        episode = []
        state = env.reset()
        done = False

        # Generate an episode using the behavior policy
        while not done:
            action = behavior_policy(state, epsilon)
            next_state, reward, done, _ = env.step(action)
            episode.append((state, action, reward))
            state = next_state

        # Initialize the return and importance sampling weight
        G = 0
        W = 1  # Importance sampling weight
        visited_state_action = set()

        # Loop backward over the episode
        for t in reversed(range(len(episode))):
            state_t, action_t, reward_t = episode[t]
            G = gamma * G + reward_t  # Accumulate discounted return
            state_action = (state_t, action_t)

            # Update cumulative weights for importance sampling
            C[state_action] += W

            # Update Q-values using weighted importance sampling
            Q[state_action] += (W / C[state_action]) * (G - Q[state_action])

            # Update the target policy to be greedy with respect to the Q-values
            target_policy[state_t] = max(all_actions, key=lambda a: Q[(state_t, a)])

            # If the behavior action differs from the greedy action, stop updating
            if action_t != target_policy[state_t]:
                break

            # Update importance sampling weight
            W *= 1.0  # Since we use deterministic target policy, rho (probability ratio) is 1

        epsilon = max(0.01, epsilon * 0.995)  # Decay epsilon over time

        # Render the environment every 100 episodes
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
        action = target_policy[state]  # Use the learned target policy (greedy)
        next_state, reward, done, _ = env.step(action)
        env.render()
        time.sleep(0.5)
        state = next_state

    env.close()

if __name__ == '__main__':
    main()

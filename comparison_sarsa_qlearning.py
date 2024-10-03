import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
import random
import time

# Assuming the environment definition is provided (Maze environment)
from maze import Maze  # Import your Maze environment here

# Epsilon-greedy policy for action selection
def epsilon_greedy_policy(Q, state, n_actions, epsilon=0.1):
    if random.uniform(0, 1) < epsilon:
        return np.random.randint(n_actions)  # Explore
    else:
        return np.argmax(Q[state])  # Exploit (Best action based on Q-values)

# SARSA Algorithm Implementation with intermediate print statements
def sarsa(env, num_episodes, alpha=0.1, gamma=0.9, epsilon=0.1):
    Q = defaultdict(lambda: np.zeros(env.action_space.n))  # Initialize Q-table
    rewards_per_episode = []

    for episode in range(num_episodes):
        state = env.reset()
        action = epsilon_greedy_policy(Q, state, env.action_space.n, epsilon)
        total_reward = 0
        done = False

        print(f"Episode {episode + 1}/{num_episodes} - SARSA")  # Print episode number

        while not done:
            next_state, reward, done, _ = env.step(action)
            next_action = epsilon_greedy_policy(Q, next_state, env.action_space.n, epsilon)

            # Print intermediate state, action, reward, and Q-value
            print(f"State: {state}, Action: {action}, Reward: {reward}")
            print(f"Q[{state}][{action}] before update: {Q[state][action]}")

            # Update the Q-value
            Q[state][action] += alpha * (reward + gamma * Q[next_state][next_action] - Q[state][action])

            print(f"Q[{state}][{action}] after update: {Q[state][action]}")

            total_reward += reward
            state, action = next_state, next_action

        rewards_per_episode.append(total_reward)

        print(f"Total reward for episode {episode + 1}: {total_reward}\n")

    return Q, rewards_per_episode

# Q-Learning Algorithm Implementation with intermediate print statements
def q_learning(env, num_episodes, alpha=0.1, gamma=0.9, epsilon=0.1):
    Q = defaultdict(lambda: np.zeros(env.action_space.n))  # Initialize Q-table
    rewards_per_episode = []

    for episode in range(num_episodes):
        state = env.reset()
        total_reward = 0
        done = False

        print(f"Episode {episode + 1}/{num_episodes} - Q-Learning")  # Print episode number

        while not done:
            action = epsilon_greedy_policy(Q, state, env.action_space.n, epsilon)
            next_state, reward, done, _ = env.step(action)

            # Print intermediate state, action, reward, and Q-value
            print(f"State: {state}, Action: {action}, Reward: {reward}")
            print(f"Q[{state}][{action}] before update: {Q[state][action]}")

            # Update the Q-value using the maximum Q-value for the next state (Q-learning)
            best_next_action = np.argmax(Q[next_state])
            Q[state][action] += alpha * (reward + gamma * Q[next_state][best_next_action] - Q[state][action])

            print(f"Q[{state}][{action}] after update: {Q[state][action]}")

            total_reward += reward
            state = next_state

        rewards_per_episode.append(total_reward)

        print(f"Total reward for episode {episode + 1}: {total_reward}\n")

    return Q, rewards_per_episode

# Plot comparison
def plot_comparative_results(sarsa_rewards, q_learning_rewards, num_episodes):
    plt.plot(range(num_episodes), sarsa_rewards, label='SARSA', color='blue')
    plt.plot(range(num_episodes), q_learning_rewards, label='Q-Learning', color='red')
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.title('Comparative Study: SARSA vs Q-Learning')
    plt.legend()
    plt.show()

# Main function to run the comparison
def main():
    num_episodes = 100  # Number of episodes for learning
    env = Maze()  # Initialize your environment

    # Run SARSA
    print("Running SARSA...\n")
    _, sarsa_rewards = sarsa(env, num_episodes)

    # Run Q-Learning
    print("Running Q-Learning...\n")
    _, q_learning_rewards = q_learning(env, num_episodes)

    # Plot the results
    plot_comparative_results(sarsa_rewards, q_learning_rewards, num_episodes)

if __name__ == "__main__":
    main()

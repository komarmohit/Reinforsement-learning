import numpy as np
import time
import pygame
from random import randint
from maze import *
import matplotlib.pyplot as plt  # Import matplotlib for plotting

env = Maze()  # Create the environment
action_values = np.zeros((5, 5, 4))  # Create the Q(s,a) table

# Define the epsilon-greedy policy
def policy(state, epsilon=0.2):
    if np.random.random() < epsilon:
        return np.random.randint(4)  # Random action for exploration
    else:
        av = action_values[state]
        return np.random.choice(np.flatnonzero(av == av.max()))  # Action with highest Q-value

# Function to plot the action values (Q-table)
def plot_action_values(action_values):
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    actions = ['Up', 'Right', 'Down', 'Left']

    for i, ax in enumerate(axes):
        cax = ax.matshow(action_values[:, :, i], cmap="viridis")
        fig.colorbar(cax, ax=ax)
        ax.set_title(f'Action: {actions[i]}')
    
    plt.show()

# Function to plot the current policy
def Plot_policy(action_values, environment_img):
    policy = np.argmax(action_values, axis=2)  # Choose the best action for each state
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.imshow(environment_img)  # Plot the maze/environment

    for i in range(action_values.shape[0]):
        for j in range(action_values.shape[1]):
            action = policy[i, j]
            if action == 0:  # Up
                ax.arrow(j, i, 0, -0.3, head_width=0.2, head_length=0.2, fc='red', ec='red')
            elif action == 1:  # Right
                ax.arrow(j, i, 0.3, 0, head_width=0.2, head_length=0.2, fc='green', ec='green')
            elif action == 2:  # Down
                ax.arrow(j, i, 0, 0.3, head_width=0.2, head_length=0.2, fc='blue', ec='blue')
            elif action == 3:  # Left
                ax.arrow(j, i, -0.3, 0, head_width=0.2, head_length=0.2, fc='yellow', ec='yellow')

    plt.show()

# Plot the action values
plot_action_values(action_values)

# Render the environment and plot the policy
Plot_policy(action_values, env.render(mode='rgb_array'))

# implement the sarsa
def sarsa(action_values, policy, episodes, alpha=0.1, gamma=0.99, epsilon=0.2):

    for episode in range(1, episodes + 1):
        state = env.reset()
        action = policy(state, epsilon)
        done = False

        while not done:
            next_state, reward, done, _ = env.step(action)
            next_action = policy(next_state, epsilon)

            qsa = action_values[state][action]
            next_qsa = action_values[next_state][next_action]
            action_values[state][action] = qsa + alpha * (reward +gamma * next_qsa - qsa)
            state = next_state
            action = next_action
sarsa(action_values, policy, 100)         

plot_action_values(action_values)
plot_policy(action_values, env.render(mode='rgb_array'))
test_agent(env, policy)
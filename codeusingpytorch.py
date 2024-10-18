import torch
import numpy as np
import matplotlib.pyplot as plt

# Data points (x, y)
data = np.array([[-1, 0.5], [1, 2], [1.5, 4], [3, 3]])

# Separate the data into x and y
x = data[:, 0]
y = data[:, 1]

# Convert x and y to PyTorch tensors
x = torch.tensor(x, dtype=torch.float32).reshape(-1, 1)  # Reshape for compatibility
y = torch.tensor(y, dtype=torch.float32).reshape(-1, 1)

# Initialize weights (c for intercept, m for slope)
c = torch.tensor(0.0, requires_grad=True)  # Intercept (w0)
m = torch.tensor(0.0, requires_grad=True)  # Slope (w1)

# Learning rate
learning_rate = 0.01

# Number of iterations for gradient descent
num_iterations = 1000

# Function to compute mean squared error
def compute_loss(y_pred, y):
    return torch.mean((y_pred - y) ** 2)

# Lists to store the loss at each step
losses = []

# Gradient Descent Algorithm
for i in range(num_iterations):
    # Predict y using the current weights
    y_pred = c + m * x

    # Compute the loss
    loss = compute_loss(y_pred, y)

    # Perform backpropagation to compute gradients
    loss.backward()

    # Update weights using gradients
    with torch.no_grad():
        c -= learning_rate * c.grad
        m -= learning_rate * m.grad

    # Zero the gradients after updating
    c.grad.zero_()
    m.grad.zero_()

    # Store the loss for plotting
    losses.append(loss.item())

    # Print intermediate results every 100 iterations
    if i % 100 == 0:
        print(f"Step {i}: c (intercept) = {c.item():.4f}, m (slope) = {m.item():.4f}, Loss = {loss.item():.4f}")

# Print the final weights
print(f"\nOptimal intercept (c): {c.item():.4f}")
print(f"Optimal slope (m): {m.item():.4f}")

# Plot the data points and the best fit line
plt.scatter(x.numpy(), y.numpy(), color='red', label='Data Points')
plt.plot(x.numpy(), (c + m * x).detach().numpy(), color='blue', label='Best Fit Line')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Optimal Weights Linear Regression (PyTorch)')
plt.legend()
plt.grid(True)
plt.show()

# Plot validation loss vs. steps
plt.plot(range(num_iterations), losses, color='purple')
plt.xlabel('Iteration')
plt.ylabel('Loss')
plt.title('Validation Loss vs Steps (PyTorch)')
plt.grid(True)
plt.show()

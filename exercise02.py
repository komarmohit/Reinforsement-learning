import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter
from sklearn.model_selection import train_test_split

# Generate 2D data for Z = 2x^2 + 4y^3
x = torch.linspace(-2, 2, 100)
y = torch.linspace(-2, 2, 100)
X, Y = torch.meshgrid(x, y)
Z = 2 * X**2 + 4 * Y**3 + torch.randn(X.size()) * 0.5  # Adding some noise

# Flatten data and prepare it as a feature matrix
X_flat = X.flatten().reshape(-1, 1)
Y_flat = Y.flatten().reshape(-1, 1)
Z_flat = Z.flatten().reshape(-1, 1)
features = torch.cat((X_flat, Y_flat), dim=1)

# Split the data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(features, Z_flat, test_size=0.2, random_state=42)

class NonLinearRegression2D(nn.Module):
    def __init__(self):
        super(NonLinearRegression2D, self).__init__()
        # Define a simple neural network with one hidden layer
        self.hidden = nn.Linear(2, 10)  # 2 input features (x and y), 10 hidden units
        self.relu = nn.ReLU()           # Non-linear activation function
        self.output = nn.Linear(10, 1)  # Output layer with 1 output feature for Z

    def forward(self, xy):
        x = self.hidden(xy)   # First hidden layer
        x = self.relu(x)      # Apply non-linear activation
        x = self.output(x)    # Output layer
        return x

# Initialize model, loss function, and optimizer
non_linear_model = NonLinearRegression2D()
criterion = nn.MSELoss()  # Mean Squared Error loss
optimizer = torch.optim.SGD(non_linear_model.parameters(), lr=0.01)  # Stochastic Gradient Descent

# Set up TensorBoard writer
writer = SummaryWriter('runs/2d_regression_experiments')

# Visualization function
def plot_predictions(epoch, model, X_train, y_train, X_val, y_val):
    plt.clf()  # Clear the previous plot
    model.eval()
    train_pred = model(X_train).detach().numpy()
    val_pred = model(X_val).detach().numpy()

    plt.scatter(X_train[:, 0].numpy(), y_train.numpy(), label='Training data', color='blue')
    plt.scatter(X_val[:, 0].numpy(), y_val.numpy(), label='Validation data', color='orange')
    plt.scatter(X_train[:, 0].numpy(), train_pred, label=f'Train Prediction at epoch {epoch}', color='green')
    plt.scatter(X_val[:, 0].numpy(), val_pred, label=f'Val Prediction at epoch {epoch}', color='red')

    plt.legend()
    plt.title(f'Prediction vs Actual Data at Epoch {epoch}')
    plt.pause(0.1)  # Pause to update the plot

# Prepare to visualize in real-time
plt.ion()  # Enable interactive mode

# Train the non-linear model and visualize predictions at intervals
epochs = 5000
for epoch in range(epochs):
    # Train mode
    non_linear_model.train()
    y_train_pred = non_linear_model(X_train)
    train_loss = criterion(y_train_pred, y_train)

    optimizer.zero_grad()  # Clear gradients
    train_loss.backward()  # Backpropagation
    optimizer.step()  # Update weights

    # Validation mode
    non_linear_model.eval()  # Set the model to evaluation mode for validation
    with torch.no_grad():  # Disable gradient calculation
        y_val_pred = non_linear_model(X_val)
        val_loss = criterion(y_val_pred, y_val)

    # Log the training and validation loss to TensorBoard
    writer.add_scalar('Loss/train', train_loss.item(), epoch)
    writer.add_scalar('Loss/validation', val_loss.item(), epoch)

    # Visualize every 100 epochs
    if (epoch + 1) % 100 == 0:
        print(f'Epoch {epoch+1}/{epochs}, Training Loss: {train_loss.item():.4f}, Validation Loss: {val_loss.item():.4f}')
        plot_predictions(epoch + 1, non_linear_model, X_train, y_train, X_val, y_val)

plt.ioff()  # Disable interactive mode after training
plt.show()  # Show the final plot

# Close the TensorBoard writer
writer.close()

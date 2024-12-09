import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.preprocessing import StandardScaler

# Constants
NUM_ANTENNAS = 8  # Number of antennas in the base station
NUM_USERS = 4     # Number of users
GRID_SIZE = 10    # Size of the simulation grid

# Generate random positions for users in a 2D space
np.random.seed(42)  # For reproducibility
user_positions = np.random.uniform(-GRID_SIZE, GRID_SIZE, (NUM_USERS, 2))

# Base station at the origin
base_station_position = np.array([0, 0])

# Visualize the setup
plt.figure(figsize=(8, 8))
plt.scatter(user_positions[:, 0], user_positions[:, 1], label="Users", color='blue')
plt.scatter(base_station_position[0], base_station_position[1], label="Base Station", color='red')
plt.xlim(-GRID_SIZE, GRID_SIZE)
plt.ylim(-GRID_SIZE, GRID_SIZE)
plt.axhline(0, color='black', linestyle='--', linewidth=0.5)
plt.axvline(0, color='black', linestyle='--', linewidth=0.5)
plt.legend()
plt.title("5G Base Station and User Positions")
plt.xlabel("X Coordinate")
plt.ylabel("Y Coordinate")
plt.grid()
plt.show()


def calculate_snr(user_position, antenna_position, noise_power=1e-3):
    """
    Calculate the Signal-to-Noise Ratio (SNR) for a given user and antenna position.
    """
    distance = np.linalg.norm(user_position - antenna_position)
    signal_power = 1 / (distance ** 2 + 1e-6)  # Inverse square law
    snr = 10 * np.log10(signal_power / noise_power)
    return snr


# Antenna positions in 2D space (antennas placed along x-axis, y = 0)
antenna_positions = np.array([[pos, 0] for pos in np.linspace(-GRID_SIZE, GRID_SIZE, NUM_ANTENNAS)])

# Calculate SNRs for each user-antenna pair
snrs = []
for user in user_positions:
    user_snrs = []
    for antenna in antenna_positions:
        user_snrs.append(calculate_snr(user, antenna))
    snrs.append(user_snrs)

# Create the SNR matrix
snr_matrix = np.array(snrs)
print("SNR Matrix (Users x Antennas):")
print(snr_matrix)

# Prepare data
X = []  # Feature set
y = []  # Target SNRs

# Create feature-target pairs for all user-antenna combinations
for user_index, user in enumerate(user_positions):
    for antenna_index, antenna in enumerate(antenna_positions):
        X.append(np.concatenate([user, antenna]))  # Combine user and antenna positions
        y.append(snr_matrix[user_index, antenna_index])  # Corresponding SNR value

X = np.array(X)  # Convert to NumPy array
y = np.array(y)  # Convert to NumPy array

# Normalize features using StandardScaler
scaler = StandardScaler()
X_norm = scaler.fit_transform(X)

# Define neural network
model = Sequential([
    Dense(16, activation='relu', input_shape=(X_norm.shape[1],)),
    Dense(8, activation='relu'),
    Dense(1, activation='linear')
])

model.compile(optimizer='adam', loss='mse', metrics=['mae'])

# Train model
model.fit(X_norm, y, epochs=50, batch_size=8, verbose=1)

# Predict optimized SNR
optimized_snr = model.predict(X_norm)
print("Optimized SNR Predictions:")
print(optimized_snr)

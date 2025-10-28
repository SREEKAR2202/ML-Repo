import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

# Initialize random seed for consistent results
np.random.seed(42)

# Define noise parameters
NOISE_STD_X = 0.25
NOISE_STD_Y = 0.25
MEASUREMENT_NOISE = 0.3

# Contour visualization levels
VISUALIZATION_LEVELS = np.geomspace(0.0001, 250, 100)

def sample_position_in_circle():
    """Generate random coordinates within unit circle"""
    radius = np.sqrt(np.random.uniform(0, 1))
    angle = np.random.uniform(0, 2) * np.pi
    x_coord = radius * np.cos(angle)
    y_coord = radius * np.sin(angle)
    return np.array([x_coord, y_coord])

def compute_distance_measurements(num_landmarks, actual_position):
    """Calculate distance measurements from all landmarks"""
    measurements = []
    for landmark_idx in range(num_landmarks):
        landmark_coords = calculate_landmark_position(landmark_idx, num_landmarks)
        distance = measure_distance(landmark_coords, actual_position)
        measurements.append(distance)
    return measurements

def calculate_landmark_position(index, total_landmarks):
    """Determine landmark position on unit circle"""
    angular_position = 2 * np.pi / total_landmarks * index
    x_position = np.cos(angular_position)
    y_position = np.sin(angular_position)
    return np.array([x_position, y_position])

def measure_distance(landmark_coords, true_coords):
    """Generate noisy distance measurement"""
    actual_distance = np.linalg.norm(true_coords - landmark_coords)
    
    # Add Gaussian noise with rejection sampling for non-negative values
    max_attempts = 100
    for attempt in range(max_attempts):
        gaussian_noise = np.random.normal(0, MEASUREMENT_NOISE)
        noisy_distance = actual_distance + gaussian_noise
        if noisy_distance >= 0:
            return noisy_distance
    
    # Return noiseless measurement if all attempts fail
    return actual_distance

def visualize_objective_function(distance_data, true_location):
    """Create contour plot of MAP objective function"""
    # Create grid for evaluation
    x_grid = np.linspace(-2, 2, 128)
    y_grid = np.linspace(-2, 2, 128)
    grid_coords = np.meshgrid(x_grid, y_grid)
    
    # Evaluate objective function
    objective_values = evaluate_map_objective(grid_coords, distance_data)

    # Set up plot style
    plt.style.use('default')
    fig, axes = plt.subplots(figsize=(10, 8))

    # Draw unit circle boundary
    boundary_circle = plt.Circle(
        (0, 0), 1, 
        color='#2F4F4F', 
        fill=False, 
        linewidth=2,
        linestyle='--',
        label='Unit Circle Boundary'
    )
    axes.add_artist(boundary_circle)

    # Create contour plot
    contour_plot = plt.contour(
        grid_coords[0], grid_coords[1], 
        objective_values, 
        cmap='viridis',
        levels=VISUALIZATION_LEVELS,
        alpha=0.7
    )

    # Plot landmarks and range circles
    num_sensors = len(distance_data)
    for sensor_idx, measured_range in enumerate(distance_data):
        sensor_coords = calculate_landmark_position(sensor_idx, num_sensors)
        x_sensor, y_sensor = sensor_coords
        
        # Plot landmark marker
        if sensor_idx == 0:
            plt.plot(
                x_sensor, y_sensor, 
                'D', 
                color='#FF8C00', 
                markersize=10,
                markerfacecolor='none',
                markeredgewidth=2,
                label='Sensor Location'
            )
        else:
            plt.plot(
                x_sensor, y_sensor, 
                'D', 
                color='#FF8C00', 
                markersize=10,
                markerfacecolor='none',
                markeredgewidth=2
            )
        
        # Draw measurement circle
        measurement_circle = plt.Circle(
            (x_sensor, y_sensor), 
            measured_range, 
            color='#4169E1', 
            alpha=0.15, 
            fill=False,
            linestyle=':',
            linewidth=1.5,
            label='Measurement Range' if sensor_idx == 0 else ""
        )
        axes.add_artist(measurement_circle)

    # Mark true position
    plt.plot(
        true_location[0], true_location[1], 
        '*', 
        color='#DC143C', 
        markersize=15,
        markeredgecolor='black',
        markeredgewidth=0.5,
        label='Actual Position'
    )

    # Configure plot
    axes.set_xlabel("X Coordinate", fontsize=11, fontweight='bold')
    axes.set_ylabel("Y Coordinate", fontsize=11, fontweight='bold')
    axes.set_title(
        f"Objective Function Contours for MAP Estimation (Sensors: {num_sensors})",
        fontsize=12,
        fontweight='bold',
        pad=15
    )
    axes.set_xlim((-2, 2))
    axes.set_ylim((-2, 2))
    axes.set_aspect('equal')
    axes.grid(True, alpha=0.2, linestyle='-', linewidth=0.5)

    # Add legend
    plt.legend(
        loc='upper right',
        framealpha=0.9,
        edgecolor='black',
        fontsize=9
    )

    # Add colorbar
    colorbar = plt.colorbar(contour_plot, label='Objective Function Value')
    colorbar.ax.tick_params(labelsize=9)

    plt.tight_layout()
    plt.show()

def evaluate_map_objective(position_grid, measurements):
    """Compute MAP objective function over grid"""
    # Reshape grid for computation
    grid_reshaped = np.expand_dims(
        np.transpose(position_grid, axes=(1, 2, 0)), 
        axis=len(np.shape(position_grid))-1
    )
    
    # Calculate prior term
    covariance_inverse = np.linalg.inv(
        np.array([[NOISE_STD_X**2, 0], [0, NOISE_STD_Y**2]])
    )
    prior_term = np.matmul(grid_reshaped, covariance_inverse)
    prior_term = np.matmul(prior_term, np.swapaxes(grid_reshaped, 2, 3))
    prior_term = np.squeeze(prior_term)
    
    # Calculate likelihood term
    likelihood_sum = 0
    total_sensors = len(measurements)
    
    for sensor_index, range_value in enumerate(measurements):
        sensor_position = calculate_landmark_position(sensor_index, total_sensors)
        
        # Compute distances from grid points to landmark
        distance_field = np.linalg.norm(
            grid_reshaped - sensor_position[None, None, None, :], 
            axis=3
        )
        
        # Accumulate squared residuals
        residual_term = (range_value - distance_field)**2 / MEASUREMENT_NOISE**2
        likelihood_sum += np.squeeze(residual_term)

    return prior_term + likelihood_sum

# ==================== MAIN EXECUTION ====================
# Generate random true position
actual_target_position = sample_position_in_circle()

# Test with different numbers of landmarks
sensor_configurations = [1, 2, 3, 4]

for num_sensors in sensor_configurations:
    # Generate measurements
    range_data = compute_distance_measurements(num_sensors, actual_target_position)
    
    # Visualize results
    visualize_objective_function(range_data, actual_target_position)
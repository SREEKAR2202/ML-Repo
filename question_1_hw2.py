import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
from sklearn.metrics import roc_curve, auc, accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import PolynomialFeatures

# Define class prior probabilities
prior_class0 = 0.6
prior_class1 = 0.4

# Define parameters for Gaussian mixture components of Class 0
mean_c0_comp1 = np.array([-0.9, -1.1])
covariance_c0_comp1 = np.array([[0.75, 0], [0, 1.25]])

mean_c0_comp2 = np.array([0.8, 0.75])
covariance_c0_comp2 = np.array([[0.75, 0], [0, 1.25]])

# Define parameters for Gaussian mixture components of Class 1
mean_c1_comp1 = np.array([-1.1, 0.9])
covariance_c1_comp1 = np.array([[0.75, 0], [0, 1.25]])

mean_c1_comp2 = np.array([0.9, -0.75])
covariance_c1_comp2 = np.array([[0.75, 0], [0, 1.25]])


def create_dataset(num_samples):
    """Generate synthetic data from Gaussian mixture models"""
    class_labels = np.random.choice([0, 1], size=num_samples, p=[prior_class0, prior_class1])
    feature_vectors = []
    
    for label in class_labels:
        if label == 0:
            # Sample from Class 0 mixture
            mixture_component = np.random.choice([1, 2])
            if mixture_component == 1:
                data_point = np.random.multivariate_normal(mean_c0_comp1, covariance_c0_comp1)
            else:
                data_point = np.random.multivariate_normal(mean_c0_comp2, covariance_c0_comp2)
        else:
            # Sample from Class 1 mixture
            mixture_component = np.random.choice([1, 2])
            if mixture_component == 1:
                data_point = np.random.multivariate_normal(mean_c1_comp1, covariance_c1_comp1)
            else:
                data_point = np.random.multivariate_normal(mean_c1_comp2, covariance_c1_comp2)
        
        feature_vectors.append(data_point)
    
    return np.array(feature_vectors), class_labels


# Generate training datasets with different sizes
features_train50, labels_train50 = create_dataset(50)
features_train500, labels_train500 = create_dataset(500)
features_train5000, labels_train5000 = create_dataset(5000)

# Generate validation dataset
features_val, labels_val = create_dataset(10000)


# Visualize all datasets
fig, axes = plt.subplots(2, 2, figsize=(15, 12))
dataset_info = [
    (features_train50, labels_train50, 'Training Set (N=50)', axes[0, 0]),
    (features_train500, labels_train500, 'Training Set (N=500)', axes[0, 1]),
    (features_train5000, labels_train5000, 'Training Set (N=5000)', axes[1, 0]),
    (features_val, labels_val, 'Validation Set (N=10000)', axes[1, 1])
]

for features, labels, plot_title, ax in dataset_info:
    class0_mask = labels == 0
    class1_mask = labels == 1
    ax.scatter(features[class0_mask, 0], features[class0_mask, 1], 
               label='Class 0', alpha=0.6, color='blue')
    ax.scatter(features[class1_mask, 0], features[class1_mask, 1], 
               label='Class 1', alpha=0.6, color='red')
    ax.set_title(plot_title)
    ax.set_xlabel('Feature 1')
    ax.set_ylabel('Feature 2')
    ax.legend()
    ax.grid(True)

plt.tight_layout()
plt.show()


# PART 1: Implement theoretically optimal Bayes classifier

def likelihood_class0(data_points):
    """Calculate likelihood for Class 0 using Gaussian mixture"""
    component1_likelihood = multivariate_normal.pdf(data_points, mean=mean_c0_comp1, cov=covariance_c0_comp1)
    component2_likelihood = multivariate_normal.pdf(data_points, mean=mean_c0_comp2, cov=covariance_c0_comp2)
    return 0.5 * component1_likelihood + 0.5 * component2_likelihood


def likelihood_class1(data_points):
    """Calculate likelihood for Class 1 using Gaussian mixture"""
    component1_likelihood = multivariate_normal.pdf(data_points, mean=mean_c1_comp1, cov=covariance_c1_comp1)
    component2_likelihood = multivariate_normal.pdf(data_points, mean=mean_c1_comp2, cov=covariance_c1_comp2)
    return 0.5 * component1_likelihood + 0.5 * component2_likelihood


def calculate_posterior_probability(data_points):
    """Compute posterior probability P(Class=1|X) using Bayes theorem"""
    likelihood_c0 = likelihood_class0(data_points)
    likelihood_c1 = likelihood_class1(data_points)
    
    evidence = likelihood_c0 * prior_class0 + likelihood_c1 * prior_class1
    posterior_class1 = (likelihood_c1 * prior_class1) / evidence
    
    return posterior_class1


# Apply Bayes classifier to validation set
posterior_probabilities = calculate_posterior_probability(features_val)
bayes_predictions = (posterior_probabilities > 0.5).astype(int)

# Evaluate performance
min_error_probability = 1 - accuracy_score(labels_val, bayes_predictions)
print(f"Minimum Probability of Error (min-P(error)): {min_error_probability:.4f}")

# Generate ROC curve
false_pos_rate, true_pos_rate, threshold_values = roc_curve(labels_val, posterior_probabilities)
area_under_curve = auc(false_pos_rate, true_pos_rate)

# Identify the operating point at threshold 0.5
threshold_idx = np.argmin(np.abs(threshold_values - 0.5))
fpr_at_threshold = false_pos_rate[threshold_idx]
tpr_at_threshold = true_pos_rate[threshold_idx]

# Visualize ROC curve
plt.figure()
plt.plot(false_pos_rate, true_pos_rate, label=f"ROC curve (area = {area_under_curve:.2f})")
plt.scatter(fpr_at_threshold, tpr_at_threshold, color='red', label='Min P(error) point')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve for Theoretically Optimal Classifier')
plt.legend(loc="lower right")
plt.grid(True)
plt.show()

# Create decision boundary visualization
x_range = [features_val[:, 0].min() - 1, features_val[:, 0].max() + 1]
y_range = [features_val[:, 1].min() - 1, features_val[:, 1].max() + 1]
x_grid, y_grid = np.meshgrid(np.linspace(x_range[0], x_range[1], 300),
                              np.linspace(y_range[0], y_range[1], 300))
grid_points = np.c_[x_grid.ravel(), y_grid.ravel()]
grid_posteriors = calculate_posterior_probability(grid_points)
decision_surface = grid_posteriors.reshape(x_grid.shape)

# Plot decision regions and classified points
plt.figure(figsize=(8, 6))
plt.contourf(x_grid, y_grid, decision_surface > 0.5, alpha=0.5, 
             levels=[0, 0.5, 1], colors=['Darkblue', 'Darkred'])

# Identify correct and incorrect classifications
correct_c0 = (labels_val == 0) & (bayes_predictions == 0)
correct_c1 = (labels_val == 1) & (bayes_predictions == 1)
incorrect_c0 = (labels_val == 0) & (bayes_predictions == 1)
incorrect_c1 = (labels_val == 1) & (bayes_predictions == 0)

plt.scatter(features_val[correct_c0, 0], features_val[correct_c0, 1], 
            label='Correct Class 0', alpha=0.5, marker='o', color='blue')
plt.scatter(features_val[correct_c1, 0], features_val[correct_c1, 1], 
            label='Correct Class 1', alpha=0.5, marker='o', color='red')
plt.scatter(features_val[incorrect_c0, 0], features_val[incorrect_c0, 1], 
            label='Incorrect Class 0', alpha=0.5, marker='x', color='cyan')
plt.scatter(features_val[incorrect_c1, 0], features_val[incorrect_c1, 1], 
            label='Incorrect Class 1', alpha=0.5, marker='x', color='magenta')

plt.title('Decision Boundary of Theoretically Optimal Classifier')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.legend()
plt.show()


# PART 2: Train logistic regression classifiers

def apply_polynomial_features(data_matrix):
    """Transform features to include quadratic terms"""
    transformer = PolynomialFeatures(degree=2, include_bias=False)
    return transformer.fit_transform(data_matrix)


# Train and evaluate logistic-linear classifiers
training_datasets = [
    (features_train50, labels_train50, 50),
    (features_train500, labels_train500, 500),
    (features_train5000, labels_train5000, 5000)
]

linear_classifiers = []
linear_predictions_scores = []

for train_features, train_labels, sample_count in training_datasets:
    model = LogisticRegression(solver='lbfgs')
    model.fit(train_features, train_labels)
    
    validation_predictions = model.predict(features_val)
    model_error_rate = 1 - accuracy_score(labels_val, validation_predictions)
    print(f"Logistic-Linear Classifier with {sample_count} samples - Error Rate: {model_error_rate:.4f}")
    
    probability_scores = model.predict_proba(features_val)[:, 1]
    
    linear_classifiers.append(model)
    linear_predictions_scores.append(probability_scores)


# Transform datasets for quadratic classifiers
features_train50_poly = apply_polynomial_features(features_train50)
features_train500_poly = apply_polynomial_features(features_train500)
features_train5000_poly = apply_polynomial_features(features_train5000)
features_val_poly = apply_polynomial_features(features_val)

quadratic_classifiers = []
quadratic_predictions_scores = []

training_datasets_poly = [
    (features_train50_poly, labels_train50, 50),
    (features_train500_poly, labels_train500, 500),
    (features_train5000_poly, labels_train5000, 5000)
]

for train_features_poly, train_labels, sample_count in training_datasets_poly:
    model = LogisticRegression(solver='lbfgs', max_iter=1000)
    model.fit(train_features_poly, train_labels)
    
    validation_predictions = model.predict(features_val_poly)
    model_error_rate = 1 - accuracy_score(labels_val, validation_predictions)
    print(f"Logistic-Quadratic Classifier with {sample_count} samples - Error Rate: {model_error_rate:.4f}")
    
    probability_scores = model.predict_proba(features_val_poly)[:, 1]
    
    quadratic_classifiers.append(model)
    quadratic_predictions_scores.append(probability_scores)


# Visualize ROC curves for linear classifiers
plt.figure()
sample_sizes = ['50 samples', '500 samples', '5000 samples']
for idx, sample_label in enumerate(sample_sizes):
    fpr, tpr, _ = roc_curve(labels_val, linear_predictions_scores[idx])
    roc_area = auc(fpr, tpr)
    plt.plot(fpr, tpr, label=f'Linear ({sample_label}), AUC={roc_area:.2f}')

plt.title('ROC Curves for Logistic-Linear Classifiers')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend(loc='lower right')
plt.grid(True)
plt.show()


# Visualize ROC curves for quadratic classifiers
plt.figure()
for idx, sample_label in enumerate(sample_sizes):
    fpr, tpr, _ = roc_curve(labels_val, quadratic_predictions_scores[idx])
    roc_area = auc(fpr, tpr)
    plt.plot(fpr, tpr, label=f'Quadratic ({sample_label}), AUC={roc_area:.2f}')

plt.title('ROC Curves for Logistic-Quadratic Classifiers')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend(loc='lower right')
plt.grid(True)
plt.show()


def visualize_classifier_boundary(trained_model, input_features, true_labels, 
                                   figure_title, is_quadratic=False):
    """Plot decision boundary and classification results"""
    feature_data = input_features.copy()
    
    x_bounds = [feature_data[:, 0].min() - 1, feature_data[:, 0].max() + 1]
    y_bounds = [feature_data[:, 1].min() - 1, feature_data[:, 1].max() + 1]
    
    x_mesh, y_mesh = np.meshgrid(np.linspace(x_bounds[0], x_bounds[1], 300),
                                  np.linspace(y_bounds[0], y_bounds[1], 300))
    mesh_points = np.c_[x_mesh.ravel(), y_mesh.ravel()]
    
    if is_quadratic:
        mesh_points = apply_polynomial_features(mesh_points)
        feature_data = apply_polynomial_features(feature_data)
    
    boundary_predictions = trained_model.predict(mesh_points)
    decision_grid = boundary_predictions.reshape(x_mesh.shape)
    
    plt.contourf(x_mesh, y_mesh, decision_grid, alpha=0.5, 
                 levels=np.linspace(0, 1, 3), colors=['Darkblue', 'Darkred'])
    
    model_predictions = trained_model.predict(feature_data)
    
    correctly_classified_c0 = (true_labels == 0) & (model_predictions == 0)
    correctly_classified_c1 = (true_labels == 1) & (model_predictions == 1)
    misclassified_c0 = (true_labels == 0) & (model_predictions == 1)
    misclassified_c1 = (true_labels == 1) & (model_predictions == 0)
    
    plt.scatter(input_features[correctly_classified_c0, 0], input_features[correctly_classified_c0, 1], 
                c='blue', marker='o', label='Correct Class 0', edgecolors='k')
    plt.scatter(input_features[correctly_classified_c1, 0], input_features[correctly_classified_c1, 1], 
                c='red', marker='o', label='Correct Class 1', edgecolors='k')
    plt.scatter(input_features[misclassified_c0, 0], input_features[misclassified_c0, 1], 
                c='cyan', marker='x', label='Incorrect Class 0', edgecolors='k')
    plt.scatter(input_features[misclassified_c1, 0], input_features[misclassified_c1, 1], 
                c='magenta', marker='x', label='Incorrect Class 1', edgecolors='k')
    
    plt.title(figure_title)
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.legend()
    plt.show()


# Visualize decision boundaries for best linear classifier
visualize_classifier_boundary(linear_classifiers[2], features_train5000, labels_train5000, 
                              'Logistic-Linear Decision Boundary (5000 samples)')
visualize_classifier_boundary(linear_classifiers[2], features_val, labels_val, 
                              'Logistic-Linear Decision Boundary (10k samples)')

# Visualize decision boundaries for best quadratic classifier
visualize_classifier_boundary(quadratic_classifiers[2], features_train5000, labels_train5000, 
                              'Logistic-Quadratic Decision Boundary (5000 samples)', is_quadratic=True)
visualize_classifier_boundary(quadratic_classifiers[2], features_val, labels_val, 
                              'Logistic-Quadratic Decision Boundary (10k samples)', is_quadratic=True)
# q2_partA_all_in_one.py
# ------------------------------------------------------------
# Q2 — Part A (MAP, 0–1 loss) with:
#   • dataset generation (10k samples, 4 Gaussians in R^2)
#   • MAP decisions and confusion matrix P(D=i | L=j)
#   • green/red scatter plot (correct/incorrect)
#   • ALSO: average minimum expected risk under Λ (same data)
# ------------------------------------------------------------

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal

# np.random.seed(42)


# Problem setup

K = 4
priors = np.full(K, 0.25)  # equal priors

# Means (class 4 in the center so it overlaps most)
means = [
    [-2.0,  0.0],  # class 1
    [ 2.0,  0.0],  # class 2
    [ 0.0,  2.0],  # class 3
    [ 0.0,  0.0],  # class 4 (most overlap)
]
# Covariances (class 4 a bit wider)
covs = [
    [[0.8,  0.2],  [0.2,  0.6]],    # class 1
    [[0.7, -0.1],  [-0.1, 0.7]],    # class 2
    [[0.6,  0.25], [0.25, 0.9]],    # class 3
    [[1.2,  0.5],  [0.5,  1.2]],    # class 4
]

# Loss matrix for Part B-style risk (rows=decision i, cols=true label j)
Lambda = np.array([
    [0, 10, 10, 100],
    [1,  0, 10, 100],
    [1,  1,  0, 100],
    [1,  1,  1,   0]
], dtype=float)


# 1) Making 10,000 samples

N = 10_000
labels = np.random.choice(K, size=N, p=priors)
X = np.zeros((N, 2))
for j in range(K):
    mask = (labels == j)
    X[mask] = np.random.multivariate_normal(mean=means[j],
                                            cov=covs[j],
                                            size=np.sum(mask))

# Saving for later
np.savez("q2_dataset.npz", X=X, labels=labels,
         means=np.array(means), covs=np.array(covs), priors=priors)

#confusion matrix P(D=i | L=j)
def confusion_P_D_given_L(y_true, y_pred, K):
    """
    Returns confusion matrix with columns = true label j,
    rows = decision i, so each column sums to 1:
        C[i, j] = P(D=i | L=j).
    """
    C = np.zeros((K, K), dtype=float)
    for j in range(K):
        mask = (y_true == j)
        if mask.any():
            preds = y_pred[mask]
            for i in range(K):
                C[i, j] = np.mean(preds == i)
    return C

# -----------------------------
# 3) MAP (0–1 loss) classification
# -----------------------------
# log p(x|j) for all classes (N x K)
logp = np.column_stack([
    multivariate_normal.logpdf(X, mean=means[j], cov=covs[j])
    for j in range(K)
])
# log posterior up to constant: log p(x|j) + log π_j
log_post = logp + np.log(priors)
# MAP decision
dec_map = np.argmax(log_post, axis=1)

# Confusion matrix (decimal formatting)
conf_map = confusion_P_D_given_L(labels, dec_map, K)
np.set_printoptions(precision=4, suppress=True)  # pretty decimals
print("Part A — Confusion Matrix  P(D=i | L=j):")
print(conf_map)
print("Column sums (should be ≈ 1):", conf_map.sum(axis=0))
print(f"Overall Accuracy (MAP, 0–1 loss): {np.mean(dec_map == labels):.4f}\n")

# -----------------------------
# 4) ALSO compute average minimum expected risk under Λ on the SAME data
#    (Bayes ERM rule for the given loss matrix)
# -----------------------------
# First get plain pdfs for numerical clarity
pdfs = np.column_stack([
    multivariate_normal.pdf(X, mean=means[j], cov=covs[j])
    for j in range(K)
])
# Unnormalized posteriors: w_j(x) = π_j p(x|j)
w = pdfs * priors  # (N x K)

# For each decision i, expected risk R_i(x) = sum_j Λ[i,j] * w_j(x)
risk = w @ Lambda.T  # (N x K)
dec_erm = np.argmin(risk, axis=1)

avg_min_expected_risk = float(np.mean(Lambda[dec_erm, labels]))
print(f"Average minimum expected risk under Λ (on same data): {avg_min_expected_risk:.6f}")

# (Optional) show Λ-ERM confusion too:
conf_erm = confusion_P_D_given_L(labels, dec_erm, K)
print("Λ-ERM Confusion Matrix  P(D=i | L=j):")
print(conf_erm)
# print("Column sums (should be ≈ 1):", conf_erm.sum(axis=0))

# -----------------------------
# 5) Visualization (green = correct, red = wrong for MAP)
# -----------------------------
markers = ['o', 's', '^', 'D']
plt.figure(figsize=(7.2, 6.4))
for j in range(K):
    mask = (labels == j)
    good = mask & (dec_map == j)
    bad  = mask & (dec_map != j)
    if good.any():
        plt.scatter(X[good,0], X[good,1], s=15, marker=markers[j],
                    c='green', alpha=0.6)
    if bad.any():
        plt.scatter(X[bad,0],  X[bad,1],  s=15, marker=markers[j],
                    c='red',   alpha=0.6)

# legend with class shapes
for j in range(K):
    plt.scatter([], [], marker=markers[j], c='k', label=f"Class {j+1}")

plt.legend(title="True class", ncol=2, fontsize=9)
plt.title("MAP green=correct, red=wrong")
plt.xlabel("x₁"); plt.ylabel("x₂"); plt.grid(alpha=0.3)
plt.tight_layout(); plt.show()

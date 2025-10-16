import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import solve

# ---- 1) Load the SAME dataset (from Part A) ----
Z = np.load("q1_dataset.npz")  # produced earlier
X, y = Z["X"], Z["y"]          # y ∈ {0,1}
p0, p1 = float(Z["p0"]), float(Z["p1"])

# ---- 2) Estimate class means & covariances (equal weighting note below) ----
X0, X1 = X[y==0], X[y==1]
m0_hat = X0.mean(axis=0)
m1_hat = X1.mean(axis=0)

# sample covariances (rowvar=False). We use unbiased (N-1) denominator by default.
C0_hat = np.cov(X0, rowvar=False)
C1_hat = np.cov(X1, rowvar=False)

# Within-class scatter (equal class weights per instructions):
Sw = C0_hat + C1_hat
# Between-class scatter direction:
mean_diff = (m1_hat - m0_hat)

# Optional tiny regularization if Sw is near-singular
eps = 1e-8
Sw_reg = Sw + eps * np.eye(Sw.shape[0])

# ---- 3) Fisher LDA direction ----
w = solve(Sw_reg, mean_diff)          # proportional to S_W^{-1}(m1 - m0)
# scale doesn't matter; the threshold will absorb it
# Normalize just to keep numbers tame
w = w / np.linalg.norm(w)

# ---- 4) Project data to 1D and sweep threshold τ ----
scores = X @ w  # y = w^T x

# Build thresholds from data (exact finite-sample sweep)
order = np.argsort(scores)          # ascending
s_sorted = scores[order]
y_sorted = y[order]
N = len(y)
N0 = np.sum(y==0); N1 = np.sum(y==1)

# If we predict class 1 for scores > τ, then as τ moves from +∞ -> -∞,
# more points become class 1. We'll scan cutpoints between sorted scores.
# Cumulative counts from the right for the ">" rule:
y_rev = y_sorted[::-1]
tp_cum = np.cumsum(y_rev==1)  # number of positives predicted as threshold moves down
fp_cum = np.cumsum(y_rev==0)

TP = np.concatenate(([0], tp_cum))   # include endpoint τ=+∞ (all predicted 0)
FP = np.concatenate(([0], fp_cum))

PD  = TP / N1
PFA = FP / N0
Pe  = PFA * p0 + (1 - PD) * p1

# Find empirical min error and its τ*
k_star = Pe.argmin()  # index in the cumulative arrays
# Convert index to threshold between adjacent scores
# k=0 -> τ=+∞ ; k=N -> τ=-∞ ; otherwise midpoint between the two adjacent sorted scores (from right scanning)
def idx_to_tau(k):
    if k == 0:            # predict all 0
        return np.inf
    if k == N:            # predict all 1
        return -np.inf
    # k in [1..N-1]: threshold is midpoint between the (N-k)th and (N-k+1)th sorted scores
    a = s_sorted[N - k]     # upper element
    b = s_sorted[N - k - 1] # lower element
    return 0.5*(a + b)

tau_star = idx_to_tau(k_star)
PD_star, PFA_star, Pe_star = PD[k_star], PFA[k_star], Pe[k_star]

# For reference: “balanced” (Bayes for equal priors/variances on projected 1D isn’t directly applicable);
# we’ll still print the τ that equals the midpoint between projected class means:
tau_mid = 0.5 * ((w @ m0_hat) + (w @ m1_hat))

# ---- 5) Plot ROC and mark min-Pe point ----
plt.figure(figsize=(6.8, 6.8))
plt.plot(PFA, PD, lw=2, label="ROC (Fisher LDA, empirical)")
plt.plot([0,1], [0,1], 'k--', lw=1, label="Chance")

plt.scatter(PFA_star, PD_star, s=90, facecolors='none', edgecolors='crimson',
            linewidths=2, label=f"Min $P_e$ (τ*={tau_star:.3g})")
plt.text(PFA_star+0.015, PD_star-0.05, f"$P_e^*={Pe_star:.4f}$",
         color='crimson', weight='bold')

plt.xlabel(r"False Alarm  $P(D{=}1\mid L{=}0)$")
plt.ylabel(r"Detection    $P(D{=}1\mid L{=}1)$")
plt.title("ROC — Fisher LDA (threshold on $w^T x$)")
plt.xlim(0,1); plt.ylim(0,1)
plt.grid(alpha=0.3); plt.legend(loc="lower right")
plt.tight_layout(); plt.show()

# ---- 6) Console summary for your report ----
print("== Fisher LDA (from data) ==")
print(f"w = {w}")
print(f"tau* (min Pe) = {tau_star:.6g}")
print(f"Pe*           = {Pe_star:.6f}")
print(f"PD(τ*)        = {PD_star:.6f}")
print(f"PFA(τ*)       = {PFA_star:.6f}")
print(f"tau_mid (midpoint of projected means) = {tau_mid:.6g}")

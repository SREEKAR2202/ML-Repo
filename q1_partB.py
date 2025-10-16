import numpy as np
import matplotlib.pyplot as plt

# --- Load the same dataset from Part A ---
Z = np.load("q1_dataset.npz")  # same 10k samples
X, y = Z["X"], Z["y"]
N0, N1 = np.sum(y == 0), np.sum(y == 1)

# --- Priors & parameters ---
p0, p1 = 0.65, 0.35
m0 = np.array([-0.5, -0.5, -0.5])
m1 = np.array([1.0, 1.0, 1.0])

# --- (A) True model (from Part A) log-likelihood ratio ---
from scipy.stats import multivariate_normal
C0 = np.array([[1, -0.5, 0.3], [-0.5, 1, -0.5], [0.3, -0.5, 1]])
C1 = np.array([[1, 0.3, -0.2], [0.3, 1, 0.3], [-0.2, 0.3, 1]])
logp0 = multivariate_normal.logpdf(X, mean=m0, cov=C0)
logp1 = multivariate_normal.logpdf(X, mean=m1, cov=C1)
llr_true = logp1 - logp0

# --- (B) Naive Bayes model (identity covariance) ---
d0 = np.sum((X - m0) ** 2, axis=1)
d1 = np.sum((X - m1) ** 2, axis=1)
llr_nb = 0.5 * (d0 - d1)

# --- Sweep gamma ---
gammas = np.logspace(-6, 6, 800)
def roc_curve(llr):
    PD, PFA, Pe = [], [], []
    for lg in np.log(gammas):
        yhat = (llr > lg).astype(int)
        tp = np.sum((yhat == 1) & (y == 1))
        fp = np.sum((yhat == 1) & (y == 0))
        pd, pfa = tp / N1, fp / N0
        PD.append(pd)
        PFA.append(pfa)
        Pe.append(pfa * p0 + (1 - pd) * p1)
    return np.array(PFA), np.array(PD), np.array(Pe)

PFA_true, PD_true, Pe_true = roc_curve(llr_true)
PFA_nb, PD_nb, Pe_nb = roc_curve(llr_nb)

# --- Thresholds ---
gamma_ideal = p0 / p1  # 1.8571
imin_true = Pe_true.argmin()
imin_nb = Pe_nb.argmin()

# --- Extract values ---
gamma_practical_true = gammas[imin_true]
gamma_practical_nb = gammas[imin_nb]
i_ideal_true = np.argmin(np.abs(gammas - gamma_ideal))
i_ideal_nb = np.argmin(np.abs(gammas - gamma_ideal))

# --- True-pdf ROC markers ---
pfa_ideal_true, pd_ideal_true, pe_ideal_true = (
    PFA_true[i_ideal_true],
    PD_true[i_ideal_true],
    Pe_true[i_ideal_true],
)
pfa_prac_true, pd_prac_true, pe_prac_true = (
    PFA_true[imin_true],
    PD_true[imin_true],
    Pe_true[imin_true],
)

# --- Naive Bayes ROC markers ---
pfa_ideal_nb, pd_ideal_nb, pe_ideal_nb = (
    PFA_nb[i_ideal_nb],
    PD_nb[i_ideal_nb],
    Pe_nb[i_ideal_nb],
)
pfa_prac_nb, pd_prac_nb, pe_prac_nb = (
    PFA_nb[imin_nb],
    PD_nb[imin_nb],
    Pe_nb[imin_nb],
)

# --- Plot ROC Comparison ---
plt.figure(figsize=(7, 6))
plt.plot(PFA_true, PD_true, 'b', lw=2, label='True Model ERM')
plt.plot(PFA_nb, PD_nb, 'r', lw=2, label='Naive Bayes (Identity Cov)')

plt.plot([0, 1], [0, 1], 'k--', lw=1, label='Chance')

# True model markers
plt.scatter(pfa_ideal_true, pd_ideal_true, s=100, edgecolors='orange', facecolors='none', lw=2)
plt.text(pfa_ideal_true+0.01, pd_ideal_true-0.04, f"Ideal γ={gamma_ideal:.2f}", color='orange', fontsize=9)
plt.scatter(pfa_prac_true, pd_prac_true, s=100, edgecolors='blue', facecolors='none', lw=2)
plt.text(pfa_prac_true+0.01, pd_prac_true+0.02, f"Emp γ*={gamma_practical_true:.2f}", color='blue', fontsize=9)

# Naive Bayes markers
plt.scatter(pfa_ideal_nb, pd_ideal_nb, s=100, edgecolors='gold', facecolors='none', lw=2)
plt.text(pfa_ideal_nb+0.01, pd_ideal_nb-0.04, f"NB Ideal γ={gamma_ideal:.2f}", color='gold', fontsize=9)
plt.scatter(pfa_prac_nb, pd_prac_nb, s=100, edgecolors='red', facecolors='none', lw=2)
plt.text(pfa_prac_nb+0.01, pd_prac_nb+0.02, f"NB Emp γ*={gamma_practical_nb:.2f}", color='red', fontsize=9)

plt.xlabel("False Alarm  $P(D=1|L=0)$")
plt.ylabel("Detection  $P(D=1|L=1)$")
plt.title("ROC Comparison — True Model vs Naive Bayes\nwith Ideal & Empirical γ Points")
plt.legend(loc="lower right", fontsize=9)
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()

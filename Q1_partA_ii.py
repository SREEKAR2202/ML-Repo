import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal

# loading dataset
N  = 10000
p0, p1 = 0.65, 0.35
rng = np.random.default_rng(42)

# parameters
m0 = np.array([-0.5, -0.5, -0.5])
C0 = np.array([[ 1.0, -0.5,  0.3],
               [-0.5,  1.0, -0.5],
               [ 0.3, -0.5,  1.0]])
m1 = np.array([1.0, 1.0, 1.0])
C1 = np.array([[ 1.0,  0.3, -0.2],
               [ 0.3,  1.0,  0.3],
               [-0.2,  0.3,  1.0]])

# Labels
y = (rng.random(N) >= p0).astype(int)
N0, N1 = np.sum(y == 0), np.sum(y == 1)

# Samples
X = np.zeros((N, 3))
X[y == 0] = rng.multivariate_normal(m0, C0, size=N0)
X[y == 1] = rng.multivariate_normal(m1, C1, size=N1)

# ======================================================
# Likelihood ratio test implementation
# ======================================================
logp0 = multivariate_normal.logpdf(X, mean=m0, cov=C0)
logp1 = multivariate_normal.logpdf(X, mean=m1, cov=C1)
llr   = logp1 - logp0  # log-likelihood ratio

# (threshold) values
gammas = np.logspace(-4, 4, 400)
log_gammas = np.log(gammas)

P_D, P_FA, Pe = [], [], []

for lg in log_gammas:
    y_pred = (llr > lg).astype(int)
    tp = np.sum((y_pred == 1) & (y == 1))   # True positives
    fp = np.sum((y_pred == 1) & (y == 0))   # False positives
    pd = tp / N1 if N1 > 0 else 0
    pfa = fp / N0 if N0 > 0 else 0
    P_D.append(pd)
    P_FA.append(pfa)
    Pe.append(pfa * p0 + (1 - pd) * p1)

P_D, P_FA, Pe = np.array(P_D), np.array(P_FA), np.array(Pe)

# key points
i_min = np.argmin(Pe)
gamma_star = gammas[i_min]
Pe_min = Pe[i_min]

bayes_gamma = p0 / p1
i_bayes = np.abs(gammas - bayes_gamma).argmin()

# Plot ROC
plt.figure(figsize=(6,6))
plt.plot(P_FA, P_D, label='ROC (ERM Classifier)', linewidth=2)
plt.plot([0,1], [0,1], 'k--', label='Classifier (Random)')
plt.xlabel('False Alarm  P(D=1 | L=0)')
plt.ylabel('Detection   P(D=1 | L=1)')
plt.title('ROC Curve — Minimum Expected Risk Classifier')
plt.legend(loc='lower right')
plt.grid(True, alpha=0.2)
plt.tight_layout()
plt.show()

# partA_wine_gaussian_map.py
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import multivariate_normal
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# ---------- PATH ----------
WINE_CSV = Path(r"C:\Users\keert\Downloads\wine_quality\winequality-white.csv")

# ---------- helpers ----------
def choose_lambda_from_hint(cov_stack, alpha=0.03):
    Cavg = cov_stack.mean(axis=0)
    s = np.linalg.svd(Cavg, compute_uv=False)
    rank = (s > 1e-10).sum()
    return float(alpha) * float(np.trace(Cavg)) / max(rank, 1)

def confusion_P_D_given_L(y_true, y_pred, K):
    C = np.zeros((K, K), float)
    for j in range(K):
        m = (y_true == j)
        if m.any():
            C[:, j] = np.bincount(y_pred[m], minlength=K) / m.sum()
    return C

def plot_confusion(conf, title, class_names):
    plt.figure(figsize=(6.4, 5.6))
    plt.imshow(conf, interpolation="nearest")
    plt.title(title); plt.colorbar()
    plt.xlabel("True class  j"); plt.ylabel("Decision  i")
    plt.xticks(range(len(class_names)), class_names, rotation=45, ha="right")
    plt.yticks(range(len(class_names)), class_names)
    plt.tight_layout(); plt.show()

def pca2_scatter(X, y, title, class_names):
    # simple 2D PCA using SVD to avoid external deps
    Xc = X - X.mean(axis=0, keepdims=True)
    U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    X2 = U[:, :2] * S[:2]
    plt.figure(figsize=(6.8, 5.6))
    for k in range(len(class_names)):
        plt.scatter(X2[y==k,0], X2[y==k,1], s=10, alpha=0.6, label=class_names[k])
    plt.title(title + " — PCA(2D)")
    plt.xlabel("PC1"); plt.ylabel("PC2"); plt.grid(alpha=0.3)
    plt.legend(ncol=3, fontsize=8); plt.tight_layout(); plt.show()

# ---------- load wine (white only) ----------
df = pd.read_csv(WINE_CSV, sep=';')
X = df.drop(columns=['quality']).to_numpy(float)
y_raw = df['quality'].to_numpy(int)       # labels like {3..9}
classes = np.sort(np.unique(y_raw))
map2zero = {c:i for i,c in enumerate(classes)}   # 0..K-1
y = np.array([map2zero[c] for c in y_raw], int)
class_names = [f"q={c}" for c in classes]

print(f"Wine (white) loaded: {X.shape[0]} samples × {X.shape[1]} features, K={len(classes)}")

# ---------- z-score, estimate Gaussians, regularize, MAP ----------
scaler = StandardScaler()
Xs = scaler.fit_transform(X)

K = len(classes)
N, d = Xs.shape
priors = np.array([(y == k).mean() for k in range(K)])

means = np.zeros((K, d)); covs = np.zeros((K, d, d))
for k in range(K):
    Xk = Xs[y==k]
    means[k] = Xk.mean(axis=0)
    covs[k]  = np.cov(Xk, rowvar=False)

lam = choose_lambda_from_hint(covs, alpha=0.03)   # tweak 0.02–0.05 if needed
covs_reg = covs + lam * np.eye(d)

mvns = [multivariate_normal(mean=means[k], cov=covs_reg[k], allow_singular=False)
        for k in range(K)]

log_posts = np.column_stack([mvns[k].logpdf(Xs) + np.log(priors[k] + 1e-12)
                             for k in range(K)])
yhat = np.argmax(log_posts, axis=1)

# ---------- metrics ----------
acc = (yhat == y).mean()
Pe  = 1.0 - acc   # minimum expected risk under 0–1 loss
conf = confusion_P_D_given_L(y, yhat, K)

np.set_printoptions(precision=3, suppress=True)
print(f"Accuracy: {acc*100:.2f}% | Probability of error Pe: {Pe:.4f} | λ={lam:.6f}")
print("Confusion P(D=i | L=j):\n", conf)
print("Column sums:", conf.sum(axis=0))

# ---------- visuals ----------
# plot_confusion(conf, "Wine (white) — Gaussian MAP, regularized", class_names)
pca2_scatter(X, y, "Wine (white)-Data Distribution",class_names )

# partB_har_gaussian_map.py
import numpy as np
from pathlib import Path
from scipy.stats import multivariate_normal
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import zipfile
import matplotlib.pyplot as plt

# ---------- PATH ----------
HAR_ZIP = Path(r"C:\Users\keert\Downloads\human_activity\UCI HAR Dataset.zip")  

# ---------- helpers (same as Part A, re-copied for stand-alone script) ----------
def choose_lambda_from_hint(cov_stack, alpha=0.02):
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
    plt.xticks(range(len(class_names)), class_names, rotation=30, ha="right")
    plt.yticks(range(len(class_names)), class_names)
    plt.tight_layout(); plt.show()

def pca2_scatter(X, y, title, class_names):
    plt.figure(figsize=(6.8, 5.6))
    for k in range(len(class_names)):
        plt.scatter(X[y==k,0], X[y==k,1], s=8, alpha=0.6, label=class_names[k])
    plt.title(title + " — PCA(2D)")
    plt.xlabel("PC1"); plt.ylabel("PC2"); plt.grid(alpha=0.3)
    plt.legend(ncol=3, fontsize=8); plt.tight_layout(); plt.show()

# ---------- load HAR from zip ----------
with zipfile.ZipFile(HAR_ZIP, 'r') as zf:
    names = zf.namelist()
    base = ""
    for n in names:
        if n.endswith("train/X_train.txt"):
            base = n.split("train/X_train.txt")[0] + "train/"
            break
    if not base:
        raise FileNotFoundError("train/X_train.txt not found inside HAR zip.")

    def read_txt(name):
        with zf.open(name) as f:
            return np.loadtxt(f)

    Xtr = read_txt(base + "X_train.txt")
    ytr = read_txt(base + "y_train.txt").astype(int).ravel()
    Xte = read_txt(base.replace("train/", "test/") + "X_test.txt")
    yte = read_txt(base.replace("train/", "test/") + "y_test.txt").astype(int).ravel()

X = np.vstack([Xtr, Xte])
y = np.concatenate([ytr, yte]) - 1    # labels 0..5
class_names = ["WALK","UP","DOWN","SIT","STAND","LAY"]
print(f"HAR loaded: {X.shape[0]} samples × {X.shape[1]} features, K={len(class_names)}")

# ---------- PCA (recommended for HAR) ----------
PCA_DIM = 70  # try 50–100; tune if needed
pca = PCA(n_components=PCA_DIM, svd_solver="full")
X_pca = pca.fit_transform(X)

# ---------- z-score, estimate Gaussians, regularize, MAP ----------
scaler = StandardScaler()
Xs = scaler.fit_transform(X_pca)

K = len(class_names)
N, d = Xs.shape
priors = np.array([(y == k).mean() for k in range(K)])

means = np.zeros((K, d)); covs = np.zeros((K, d, d))
for k in range(K):
    Xk = Xs[y==k]
    means[k] = Xk.mean(axis=0)
    covs[k]  = np.cov(Xk, rowvar=False)

lam = choose_lambda_from_hint(covs, alpha=0.02)   # smaller α for stability
covs_reg = covs + lam * np.eye(d)

mvns = [multivariate_normal(mean=means[k], cov=covs_reg[k], allow_singular=False)
        for k in range(K)]

log_posts = np.column_stack([mvns[k].logpdf(Xs) + np.log(priors[k] + 1e-12)
                             for k in range(K)])
yhat = np.argmax(log_posts, axis=1)

# ---------- metrics ----------
acc = (yhat == y).mean()
Pe  = 1.0 - acc
conf = confusion_P_D_given_L(y, yhat, K)

np.set_printoptions(precision=3, suppress=True)
print(f"Accuracy: {acc*100:.2f}% | Probability of error Pe: {Pe:.4f} | λ={lam:.6f}")
print("Confusion P(D=i | L=j):\n", conf)
print("Column sums:", conf.sum(axis=0))

# ---------- visuals ----------
# plot_confusion(conf, "HAR — Gaussian MAP (PCA + regularized)", class_names)
# show PCA coordinates used (nice for intuition)
pca2_scatter(X_pca[:, :2], y, "HAR", class_names)

import numpy as np
import matplotlib.pyplot as plt

# --- config ---
N  = 10000
p0, p1 = 0.65, 0.35
# rng = np.random.default_rng(42)  # fixed seed for reusability

# Labels: 0 with prob p0, 1 with prob p1
u = (rng.random(N) >= p0).astype(int)  # matches MATLAB logic
N0 = np.sum(u == 0)
N1 = np.sum(u == 1)

# Class parameters
mu0 = np.array([-0.5, -0.5, -0.5])
Sigma0 = np.array([[ 1.0, -0.5,  0.3],
                   [-0.5,  1.0, -0.5],
                   [ 0.3, -0.5,  1.0]])

mu1 = np.array([1.0, 1.0, 1.0])
Sigma1 = np.array([[ 1.0,  0.3, -0.2],
                   [ 0.3,  1.0,  0.3],
                   [-0.2,  0.3,  1.0]])

# Draw samples
r0 = rng.multivariate_normal(mu0, Sigma0, size=N0)
r1 = rng.multivariate_normal(mu1, Sigma1, size=N1)

# Assemble in original order (so X[i] matches label u[i])
X = np.empty((N, 3))
X[u == 0] = r0
X[u == 1] = r1
y = u  # labels in {0,1}

'''
# Save once; reuse this file for all later parts
np.savez('q1_dataset.npz',
         X=X, y=y, mu0=mu0, Sigma0=Sigma0, mu1=mu1, Sigma1=Sigma1, p0=p0, p1=p1)
'''

# 3D plot
fig = plt.figure(figsize=(6,6))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(X[y==0,0], X[y==0,1], X[y==0,2], s=4, marker='.', label='Class 0')
ax.scatter(X[y==1,0], X[y==1,1], X[y==1,2], s=4, marker='.', label='Class 1')
ax.set_xlabel('x1'); ax.set_ylabel('x2'); ax.set_zlabel('x3'); ax.set_title('Q1 data')
ax.legend(loc='best')
# make axes equal-ish
mins = X.min(axis=0); maxs = X.max(axis=0)
ax.set_box_aspect((maxs - mins))
plt.tight_layout(); plt.show()

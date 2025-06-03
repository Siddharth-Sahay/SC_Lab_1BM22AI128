import numpy as np

# AND gate dataset
X = np.array([[0,0], [0,1], [1,0], [1,1]])
y = np.array([-1, -1, -1, 1])  # AND logic (-1 for 0, 1 for 1)

# Add bias term
Xb = np.hstack((X, np.ones((4,1))))

# Hebb's learning rule: w += x * y
w = np.zeros(3)
for i in range(len(Xb)):
    w += Xb[i] * y[i]

# Test
print("Hebb's Rule Weights:", w)
for i in range(len(Xb)):
    out = np.sign(np.dot(Xb[i], w))
    print(f"Input: {X[i]}, Output: {out}")

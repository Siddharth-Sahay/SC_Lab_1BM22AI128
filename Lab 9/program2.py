import numpy as np

# AND gate dataset
X = np.array([[0,0], [0,1], [1,0], [1,1]])
y = np.array([0, 0, 0, 1])  # Output for AND

# Add bias term
Xb = np.hstack((X, np.ones((4,1))))

# Delta rule training
w = np.zeros(3)
lr = 0.1
for epoch in range(10):
    for i in range(len(Xb)):
        out = np.dot(Xb[i], w)
        error = y[i] - out
        w += lr * error * Xb[i]

# Test
print("\nDelta Rule Weights:", w)
for i in range(len(Xb)):
    out = np.dot(Xb[i], w)
    print(f"Input: {X[i]}, Output: {round(out)}")

import numpy as np

def f(x): return x**2  # Objective function

wolves = np.random.uniform(-10, 10, 5)  # 5 grey wolves (1D)
alpha, beta, delta = None, None, None

for iter in range(20):
    sorted_idx = np.argsort(f(wolves))
    alpha, beta, delta = wolves[sorted_idx[:3]]

    a = 2 - iter * (2 / 20)  # linearly decrease from 2 to 0
    for i in range(len(wolves)):
        for leader in [alpha, beta, delta]:
            r1, r2 = np.random.rand(), np.random.rand()
            A = a * (2*r1 - 1)
            C = 2 * r2
            D = abs(C * leader - wolves[i])
            X = leader - A * D
            wolves[i] = (wolves[i] + X) / 2  # average with current position

    best = alpha
    print(f"Iter {iter+1}: Best = {best:.4f}, f(Best) = {f(best):.4f}")

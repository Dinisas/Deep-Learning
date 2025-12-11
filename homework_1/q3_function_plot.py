import numpy as np
import matplotlib.pyplot as plt

# ---------------------------
# Softmax
# ---------------------------
def softmax_p2(t):
    return np.exp(t) / (1 + np.exp(t))

def softmax_p2_prime(t):
    s = softmax_p2(t)
    return s * (1 - s)


# ---------------------------
# Sparsemax
# ---------------------------
def sparsemax_p2(t):
    t = np.array(t)
    p = np.zeros_like(t)

    # Regions:
    # t ≤ -1 → 0
    mask_mid = (t > -1) & (t < 1)
    p[mask_mid] = (t[mask_mid] + 1) / 2
    p[t >= 1] = 1
    return p

def sparsemax_p2_prime(t):
    t = np.array(t)
    dp = np.zeros_like(t)
    dp[(t > -1) & (t < 1)] = 0.5
    return dp


# ---------------------------
# Relumax_b
# ---------------------------
def relumax_p2(t, b):
    t = np.array(t)
    p = np.zeros_like(t)

    # Case t <= -b → 0
    mask1 = t <= -b
    # Case -b < t < 0
    mask2 = (t > -b) & (t < 0)
    # Case 0 <= t < b
    mask3 = (t >= 0) & (t < b)
    # Case t >= b → 1
    mask4 = t >= b

    p[mask2] = (t[mask2] + b) / (t[mask2] + 2*b)
    p[mask3] = b / (2*b - t[mask3])
    p[mask4] = 1
    return p

def relumax_p2_prime(t, b):
    t = np.array(t)
    dp = np.zeros_like(t)

    mask2 = (t > -b) & (t < 0)
    dp[mask2] = b / (t[mask2] + 2*b)**2

    mask3 = (t > 0) & (t < b)
    dp[mask3] = b / (2*b - t[mask3])**2
    return dp


# ---------------------------
# Plotting
# ---------------------------
t = np.linspace(-3, 3, 400)
b = 1.0

plt.figure(figsize=(14, 10))

# ---- p2(t)
plt.subplot(2, 1, 1)
plt.plot(t, softmax_p2(t), label="Softmax $p_2(t)$")
plt.plot(t, sparsemax_p2(t), label="Sparsemax $p_2(t)$")
plt.plot(t, relumax_p2(t, b), label=f"Relumax$_b$ $p_2(t)$ (b={b})")
plt.title("Activations $p_2(t)$")
plt.grid(True)
plt.legend()

# ---- p2'(t)
plt.subplot(2, 1, 2)
plt.plot(t, softmax_p2_prime(t), label="Softmax $p_2'(t)$")
plt.plot(t, sparsemax_p2_prime(t), label="Sparsemax $p_2'(t)$")
plt.plot(t, relumax_p2_prime(t, b), label=f"Relumax$_b$ $p_2'(t)$ (b={b})")
plt.title("Derivatives $p_2'(t)$")
plt.grid(True)
plt.legend()

plt.show()

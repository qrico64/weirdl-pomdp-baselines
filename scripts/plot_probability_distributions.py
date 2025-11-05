import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm, uniform

# Parameters
mean = 0
std = 0.4
low, high = -0.4, 0.4

# Range for plotting
x = np.linspace(-1.5, 1.5, 500)

# Compute PDFs
gaussian_pdf = norm.pdf(x, loc=mean, scale=std)
uniform_pdf = uniform.pdf(x, loc=low, scale=(high - low))

# Plot
plt.figure(figsize=(8, 5))
plt.plot(x, gaussian_pdf, label=r"$\mathcal{N}(0, 0.4^2)$", linewidth=2)
plt.plot(x, uniform_pdf, label=r"$U(-0.4, 0.4)$", linewidth=2)

# Formatting
plt.title("Gaussian vs. Uniform", fontsize=14)
plt.xlabel("x", fontsize=12)
plt.ylabel("probability density", fontsize=12)
plt.legend(fontsize=12)
plt.grid(True, linestyle='--', alpha=0.6)

# Show
plt.tight_layout()
out = "viz/nov5/nov5_normal_vs_uniform.png"
plt.savefig(out)
print(out + "\n")

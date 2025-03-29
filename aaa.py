import matplotlib.pyplot as plt

from cgtnnlib.NoiseGenerator import stable_noise_func

def plot_histogram_pdf(data, title="Histogram PDF", bins=10):
    """Plots a histogram of the data, normalized to approximate the PDF."""
    plt.hist(data, bins=bins, density=True, alpha=0.6, color='skyblue')  # density=True normalizes
    plt.title(title)
    plt.xlabel("Value")
    plt.ylabel("Probability Density")
    plt.show()

plot_histogram_pdf(stable_noise_func(alpha=2, beta=0, size=100))
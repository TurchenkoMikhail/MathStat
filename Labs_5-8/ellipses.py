import scipy.stats as sps
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.transforms as transforms
import statistics as stats
from matplotlib.patches import Ellipse

sizes = [20, 60, 100]
rhos = [0, 0.5, 0.9]
repetitions = 1000

def normal(size, rho):
    return sps.multivariate_normal.rvs([0, 0], [[1.0, rho], [rho, 1.0]], size=size)

def build_ellipse(x, y, ax):
    cov = np.cov(x, y)
    pearson = cov[0, 1] / np.sqrt(cov[0, 0] * cov[1, 1])

    rad_x = np.sqrt(1 + pearson)
    rad_y = np.sqrt(1 - pearson)
    ellipse = Ellipse((0, 0), width=rad_x * 2, height=rad_y * 2, facecolor='none', edgecolor='red')

    scale_x = np.sqrt(cov[0, 0]) * 3
    mean_x = np.mean(x)

    scale_y = np.sqrt(cov[1, 1]) * 3
    mean_y = np.mean(y)

    transform = transforms.Affine2D().rotate_deg(45).scale(scale_x, scale_y).translate(mean_x, mean_y)
    ellipse.set_transform(transform + ax.transData)
    return ax.add_patch(ellipse)

def show_ellipse(size):
    fig, ax = plt.subplots(1, 3)
    titles = ["rho = 0", "rho = 0.5", "rho = 0.9"]

    for i in range(len(rhos)):
        sample = normal(size, rhos[i])
        x, y = sample[:, 0], sample[:, 1]
        build_ellipse(x, y, ax[i])
        ax[i].grid()
        ax[i].scatter(x, y, s=5)
        ax[i].set_title(titles[i])
    plt.suptitle("n = " + str(size))
    plt.show()

for size in sizes:
    show_ellipse(size)
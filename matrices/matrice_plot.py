import numpy as np
import matplotlib.pyplot as plt

matrice = np.loadtxt('DBSCAN_dim8_auc_3_7_versionMID')


lignes, colonnes = matrice.shape
x = np.linspace(2, 10, colonnes)
y = np.linspace(0.5, 12 , lignes)
X, Y = np.meshgrid(x, y)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

surface = ax.plot_surface(X, Y, matrice, cmap='inferno', edgecolor='none')
fig.colorbar(surface, ax=ax, shrink=0.5, aspect=5)

ax.set_xlabel('min_samples')
ax.set_ylabel('eps')
ax.set_zlabel('AUC')

plt.show()

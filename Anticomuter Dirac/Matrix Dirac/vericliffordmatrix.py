import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Matrizes de Pauli
sigma_x = np.array([[0, 1], [1, 0]], dtype=complex)
sigma_y = np.array([[0, -1j], [1j, 0]], dtype=complex)
sigma_z = np.array([[1, 0], [0, -1]], dtype=complex)

# Identidade e zero 2x2
I2 = np.eye(2, dtype=complex)
Z2 = np.zeros((2, 2), dtype=complex)

# Matrizes de Dirac (representação padrão)
gamma_0 = np.block([[I2, Z2], [Z2, -I2]])
gamma_1 = np.block([[Z2, sigma_x], [-sigma_x, Z2]])
gamma_2 = np.block([[Z2, sigma_y], [-sigma_y, Z2]])
gamma_3 = np.block([[Z2, sigma_z], [-sigma_z, Z2]])

# Lista com todas as gammas
gamma = [gamma_0, gamma_1, gamma_2, gamma_3]

# Inicializa a matriz dos anticomutadores
anticom_values = np.zeros((4, 4), dtype=float)

# Calcula os anticomutadores normalizados
for mu in range(4):
    for nu in range(4):
        A = gamma[mu] @ gamma[nu] + gamma[nu] @ gamma[mu]
        value = np.trace(A).real / 4  # Normaliza pelo tamanho
        anticom_values[mu, nu] = value

# Coordenadas para o gráfico
x, y = np.meshgrid(np.arange(4), np.arange(4))
z = np.zeros_like(x)
dz = anticom_values

# Cria o gráfico 3D
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')

# Mapa de cores
colors = plt.cm.coolwarm((dz.flatten() - dz.min()) / (dz.max() - dz.min()))

# Barras 3D
ax.bar3d(x.flatten(), y.flatten(), z.flatten(), 1, 1, dz.flatten(),
         shade=True, color=colors)

# Rótulos
ax.set_xticks(range(4))
ax.set_yticks(range(4))
ax.set_xticklabels([r'$\gamma^0$', r'$\gamma^1$', r'$\gamma^2$', r'$\gamma^3$'])
ax.set_yticklabels([r'$\gamma^0$', r'$\gamma^1$', r'$\gamma^2$', r'$\gamma^3$'])
ax.set_zlabel('Valor do Anticomutador')
ax.set_title('Anticomutadores {γ^μ, γ^ν} - Gráfico 3D')

plt.tight_layout()
plt.show()

import numpy as np
from scipy.integrate import simps
from scipy.constants import c
import cmath
import matplotlib.pyplot as plt
import seaborn as sb
#%matplotlib widget

class PolarizationState:
    def __init__(self, horizontal, vertical):
        self.horizontal = horizontal
        self.vertical = vertical

class Pointeur:
    def __init__(self, zeta, decal, function):
        self.zeta = zeta
        self.decal = decal
        self.function = function

# L'écart d'un laser pulsé: dans le dégrée temporelle
wavelength = 640e-9  # nm
largeur = 10e-9  # ns
MIN_E = -largeur
MAX_E = largeur
N = 1000
angle_pre_selection = np.pi / 3
angle_post_selection = np.pi/4
ellipse_pre = 0
phi_x = 0
phi_y = 0
frequence = c / wavelength
freq_angulaire = 2 * np.pi * frequence
time = np.linspace(MIN_E, MAX_E, N)

decal = np.linspace(0, 3, N)

def pointeur(t, sigma, wavelength, z):
    k = (2 * np.pi) / wavelength
    w = 2 * np.pi * c / wavelength
    E_0 = (1 / np.sqrt(np.sqrt(2 * np.pi) * sigma))
    return E_0 * np.exp(-np.square(t / (2 * sigma))) * np.exp(1j * (k * z - w * t))

pointeur_vals = pointeur(time, largeur, wavelength, 0)

a = np.cos(angle_pre_selection) * (np.cos(phi_x) + 1j * np.sin(phi_x))
b = np.sin(angle_pre_selection) * (np.cos(phi_y) + 1j * np.sin(phi_y))
u = np.cos(angle_post_selection) * (np.cos(phi_x) + 1j * np.sin(phi_x))
v = np.sin(angle_post_selection) * (np.cos(phi_x) + 1j * np.sin(phi_x))

polarisation = PolarizationState(a, b)

def interaction_operator(d):
    return np.exp(1j * d*2*np.pi/wavelength)

U = interaction_operator(decal)

post_polar = PolarizationState(u, v)

# Calculate intensity profile
intensity_profile = np.zeros([N,N])

for i in range(N):
    for j in range(N):
        intensity_H = np.abs(
            np.conjugate(post_polar.horizontal) * polarisation.horizontal * pointeur_vals[j]
        ) ** 2
        intensity_V = np.abs(
            np.conjugate(post_polar.vertical) * polarisation.vertical * pointeur_vals[j]
        ) ** 2
        intensity_profile[i, j] = (1 / np.sqrt(2)) * (intensity_H + intensity_V) * (1 + interaction_operator(decal[i]))

# Create meshgrid for plotting
time_mesh, distance_mesh = np.meshgrid(time, decal)

# Create 3D plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(time_mesh, distance_mesh, intensity_profile.reshape(N, N))

# Set labels and title
ax.set_xlabel('Time')
ax.set_ylabel('Distance')
ax.set_zlabel('Intensity')
ax.set_title('Intensity vs Time and Distance')

# Show the plot
plt.show()

fig = plt.figure()
plt.plot(decal, intensity_profile)
plt.show()
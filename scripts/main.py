import numpy as np
import matplotlib.pyplot as plt
import pyvista as pv

class Asteroid:
    def __init__(self, name, r_a, r_p, a, e, i, LAAN, om, T, v_avg, D, extent, mass, rho, T_rot):
        self.name = name
        self.r_a = r_a
        self.r_p = r_p
        self.a = a
        self.e = e
        self.i = i
        self.LAAN = LAAN
        self.om = om
        self.T = T
        self.v_avg = v_avg
        self.D = D
        self.extent = extent
        self.mass = mass
        self.rho = rho
        self.T_rot = T_rot


### Itokawa asteroid data (25413) ###

## Orbital Data ##

# AU #

r_a = 1.6951
r_p = 0.9533
a = 1.3242

# km #

r_a = 2.5358e8
r_p = 1.4251e8
a = 1.981e8

e = 0.2800889

i = 1.6238 # degrees
i = np.radians(i) # radians

LAAN = 69.077 # degrees
LAAN = np.radians(LAAN) # radians

om = 162.82 # degrees
om = np.radians(om) # radians\

T = 1.517 # years
T = 556.581 # days
T = T * 24 * 3600 # seconds

v_avg = 25.37 # km/s

## Physical Characteristics ##

D = 0.33 # km
extent = [0.535, 0.294, 0.209] # km
mass = 3.147e10 # kg
rho = 1.673 # g/cm^3
rho = rho * 1e3 # kg/m^3
T_rot = 12.1324 # hours
T_rot = T_rot * 3600 # seconds

### end ###

# https://3d-asteroids.space/asteroids/25143-Itokawa #


Itokawa = Asteroid('Itokawa', r_a, r_p, a, e, i, LAAN, om, T, v_avg, D, extent, mass, rho, T_rot)

mesh = pv.read('data/itokawa_50k.ply')
mesh.plot()
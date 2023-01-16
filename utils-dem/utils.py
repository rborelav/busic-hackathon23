import taichi as ti
import math
import os

# NUMERICAL STABILITY
def pwave_timestep(radius, youngs_mod, density, safety_coeff=0.3):
    return safety_coeff*radius*math.sqrt(density/youngs_mod)

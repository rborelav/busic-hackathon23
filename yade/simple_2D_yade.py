# -*- encoding=utf-8 -*-
from yade import plot,pack,timing
import time, sys, os, copy
sys.path.append(os.getcwd())
import random as r

# profiling activated
O.timingEnabled=True

# define material properties (frictionless material)
sphere_mat = O.materials.append(FrictMat(density=100.,young=1e4,poisson=1,frictionAngle=0.,label='sphere_mat'))

# create 2D pack
n = 512000
grid_n = 130
grid_size = 1.0 / grid_n  # Simulation domain of size [0, 1]

grain_r_min = 0.002
grain_r_max = 0.003

assert grain_r_max * 2 < grid_size

sp = []
padding = 0.01
region_width = 1.0 - 2*padding
for i in range(n):
    l = i * grid_size
    sp.append(O.bodies.append(utils.sphere((l % region_width + padding + grid_size * r.random() * 0.1, 0., l // region_width * grid_size + 0.01), r.random() * (grain_r_max - grain_r_min) + grain_r_min)))

for b in O.bodies:
    b.state.blockedDOFs = 'yXZ'

print(utils.aabbExtrema())


wall = utils.wall( 0 ,0, sense=0, material=-1)
O.bodies.append(wall)
wall = utils.wall( 1, 0, sense=0, material=-1)
O.bodies.append(wall)
wall = utils.wall( 0, 2, sense=0, material=-1)
O.bodies.append(wall)
wall = utils.wall( 32, 2, sense=0, material=-1)
O.bodies.append(wall)

O.dt = 0.0001

O.engines=[
	ForceResetter(),
	InsertionSortCollider([Bo1_Sphere_Aabb(), Bo1_Wall_Aabb()],verletDist=.000001*grain_r_min),
	InteractionLoop(
		[Ig2_Sphere_Sphere_ScGeom(),
		Ig2_Wall_Sphere_ScGeom()],
		[Ip2_FrictMat_FrictMat_FrictPhys()],
		[Law2_ScGeom_FrictPhys_CundallStrack()],
	),
	NewtonIntegrator(gravity = (0.,0.,-9.81), damping = 0),
	]
	
O.run(10000)
O.wait()
timing.stats()



"""
This is a modified version of the basic DEM simulation implemented in Taichi.
The modifications include:
- 16000 particles instead of ~8000
- Timer module used to measure the performance of each sim component
- Utils module implements P-Wave timestep calculation used to calculate an appropriate time step
- Contacts are made size invariant
- code is extended to 3D
"""
# activate python virtual environment: .\taichi-env\Scripts\activate
import taichi as ti
import math
import os, sys
import time
#from cupyx.profiler import benchmark

# add folders with python modules
cwd = r''+os.getcwd()
sys.path.append(cwd+r'/utils-dem/')
sys.path.append(cwd+r'/profiling/')

# add user defined modules
import utils
from timer_implementation import Timer #--> ADDED

func_timer = Timer() #--> ADDED
ti.init(arch=ti.cpu)
#vec = ti.math.vec2
SAVE_FRAMES = False

window_size = 640  # Number of pixels of the window
n = 8000 #8192  # Number of grains

density = 2700.0 # quartz density
youngs_mod = 1e9 # quartz youngs mod
rest_coeff = 0.001
tangential_ratio = 0.5
gravity = -9.81
substeps = 1
vec3 = ti.types.vector(3, ti.f32)


# ==========================================================
# PARTICLES
# ==========================================================
@ti.dataclass
#Initializing the features associated with each particle
class Grain:
    p: vec3      # Position
    rot: vec3    # Rotation
    m: ti.f32    # Mass
    r: ti.f32    # Radius
    v: vec3      # Velocity
    angv: vec3   # Angular velocity
    angaccel: vec3
    a: vec3      # Acceleration
    t: vec3      # Torque
    f: vec3      # Force
    mu: ti.f32  # Friction coefficient

gf = Grain.field(shape=(n, ))

# Defining the grid size, number of grids, and particle size range.
# The grid spans the sample domain. Each grid accomodates atleast 2 particles
grid_n = 130
grid_size = 1.0 / grid_n  # Simulation domain of size [0, 1]
print(f"Grid size: {grid_n}x{grid_n}x{grid_n}")
grain_r_min = 0.002
grain_r_max = 0.002
assert grain_r_max * 2 < grid_size

#computing the timestep
dt = utils.pwave_timestep(grain_r_min, youngs_mod, density, 1) # Larger dt might lead to unstable results.
print('Timestep is: '+ str(dt))


# ==========================================================
# PACKING GENERATION
# ==========================================================

#function to initialize all features of the Grain class
@ti.kernel
def init():
    for i in gf:
        # Spread grains in a restricted area.
        l = i * grid_size
        padding = 0.01
        """
        region_width = 1.0 - padding * 2
        pos = vec(l % region_width + padding + grid_size * ti.random() * 0.2,
                  l // region_width * grid_size + 0.3)
        """
        region_width_depth = 1.0 - 2*padding
        pos = vec3(l % region_width_depth + padding + grid_size * ti.random() * 0.1,
                  l // region_width_depth * grid_size + 0.01,
                  l % region_width_depth + padding + grid_size * ti.random() * 0.1)
        gf[i].p = pos
        gf[i].r = ti.random() * (grain_r_max - grain_r_min) + grain_r_min
        gf[i].m = density * math.pi * gf[i].r**2


# ==========================================================
# INTERACTIONS
# ==========================================================

@ti.dataclass
#Initializing the features associated with each interaction
class Interaction:
    # contact point
    cp: vec3            # Contact point
    # normal force vars
    normal: vec3        # Normal direction
    un: ti.f32          # Particle overlap
    kn: ti.f32          # Normal stiffness
    fn: vec3            # Normal force
    # tangential force vars
    tangent: vec3       # Tangential direction
    ut: ti.f32          # Tangential displacement
    kt: ti.f32          # Tangential stiffness
    ft: vec3            # Tangential force
intf = Interaction.field(shape=(n,n))


# ==========================================================
# MOTION INTEGRATION
# ==========================================================

#function tio update the position, velocity , and acceleration of particles after every iteration or a substep of iterations
@ti.kernel
def update():
    dt2 = 0.5 * dt**2
    dt_half = 0.5 * dt
    # ti.loop_config(block_dim=256)
    for i in gf:
        # Calculate linear motion component
        a = gf[i].f / gf[i].m
        gf[i].v += (gf[i].a + a) * dt_half
        gf[i].p += gf[i].v * dt + a * dt2
        gf[i].a = a
        # Calculate rotational component
        inertia = 2/5 * gf[i].m * (gf[i].r**2)
        angaccel = gf[i].t/inertia
        gf[i].angv += (gf[i].angaccel + angaccel) * dt_half
        gf[i].angaccel = angaccel


# ==========================================================
# BOUNDARY CONDITIONS
# ==========================================================
#function to apply boundary conditions to the sample domain

@ti.kernel
def apply_bc():
    bounce_coef = 0.3  # Velocity damping
    for i in gf:
        x = gf[i].p[0]
        y = gf[i].p[1]
        z = gf[i].p[2]

        if y - gf[i].r < 0:
            gf[i].p[1] = gf[i].r
            gf[i].v[1] *= -bounce_coef

        elif y + gf[i].r > 1.0:
            gf[i].p[1] = 1.0 - gf[i].r
            gf[i].v[1] *= -bounce_coef

        if z - gf[i].r < 0:
            gf[i].p[2] = gf[i].r
            gf[i].v[2] *= -bounce_coef

        elif z + gf[i].r > 1.0:
            gf[i].p[2] = 1.0 - gf[i].r
            gf[i].v[2] *= -bounce_coef

        if x - gf[i].r < 0:
            gf[i].p[0] = gf[i].r
            gf[i].v[0] *= -bounce_coef

        elif x + gf[i].r > 1.0:
            gf[i].p[0] = 1.0 - gf[i].r
            gf[i].v[0] *= -bounce_coef


# ==========================================================
# CONTACT PHYSICS
# ==========================================================
#function used in collision detection between 2 particles to
#update the forces between contacting particles

@ti.func
def resolve(i, j):
    cp = vec3(0.,0.,0.)
    normal = vec3(0.,0.,0.)
    overlap = 0.
    kn = 0.
    fn = vec3(0.,0.,0.)
    tangent = vec3(0.,0.,0.)
    ut = 0.
    kt = 0.
    ft = vec3(0.,0.,0.)
    # compute relative position, distance and overlap
    rel_pos = gf[j].p - gf[i].p
    dist = ti.sqrt(rel_pos[0]**2 + rel_pos[1]**2 + + rel_pos[2]**2)
    overlap = -dist + gf[i].r + gf[j].r  # delta = d - 2 * r
    if overlap > 0:  # in contact
        # Normal force calculation
        normal = rel_pos / dist
        kn = youngs_mod * 2 * (gf[i].r*gf[j].r)/(gf[i].r + gf[j].r)
        f1 = normal * overlap * kn
        # Normal damping force
        M = (gf[i].m * gf[j].m) / (gf[i].m + gf[j].m)
        K = kn
        C = 2.*(1./ti.sqrt(1.+(math.pi/ti.log(rest_coeff))**2)
                  ) * ti.sqrt(K * M)
        V = (gf[j].v - gf[i].v) * normal
        f2 = C * V * normal
        fn = f2 - f1
        # Contact point
        cp = gf[i].p + normal*(gf[i].r - 0.5*overlap)
        # Calculate tangential stiffness
        kt = kn*tangential_ratio
        # Calculate frictional coefficient
        fric_coeff = ti.min(gf[i].mu, gf[j].mu)
        tangent = vec3(0., 0., 0.)
        # Calculate relative velocity (https://gitlab.com/yade-dev/trunk/-/blob/master/pkg/dem/ScGeom.cpp)
        ci_vel = gf[i].v + gf[i].angv.cross(cp - gf[i].p)
        cj_vel = gf[j].v + gf[j].angv.cross(cp - gf[j].p)
        relative_vel = cj_vel - ci_vel
        relative_vel_n = normal.dot(relative_vel) * normal
        relative_vel_t = relative_vel - relative_vel_n
        # Calculate tangential increment
        tangential_inc = relative_vel_t * dt
        # Sliding direction
        if tangential_inc.norm() > 0:
            tangent = tangential_inc/tangential_inc.norm()
        # Calculate frictional force increment
        f3 = tangential_inc * kt
        # Get potential tangential force
        f4 = intf[i, j].ft - f3
        # Calculate max frictional force
        ft_max = fric_coeff*(fn.norm())
        # Calculate the final tagential force
        ft = ti.min(f4.norm(),fn.norm())*tangent
        # Apply force to particle
        gf[i].f += fn - ft
        gf[j].f -= fn - ft
        # Apply torque to particle (https://gitlab.com/yade-dev/trunk/-/blob/master/pkg/dem/ElasticContactLaw.cpp#L111)
        gf[i].t += (gf[i].r - 0.5*overlap)*normal.cross(fn - ft)
        gf[j].t += (gf[j].r - 0.5*overlap)*normal.cross(-fn + ft)
    else:
        cp = vec3(0.,0.,0.)
        normal = vec3(0.,0.,0.)
        overlap = 0
        kn = 0
        fn = vec3(0.,0.,0.)
        tangent = vec3(0.,0.,0.)
        ut = 0
        kt = 0
        ft = vec3(0.,0.,0.)
    # Store data in the interactions field
    intf[i, j].cp = cp
    intf[i, j].normal = normal
    intf[i, j].un = overlap
    intf[i, j].kn = kn
    intf[i, j].fn = fn
    intf[i, j].tangent = tangent
    intf[i, j].ut = ut
    intf[i, j].kt = kt
    intf[i, j].ft = ft


# ==========================================================
# COLLISION DETECTION
# ==========================================================

list_head = ti.field(dtype=ti.i32, shape=grid_n * grid_n * grid_n)
list_cur = ti.field(dtype=ti.i32, shape=grid_n * grid_n * grid_n)
list_tail = ti.field(dtype=ti.i32, shape=grid_n * grid_n * grid_n)

grain_count = ti.field(dtype=ti.i32,
                       shape=(grid_n, grid_n, grid_n),
                       name="grain_count")
column_sum = ti.field(dtype=ti.i32, shape=grid_n, name="column_sum")
prefix_sum = ti.field(dtype=ti.i32, shape=(grid_n, grid_n, grid_n), name="prefix_sum")
particle_id = ti.field(dtype=ti.i32, shape=n, name="particle_id")


#function to detect if neighbouring particles are in contact with each other and resolve forces between them
@ti.kernel
def contact(gf: ti.template()):
    '''
    Handle the collision between grains.
    '''
    # Apply gravity to all particles
    for i in gf:
        gf[i].f = vec3(0., 0, gravity * gf[i].m)

    #count grains
    grain_count.fill(0)
    for i in range(n):
        grid_idx = ti.floor(gf[i].p * grid_n, int)#cell index for each particle
        grain_count[grid_idx] += 1#incrementing grain count for that cell by 1

    # #number of particles in a column of the sample grid
    # for i in range(grid_n):
    #     sum = 0
    #     for j in range(grid_n):
    #         sum += grain_count[i, j]
    #     column_sum[i] = sum

    # #computes incremental number of particles in the grid when arranged linearly
    # prefix_sum[0, 0] = 0
    # ti.loop_config(serialize=True)
    # for i in range(1, grid_n):
    #     prefix_sum[i, 0] = prefix_sum[i - 1, 0] + column_sum[i - 1]

    # for i in range(grid_n):
    #     for j in range(grid_n):
    #         if j == 0:
    #             prefix_sum[i, j] += grain_count[i, j]
    #         else:
    #             prefix_sum[i, j] = prefix_sum[i, j - 1] + grain_count[i, j]


    #compute prefix sum in a better way, instead of above commented part
    # Assign the first element to the first element of prefix_sum
    prefix_sum[0,0,0] = grain_count[0,0,0]

    for i,j,k in ti.ndrange(grain_count.shape[0], grain_count.shape[1], grain_count.shape[2]):
        prefix_sum[i,j,k] = prefix_sum[i-1,j,k] + prefix_sum[i,j-1,k] - prefix_sum[i-1,j-1,k] + grain_count[i,j,k]

    #compute list_head and list_tail of cumulative sum
    for i in range(grid_n):
         for j in range(grid_n):
             for k in range(grid_n):
                linear_idx = i * grid_n * grid_n + j* grid_n + k
                list_head[linear_idx] = prefix_sum[i, j,k] - grain_count[i, j,k]
                list_cur[linear_idx] = list_head[linear_idx]
                list_tail[linear_idx] = prefix_sum[i, j,k]

    #particle_id is ordering the particle ids based on their linear position
    for i in range(n):
        grid_idx = ti.floor(gf[i].p * grid_n, int)
        linear_idx = grid_idx[0] * grid_n * grid_n + grid_idx[1] * grid_n + grid_idx[2]
        grain_location = ti.atomic_add(list_cur[linear_idx], 1)
        particle_id[grain_location] = i

    # Brute-force collision detection
    '''
    for i in range(n):
        for j in range(i + 1, n):
            resolve(i, j)
    '''

    # Fast collision detection
    for i in range(n):
        grid_idx = ti.floor(gf[i].p * grid_n, int)
        x_begin = ti.max(grid_idx[0] - 1, 0)
        x_end = ti.min(grid_idx[0] + 2, grid_n)

        y_begin = ti.max(grid_idx[1] - 1, 0)
        y_end = ti.min(grid_idx[1] + 2, grid_n)

        z_begin = ti.max(grid_idx[2] - 1, 0)
        z_end = ti.min(grid_idx[2] + 2, grid_n)

        for neigh_i in range(x_begin, x_end):
            for neigh_j in range(y_begin, y_end):
                for neigh_k in range(z_begin, z_end):
                    neigh_linear_idx = neigh_i * grid_n * grid_n + neigh_j * grid_n + neigh_k
                    for p_idx in range(list_head[neigh_linear_idx],
                                       list_tail[neigh_linear_idx]):
                        j = particle_id[p_idx]
                        if i < j:
                            resolve(i, j)

func_timer.start('init')  #--> ADDED
init()
func_timer.log('init')  #--> ADDED

gui = ti.GUI('Taichi DEM', (window_size, window_size), show_gui=False)
""""""
step = 0

if SAVE_FRAMES:
    os.makedirs('output', exist_ok=True)
"""
pos = gf.p.to_numpy()
r = gf.r.to_numpy() * window_size
gui.circles(pos, radius=r)
if SAVE_FRAMES:
    gui.show(f'output/{step:06d}.png')
else:
    gui.show()

"""
# while gui.running:
print('Cumulative Time elapsed before executing functions: ' + str(time.perf_counter()))
while step < 10000:
    for s in range(substeps):
        func_timer.start('update')  #--> ADDED
        update()
        func_timer.log('update')  #--> ADDED

        func_timer.start('apply_bc')  #--> ADDED
        apply_bc()
        func_timer.log('apply_bc')  #--> ADDED

        func_timer.start('contact')  #--> ADDED
        contact(gf)
        func_timer.log('contact')  #--> ADDED

    pos = gf.p.to_numpy()
    r = gf.r.to_numpy() * window_size
    """
    gui.circles(pos, radius=r)
    if SAVE_FRAMES:
        gui.show(f'output/{step:06d}.png')
    else:
        gui.show()
    """
    step += 1
print('Cumulative Time elapsed after executing all functions: ' + str(time.perf_counter()))
func_timer.output_log('basic-sim-prof.txt')
print(benchmark(update, (), n_repeat=100))

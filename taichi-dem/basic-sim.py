"""
This version of the file implements the following changes:
- 16000 particles instead of ~8000
- Timer module used to measure the performance of each sim
component
- Utils module implements P-Wave timestep calculation used
to calculate an appropriate time step
- Contacts are made size invariant
"""
# activate python virtual environment: .\taichi-env\Scripts\activate
import taichi as ti
import math
import os, sys
import time
from cupyx.profiler import benchmark

import numpy as np
import cupy as cp
import copy
import torch

# add folders with python modules
cwd = r''+os.getcwd()
sys.path.append(cwd+r'/utils-dem')
sys.path.append(cwd+r'/profiling')

# add user defined modules
import utils
from timer_implementation import Timer #--> ADDED
from timer_implementation import DEMSolverStatistics

func_timer = Timer() #--> ADDED

ti.init(arch=ti.gpu, kernel_profiler=True)
vec = ti.math.vec2

SAVE_FRAMES = False

window_size = 640  # Number of pixels of the window
n = 16000 #8192  # Number of grains

density = 2700.0
youngs_mod = 1e9
restitution_coef = 0.001
gravity = -9.81
substeps = 1
num_steps = 10000

@ti.dataclass
class Grain:
    p: vec  # Position
    m: ti.f32  # Mass
    r: ti.f32  # Radius
    v: vec  # Velocity
    a: vec  # Acceleration
    f: vec  # Force

gf = Grain.field(shape=(n, ))

grid_n = 130
grid_size = 1.0 / grid_n  # Simulation domain of size [0, 1]
print(f"Grid size: {grid_n}x{grid_n}")

grain_r_min = 0.002
grain_r_max = 0.003

assert grain_r_max * 2 < grid_size

dt = utils.pwave_timestep(grain_r_min, youngs_mod, density, 1) # Larger dt might lead to unstable results.
print(dt)

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
        region_width = 1.0 - 2*padding
        pos = vec(l % region_width + padding + grid_size * ti.random() * 0.1,
                  l // region_width * grid_size + 0.01)
        gf[i].p = pos
        gf[i].r = ti.random() * (grain_r_max - grain_r_min) + grain_r_min
        gf[i].m = density * math.pi * gf[i].r**2

@ti.kernel
def update():
    dt2 = 0.5 * dt**2
    dt_half = 0.5 * dt 
    # ti.loop_config(block_dim=256)
    for i in gf:
        a = gf[i].f / gf[i].m
        gf[i].v += (gf[i].a + a) * dt_half
        gf[i].p += gf[i].v * dt + a * dt2
        gf[i].a = a


@ti.kernel
def apply_bc():
    bounce_coef = 0.3  # Velocity damping
    for i in gf:
        x = gf[i].p[0]
        y = gf[i].p[1]

        if y - gf[i].r < 0:
            gf[i].p[1] = gf[i].r
            gf[i].v[1] *= -bounce_coef

        elif y + gf[i].r > 1.0:
            gf[i].p[1] = 1.0 - gf[i].r
            gf[i].v[1] *= -bounce_coef

        if x - gf[i].r < 0:
            gf[i].p[0] = gf[i].r
            gf[i].v[0] *= -bounce_coef

        elif x + gf[i].r > 1.0:
            gf[i].p[0] = 1.0 - gf[i].r
            gf[i].v[0] *= -bounce_coef

@ti.func
def resolve(i, j):
    rel_pos = gf[j].p - gf[i].p
    dist = ti.sqrt(rel_pos[0]**2 + rel_pos[1]**2)
    delta = -dist + gf[i].r + gf[j].r  # delta = d - 2 * r
    if delta > 0:  # in contact
        normal = rel_pos / dist
        stiffness = youngs_mod * 2 * (gf[i].r*gf[j].r)/(gf[i].r + gf[j].r) # ---> ADDED
        f1 = normal * delta * stiffness
        # Damping force
        M = (gf[i].m * gf[j].m) / (gf[i].m + gf[j].m)
        K = stiffness
        C = 2. * (1. / ti.sqrt(1. + (math.pi / ti.log(restitution_coef))**2)
                  ) * ti.sqrt(K * M)
        V = (gf[j].v - gf[i].v) * normal
        f2 = C * V * normal
        gf[i].f += f2 - f1
        gf[j].f -= f2 - f1


list_head = ti.field(dtype=ti.i32, shape=grid_n * grid_n)
list_cur = ti.field(dtype=ti.i32, shape=grid_n * grid_n)
list_tail = ti.field(dtype=ti.i32, shape=grid_n * grid_n)

grain_count = ti.field(dtype=ti.i32,
                       shape=(grid_n, grid_n),
                       name="grain_count")
column_sum = ti.field(dtype=ti.i32, shape=grid_n, name="column_sum")
prefix_sum = ti.field(dtype=ti.i32, shape=(grid_n, grid_n), name="prefix_sum")
particle_id = ti.field(dtype=ti.i32, shape=n, name="particle_id")


@ti.kernel
def contact(gf: ti.template()):
    '''
    Handle the collision between grains.
    '''
    for i in gf:
        gf[i].f = vec(0., gravity * gf[i].m)  # Apply gravity.

    grain_count.fill(0)

    for i in range(n):
        grid_idx = ti.floor(gf[i].p * grid_n, int)
        grain_count[grid_idx] += 1

    # for i in range(grid_n):
    #     sum = 0
    #     for j in range(grid_n):
    #         sum += grain_count[i, j]
    #     column_sum[i] = sum


@ti.kernel
def contact2(gf: ti.template(), grain_count_cp: ti.i32):
    prefix_sum[0, 0] = 0

    ti.loop_config(serialize=True)
    for i in range(1, grid_n):
        prefix_sum[i, 0] = prefix_sum[i - 1, 0] + column_sum[i - 1]

    grain_count

    for i in range(grid_n):
        for j in range(grid_n):
            if j == 0:
                prefix_sum[i, j] += grain_count[i, j]
            else:
                prefix_sum[i, j] = prefix_sum[i, j - 1] + grain_count[i, j]

            linear_idx = i * grid_n + j

            list_head[linear_idx] = prefix_sum[i, j] - grain_count[i, j]
            list_cur[linear_idx] = list_head[linear_idx]
            list_tail[linear_idx] = prefix_sum[i, j]

    

    # Brute-force collision detection
    """
    for i in range(n):
        for j in range(i + 1, n):
            resolve(i, j)
    """
    # print(list_head)

@ti.kernel
def collision_detect(gf:ti.template(), list_head: ti.template(), list_tail: ti.template(), list_cur: ti.template()):

    for i in range(n):
        grid_idx = ti.floor(gf[i].p * grid_n, int)
        linear_idx = grid_idx[0] * grid_n + grid_idx[1]
        grain_location = ti.atomic_add(list_cur[linear_idx], 1)
        particle_id[grain_location] = i
    # Fast collision detection
    # for i in range(n):
    #     grid_idx = ti.floor(gf[i].p * grid_n, int)
    #     x_begin = ti.max(grid_idx[0] - 1, 0)
    #     x_end = ti.min(grid_idx[0] + 2, grid_n)

    #     y_begin = ti.max(grid_idx[1] - 1, 0)
    #     y_end = ti.min(grid_idx[1] + 2, grid_n)

    #     for neigh_i in range(x_begin, x_end):
    #         for neigh_j in range(y_begin, y_end):
    #             neigh_linear_idx = neigh_i * grid_n + neigh_j
    #             for p_idx in range(list_head[neigh_linear_idx],
    #                                list_tail[neigh_linear_idx]):
    #                 j = particle_id[p_idx]
    #                 if i < j:
    #                     resolve(i, j)


# def run_sim():

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
start_time = time.perf_counter()
# print(time.perf_counter())
r = gf.r.to_numpy() * window_size # move out of for loop

# mult = n * substeps * 2
# # f = open("position.txt", 'a')
# fp = np.memmap(f"position.dat", dtype="float32", mode='w+', shape=(mult*num_steps))
# pos_list = []
# batch_idx = 1
# batch_size = 1000
statistcs = DEMSolverStatistics()
list_head_tc = torch.zeros(grid_n*grid_n, dtype=torch.int32, device='cuda')
while step < num_steps:
    for s in range(substeps):
        statistcs.SolveTime.tick()
        # func_timer.start('update')  #--> ADDED
        statistcs.UpdateTime.tick()
        update()
        statistcs.UpdateTime.tick()
        # func_timer.log('update')  #--> ADDED

        # func_timer.start('apply_bc')  #--> ADDED
        statistcs.Apply_bcTime.tick()
        apply_bc()
        statistcs.Apply_bcTime.tick()
        # func_timer.log('apply_bc')  #--> ADDED

        # func_timer.start('contact')  #--> ADDED
        statistcs.ContactTime.tick()
        # contact(gf)
        # grain_count_tc = grain_count.to_torch(device='cuda')
        # # print(grain_count_cp.shape)
        # # zeroA = torch.linspace(0, 1, steps=1, dtype=torch.int32, device='cuda')
        # list_tail_tc = torch.cumsum(grain_count_tc, dim=0, dtype=torch.int32)
        # list_tail_tc = torch.flatten(list_tail_tc)
        # list_head_tc[1:grid_n*grid_n] = list_tail_tc[0:grid_n*grid_n-1]
        # # list_head_np = cp.asnumpy(list_head_cp)
        

        # list_curr_tc = list_head_tc.clone()
        # # # print(list_curr_np.shape)
        # # # print(type(list_head_np[0]))
        # # # print(f"list head: {list_head_np[100:200]}")
        # # # print(f"list cur: {list_curr_np[100:200]}")
        # # # print(f"list tail: {list_tail_cp[100:200]}")
        # list_head.from_torch(list_head_tc)
        # list_cur.from_torch(list_curr_tc)
        # list_tail.from_torch(list_tail_tc)
        # collision_detect(gf, list_head, list_tail, list_cur)
        # contact2(gf, grain_count_cp)
        statistcs.ContactTime.tick()
        statistcs.SolveTime.tick()
        # print([list_head[idx] for idx in range(500)])
        # print(list_head[:500])
        # break
        # func_timer.log('contact')  #--> ADDED
    # break
    pos = gf.p.to_numpy()
    if step == 5000:
        statistcs.report_avg(step)
    # gf.p.to_numpy().tofile(f, format="%f")
    # cp_array = cp.asarray(gf.p.to_numpy())
    # pos_list.append(np.reshape(gf.p.to_numpy(), -1))
        
    # if batch_idx == batch_size:
    #     pos_flat = np.concatenate(pos_list, 0)
    #     fp[step//batch_size*mult*batch_size:(step//batch_size+1)*mult*batch_size] = pos_flat[:]
    #     pos_list = []
    #     batch_idx = 0
    # batch_idx += 1

    # r = gf.r.to_numpy() * window_size

    """
    gui.circles(pos, radius=r)
    if SAVE_FRAMES:
        gui.show(f'output/{step:06d}.png')
    else:
        gui.show()
    """
    step += 1

# del fp
time_used = time.perf_counter()-start_time
print(f"time used is {time_used}")
    # func_timer.output_log('basic-sim-prof.txt')
    # print(benchmark(update, (), n_repeat=100))
    # ti.profiler.print_scoped_profiler_info()
ti.profiler.print_kernel_profiler_info()
    # return time_used
# print(benchmark(update, (), n_repeat=100))
# print(benchmark(apply_bc, (), n_repeat=100))
# print(benchmark(contact, (gf,), n_repeat=100))

# time_list = []
# for i in range(1):
#     time_list.append(run_sim())

# print(np.mean(time_list))
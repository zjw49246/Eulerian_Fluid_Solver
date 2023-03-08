# References:
# http://developer.download.nvidia.com/books/HTML/gpugems/gpugems_ch38.html
# https://github.com/PavelDoGreat/WebGL-Fluid-Simulation
# https://www.bilibili.com/video/BV1ZK411H7Hc?p=4
# https://github.com/ShaneFX/GAMES201/tree/master/HW01

import argparse

import numpy as np

import math

import taichi as ti

# How to run:
#   `python stable_fluid.py`: use the jacobi iteration to solve the linear system.
#   `python stable_fluid.py -S`: use a sparse matrix to do so.
parser = argparse.ArgumentParser()
parser.add_argument('-S',
                    '--use-sp-mat',
                    action='store_true',
                    help='Solve Poisson\'s equation by using a sparse matrix')
parser.add_argument('-a',
                    '--arch',
                    required=False,
                    default="cpu",
                    dest='arch',
                    type=str,
                    help='The arch (backend) to run this example on')
args, unknowns = parser.parse_known_args()

test_mode = True
use_MacCormack = False
use_conjugate_gradients = True
use_warm_starting = True
use_jacobi_preconditioner = True
use_MGPCG = False
p_conjugate_gradients_iters = 200
vc_jacobi_iters = 500
gravity = 0
print_residual = True

res = 512
res_sq_recip = 1.0 / (res * res)
one_third = 1 / 3
one_64 = 1 / 64
dt = 0.03
p_jacobi_iters = 20  # 40 for a quicker but less accurate result
f_strength = 2000.0
curl_strength = 0
time_c = 2
maxfps = 60
dye_decay = 1 - 1 / (maxfps * time_c)
force_radius = res / 2.0
debug = False
vc_num_level = 4

use_sparse_matrix = False
arch = "gpu"
if arch in ["x64", "cpu", "arm64"]:
    ti.init(arch=ti.cpu)
elif arch in ["cuda", "gpu"]:
    ti.init(arch=ti.cuda)
else:
    raise ValueError('Only CPU and CUDA backends are supported for now.')

if use_sparse_matrix:
    print('Using sparse matrix')
elif use_MGPCG:
    print('Using MGPCG')
elif use_conjugate_gradients:
    print(f'Using conjugate gradients {p_conjugate_gradients_iters} iterations')
else:
    print(f'Using jacobi iteration {p_jacobi_iters} iterations')

_velocities = ti.Vector.field(2, float, shape=(res, res))
_new_velocities = ti.Vector.field(2, float, shape=(res, res))
velocity_divs = ti.field(float, shape=(res, res))
velocity_curls = ti.field(float, shape=(res, res))
_pressures = ti.field(float, shape=(res, res))
_new_pressures = ti.field(float, shape=(res, res))
_dye_buffer = ti.Vector.field(3, float, shape=(res, res))
_new_dye_buffer = ti.Vector.field(3, float, shape=(res, res))
# A_cg = ti.field(int, shape=(res * res, 5))
r_cg_cur = ti.field(float, shape=(res * res))
mu_cg = ti.field(ti.f64, shape=1)
r_cg_nxt = ti.field(float, shape=(res * res))
p_cg_cur = ti.field(float, shape=(res * res))
p_cg_nxt = ti.field(float, shape=(res * res))
Ap_cg = ti.field(float, shape=(res * res))
x_cg = ti.field(float, shape=(res * res))
u_vc = ti.field(float)
b_vc = ti.field(float)
vc_snode = ti.root.dense(ti.i, vc_num_level + 1).dense(ti.j, res * res) # total of (vc_num_level + 1) levels
vc_snode.place(u_vc)
vc_snode.place(b_vc)
u_vc_top_level = ti.field(float, shape=(res * res))
r_vc = ti.field(float, shape=(res * res))


class TexPair:
    def __init__(self, cur, nxt):
        self.cur = cur
        self.nxt = nxt

    def swap(self):
        self.cur, self.nxt = self.nxt, self.cur


velocities_pair = TexPair(_velocities, _new_velocities)
pressures_pair = TexPair(_pressures, _new_pressures)
dyes_pair = TexPair(_dye_buffer, _new_dye_buffer)
r_cg_pair = TexPair(r_cg_cur, r_cg_nxt)
p_cg_pair = TexPair(p_cg_cur, p_cg_nxt)

if use_sparse_matrix:
    # use a sparse matrix to solve Poisson's pressure equation.
    @ti.kernel
    def fill_laplacian_matrix(laplacian_matrix: ti.types.sparse_matrix_builder()):
        for i, j in ti.ndrange(res, res):
            row = i * res + j
            center = 0.0
            if j != 0:
                laplacian_matrix[row, row - 1] += -1.0
                center += 1.0
            if j != res - 1:
                laplacian_matrix[row, row + 1] += -1.0
                center += 1.0
            if i != 0:
                laplacian_matrix[row, row - res] += -1.0
                center += 1.0
            if i != res - 1:
                laplacian_matrix[row, row + res] += -1.0
                center += 1.0
            laplacian_matrix[row, row] += center

    N = res * res
    K = ti.linalg.SparseMatrixBuilder(N, N, max_num_triplets=N * 6)
    F_b = ti.ndarray(ti.f32, shape=N)

    fill_laplacian_matrix(K)
    L = K.build()
    solver = ti.linalg.SparseSolver(solver_type="LLT")
    solver.analyze_pattern(L)
    solver.factorize(L)

# if use_conjugate_gradients:
#     @ti.kernel
#     def fill_laplacian_matrix(laplacian_matrix: ti.template()):
#         for i, j in ti.ndrange(res, res):
#             row = i * res + j
#             count = 0
#             if j != 0:
#                 count += 1
#                 laplacian_matrix[row, count] = row - 1
#             if j != res - 1:
#                 count += 1
#                 laplacian_matrix[row, count] = row + 1
#             if i != 0:
#                 count += 1
#                 laplacian_matrix[row, count] = row - res
#             if i != res - 1:
#                 count += 1
#                 laplacian_matrix[row, count] = row + res
#             laplacian_matrix[row, 0] = row
#             if count < 4:
#                 for k in range(count + 1, 5):
#                     laplacian_matrix[row, k] = -1
#     fill_laplacian_matrix(A_cg)
#     print(A_cg)

@ti.func
def sample(qf, u, v):
    I = ti.Vector([int(u), int(v)])
    I = ti.max(0, ti.min(res - 1, I))
    return qf[I]


@ti.func
def lerp(vl, vr, frac):
    # frac: [0.0, 1.0]
    return vl + frac * (vr - vl)


@ti.func
def bilerp(vf, p):
    u, v = p
    s, t = u - 0.5, v - 0.5
    # floor
    iu, iv = ti.floor(s), ti.floor(t)
    # fract
    fu, fv = s - iu, t - iv
    a = sample(vf, iu, iv)
    b = sample(vf, iu + 1, iv)
    c = sample(vf, iu, iv + 1)
    d = sample(vf, iu + 1, iv + 1)
    return lerp(lerp(a, b, fu), lerp(c, d, fu), fv)


# 3rd order Runge-Kutta
@ti.func
def backtrace(vf: ti.template(), p, dt_: ti.template()):
    v1 = bilerp(vf, p)
    p1 = p - 0.5 * dt_ * v1
    v2 = bilerp(vf, p1)
    p2 = p - 0.75 * dt_ * v2
    v3 = bilerp(vf, p2)
    p -= dt_ * ((2 / 9) * v1 + (1 / 3) * v2 + (4 / 9) * v3)
    return p


@ti.kernel
def advect(vf: ti.template(), qf: ti.template(), new_qf: ti.template()):
    for i, j in vf:
        p = ti.Vector([i, j]) + 0.5
        if use_MacCormack:
            p_back = backtrace(vf, p, dt)
            p_back_and_forth = backtrace(vf, p_back, -dt)
            p_error = (p_back_and_forth - p)
            p = p_back + p_error
        else:
            p = backtrace(vf, p, dt)
        new_qf[i, j] = bilerp(qf, p) * dye_decay


@ti.kernel
def apply_impulse(vf: ti.template(), dyef: ti.template(),
                  imp_data: ti.types.ndarray()):
    g_dir = -ti.Vector([0, gravity]) * 300
    for i, j in vf:
        omx, omy = imp_data[2], imp_data[3]
        mdir = ti.Vector([imp_data[0], imp_data[1]])
        dx, dy = (i + 0.5 - omx), (j + 0.5 - omy)
        d2 = dx * dx + dy * dy
        # dv = F * dt
        factor = ti.exp(-d2 / force_radius)

        dc = dyef[i, j]
        a = dc.norm()

        momentum = (mdir * f_strength * factor + g_dir * a / (1 + a)) * dt
        # momentum = (mdir * f_strength * factor) * dt

        v = vf[i, j]
        vf[i, j] = v + momentum
        # add dye
        if mdir.norm() > 0.5:
            dc += ti.exp(-d2 * (4 / (res / 15)**2)) * ti.Vector(
                [imp_data[4], imp_data[5], imp_data[6]])

        dyef[i, j] = dc


@ti.kernel
def divergence(vf: ti.template()):
    for i, j in vf:
        vl = sample(vf, i - 1, j)
        vr = sample(vf, i + 1, j)
        vb = sample(vf, i, j - 1)
        vt = sample(vf, i, j + 1)
        vc = sample(vf, i, j)
        if i == 0:
            vl.x = -vc.x
        if i == res - 1:
            vr.x = -vc.x
        if j == 0:
            vb.y = -vc.y
        if j == res - 1:
            vt.y = -vc.y
        velocity_divs[i, j] = (vr.x - vl.x + vt.y - vb.y) * 0.5


@ti.kernel
def vorticity(vf: ti.template()):
    for i, j in vf:
        vl = sample(vf, i - 1, j)
        vr = sample(vf, i + 1, j)
        vb = sample(vf, i, j - 1)
        vt = sample(vf, i, j + 1)
        velocity_curls[i, j] = (vr.y - vl.y - vt.x + vb.x) * 0.5


@ti.kernel
def pressure_jacobi(pf: ti.template(), new_pf: ti.template()):
    for i, j in pf:
        pl = sample(pf, i - 1, j)
        pr = sample(pf, i + 1, j)
        pb = sample(pf, i, j - 1)
        pt = sample(pf, i, j + 1)
        div = velocity_divs[i, j]
        new_pf[i, j] = (pl + pr + pb + pt - div) * 0.25


@ti.kernel
def subtract_gradient(vf: ti.template(), pf: ti.template()):
    for i, j in vf:
        pl = sample(pf, i - 1, j)
        pr = sample(pf, i + 1, j)
        pb = sample(pf, i, j - 1)
        pt = sample(pf, i, j + 1)
        vf[i, j] -= 0.5 * ti.Vector([pr - pl, pt - pb])


@ti.kernel
def enhance_vorticity(vf: ti.template(), cf: ti.template()):
    # anti-physics visual enhancement...
    for i, j in vf:
        cl = sample(cf, i - 1, j)
        cr = sample(cf, i + 1, j)
        cb = sample(cf, i, j - 1)
        ct = sample(cf, i, j + 1)
        cc = sample(cf, i, j)
        force = ti.Vector([abs(ct) - abs(cb),
                           abs(cl) - abs(cr)]).normalized(1e-3)
        force *= curl_strength * cc
        vf[i, j] = ti.min(ti.max(vf[i, j] + force * dt, -1e3), 1e3)


@ti.kernel
def copy_divergence(div_in: ti.template(), div_out: ti.types.ndarray()):
    for I in ti.grouped(div_in):
        div_out[I[0] * res + I[1]] = -div_in[I]


@ti.kernel
def apply_pressure(p_in: ti.types.ndarray(), p_out: ti.template()):
    for I in ti.grouped(p_out):
        p_out[I] = p_in[I[0] * res + I[1]]

@ti.func
def laplacian_A_mul_p(p: ti.template(), Ap: ti.template(), res: int, mul_ct: float, mul_nb: float):
    # ct: center, nb: neighbor
    for row in Ap:
        i = row // res
        j = row % res
        Ap[row] = 0.0
        mul = 1.0
        if use_conjugate_gradients and use_jacobi_preconditioner:
            mul = 0.25
        Ap[row] += mul * mul_ct * p[row]
        if j != 0:
            Ap[row] += mul * mul_nb * p[row - 1]
        if j != res - 1:
            Ap[row] += mul * mul_nb * p[row + 1]
        if i != 0:
            Ap[row] += mul * mul_nb * p[row - res]
        if i != res - 1:
            Ap[row] += mul * mul_nb * p[row + res]

@ti.func
def vc_laplacian_A_mul_p(p: ti.template(), Ap: ti.template(), res: int, mul_ct: float, mul_nb: float, l: int):
    # ct: center, nb: neighbor
    for row in Ap:
        Ap[row] = 0.0
        if row >= res * res:
            continue

        i = row // res
        j = row % res
        mul = 1.0
        if use_conjugate_gradients and use_jacobi_preconditioner:
            mul = 0.25
        Ap[row] += mul * mul_ct * p[l, row]
        if j != 0:
            Ap[row] += mul * mul_nb * p[l, row - 1]
        if j != res - 1:
            Ap[row] += mul * mul_nb * p[l, row + 1]
        if i != 0:
            Ap[row] += mul * mul_nb * p[l, row - res]
        if i != res - 1:
            Ap[row] += mul * mul_nb * p[l, row + res]

@ti.kernel
def pressure_conjugate_gradients_pre(pf: ti.template()):
    for i, j in pf:
        row = i * res + j
        if use_warm_starting:
            pressure = sample(pf, i, j)
            x_cg[row] = pressure
        else:
            x_cg[row] = 0

    laplacian_A_mul_p(x_cg, Ap_cg, res, 4.0, -1.0)

    for i, j in pf:
        row = i * res + j
        r_cg_pair.cur[row] = -velocity_divs[i, j]
        if use_jacobi_preconditioner:
            r_cg_pair.cur[row] *= 0.25
        r_cg_pair.cur[row] -= Ap_cg[row]
        # # pressure = sample(pf, A_cg[row, 0][0], A_cg[row, 0][1])
        # # r_cg[row] -= 4 * pressure
        # for k in range(5):
        #     mul = -1.0
        #     if k == 0:
        #         mul = 4.0
        #     if use_jacobi_preconditioner:
        #         mul *= 0.25
        #     if A_cg[row, k] != -1:
        #         r_cg_pair.cur[row] -= mul * x_cg[A_cg[row, k]]
        #     else:
        #         break

    for r in p_cg_pair.cur:
        p_cg_pair.cur[r] = r_cg_pair.cur[r]

@ti.kernel
def pressure_conjugate_gradients_iter(pf: ti.template()):
    laplacian_A_mul_p(p_cg_pair.cur, Ap_cg, res, 4.0, -1.0)
    # for r in Ap_cg:
    #     Ap_cg[r] = 0
    #     for k in range(5):
    #         mul = -1.0
    #         if k == 0:
    #             mul = 4.0
    #         if use_jacobi_preconditioner:
    #             mul *= 0.25
    #         # print(f'{k} {mul}')
    #         if A_cg[r, k] != -1:
    #             Ap_cg[r] += mul * p_cg_pair.cur[A_cg[r, k]]

    nume = 0.0
    deno = 0.0
    for r in r_cg_pair.cur:
        nume += r_cg_pair.cur[r] * r_cg_pair.cur[r]
        deno += p_cg_pair.cur[r] * Ap_cg[r]
    alpha = nume / (deno + 1e-5)

    r_cg_pair_cur_norm_sq = 0.0
    r_cg_pair_nxt_norm_sq = 0.0
    for r in x_cg:
        x_cg[r] += alpha * p_cg_pair.cur[r]
        r_cg_pair.nxt[r] = r_cg_pair.cur[r] - alpha * Ap_cg[r]
        r_cg_pair_cur_norm_sq += r_cg_pair.cur[r] * r_cg_pair.cur[r]
        r_cg_pair_nxt_norm_sq += r_cg_pair.nxt[r] * r_cg_pair.nxt[r]

    if print_residual:
        print(r_cg_pair_nxt_norm_sq)
    # if ti.sqrt(r_cg_pair_nxt_norm_sq) < 1e-5:
    #     return

    beta = r_cg_pair_nxt_norm_sq / (r_cg_pair_cur_norm_sq + 1e-5)
    for r in p_cg_pair.nxt:
        p_cg_pair.nxt[r] = r_cg_pair.nxt[r] + beta * p_cg_pair.cur[r]
        p_cg_pair.cur[r] = p_cg_pair.nxt[r]
        r_cg_pair.cur[r] = r_cg_pair.nxt[r]

@ti.kernel
def pressure_conjugate_gradients_post(new_pf: ti.template()):
    for i, j in new_pf:
        row = i * res + j
        new_pf[i, j] = x_cg[row]

@ti.func
def calculate_vc_res(l: int):
    grid_length = ti.pow(2, l)
    vc_res = int(res / grid_length)

    return vc_res

@ti.func
def damped_jacobi_smooth(l: int):
    vc_res = calculate_vc_res(l)
    vc_laplacian_A_mul_p(u_vc, Ap_cg, vc_res, 4.0, -1.0, l)
    for i in Ap_cg:
        if i >= vc_res * vc_res:
            continue
        u_vc[l, i] += 2 * one_third * (b_vc[l, i] - Ap_cg[l, i])

@ti.kernel
def v_cycle_pre():
    for row in ti.ndrange(res * res):
        u_vc[0, row] = 0.0

@ti.func
def v_cycle_restrict(l: int):
    vc_res_cur = calculate_vc_res(l)
    vc_res_nxt = calculate_vc_res(l + 1)
    for i, j in ti.ndrange(vc_res_nxt, vc_res_nxt):
        row = i * vc_res_nxt + j
        b_vc[l + 1, row] = 0.0
        # get the left bottom u_h, origin at the most left bottom point
        h_i = i * 2
        h_j = j * 2
        count = 0
        mul = 3
        for m, n in ti.static(ti.ndrange((-1, 3), (-1, 3))):
            i_cur = i + m
            j_cur = j + n
            row_cur = i_cur * vc_res_cur + j_cur
            if 0 <= i_cur < vc_res_cur and 0 <= j_cur < vc_res_cur:
                if 0 <= m <= 1 and 0 <= n <= 1:
                    mul = 9
                elif m == n or (m == -1 and n == 2) or (m == 2 and n == -1):
                    mul = 1
                b_vc[l + 1, row] += mul * r_vc[row_cur]
                count += mul
        b_vc[l + 1, row] /= count

@ti.kernel
def v_cycle_up_iter(l: int):
    vc_res = calculate_vc_res(l)
    damped_jacobi_smooth(l)
    vc_laplacian_A_mul_p(u_vc, Ap_cg, vc_res, 4.0, -1.0, l)
    for i in Ap_cg:
        r_vc[i] = 0
        if i >= vc_res * vc_res:
            continue
        r_vc[i] = b_vc[l, i] - Ap_cg[i]
    v_cycle_restrict(l)

    vc_res_nxt = calculate_vc_res(l + 1)
    for row in ti.ndrange(vc_res_nxt * vc_res_nxt):
        u_vc[l + 1, row] = 0.0


@ti.kernel
def v_cycle_solve():
    vc_res = calculate_vc_res(vc_num_level)
    for row in ti.ndrange(vc_res * vc_res):
        ul = u_vc[vc_num_level, row + vc_res]
        ur = u_vc[vc_num_level, row - vc_res]
        ub = u_vc[vc_num_level, row - 1]
        ut = u_vc[vc_num_level, row + 1]
        b_val = b_vc[vc_num_level, row]
         = (ul + ur + ub + ut - b_val) * 0.25


def v_cycle():
    v_cycle_pre()
    for l in range(vc_num_level):
        v_cycle_up_iter(l)
    for _ in range(vc_jacobi_iters):





@ti.kernel
def pressure_MGPCG_pre(pf: ti.template()):
    for i, j in pf:
        row = i * res + j
        if use_warm_starting:
            pressure = sample(pf, i, j)
            x_cg[row] = pressure
        else:
            x_cg[row] = 0

    laplacian_A_mul_p(x_cg, Ap_cg, res, 4.0, -1.0)

    mu_cg[0] = 0.0
    for i, j in pf:
        row = i * res + j
        r_cg_pair.cur[row] = -velocity_divs[i, j]
        if use_jacobi_preconditioner:
            r_cg_pair.cur[row] *= 0.25
        r_cg_pair.cur[row] -= Ap_cg[row]
        mu_cg[0] += r_cg_pair.cur[row]

    mu_cg[0] *= res_sq_recip
    for row in r_cg_pair.cur:
        r_cg_pair.cur[row] -= mu_cg



def solve_pressure_sp_mat():
    copy_divergence(velocity_divs, F_b)
    x = solver.solve(F_b)
    apply_pressure(x, pressures_pair.cur)


def solve_pressure_jacobi():
    for _ in range(p_jacobi_iters):
        pressure_jacobi(pressures_pair.cur, pressures_pair.nxt)
        pressures_pair.swap()

def solve_pressure_conjugate_gradients():
    pressure_conjugate_gradients_pre(pressures_pair.cur)
    for _ in range(p_conjugate_gradients_iters):
        pressure_conjugate_gradients_iter(pressures_pair.cur)
    pressure_conjugate_gradients_post(pressures_pair.nxt)
    pressures_pair.swap()

def solve_pressure_MGPCG():
    pass

def step(mouse_data):
    advect(velocities_pair.cur, velocities_pair.cur, velocities_pair.nxt)
    advect(velocities_pair.cur, dyes_pair.cur, dyes_pair.nxt)
    velocities_pair.swap()
    dyes_pair.swap()

    apply_impulse(velocities_pair.cur, dyes_pair.cur, mouse_data)

    divergence(velocities_pair.cur)

    if curl_strength:
        vorticity(velocities_pair.cur)
        enhance_vorticity(velocities_pair.cur, velocity_curls)

    if use_MGPCG:
        solve_pressure_MGPCG()
    elif use_conjugate_gradients:
        solve_pressure_conjugate_gradients()
    else:
        solve_pressure_jacobi()

    subtract_gradient(velocities_pair.cur, pressures_pair.cur)

    if debug:
        divergence(velocities_pair.cur)
        div_s = np.sum(velocity_divs.to_numpy())
        print(f'divergence={div_s}')


class MouseDataGen:
    def __init__(self, test_mode=False):
        self.prev_mouse = None
        self.prev_color = None
        self.test_mode = test_mode
        self.test_frame = 0
        self.max_test_frame = 1000

    def __call__(self, gui):
        # [0:2]: normalized delta direction
        # [2:4]: current mouse xy
        # [4:7]: color
        mouse_data = np.zeros(8, dtype=np.float32)
        if self.test_mode:
            self.test_frame += 1
            if self.test_frame % 1 == 0:
                # mxy = np.array([0.1 * self.test_frame, 0.15 * self.test_frame], dtype=np.float32) * res
                mxy = np.array([0.5, 0], dtype=np.float32) * res
                if self.prev_mouse is None:
                    self.prev_mouse = mxy
                    # Set lower bound to 0.3 to prevent too dark colors
                    self.prev_color = (np.array([1, 1, 1]) * 0.7) + 0.3
                else:
                    mdir = ti.Vector([0.1 * ti.math.sin(self.test_frame * 5 / 180 * ti.math.pi), 1])
                    mdir = mdir / (np.linalg.norm(mdir) + 1e-5)
                    mouse_data[0], mouse_data[1] = mdir[0], mdir[1]
                    mouse_data[2], mouse_data[3] = mxy[0], mxy[1]
                    mouse_data[4:7] = self.prev_color
                    self.prev_mouse = mxy
            return mouse_data

        if gui.is_pressed(ti.GUI.LMB):
            mxy = np.array(gui.get_cursor_pos(), dtype=np.float32) * res
            if self.prev_mouse is None:
                self.prev_mouse = mxy
                # Set lower bound to 0.3 to prevent too dark colors
                self.prev_color = (np.random.rand(3) * 0.7) + 0.3
            else:
                mdir = mxy - self.prev_mouse
                mdir = mdir / (np.linalg.norm(mdir) + 1e-5)
                mouse_data[0], mouse_data[1] = mdir[0], mdir[1]
                mouse_data[2], mouse_data[3] = mxy[0], mxy[1]
                mouse_data[4:7] = self.prev_color
                self.prev_mouse = mxy
        else:
            self.prev_mouse = None
            self.prev_color = None

        return mouse_data


def reset():
    velocities_pair.cur.fill(0)
    pressures_pair.cur.fill(0)
    dyes_pair.cur.fill(0)


def main():
    global debug, curl_strength
    visualize_d = True  #visualize dye (default)
    visualize_v = False  #visualize velocity
    visualize_c = False  #visualize curl

    paused = False

    gui = ti.GUI('Stable Fluid', (res, res))
    md_gen = MouseDataGen(test_mode)

    while gui.running:
        if gui.get_event(ti.GUI.PRESS):
            e = gui.event
            if e.key == ti.GUI.ESCAPE:
                break
            elif e.key == 'r':
                paused = False
                reset()
            elif e.key == 's':
                if curl_strength:
                    curl_strength = 0
                else:
                    curl_strength = 7
            elif e.key == 'v':
                visualize_v = True
                visualize_c = False
                visualize_d = False
            elif e.key == 'd':
                visualize_d = True
                visualize_v = False
                visualize_c = False
            elif e.key == 'c':
                visualize_c = True
                visualize_d = False
                visualize_v = False
            elif e.key == 'p':
                paused = not paused
            elif e.key == 'd':
                debug = not debug

        # Debug divergence:
        # print(max((abs(velocity_divs.to_numpy().reshape(-1)))))

        if not paused:
            mouse_data = md_gen(gui)
            step(mouse_data)
        if visualize_c:
            vorticity(velocities_pair.cur)
            gui.set_image(velocity_curls.to_numpy() * 0.03 + 0.5)
        elif visualize_d:
            gui.set_image(dyes_pair.cur)
        elif visualize_v:
            gui.set_image(velocities_pair.cur.to_numpy() * 0.01 + 0.5)
        gui.show()


if __name__ == '__main__':
    main()
import cmath
import math
from itertools import combinations

import numpy as np

exp = cmath.exp
sin = cmath.sin
cos = cmath.cos
sqrt = cmath.sqrt
phase = cmath.phase

M = 1.0
l = 2
s = 2


def set_parameters(M_value=None, l_value=None, s_value=None):
    global M, l, s

    if M_value is not None:
        M = M_value
    if l_value is not None:
        l = l_value
    if s_value is not None:
        s = s_value

    return {"M": M, "l": l, "s": s}


def get_parameters():
    return {"M": M, "l": l, "s": s}


def VGR_minus(r):  # RW potential of GR
    return l * (l + 1) / r**2 - 2 * (s**2 - 1) * M / r**3

# the potential delta V is obtained from https://arxiv.org/abs/2404.11110

def V_minus(r, epsilon):  # RW potential of beyond GR
    rh = 2 * M * (1 - 5 * epsilon / 16)
    return (1 - rh / r) * (VGR_minus(r) + epsilon * delta_V(r))


def v_minus(ell):
    return np.array(
        [
            -5 * ell * (ell + 1) / 8,
            -5 * (ell**2 + ell - 3) / 4,
            -5 * (ell**2 + ell - 3) / 2,
            -5 * (ell**2 + ell - 3),
            1430 * ell * (ell + 1) - 8610,
            41460 - 3332 * ell * (ell + 1),
            -48192,
        ]
    )


def delta_V(r):
    return (1 / r**2) * sum(v_minus(l)[j] * (M / r) ** (j + 1) for j in range(7))


def Q_minus(r, omega, epsilon):
    rh = 2 * M * (1 - 5 * epsilon / 16)
    cs2 = 1 - 288 * epsilon * (1 - rh / r) * M**5 / r**5
    return omega**2 / cs2 - V_minus(r, epsilon)

# Note that since x(r) and r(x) is not single-valued, we only use this function to determine x_m=x(r_m) at r_m=3M in this program. Be cautious when using this function for other purposes, as it may lead to incorrect results.
def x(r, epsilon):
    rh = 2 * M * (1 - 5 * epsilon / 16)
    log = lambda z: cmath.log(z)
    return (
        r
        + rh * log(r / rh - 1)
        + epsilon
        * (
            (
                2
                * rh
                * (
                    -104 * M**5 * r * (44 * M - 5 * rh) * rh**2
                    - 3432 * M**6 * rh**3
                    - 78 * M**4 * r**2 * rh * (88 * M**2 - 10 * M * rh - 5 * rh**2)
                    + 3 * M**3 * r**3 * (-4576 * M**3 + 520 * M**2 * rh + 260 * M * rh**2 + 5 * rh**3)
                )
            )
            / (312 * rh**5 * r**4)
            + (
                6
                * M**2
                * (4576 * M**4 - 520 * M**3 * rh - 260 * M**2 * rh**2 - 5 * M * rh**3 - 65 * rh**4)
                * log(r)
            )
            / (312 * rh**5)
            - (
                3
                * M
                * (9152 * M**5 - 1040 * M**4 * rh - 520 * M**3 * rh**2 - 10 * M**2 * rh**3 - 130 * M * rh**4 - 65 * rh**5)
                * log(r - rh)
            )
            / (312 * rh**5)
        )
    )


def dx_dr(r, epsilon):
    rh = 2 * M * (1 - 5 * epsilon / 16)
    return (1 - rh / r) ** (-1) * (
        1
        + epsilon
        * (
            5 * M / (8 * r)
            + 5 * M**2 / (4 * r**2)
            + 5 * M**3 / (52 * r**3)
            + 5 * M**4 / r**4
            + 10 * M**5 / r**5
            - 88 * M**6 / r**6
        )
    )


def dr_dx(r, epsilon):
    return 1 / dx_dr(r, epsilon)


def find_rmin_from_rm_RK4(epsilon, omega, rm, rhom, rho_min, n_steps=50000):
    h = (rho_min - rhom) / n_steps
    r = complex(rm)
    beta = -cmath.phase(omega)
    phase_factor = exp(1j * beta)

    def rhs(r_value):
        return phase_factor / dx_dr(r_value, epsilon)

    for _ in range(n_steps):
        k1 = rhs(r)
        k2 = rhs(r + 0.5 * h * k1)
        k3 = rhs(r + 0.5 * h * k2)
        k4 = rhs(r + h * k3)
        r += h * (k1 + 2 * k2 + 2 * k3 + k4) / 6

    return r


def integrate_P_r_RK4(omega, epsilon, rhom, rho_min, r_min, P_min, n_steps=10000):
    h = (rhom - rho_min) / n_steps
    beta = -cmath.phase(omega)
    phase_factor = exp(1j * beta)

    rho = rho_min
    r = complex(r_min)
    P = complex(P_min)

    rho_list = [rho]
    r_list = [r]
    P_list = [P]

    def rhs(r_value, P_value, epsilon_value=epsilon):
        dP_drho = -phase_factor * (P_value**2 + Q_minus(r_value, omega, epsilon_value))
        dr_drho = phase_factor / dx_dr(r_value, epsilon_value)
        return dP_drho, dr_drho

    for _ in range(n_steps):
        k1P, k1r = rhs(r, P, epsilon)
        k2P, k2r = rhs(r + 0.5 * h * k1r, P + 0.5 * h * k1P, epsilon)
        k3P, k3r = rhs(r + 0.5 * h * k2r, P + 0.5 * h * k2P, epsilon)
        k4P, k4r = rhs(r + h * k3r, P + h * k3P, epsilon)

        P += h * (k1P + 2 * k2P + 2 * k3P + k4P) / 6
        r += h * (k1r + 2 * k2r + 2 * k3r + k4r) / 6
        rho += h

        rho_list.append(rho)
        r_list.append(r)
        P_list.append(P)

    return rho_list, r_list, P_list


def P_to_Prufer(P, x_value, omega):
    return -omega * x_value + 1 / (2j) * cmath.log((1j * P - omega) / (1j * P + omega))


def Prufer_to_P(Prufer, x_value, omega):
    return omega / cmath.tan(omega * x_value + Prufer)


class QNMReached(Exception):
    def __init__(self, rho, r, Prufer, phase_arg):
        self.rho = rho
        self.r = r
        self.Prufer = Prufer
        self.phase_arg = phase_arg


class BadBranchReached(Exception):
    def __init__(self, rho, r, Prufer, phase_arg):
        self.rho = rho
        self.r = r
        self.Prufer = Prufer
        self.phase_arg = phase_arg


def choose_n_steps_outer(omega, rho_m, rho_max, points_per_period=150, n_steps_minimum=50000):
    length = abs(rho_max - rho_m)
    n_est = int(points_per_period * abs(omega) * length / math.pi)
    return max(n_est, n_steps_minimum)


def integrate_Prufer_r_RK4(
    omega,
    epsilon,
    rho_m,
    rho_max,
    r_m,
    Prufer_m,
    n_steps=50000,
    qnm_cutoff=150,
    print_termination=True,
):
    h = (rho_max - rho_m) / n_steps

    beta = -phase(omega)
    phase_factor = exp(1j * beta)

    rho = rho_m
    r = complex(r_m)
    Prufer = complex(Prufer_m)

    rho_list = [rho]
    r_list = [r]
    Prufer_list = [Prufer]

    xm = x(r_m, epsilon)

    def rhs(rho_value, r_value, Prufer_value):
        phase_arg = omega * xm + abs(omega) * rho_value + Prufer_value
# if phase_arg.imag > qnm_cutoff, A_in/A_out=exp(-2i*zeta) blows up.
        if phase_arg.imag < -qnm_cutoff:
            raise QNMReached(rho_value, r_value, Prufer_value, phase_arg)
        if phase_arg.imag > qnm_cutoff:
            raise BadBranchReached(rho_value, r_value, Prufer_value, phase_arg)

        dPrufer_drho = -phase_factor * (omega - Q_minus(r_value, omega, epsilon) / omega) * sin(phase_arg) ** 2
        dr_drho = phase_factor / dx_dr(r_value, epsilon)
        return dPrufer_drho, dr_drho

    reached_qnm = False
    bad_branch = False

    try:
        for _ in range(n_steps):
            k1P, k1r = rhs(rho, r, Prufer)
            k2P, k2r = rhs(rho + 0.5 * h, r + 0.5 * h * k1r, Prufer + 0.5 * h * k1P)
            k3P, k3r = rhs(rho + 0.5 * h, r + 0.5 * h * k2r, Prufer + 0.5 * h * k2P)
            k4P, k4r = rhs(rho + h, r + h * k3r, Prufer + h * k3P)

            Prufer += h * (k1P + 2 * k2P + 2 * k3P + k4P) / 6
            r += h * (k1r + 2 * k2r + 2 * k3r + k4r) / 6
            rho += h

            rho_list.append(rho)
            r_list.append(r)
            Prufer_list.append(Prufer)
    except QNMReached as event:
        reached_qnm = True
        if print_termination:
            print(f"rho terminates at {event.rho} because phase_arg -> -i infinity")
        rho_list.append(event.rho)
        r_list.append(event.r)
        Prufer_list.append(event.Prufer)
    except BadBranchReached as event:
        bad_branch = True
        if print_termination:
            print(f"rho terminates at {event.rho} because phase_arg -> +i infinity")
        rho_list.append(event.rho)
        r_list.append(event.r)
        Prufer_list.append(event.Prufer)

    return rho_list, r_list, Prufer_list, reached_qnm, bad_branch


def compute_Ain_Aout_and_zeta(
    omega,
    epsilon,
    r_m=None,
    rho_m=0,
    rho_min=None,
    rho_max=None,
    n_steps_min=40000,
    n_steps_max=None,
    points_per_period=150,
    qnm_cutoff=150,
    bad_branch_penalty=1e100,
    print_termination=False,
):
    omega = complex(omega)
    beta = -phase(omega)

    if r_m is None:
        r_m = 3 * M
    if rho_min is None:
        rho_min = -50 * M
    if rho_max is None:
        rho_max = 10**4 * M
    if n_steps_max is None:
        n_steps_max = choose_n_steps_outer(
            omega=omega,
            rho_m=rho_m,
            rho_max=rho_max,
            points_per_period=points_per_period,
            n_steps_minimum=50000,
        )

    x_m = x(r_m, epsilon)
    r_min = find_rmin_from_rm_RK4(epsilon=epsilon, omega=omega, rm=r_m, rhom=rho_m, rho_min=rho_min, n_steps=n_steps_min)
    P_min = -1j * sqrt(Q_minus(r_min, omega, epsilon))

    rho_list_1, r_list_1, P_list_1 = integrate_P_r_RK4(
        omega=omega,
        epsilon=epsilon,
        rhom=rho_m,
        rho_min=rho_min,
        r_min=r_min,
        P_min=P_min,
        n_steps=n_steps_min,
    )

    P_m = P_list_1[-1]
    r_m_check = r_list_1[-1]

    Prufer_m = P_to_Prufer(P_m, x_m + rho_list_1[-1] * exp(1j * beta), omega)

    rho_list_2, r_list_2, Prufer_list_2, reached_qnm, bad_branch = integrate_Prufer_r_RK4(
        omega=omega,
        epsilon=epsilon,
        rho_m=rho_m,
        rho_max=rho_max,
        r_m=r_m_check,
        Prufer_m=Prufer_m,
        n_steps=n_steps_max,
        qnm_cutoff=qnm_cutoff,
        print_termination=print_termination,
    )

    zeta = Prufer_list_2[-1]

    if reached_qnm or zeta.imag < -qnm_cutoff:
        Ain_Aout = -exp(-2j * zeta.real) * exp(-2 * qnm_cutoff)
    elif bad_branch or zeta.imag > qnm_cutoff:
        Ain_Aout = bad_branch_penalty * exp(1j * zeta.real)
    else:
        Ain_Aout = -exp(-2j * zeta)

    return Ain_Aout, zeta


def muller_root_complex(
    F,
    z0,
    z1,
    z2,
    tol=1e-10,
    step_tol=None,
    max_iter=50,
    min_iter=4,
    verbose=False,
    min_sep=1e-14,
    max_step=None,
):
    z0 = complex(z0)
    z1 = complex(z1)
    z2 = complex(z2)

    if step_tol is None:
        step_tol = tol

    last_dz = None

    for n in range(max_iter):
        F0 = F(z0)
        F1 = F(z1)
        F2 = F(z2)

        if verbose:
            print("Muller iteration", n)
            print("z0 =", z0, "A_in/A_out(z0) =", F0, "|A_in/A_out(z0)| =", abs(F0))
            print("z1 =", z1, "A_in/A_out(z1) =", F1, "|A_in/A_out(z1)| =", abs(F1))
            print("z2 =", z2, "A_in/A_out(z2) =", F2, "|A_in/A_out(z2)| =", abs(F2))
            if last_dz is not None:
                print("|delta z| =", abs(last_dz))
            print()

        h0 = z1 - z0
        h1 = z2 - z1
        scale = max(1.0, abs(z0), abs(z1), abs(z2))

        if abs(h0) < min_sep * scale:
            z1 = z0 + min_sep * scale * (1.0 + 1.0j)
            F1 = F(z1)
            h0 = z1 - z0
        if abs(h1) < min_sep * scale:
            z2 = z1 + min_sep * scale * (1.0 - 1.0j)
            F2 = F(z2)
            h1 = z2 - z1

        if abs(h0 + h1) < min_sep * scale:
            if abs(F2 - F1) > min_sep * max(1.0, abs(F2), abs(F1)):
                dz = -F2 * (z2 - z1) / (F2 - F1)
            else:
                dz = min_sep * scale * (1.0 + 1.0j)
        else:
            delta0 = (F1 - F0) / h0
            delta1 = (F2 - F1) / h1

            a = (delta1 - delta0) / (h1 + h0)
            b = a * h1 + delta1
            c = F2

            disc = sqrt(b * b - 4 * a * c)
            denom_plus = b + disc
            denom_minus = b - disc
            denom = denom_plus if abs(denom_plus) > abs(denom_minus) else denom_minus

            if abs(denom) < min_sep * max(1.0, abs(b), abs(disc)):
                if abs(F2 - F1) > min_sep * max(1.0, abs(F2), abs(F1)):
                    dz = -F2 * (z2 - z1) / (F2 - F1)
                else:
                    dz = min_sep * scale * (1.0 + 1.0j)
            else:
                dz = -2 * c / denom

        if max_step is not None and abs(dz) > max_step:
            dz = dz / abs(dz) * max_step

        z3 = z2 + dz
        last_dz = dz

        if n >= min_iter and abs(F2) < tol and abs(dz) < step_tol * max(1.0, abs(z3)):
            if verbose:
                print("Muller converged.")
                print("Final |A_in/A_out| =", abs(F2))
                print("Final |delta z| =", abs(dz))
            return z3

        z0, z1, z2 = z1, z2, z3

    raise RuntimeError("Muller method did not converge.")


def find_QNM_muller(
    omega_ini,
    epsilon,
    delta=None,
    tol=1e-10,
    step_tol=None,
    max_iter=50,
    min_iter=2,
    r_m=None,
    rho_m=0,
    rho_min=None,
    rho_max=None,
    n_steps_min=40000,
    n_steps_max=None,
    points_per_period=150,
    qnm_cutoff=150,
    bad_branch_penalty=1e100,
    verbose=False,
    print_termination=False,
    max_step=None,
):
    omega_ini = complex(omega_ini)

    if delta is None:
        delta = 5e-4 * max(1.0, abs(omega_ini))
    if max_step is None:
        max_step = 20 * abs(delta)
    if step_tol is None:
        step_tol = 10 * tol

    omega0 = omega_ini
    omega1 = omega_ini + delta
    omega2 = omega_ini + 1j * delta

    def F(omega):
        Ain_Aout, _ = compute_Ain_Aout_and_zeta(
            omega=omega,
            epsilon=epsilon,
            r_m=r_m,
            rho_m=rho_m,
            rho_min=rho_min,
            rho_max=rho_max,
            n_steps_min=n_steps_min,
            n_steps_max=n_steps_max,
            points_per_period=points_per_period,
            qnm_cutoff=qnm_cutoff,
            bad_branch_penalty=bad_branch_penalty,
            print_termination=print_termination,
        )
        return Ain_Aout

    omega_qnm = muller_root_complex(
        F,
        omega0,
        omega1,
        omega2,
        tol=tol,
        step_tol=step_tol,
        max_iter=max_iter,
        min_iter=min_iter,
        verbose=verbose,
        max_step=max_step,
    )

    print("Muller found omega_qnm =", omega_qnm)

    Ain_Aout_qnm, zeta_qnm = compute_Ain_Aout_and_zeta(
        omega=omega_qnm,
        epsilon=epsilon,
        r_m=r_m,
        rho_m=rho_m,
        rho_min=rho_min,
        rho_max=rho_max,
        n_steps_min=n_steps_min,
        n_steps_max=n_steps_max,
        points_per_period=points_per_period,
        qnm_cutoff=qnm_cutoff,
        bad_branch_penalty=bad_branch_penalty,
        print_termination=print_termination,
    )

    print("A_in/A_out_qnm =", Ain_Aout_qnm)
    print("zeta =", zeta_qnm)
    print("|A_in/A_out_qnm| =", abs(Ain_Aout_qnm))

    return omega_qnm, Ain_Aout_qnm, zeta_qnm


def find_QNM_list_muller(
    epsilon_list,
    omega_ini,
    delta=None,
    tol=1e-10,
    max_iter=50,
    r_m=None,
    rho_m=0,
    rho_min=None,
    rho_max=None,
    n_steps_min=40000,
    n_steps_max=None,
    points_per_period=150,
    qnm_cutoff=150,
    bad_branch_penalty=1e100,
    verbose=False,
    print_termination=False,
    max_step=None,
    use_previous_as_initial=True,
):
    omega_qnm_list = []
    omega_guess = complex(omega_ini)

    for eps in epsilon_list:
        print("Finding QNM for epsilon =", eps)

        omega_qnm, ratio, zeta = find_QNM_muller(
            omega_ini=omega_guess,
            epsilon=eps,
            delta=delta,
            tol=tol,
            max_iter=max_iter,
            r_m=r_m,
            rho_m=rho_m,
            rho_min=rho_min,
            rho_max=rho_max,
            n_steps_min=n_steps_min,
            n_steps_max=n_steps_max,
            points_per_period=points_per_period,
            qnm_cutoff=qnm_cutoff,
            bad_branch_penalty=bad_branch_penalty,
            verbose=verbose,
            print_termination=print_termination,
            max_step=max_step,
        )

        omega_qnm_list.append(omega_qnm)

        print("omega_qnm =", omega_qnm)
        print("A_in/A_out =", ratio)
        print("zeta =", zeta)
        print()

        if use_previous_as_initial:
            omega_guess = omega_qnm

    return omega_qnm_list


def dQ_minus_domega(r, omega, epsilon, h=1e-6):
    return (Q_minus(r, omega + h, epsilon) - Q_minus(r, omega - h, epsilon)) / (2 * h)


def integrate_PLminus_OmegaLminus_RK4(
    omega,
    epsilon,
    rho_m=0,
    rho_min=None,
    r_m=None,
    n_steps=40000,
    domega_step=1e-6,
):
    omega = complex(omega)

    if r_m is None:
        r_m = 3 * M
    if rho_min is None:
        rho_min = -50 * M

    beta = -phase(omega)
    phase_factor = exp(1j * beta)

    r_min = find_rmin_from_rm_RK4(
        epsilon=epsilon,
        omega=omega,
        rm=r_m,
        rhom=rho_m,
        rho_min=rho_min,
        n_steps=n_steps,
    )

    sqrtQ_min = sqrt(Q_minus(r_min, omega, epsilon))
    if (omega.conjugate() * sqrtQ_min).real > 0:
        sqrtQ_min = -sqrtQ_min

    PLminus = 1j * sqrtQ_min
    print("PL_initial =", PLminus)

    dQdw_min = dQ_minus_domega(r_min, omega, epsilon, h=domega_step)
    OmegaLminus = -dQdw_min / (2 * PLminus)

    r = complex(r_min)
    h = (rho_m - rho_min) / n_steps

    def rhs(r_value, PLminus_value, OmegaLminus_value):
        Q_val = Q_minus(r_value, omega, epsilon)
        dQdw = dQ_minus_domega(r_value, omega, epsilon, h=domega_step)
        dPL_drho = -phase_factor * (PLminus_value**2 + Q_val)
        dOmega_drho = -phase_factor * (2 * PLminus_value * OmegaLminus_value + dQdw)
        dr_drho = phase_factor / dx_dr(r_value, epsilon)
        return dPL_drho, dOmega_drho, dr_drho

    for _ in range(n_steps):
        k1P, k1O, k1r = rhs(r, PLminus, OmegaLminus)
        k2P, k2O, k2r = rhs(r + 0.5 * h * k1r, PLminus + 0.5 * h * k1P, OmegaLminus + 0.5 * h * k1O)
        k3P, k3O, k3r = rhs(r + 0.5 * h * k2r, PLminus + 0.5 * h * k2P, OmegaLminus + 0.5 * h * k2O)
        k4P, k4O, k4r = rhs(r + h * k3r, PLminus + h * k3P, OmegaLminus + h * k3O)

        PLminus += h * (k1P + 2 * k2P + 2 * k3P + k4P) / 6
        OmegaLminus += h * (k1O + 2 * k2O + 2 * k3O + k4O) / 6
        r += h * (k1r + 2 * k2r + 2 * k3r + k4r) / 6

    return r, PLminus, OmegaLminus


def cot(z):
    return cos(z) / sin(z)


def integrate_Ptilde_phi_Omega_r_RK4(
    omega,
    epsilon,
    label,
    rho_m=0,
    rho_max=None,
    r_m=None,
    n_steps=50000,
    domega_step=1e-6,
):
    omega = complex(omega)

    if r_m is None:
        r_m = 3 * M
    if rho_max is None:
        rho_max = 2 * 10**4 * M

    beta = -phase(omega)
    phase_factor = exp(1j * beta)

    r_max = find_rmin_from_rm_RK4(
        epsilon=epsilon,
        omega=omega,
        rm=r_m,
        rhom=rho_m,
        rho_min=rho_max,
        n_steps=n_steps,
    )

    x_m = x(r_m, epsilon)
    x_max = x_m + rho_max * phase_factor

    sqrtQ_max = sqrt(Q_minus(r_max, omega, epsilon))
    if (omega.conjugate() * sqrtQ_max).real < 0:
        sqrtQ_max = -sqrtQ_max

    if label == "+":
        P_R = 1j * sqrtQ_max
    elif label == "-":
        P_R = -1j * sqrtQ_max
    else:
        raise ValueError("label must be '+' or '-'")

    print("boundary condition=", label, "P_R =", P_R)

    dQdw_max = dQ_minus_domega(r_max, omega, epsilon, h=domega_step)
    Omega = -dQdw_max / (2 * P_R)
    phi = 0.0 + 0.0j
    Ptilde = P_to_Prufer(P_R, x_max, omega)

    r = complex(r_max)
    rho = rho_max
    h = (rho_m - rho_max) / n_steps

    def rhs(rho_value, r_value, Ptilde_value, phi_value, Omega_value):
        x_rho = x_m + rho_value * phase_factor
        arg = omega * x_rho + Ptilde_value
        Q_val = Q_minus(r_value, omega, epsilon)
        dQdw = dQ_minus_domega(r_value, omega, epsilon, h=domega_step)

        dPtilde_drho = -phase_factor * (omega - Q_val / omega) * sin(arg) ** 2
        dphi_drho = phase_factor * omega * cot(arg)
        dOmega_drho = -phase_factor * (2 * Omega_value * omega * cot(arg) + dQdw)
        dr_drho = phase_factor / dx_dr(r_value, epsilon)
        return dPtilde_drho, dphi_drho, dOmega_drho, dr_drho

    for _ in range(n_steps):
        k1P, k1phi, k1O, k1r = rhs(rho, r, Ptilde, phi, Omega)
        k2P, k2phi, k2O, k2r = rhs(
            rho + 0.5 * h,
            r + 0.5 * h * k1r,
            Ptilde + 0.5 * h * k1P,
            phi + 0.5 * h * k1phi,
            Omega + 0.5 * h * k1O,
        )
        k3P, k3phi, k3O, k3r = rhs(
            rho + 0.5 * h,
            r + 0.5 * h * k2r,
            Ptilde + 0.5 * h * k2P,
            phi + 0.5 * h * k2phi,
            Omega + 0.5 * h * k2O,
        )
        k4P, k4phi, k4O, k4r = rhs(rho + h, r + h * k3r, Ptilde + h * k3P, phi + h * k3phi, Omega + h * k3O)

        Ptilde += h * (k1P + 2 * k2P + 2 * k3P + k4P) / 6
        phi += h * (k1phi + 2 * k2phi + 2 * k3phi + k4phi) / 6
        Omega += h * (k1O + 2 * k2O + 2 * k3O + k4O) / 6
        r += h * (k1r + 2 * k2r + 2 * k3r + k4r) / 6
        rho += h

    return Ptilde, phi, Omega, r


def track_sqrt(z_1, z_2):
    if (z_1.conjugate() * z_2).real < 0:
        return -z_1
    return z_1


def Phi_of_omega(
    omega,
    epsilon,
    x_max,
    r_max,
    phiR_plus,
    phiR_minus,
    rho_max,
    steps_per_period=150,
    rho_end=None,
    n_steps=None,
):
    omega = complex(omega)
    x_max = complex(x_max)
    r = complex(r_max)

    phiR_plus = complex(phiR_plus)
    phiR_minus = complex(phiR_minus)

    beta = -phase(omega)
    phase_factor = exp(1j * beta)

    if rho_end is None:
        rho_end = rho_max + 2 * 10**4 * M
    if n_steps is None:
        n_steps = int((rho_end - rho_max) * steps_per_period / (2 * math.pi))

    h = (rho_end - rho_max) / n_steps
    rho = rho_max
    integral = 0.0 + 0.0j
    
    sqrtQ_max_last = track_sqrt(sqrt(Q_minus(r, omega, epsilon)), omega)

    def rhs(r_value):
        Q_val = Q_minus(r_value, omega, epsilon)
        sqrtQ_val = track_sqrt(sqrt(Q_val), sqrtQ_max_last)
        
        integrand = (
            (Q_val - omega**2)
            / (sqrtQ_val + omega)
        )

        dI_drho = integrand * phase_factor
        dr_drho = phase_factor / dx_dr(r_value, epsilon)

        return dI_drho, dr_drho

    for _ in range(n_steps):
        k1I, k1r = rhs(r)
        k2I, k2r = rhs(
            r + 0.5 * h * k1r
        )
        k3I, k3r = rhs(
            r + 0.5 * h * k2r
        )
        k4I, k4r = rhs(
            r + h * k3r
        )

        integral += h * (k1I + 2 * k2I + 2 * k3I + k4I) / 6
        r += h * (k1r + 2 * k2r + 2 * k3r + k4r) / 6
        
        sqrtQ_max_last = track_sqrt(sqrt(Q_minus(r, omega, epsilon)), sqrtQ_max_last)
        rho += h

    return (
        2j * omega * x_max
        + phiR_plus
        - phiR_minus
        - 2j * integral
    )


def calculate_Bn(
    omega_n,
    epsilon,
    r_m=None,
    rho_m=0,
    rho_min=None,
    rho_max=None,
    rho_end=None,
    n_steps_inner=80000,
    n_steps_outer=300000,
    n_steps_tail=None,
    label_R_plus="+",
    label_R_minus="-",
    domega_step=1e-6,
    steps_per_period=150,
    verbose=False,
):
    omega_n = complex(omega_n)

    if r_m is None:
        r_m = 3 * M
    if rho_min is None:
        rho_min = -40 * M
    if rho_max is None:
        rho_max = 2 * 10**4 * M
    if rho_end is None:
        rho_end = rho_max + 2 * 10**4 * M

    beta = -phase(omega_n)
    phase_factor = exp(1j * beta)
    x_m = x(r_m, epsilon)

    r_min = find_rmin_from_rm_RK4(epsilon=epsilon, omega=omega_n, rm=r_m, rhom=rho_m, rho_min=rho_min, n_steps=n_steps_inner)
    r_max = find_rmin_from_rm_RK4(epsilon=epsilon, omega=omega_n, rm=r_m, rhom=rho_m, rho_min=rho_max, n_steps=n_steps_outer)
    x_max = x_m + rho_max * phase_factor

    r_m_check, PLminus_m, OmegaL_minus_m = integrate_PLminus_OmegaLminus_RK4(
        omega=omega_n,
        epsilon=epsilon,
        rho_m=rho_m,
        rho_min=rho_min,
        r_m=r_m,
        n_steps=n_steps_inner,
        domega_step=domega_step,
    )

    Ptilde_Rplus_m, phiR_plus_m, OmegaR_plus_m, r_Rplus_m = integrate_Ptilde_phi_Omega_r_RK4(
        omega=omega_n,
        epsilon=epsilon,
        label=label_R_plus,
        rho_m=rho_m,
        rho_max=rho_max,
        r_m=r_m,
        n_steps=n_steps_outer,
        domega_step=domega_step,
    )
    Ptilde_Rminus_m, phiR_minus_m, OmegaR_minus_m, r_Rminus_m = integrate_Ptilde_phi_Omega_r_RK4(
        omega=omega_n,
        epsilon=epsilon,
        label=label_R_minus,
        rho_m=rho_m,
        rho_max=rho_max,
        r_m=r_m,
        n_steps=n_steps_outer,
        domega_step=domega_step,
    )

    PRplus_m = Prufer_to_P(Ptilde_Rplus_m, x_m, omega_n)
    PRminus_m = Prufer_to_P(Ptilde_Rminus_m, x_m, omega_n)

    Phi_n = Phi_of_omega(
        omega=omega_n,
        epsilon=epsilon,
        x_max=x_max,
        r_max=r_max,
        phiR_plus=phiR_plus_m,
        phiR_minus=phiR_minus_m,
        rho_max=rho_max,
        rho_end=rho_end,
        steps_per_period=steps_per_period,
        n_steps=n_steps_tail,
    )

    alpha_over_Aout = (OmegaR_plus_m - OmegaL_minus_m) / (PLminus_m - PRminus_m) * exp(Phi_n)
    B_n = 1 / (2 * omega_n) / alpha_over_Aout

    if verbose:
        print("beta =", beta)
        print("r_min =", r_min)
        print("r_max =", r_max)
        print("r_m_check_left =", r_m_check)
        print("r_m_check_right =", r_Rplus_m)
        print("x_m =", x_m)
        print("x_max =", x_max)
        print("P_L,-(rho_m) =", PLminus_m)
        print("Omega_L,-(rho_m) =", OmegaL_minus_m)
        print("Ptilde_R,+(rho_m) =", Ptilde_Rplus_m)
        print("Ptilde_R,-(rho_m) =", Ptilde_Rminus_m)
        print("P_R,+(rho_m) =", PRplus_m)
        print("P_R,-(rho_m) =", PRminus_m)
        print("phi_R,+(rho_m) =", phiR_plus_m)
        print("phi_R,-(rho_m) =", phiR_minus_m)
        print("Omega_R,+(rho_m) =", OmegaR_plus_m)
        print("Omega_R,-(rho_m) =", OmegaR_minus_m)
        print("Phi_n =", Phi_n)
        print("alpha_n/A_out =", alpha_over_Aout)
        print("B_n =", B_n)

    return {
        "B_n": B_n,
        "alpha_over_Aout": alpha_over_Aout,
        "Phi_n": Phi_n,
        "omega_n": omega_n,
        "epsilon": epsilon,
        "beta": beta,
        "r_min": r_min,
        "r_max": r_max,
        "x_m": x_m,
        "x_max": x_max,
        "PLminus_m": PLminus_m,
        "OmegaLminus_m": OmegaL_minus_m,
        "Ptilde_Rplus_m": Ptilde_Rplus_m,
        "Ptilde_Rminus_m": Ptilde_Rminus_m,
        "PRplus_m": PRplus_m,
        "PRminus_m": PRminus_m,
        "phiR_plus_m": phiR_plus_m,
        "phiR_minus_m": phiR_minus_m,
        "OmegaR_plus_m": OmegaR_plus_m,
        "OmegaR_minus_m": OmegaR_minus_m,
        "r_Rplus_m": r_Rplus_m,
        "r_Rminus_m": r_Rminus_m,
    }


def calculate_Bn_list(
    omega_list,
    epsilon_list,
    r_m=None,
    rho_m=0,
    rho_min=None,
    rho_max=None,
    rho_end=None,
    n_steps_inner=80000,
    n_steps_outer=300000,
    n_steps_tail=None,
    label_R_plus="+",
    label_R_minus="-",
    domega_step=1e-6,
    steps_per_period=150,
    verbose=False,
):
    """
    Calculate excitation factors B_n for paired omega and epsilon lists.

    Parameters
    ----------
    omega_list, epsilon_list
        One-dimensional iterables of equal length. Each pair
        (omega_list[i], epsilon_list[i]) is passed to calculate_Bn.

    All remaining parameters are the same as for calculate_Bn.

    Returns
    -------
    list of dict
        A list whose entries are the dictionaries returned by calculate_Bn.
    """

    omega_values = list(omega_list)
    epsilon_values = list(epsilon_list)

    if len(omega_values) != len(epsilon_values):
        raise ValueError("omega_list and epsilon_list must have the same length.")

    if len(omega_values) == 0:
        raise ValueError("At least one (omega, epsilon) pair is required.")

    results = []

    for omega_n, epsilon in zip(omega_values, epsilon_values):
        print("Calculating B_n for omega_n =", omega_n, "epsilon =", epsilon)

        result = calculate_Bn(
            omega_n=omega_n,
            epsilon=epsilon,
            r_m=r_m,
            rho_m=rho_m,
            rho_min=rho_min,
            rho_max=rho_max,
            rho_end=rho_end,
            n_steps_inner=n_steps_inner,
            n_steps_outer=n_steps_outer,
            n_steps_tail=n_steps_tail,
            label_R_plus=label_R_plus,
            label_R_minus=label_R_minus,
            domega_step=domega_step,
            steps_per_period=steps_per_period,
            verbose=verbose,
        )

        results.append(result)

        print("B_n =", result["B_n"])
        print()

    return results


def _prepare_linear_fit_arrays(epsilon_list, omega_qnm_list_epsilon):
    epsilon_array = np.asarray(epsilon_list, dtype=float)
    omega_array = np.asarray(omega_qnm_list_epsilon, dtype=complex)

    if epsilon_array.ndim != 1 or omega_array.ndim != 1:
        raise ValueError("epsilon_list and omega_qnm_list_epsilon must be one-dimensional.")

    if len(epsilon_array) != len(omega_array):
        raise ValueError("epsilon_list and omega_qnm_list_epsilon must have the same length.")

    if len(epsilon_array) == 0:
        raise ValueError("At least one data point is required.")

    return epsilon_array, omega_array


def fit_qnm_linear_with_fixed_intercept(epsilon_list, omega_qnm_list_epsilon, omega_reference):
    """
    Fit

        omega(epsilon) = k * epsilon + omega_reference

    with omega_reference kept fixed.
    """

    epsilon_array, omega_array = _prepare_linear_fit_arrays(epsilon_list, omega_qnm_list_epsilon)

    denominator = np.sum(epsilon_array**2)
    if denominator == 0:
        raise ValueError("At least one epsilon must be nonzero to determine the slope.")

    omega_reference = complex(omega_reference)
    k = np.sum(epsilon_array * (omega_array - omega_reference)) / denominator
    fitted = k * epsilon_array + omega_reference
    residuals = omega_array - fitted
    rmse = float(np.sqrt(np.mean(np.abs(residuals) ** 2)))

    return {
        "k": k,
        "b": omega_reference,
        "fitted_values": fitted,
        "residuals": residuals,
        "rmse": rmse,
    }


def compute_delta_omega_estimates(epsilon_list, omega_qnm_list_epsilon, omega_reference):
    """
    Compute pointwise estimates

        delta_Omega_i = (omega_i - omega_reference) / epsilon_i

    assuming the intercept is fixed to omega_reference.
    """

    epsilon_array, omega_array = _prepare_linear_fit_arrays(epsilon_list, omega_qnm_list_epsilon)

    if np.any(np.isclose(epsilon_array, 0.0)):
        raise ValueError("All epsilon values must be nonzero to compute pointwise delta_Omega estimates.")

    omega_reference = complex(omega_reference)
    delta_omega_array = (omega_array - omega_reference) / epsilon_array

    return {
        "epsilon": epsilon_array,
        "omega": omega_array,
        "delta_omega": delta_omega_array,
    }


def _complex_variance(values):
    values = np.asarray(values, dtype=complex)

    if len(values) == 0:
        raise ValueError("At least one value is required.")

    center = np.mean(values)
    variance = float(np.mean(np.abs(values - center) ** 2))
    return center, variance


def select_qnm_linear_fit_subset(
    epsilon_list,
    omega_qnm_list_epsilon,
    omega_reference,
    min_points=3,
    subset_mode="all",
    max_subsets=200000,
    sort_by_epsilon=True,
    intercept_tol=None,
    rmse_tol=None,
):
    """
    Select the subset of points that is most consistent with a linear fit

        omega_qnm(epsilon) = k * epsilon + b

    under the prior expectation that b should be close to omega_reference.

    Selection rule:
        1. Minimize |b_free - omega_reference|.
        2. Break ties using the RMSE of the fixed-intercept fit.
        3. Then use the RMSE of the free fit.
        4. Prefer larger subsets if the above are comparable.

    Parameters
    ----------
    epsilon_list, omega_qnm_list_epsilon
        One-dimensional arrays/lists of equal length.
    omega_reference
        Expected intercept, e.g. omega_qnm_2.
    min_points
        Minimum number of points used in each candidate subset.
    subset_mode
        "all" considers every subset with at least min_points points.
        "contiguous" only considers contiguous blocks after sorting by epsilon.
    max_subsets
        Safety cap for subset_mode="all".
    sort_by_epsilon
        If True, sort points by epsilon before generating candidates.
    intercept_tol, rmse_tol
        Absolute tolerances used to decide when two candidate fits are
        effectively tied. Within those tolerances, larger subsets are preferred.
    """

    epsilon_array, omega_array = _prepare_linear_fit_arrays(epsilon_list, omega_qnm_list_epsilon)
    omega_reference = complex(omega_reference)

    if sort_by_epsilon:
        order = np.argsort(epsilon_array)
        epsilon_array = epsilon_array[order]
        omega_array = omega_array[order]
        original_indices = order
    else:
        original_indices = np.arange(len(epsilon_array))

    num_points = len(epsilon_array)

    if min_points < 2:
        raise ValueError("min_points must be at least 2.")
    if min_points > num_points:
        raise ValueError("min_points cannot exceed the number of available points.")

    if subset_mode not in {"all", "contiguous"}:
        raise ValueError("subset_mode must be either 'all' or 'contiguous'.")

    if intercept_tol is None:
        intercept_tol = 1e-8 * max(1.0, abs(omega_reference))

    if rmse_tol is None:
        rmse_tol = 1e-8 * max(1.0, float(np.max(np.abs(omega_array))))

    candidate_index_sets = []

    if subset_mode == "all":
        total_subsets = sum(math.comb(num_points, count) for count in range(min_points, num_points + 1))
        if total_subsets > max_subsets:
            raise ValueError(
                f"Too many subsets to scan ({total_subsets}). "
                "Use subset_mode='contiguous', increase min_points, or raise max_subsets."
            )

        for count in range(min_points, num_points + 1):
            candidate_index_sets.extend(combinations(range(num_points), count))
    else:
        for start in range(num_points):
            for stop in range(start + min_points, num_points + 1):
                candidate_index_sets.append(tuple(range(start, stop)))

    candidates = []

    for indices in candidate_index_sets:
        local_indices = np.array(indices, dtype=int)
        epsilon_subset = epsilon_array[local_indices]
        omega_subset = omega_array[local_indices]

        k_free, b_free = np.polyfit(epsilon_subset, omega_subset, 1)
        fitted_free = k_free * epsilon_subset + b_free
        rmse_free = float(np.sqrt(np.mean(np.abs(omega_subset - fitted_free) ** 2)))

        fixed_fit = fit_qnm_linear_with_fixed_intercept(
            epsilon_subset,
            omega_subset,
            omega_reference,
        )

        candidate = {
            "selected_indices": original_indices[local_indices].tolist(),
            "epsilon_selected": epsilon_subset.tolist(),
            "omega_selected": omega_subset.tolist(),
            "count": len(local_indices),
            "k_free": k_free,
            "b_free": b_free,
            "intercept_error": float(abs(b_free - omega_reference)),
            "rmse_free": rmse_free,
            "k_fixed": fixed_fit["k"],
            "b_fixed": omega_reference,
            "rmse_fixed": fixed_fit["rmse"],
            "fitted_free": fitted_free,
            "fitted_fixed": fixed_fit["fitted_values"],
        }
        candidates.append(candidate)

    candidates.sort(
        key=lambda item: (
            item["intercept_error"],
            item["rmse_fixed"],
            item["rmse_free"],
            -item["count"],
        )
    )

    best_intercept = candidates[0]["intercept_error"]
    close_intercept = [
        item for item in candidates
        if item["intercept_error"] <= best_intercept + intercept_tol
    ]

    best_rmse_fixed = min(item["rmse_fixed"] for item in close_intercept)
    close_fixed_rmse = [
        item for item in close_intercept
        if item["rmse_fixed"] <= best_rmse_fixed + rmse_tol
    ]

    best_rmse_free = min(item["rmse_free"] for item in close_fixed_rmse)
    finalists = [
        item for item in close_fixed_rmse
        if item["rmse_free"] <= best_rmse_free + rmse_tol
    ]

    finalists.sort(
        key=lambda item: (
            -item["count"],
            item["intercept_error"],
            item["rmse_fixed"],
            item["rmse_free"],
        )
    )

    best = finalists[0]

    return {
        "selected_indices": best["selected_indices"],
        "epsilon_selected": best["epsilon_selected"],
        "omega_selected": best["omega_selected"],
        "k": best["k_fixed"],
        "b": best["b_fixed"],
        "k_fixed": best["k_fixed"],
        "b_fixed": best["b_fixed"],
        "k_free": best["k_free"],
        "b_free": best["b_free"],
        "intercept_error": best["intercept_error"],
        "rmse_fixed": best["rmse_fixed"],
        "rmse_free": best["rmse_free"],
        "fitted_free": best["fitted_free"],
        "fitted_fixed": best["fitted_fixed"],
        "intercept_tol": intercept_tol,
        "rmse_tol": rmse_tol,
        "candidates": candidates,
    }


def select_qnm_low_variance_subset(
    epsilon_list,
    omega_qnm_list_epsilon,
    omega_reference,
    min_points=3,
    max_removed=None,
    min_relative_variance_reduction=0.05,
    min_sigma_distance=1.5,
    sort_by_epsilon=True,
):
    """
    Iteratively remove points that contribute the most to the variance of

        delta_Omega_i = (omega_i - omega_reference) / epsilon_i.

    At each step, this function tests removing each remaining point and chooses
    the one that gives the largest variance reduction. The removal is accepted
    only if the reduction is significant enough.

    This is useful when a few problematic omega values spoil an otherwise
    stable estimate of delta_Omega.
    """

    estimate_data = compute_delta_omega_estimates(
        epsilon_list,
        omega_qnm_list_epsilon,
        omega_reference,
    )

    epsilon_array = estimate_data["epsilon"]
    omega_array = estimate_data["omega"]
    delta_omega_array = estimate_data["delta_omega"]
    omega_reference = complex(omega_reference)

    if sort_by_epsilon:
        order = np.argsort(epsilon_array)
        epsilon_array = epsilon_array[order]
        omega_array = omega_array[order]
        delta_omega_array = delta_omega_array[order]
        original_indices = order
    else:
        original_indices = np.arange(len(epsilon_array))

    if min_points < 2:
        raise ValueError("min_points must be at least 2.")

    if min_points > len(epsilon_array):
        raise ValueError("min_points cannot exceed the number of available points.")

    if max_removed is None:
        max_removed = len(epsilon_array) - min_points
    else:
        max_removed = min(max_removed, len(epsilon_array) - min_points)

    kept_mask = np.ones(len(epsilon_array), dtype=bool)
    removal_history = []

    while kept_mask.sum() > min_points and len(removal_history) < max_removed:
        kept_indices = np.flatnonzero(kept_mask)
        delta_kept = delta_omega_array[kept_indices]

        current_center, current_variance = _complex_variance(delta_kept)
        current_sigma = math.sqrt(current_variance)

        best_candidate = None

        for local_position, point_index in enumerate(kept_indices):
            trial_mask = kept_mask.copy()
            trial_mask[point_index] = False
            trial_indices = np.flatnonzero(trial_mask)
            trial_delta = delta_omega_array[trial_indices]

            trial_center, trial_variance = _complex_variance(trial_delta)
            variance_reduction = current_variance - trial_variance

            point_distance = abs(delta_omega_array[point_index] - current_center)
            sigma_distance = point_distance / current_sigma if current_sigma > 0 else 0.0
            relative_reduction = variance_reduction / current_variance if current_variance > 0 else 0.0

            candidate = {
                "point_index_sorted": int(point_index),
                "point_index_original": int(original_indices[point_index]),
                "epsilon_removed": float(epsilon_array[point_index]),
                "omega_removed": omega_array[point_index],
                "delta_omega_removed": delta_omega_array[point_index],
                "point_distance": float(point_distance),
                "sigma_distance": float(sigma_distance),
                "variance_before": current_variance,
                "variance_after": trial_variance,
                "variance_reduction": variance_reduction,
                "relative_variance_reduction": relative_reduction,
                "delta_omega_center_before": current_center,
                "delta_omega_center_after": trial_center,
            }

            if best_candidate is None or candidate["variance_reduction"] > best_candidate["variance_reduction"]:
                best_candidate = candidate

        if best_candidate is None:
            break

        if (
            best_candidate["relative_variance_reduction"] < min_relative_variance_reduction
            and best_candidate["sigma_distance"] < min_sigma_distance
        ):
            break

        kept_mask[best_candidate["point_index_sorted"]] = False
        removal_history.append(best_candidate)

    kept_indices = np.flatnonzero(kept_mask)
    epsilon_selected = epsilon_array[kept_indices]
    omega_selected = omega_array[kept_indices]
    delta_selected = delta_omega_array[kept_indices]

    delta_center, delta_variance = _complex_variance(delta_selected)
    fixed_fit = fit_qnm_linear_with_fixed_intercept(
        epsilon_selected,
        omega_selected,
        omega_reference,
    )

    removed_indices = np.flatnonzero(~kept_mask)

    return {
        "selected_indices": original_indices[kept_indices].tolist(),
        "removed_indices": original_indices[removed_indices].tolist(),
        "epsilon_selected": epsilon_selected.tolist(),
        "omega_selected": omega_selected.tolist(),
        "delta_omega_selected": delta_selected.tolist(),
        "delta_omega_mean": delta_center,
        "delta_omega_variance": delta_variance,
        "delta_omega_std": math.sqrt(delta_variance),
        "k": fixed_fit["k"],
        "b": omega_reference,
        "rmse_fixed": fixed_fit["rmse"],
        "fitted_fixed": fixed_fit["fitted_values"],
        "residuals_fixed": fixed_fit["residuals"],
        "removal_history": removal_history,
        "settings": {
            "min_points": min_points,
            "max_removed": max_removed,
            "min_relative_variance_reduction": min_relative_variance_reduction,
            "min_sigma_distance": min_sigma_distance,
            "sort_by_epsilon": sort_by_epsilon,
        },
    }


__all__ = [
    "BadBranchReached",
    "M",
    "Phi_of_omega",
    "Prufer_to_P",
    "P_to_Prufer",
    "QNMReached",
    "Q_minus",
    "VGR_minus",
    "V_minus",
    "calculate_Bn",
    "calculate_Bn_list",
    "choose_n_steps_outer",
    "compute_delta_omega_estimates",
    "compute_Ain_Aout_and_zeta",
    "cot",
    "dQ_minus_domega",
    "delta_V",
    "dr_dx",
    "dx_dr",
    "exp",
    "fit_qnm_linear_with_fixed_intercept",
    "find_QNM_list_muller",
    "find_QNM_muller",
    "find_rmin_from_rm_RK4",
    "get_parameters",
    "integrate_PLminus_OmegaLminus_RK4",
    "integrate_P_r_RK4",
    "integrate_Prufer_r_RK4",
    "integrate_Ptilde_phi_Omega_r_RK4",
    "l",
    "muller_root_complex",
    "phase",
    "s",
    "set_parameters",
    "select_qnm_low_variance_subset",
    "select_qnm_linear_fit_subset",
    "sin",
    "sqrt",
    "track_sqrt",
    "v_minus",
    "x",
]

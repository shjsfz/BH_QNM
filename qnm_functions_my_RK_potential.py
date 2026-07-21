import cmath
import importlib.util
from pathlib import Path
from typing import Any


_module_path = Path(__file__).with_name("qnm_functions.py")
_spec = importlib.util.spec_from_file_location("_qnm_functions_base_rk", _module_path)
if _spec is None or _spec.loader is None:
    raise ImportError(f"Could not load base module from {_module_path}")

_base: Any = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_base)


def _sync_parameters():
    global M, l, s
    M = _base.M
    l = _base.l
    s = _base.s


def set_parameters(M_value=None, l_value=None, s_value=None):
    result = _base.set_parameters(M_value=M_value, l_value=l_value, s_value=s_value)
    _sync_parameters()
    return result


def get_parameters():
    _sync_parameters()
    return _base.get_parameters()


def delta_V(r, omega):  # RK
    M_value = _base.M
    l_value = _base.l

    return (
        -3 * M_value**2
        / (2 * r**10)
        * (
            -3840 * M_value**6
            - 16
            * (-317 + 22 * l_value * (1 + l_value))
            * M_value**5
            * r
            + 8
            * M_value**4
            * r**2
            * (
                -209
                + 24 * l_value * (1 + l_value)
                - 16 * r**2 * omega**2
            )
            + 2
            * M_value
            * r**5
            * (5 + 2 * r**2 * omega**2)
            + r**6
            * (
                3
                - 2 * l_value * (1 + l_value)
                + 2 * r**2 * omega**2
            )
            + 4
            * M_value**3
            * (r**3 + 4 * r**5 * omega**2)
            + 2
            * M_value**2
            * (r**4 + 4 * r**6 * omega**2)
        )
    )


def V_minus(r, epsilon, omega):  # RW potential of beyond GR with RK correction
    return (1 - 2 * _base.M / r) * (_base.VGR_minus(r)) + epsilon * delta_V(r, omega)


def Q_minus(r, omega, epsilon):
    return omega**2 - V_minus(r, epsilon, omega)


def x(r, epsilon):
    M_value = _base.M
    rh = 2*M_value-epsilon*3/4*M_value
    return r + 2 * M_value * cmath.log(r/rh-1)


def dx_dr(r, epsilon):
    M_value = _base.M
    rh = 2*M_value-epsilon*3/4*M_value
    return (1+2*M_value/(r-rh))


def dr_dx(r, epsilon):
    return 1 / dx_dr(r, epsilon)


setattr(_base, "delta_V", delta_V)
setattr(_base, "V_minus", V_minus)
setattr(_base, "Q_minus", Q_minus)
setattr(_base, "x", x)
setattr(_base, "dx_dr", dx_dr)
setattr(_base, "dr_dx", dr_dx)


for _name in _base.__all__:
    if _name in {"M", "l", "s", "set_parameters", "get_parameters", "delta_V", "V_minus", "Q_minus"}:
        continue
    globals()[_name] = getattr(_base, _name)


_sync_parameters()


__all__ = list(_base.__all__)

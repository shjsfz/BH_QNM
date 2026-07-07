import importlib.util
from pathlib import Path
from typing import Any


_module_path = Path(__file__).with_name("qnm_functions.py")
_spec = importlib.util.spec_from_file_location("_qnm_functions_base_r3", _module_path)
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


def delta_V(r, omega):  # Riemann cubic
    M_value = _base.M
    l_value = _base.l
    L = l_value * (l_value + 1)

    num = (
        -34804 * M_value**3
        - 4 * (-13091 + 263 * L) * M_value**2 * r
        - 5 * (l_value - 2) * (l_value + 3) * (156 + 7 * L) * r**3
        - 8 * (-87 + L) * r**5 * omega**2
        + M_value * r**2 * (
            L * (1727 + 83 * L)
            - 6 * (4489 + 160 * r**2 * omega**2)
        )
    )

    return -24 * M_value * num / ((L - 2) * r**10)


def V_minus(r, epsilon, omega):  # RW potential of beyond GR with R3 correction
    return (1 - 2 * _base.M / r) * (_base.VGR_minus(r) + epsilon * delta_V(r, omega))


def Q_minus(r, omega, epsilon):
    return omega**2 - V_minus(r, epsilon, omega)


setattr(_base, "delta_V", delta_V)
setattr(_base, "V_minus", V_minus)
setattr(_base, "Q_minus", Q_minus)


for _name in _base.__all__:
    if _name in {"M", "l", "s", "set_parameters", "get_parameters", "delta_V", "V_minus", "Q_minus"}:
        continue
    globals()[_name] = getattr(_base, _name)


_sync_parameters()


__all__ = list(_base.__all__)

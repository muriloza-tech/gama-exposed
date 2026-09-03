"""Distribuicao normal padrao, vetorizada, sem scipy.

Implementado aqui de proposito: remove uma dependencia pesada e elimina o
risco de wheel indisponivel em versoes novas do Python. `math.erf` e exato
ate a precisao de double, entao nao ha perda numerica em relacao ao scipy.
"""

from __future__ import annotations

import math

import numpy as np
from numpy.typing import ArrayLike, NDArray

_INV_SQRT_2PI = 1.0 / math.sqrt(2.0 * math.pi)
_INV_SQRT_2 = 1.0 / math.sqrt(2.0)

# math.erf e escalar; frompyfunc o aplica elemento a elemento. Para as
# centenas de strikes de uma serie o custo e irrelevante.
_erf_vec = np.frompyfunc(math.erf, 1, 1)


def norm_pdf(x: ArrayLike) -> NDArray[np.float64]:
    """Densidade da normal padrao."""
    x = np.asarray(x, dtype=np.float64)
    return _INV_SQRT_2PI * np.exp(-0.5 * x * x)


def norm_cdf(x: ArrayLike) -> NDArray[np.float64]:
    """Acumulada da normal padrao."""
    x = np.asarray(x, dtype=np.float64)
    # frompyfunc devolve dtype=object para arrays e um escalar Python para
    # entrada 0-d; np.asarray normaliza os dois casos para float64.
    erf = np.asarray(_erf_vec(x * _INV_SQRT_2), dtype=np.float64)
    return 0.5 * (1.0 + erf)

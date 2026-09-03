"""Dias de demonstracao para o backtest -- SINTETICOS.

Serve para ver a forma do relatorio e conferir que o pipeline fecha ponta a
ponta antes de haver dado real. Os precos sao passeio aleatorio, portanto o
resultado esperado e "nenhum efeito significativo". Se o relatorio de demo
mostrasse edge, seria bug no motor.
"""

from __future__ import annotations

from datetime import date, datetime, timedelta

import numpy as np
import pandas as pd

from ..data.sources.sintetica import FonteSintetica
from ..model.profile import construir_perfil
from .engine import DiaBacktest, avaliador_de_perfil


def dias_demo(
    n_dias: int = 40,
    *,
    barras_por_dia: int = 78,
    spot_win: float = 185_000.0,
    passo_pts: float = 25.0,
    semente: int = 42,
) -> list[DiaBacktest]:
    """Gera dias com perfil de gama sintetico e precos em passeio aleatorio.

    O perfil de cada dia e recalculado a partir do spot de ABERTURA daquele
    dia, imitando o fluxo real: perfil montado antes do pregao, precos
    intradiarios avaliados contra ele.
    """
    rng = np.random.default_rng(semente)
    dias: list[DiaBacktest] = []
    d = date(2026, 3, 2)
    spot = spot_win

    while len(dias) < n_dias:
        if d.weekday() < 5:
            # Perfil do dia, a partir do subjacente equivalente ao spot WIN.
            ratio = 1_000.0
            fonte = FonteSintetica(
                spot=spot / ratio,
                passo_strike=1.0,
                n_strikes_cada_lado=25,
                semente=int(rng.integers(1, 10**6)),
            )
            perfil = construir_perfil(fonte.buscar(d))
            avaliar = avaliador_de_perfil(perfil, ratio_win=ratio)

            passos = rng.choice([-1.0, 1.0], size=barras_por_dia) * passo_pts
            closes = spot + np.cumsum(passos)

            inicio = datetime(d.year, d.month, d.day, 10, 0)
            barras = pd.DataFrame(
                {
                    "ts": [
                        inicio + timedelta(minutes=5 * i)
                        for i in range(barras_por_dia)
                    ],
                    "close": closes,
                }
            )
            dias.append(DiaBacktest(data=d, barras=barras, avaliar_gama=avaliar))
            spot = float(closes[-1])
        d += timedelta(days=1)

    return dias

"""Relatorio do backtest.

A hierarquia visual e deliberada: a DIFERENCA contra o baseline vem antes da
taxa absoluta, porque e ela que responde se o gama informa algo. Taxa de
continuacao sozinha em destaque convida a conclusao errada.
"""

from __future__ import annotations

import numpy as np

from gama_win.backtest.engine import ResultadoBacktest, ResultadoGrupo
from .relatorio import num

LARG = 78


def _pct(v: float, decimais: int = 1) -> str:
    if not np.isfinite(v):
        return "n/d"
    return f"{num(v * 100, decimais)}%"


def _pts(v: float) -> str:
    if not np.isfinite(v):
        return "n/d"
    sinal = "+" if v > 0 else ""
    return f"{sinal}{num(v, 1)} pts"


def _linha_grupo(g: ResultadoGrupo, baseline: bool = False) -> list[str]:
    c = g.continuacao
    ic = (
        f"[{_pct(c.ic_inferior)} a {_pct(c.ic_superior)}]"
        if c.ic_contem
        else "[IC indisponivel]"
    )

    linhas = [
        f"  {g.rotulo}",
        f"     continuacao ..... {_pct(c.taxa)} {ic}  n={c.n} em {c.n_blocos} dia(s)",
    ]

    if not baseline:
        lo, hi = g.dif_ic
        marca = "SIGNIFICATIVA" if g.dif_e_significativa else "nao significativa"
        ic_dif = (
            f"[{_pct(lo)} a {_pct(hi)}]"
            if np.isfinite(lo) and np.isfinite(hi)
            else "[IC indisponivel]"
        )
        linhas.append(
            f"     vs baseline ..... {_pct(g.dif_vs_baseline)} {ic_dif}  -> {marca}"
        )

    d = g.retorno_direcional
    lo_r, hi_r = g.ret_ic
    ic_r = (
        f"[{_pts(lo_r)} a {_pts(hi_r)}]"
        if np.isfinite(lo_r) and np.isfinite(hi_r)
        else ""
    )
    linhas.append(
        f"     retorno medio ... {_pts(d.media)} {ic_r}".rstrip()
    )
    linhas.append(
        f"     distribuicao .... p10 {_pts(d.p10)} | mediana {_pts(d.mediana)} "
        f"| p90 {_pts(d.p90)}"
    )
    return linhas


def relatorio_backtest(r: ResultadoBacktest) -> str:
    cfg = r.config
    linhas: list[str] = [
        "=" * LARG,
        "  BACKTEST :: O GAMA INFORMA ALGO ALEM DA DIRECAO RECENTE?",
        "=" * LARG,
        f"  Periodo ........ {r.periodo[0].isoformat()} a "
        f"{r.periodo[1].isoformat()}  ({r.n_dias} dia(s))",
        f"  Eventos ........ {num(r.n_eventos, 0)}",
        f"  Horizonte ...... {cfg.horizonte_barras} barra(s) a frente",
        f"  Lookback ....... {cfg.lookback_barras} barra(s) para a direcao",
        f"  Limiar mov. .... {num(cfg.limiar_movimento_pts, 1)} pts",
        f"  Bootstrap ...... {num(cfg.n_reamostras, 0)} reamostras de blocos "
        "de dia (IC 95%)",
        "",
        "  Continuacao = o movimento das ultimas barras prosseguiu no",
        "  horizonte. Retorno direcional positivo = continuou; negativo =",
        "  reverteu. 'vs baseline' e a diferenca contra a taxa incondicional",
        "  do MESMO periodo, com IC pareado -- se ela nao exclui zero, o",
        "  gama nao esta explicando nada alem de momentum.",
        "",
        "-" * LARG,
        "  BASELINE",
        "-" * LARG,
    ]
    linhas += _linha_grupo(r.baseline, baseline=True)

    if r.por_regime:
        linhas += ["", "-" * LARG, "  POR REGIME DE GAMA", "-" * LARG]
        for g in r.por_regime.values():
            linhas += _linha_grupo(g)
            linhas.append("")
        linhas.pop()

    if r.por_regime_e_direcao:
        linhas += ["", "-" * LARG, "  POR REGIME E DIRECAO", "-" * LARG]
        for g in r.por_regime_e_direcao.values():
            linhas += _linha_grupo(g)
            linhas.append("")
        linhas.pop()

    if any(r.descartados.values()):
        linhas += ["", "-" * LARG, "  DESCARTES", "-" * LARG]
        for motivo, n in r.descartados.items():
            if n:
                linhas.append(f"  {motivo.replace('_', ' ')} ... {num(n, 0)}")

    if r.avisos:
        linhas += ["", "-" * LARG, "  AVISOS", "-" * LARG]
        for a in r.avisos:
            linhas.append(f"  ! {a}")

    linhas.append("=" * LARG)
    return "\n".join(linhas)

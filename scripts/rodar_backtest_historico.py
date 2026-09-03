"""Backtest com historico real da B3 + intradiario do Ibovespa.

Regra anti-lookahead que estrutura o script: para avaliar o intradiario do
dia D, o perfil de gama vem do arquivo de D-1. O arquivo da B3 datado D
contem as posicoes do FECHAMENTO de D, que ninguem tinha durante o pregao de
D. Usar o proprio dia seria olhar o futuro.

Uso:
    python scripts/rodar_backtest_historico.py
"""

from __future__ import annotations

import json
import os
import sys
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ))

# Certificados do Windows, para o curl_cffi do yfinance.
_pem = RAIZ / "data" / "ca-bundle.pem"
if _pem.exists():
    for var in ("CURL_CA_BUNDLE", "SSL_CERT_FILE", "REQUESTS_CA_BUNDLE"):
        os.environ.setdefault(var, str(_pem))

import truststore  # noqa: E402

truststore.inject_into_ssl()

from gama_win.backtest.engine import (  # noqa: E402
    ConfigBacktest,
    DiaBacktest,
    avaliador_de_perfil,
    rodar_backtest,
)
from gama_win.data.historico import (  # noqa: E402
    HistoricoError,
    carregar_catalogo,
    chain_do_snapshot,
    construir_catalogo,
    construir_snapshot,
    datas_de_amostragem,
)
from gama_win.data.sources.b3_arquivos import B3Error  # noqa: E402
from gama_win.model.profile import construir_perfil  # noqa: E402
from gama_win.view.relatorio_backtest import relatorio_backtest  # noqa: E402

ATIVO = "BOVA11"
RATE = 0.105
DIV = 0.035
CACHE_BARRAS = RAIZ / "data" / "barras_ibov_5m.csv"


def log(msg: str) -> None:
    print(msg, flush=True)


# ------------------------------------------------------------- barras ---


def obter_barras() -> pd.DataFrame:
    """Barras de 5 minutos do Ibovespa, com cache em disco."""
    if CACHE_BARRAS.exists():
        df = pd.read_csv(CACHE_BARRAS, parse_dates=["ts"])
        log(f"barras em cache: {len(df)} linhas")
        return df

    import yfinance as yf

    h = yf.Ticker("^BVSP").history(period="60d", interval="5m")
    if h.empty:
        raise SystemExit("yfinance devolveu vazio para ^BVSP 5m")

    df = pd.DataFrame(
        {"ts": h.index.tz_localize(None), "close": h["Close"].to_numpy(float)}
    ).dropna()
    df["dia"] = df["ts"].dt.date
    CACHE_BARRAS.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(CACHE_BARRAS, index=False)
    log(f"barras baixadas: {len(df)} linhas, {df['dia'].nunique()} dias")
    return df


def dia_util_anterior(d: date, validos: set[date]) -> date | None:
    """Ultimo dia com snapshot disponivel antes de `d`."""
    c = d - timedelta(days=1)
    for _ in range(10):
        if c in validos:
            return c
        c -= timedelta(days=1)
    return None


# --------------------------------------------------------------- main ---


def main() -> int:
    barras = obter_barras()
    barras["dia"] = pd.to_datetime(barras["ts"]).dt.date
    dias_pregao = sorted(barras["dia"].unique())
    log(f"\ndias de pregao no intradiario: {len(dias_pregao)} "
        f"({dias_pregao[0]} a {dias_pregao[-1]})")

    # Snapshots necessarios: o dia anterior a cada dia de pregao.
    inicio = dias_pregao[0] - timedelta(days=6)
    fim = dias_pregao[-1]

    # -------------------------------------------------------- catalogo
    log("\n=== 1/3 catalogo de series (amostrado) ===")
    amostras = datas_de_amostragem(inicio, fim, passo_dias=10)
    log(f"amostrando cadastro em {len(amostras)} datas: "
        f"{[d.isoformat() for d in amostras]}")
    try:
        catalogo = carregar_catalogo()
        log(f"catalogo em cache: {len(catalogo)} series")
    except HistoricoError:
        catalogo = construir_catalogo(amostras, ativo=ATIVO, manter_brutos=True)
        log(f"catalogo construido: {len(catalogo)} series")

    # -------------------------------------------------------- snapshots
    log("\n=== 2/3 snapshots diarios ===")
    precisamos = sorted({d - timedelta(days=1) for d in dias_pregao} | set(dias_pregao))
    snaps: dict[date, object] = {}
    for i, d in enumerate(precisamos, 1):
        if not np.is_busday(np.datetime64(d, "D")):
            continue
        try:
            snaps[d] = construir_snapshot(d, catalogo, ativo=ATIVO)
            if i % 10 == 0 or i == len(precisamos):
                log(f"  [{i}/{len(precisamos)}] {d} ok "
                    f"(cobertura OI {snaps[d].cobertura_oi:.1%})")
        except (B3Error, HistoricoError) as exc:
            log(f"  [{i}/{len(precisamos)}] {d} PULADO: {str(exc)[:110]}")

    log(f"\nsnapshots obtidos: {len(snaps)}")
    if snaps:
        cobs = [s.cobertura_oi for s in snaps.values()]
        log(f"cobertura do OI pelo catalogo: min {min(cobs):.1%} | "
            f"mediana {float(np.median(cobs)):.1%} | max {max(cobs):.1%}")

    # ----------------------------------------------------- montar dias
    log("\n=== 3/3 montando dias do backtest ===")
    dias_bt: list[DiaBacktest] = []
    sem_snapshot = 0
    falhas = 0

    for d in dias_pregao:
        anterior = dia_util_anterior(d, set(snaps))
        if anterior is None:
            sem_snapshot += 1
            continue
        snap = snaps[anterior]
        try:
            chain = chain_do_snapshot(snap, rate=RATE, div_yield=DIV)
            perfil = construir_perfil(chain, vol_fallback=0.22)
            avaliar = avaliador_de_perfil(perfil, ratio_win=snap.ratio_win)
        except Exception as exc:
            falhas += 1
            log(f"  {d}: perfil de {anterior} falhou: {type(exc).__name__} "
                f"{str(exc)[:90]}")
            continue

        b = barras[barras["dia"] == d][["ts", "close"]].reset_index(drop=True)
        if len(b) < 20:
            continue
        dias_bt.append(DiaBacktest(data=d, barras=b, avaliar_gama=avaliar))

    log(f"dias montados: {len(dias_bt)} | sem snapshot: {sem_snapshot} | "
        f"falhas de perfil: {falhas}")

    if len(dias_bt) < 5:
        log("\nAmostra insuficiente para rodar o backtest.")
        return 1

    # ------------------------------------------------------- backtest
    for rot, cfg in (
        ("30 min a frente", ConfigBacktest(horizonte_barras=6, lookback_barras=3,
                                           limiar_movimento_pts=0.0, n_reamostras=2000)),
        ("60 min a frente", ConfigBacktest(horizonte_barras=12, lookback_barras=3,
                                           limiar_movimento_pts=0.0, n_reamostras=2000)),
        ("30 min, mov > 100 pts", ConfigBacktest(horizonte_barras=6, lookback_barras=3,
                                                 limiar_movimento_pts=100.0,
                                                 n_reamostras=2000)),
    ):
        log("\n" + "#" * 78)
        log(f"#  {rot}")
        log("#" * 78)
        r = rodar_backtest(dias_bt, cfg)
        log(relatorio_backtest(r))
        saida = RAIZ / "data" / f"backtest_{rot.replace(' ', '_').replace('>', 'gt')}.csv"
        r.eventos.to_csv(saida, index=False)
        log(f"eventos gravados em {saida.name}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

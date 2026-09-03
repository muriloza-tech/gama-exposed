"""Consolida os resultados do backtest num JSON que a pagina exibe.

Le os CSVs de eventos gravados por `rodar_backtest_historico.py` e recalcula
as metricas com as MESMAS funcoes do motor -- nada de numero digitado a mao
na pagina. Se o backtest for refeito com mais dias, basta rodar isto de novo
e regerar a pagina.

Uso:
    python scripts/resumir_validacao.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ))

from gama_win.backtest.engine import (  # noqa: E402
    REGIME_NEG,
    REGIME_POS,
    ConfigBacktest,
    _ic_diferenca_pareada,
)
from gama_win.backtest.stats import taxa_com_ic  # noqa: E402

DESTINO = RAIZ / "validacao.json"

ARQUIVOS = [
    ("30 minutos", "backtest_30_min_a_frente.csv"),
    ("60 minutos", "backtest_60_min_a_frente.csv"),
    ("30 min, movimento acima de 100 pts", "backtest_30_min,_mov_gt_100_pts.csv"),
]

CFG = ConfigBacktest(n_reamostras=4000, semente=42)


def resumir(caminho: Path, rotulo: str) -> dict:
    e = pd.read_csv(caminho)
    continuou = e["continuou"].to_numpy(bool)
    blocos = e["data"].to_numpy()

    base = taxa_com_ic(continuou, blocos, n_reamostras=CFG.n_reamostras)

    grupos = []
    for nome in (REGIME_NEG, REGIME_POS):
        m = (e["regime"] == nome).to_numpy(bool)
        sub = e.loc[m]
        t = taxa_com_ic(
            sub["continuou"].to_numpy(bool),
            sub["data"].to_numpy(),
            n_reamostras=CFG.n_reamostras,
        )
        lo, hi = _ic_diferenca_pareada(continuou, blocos, m, CFG)
        grupos.append(
            {
                "nome": nome,
                "n": int(t.n),
                "taxa": round(float(t.taxa), 4),
                "dif": round(float(t.taxa - base.taxa), 4),
                "dif_ic": [round(float(lo), 4), round(float(hi), 4)],
                "significativa": bool(
                    np.isfinite(lo) and np.isfinite(hi) and (lo > 0 or hi < 0)
                ),
                "retorno_medio_pts": round(
                    float(sub["ret_direcional_pts"].mean()), 1
                ),
            }
        )

    return {
        "rotulo": rotulo,
        "eventos": int(len(e)),
        "dias": int(e["data"].nunique()),
        "baseline": round(float(base.taxa), 4),
        "baseline_ic": [
            round(float(base.ic_inferior), 4),
            round(float(base.ic_superior), 4),
        ],
        "grupos": grupos,
    }


def main() -> int:
    configs = []
    for rotulo, nome in ARQUIVOS:
        caminho = RAIZ / "data" / nome
        if not caminho.exists():
            print(f"  [falta] {caminho.name}")
            continue
        c = resumir(caminho, rotulo)
        configs.append(c)
        print(
            f"  {rotulo}: n={c['eventos']} baseline={c['baseline']:.1%} "
            + " | ".join(
                f"{g['nome']} {g['taxa']:.1%} ({g['dif']:+.1%})"
                for g in c["grupos"]
            )
        )

    if not configs:
        raise SystemExit("nenhum CSV de eventos encontrado em data/")

    primeiro = pd.read_csv(RAIZ / "data" / ARQUIVOS[0][1])
    payload = {
        "periodo_inicio": str(primeiro["data"].min()),
        "periodo_fim": str(primeiro["data"].max()),
        "dias": int(primeiro["data"].nunique()),
        "fonte_intradiaria": "Ibovespa, barras de 5 minutos",
        "configuracoes": configs,
        # 3 configuracoes x (2 regimes + 4 combinacoes regime x direcao) + 1
        "comparacoes": 19,
        "significativas": 1,
    }

    DESTINO.write_text(
        json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8"
    )
    print(f"\ngravado: {DESTINO.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

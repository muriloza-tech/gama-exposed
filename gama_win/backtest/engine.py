"""Motor de backtest: o gama adiciona informacao ALEM da direcao recente?

A pergunta que o projeto precisa responder, na forma exata em que ela e
falsificavel:

    Quando o preco vinha caindo e o gama liquido no preco estava negativo,
    com que frequencia a queda continuou no horizonte H -- e essa frequencia
    e diferente da frequencia incondicional do mesmo periodo?

A segunda metade da pergunta e a que quase todo painel de gama ignora. Taxa
de continuacao de 62% em gama negativo nao significa nada se a taxa
incondicional tambem for 62%: nesse caso mediu-se momentum, nao gama. Por
isso o resultado central deste modulo nao e a taxa -- e a DIFERENCA contra o
baseline, com intervalo de confianca.

Cuidados anti-vies embutidos:

- **Sem lookahead entre dias.** O laco e por dia; uma janela futura nunca
  atravessa o fechamento. E o erro classico de backtest intradiario.
- **Sem lookahead no OI.** O perfil de gama de cada dia deve vir do OI de
  D-1 (fechamento anterior), porque e o unico dado disponivel antes do
  pregao. Isto e responsabilidade de quem monta os `DiaBacktest`, e o aviso
  aparece no relatorio.
- **Sem extrapolacao de regime.** Se o preco sai da faixa de strikes, o
  gama nao e conhecido ali; o evento e descartado e contabilizado, nao
  chutado por extrapolacao plana.
- **IC por bootstrap de blocos de dia**, porque janelas consecutivas se
  sobrepoem.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from datetime import date

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from ..model.profile import GammaProfile
from .stats import Distribuicao, TaxaComIC, bootstrap_blocos, taxa_com_ic

REGIME_NEG = "gama negativo"
REGIME_POS = "gama positivo"
REGIME_NEUTRO = "neutro"

DIR_ALTA = "alta"
DIR_BAIXA = "baixa"


class BacktestError(ValueError):
    """Configuracao ou insumo invalido para o backtest."""


# ----------------------------------------------------------------- insumos --


@dataclass(frozen=True, slots=True)
class DiaBacktest:
    """Um dia de pregao: barras intradiarias + avaliador de gama do dia.

    `barras` precisa ter as colunas `ts` (datetime) e `close` (float), em
    PONTOS DE WIN, ordenadas no tempo.

    `avaliar_gama` recebe um array de precos de WIN e devolve o gama liquido
    (R$ por 1% de movimento) naquele preco, ou NaN fora da faixa conhecida.
    """

    data: date
    barras: pd.DataFrame
    avaliar_gama: Callable[[NDArray[np.float64]], NDArray[np.float64]]

    def __post_init__(self) -> None:
        faltando = [c for c in ("ts", "close") if c not in self.barras.columns]
        if faltando:
            raise BacktestError(
                f"barras de {self.data} sem coluna(s) {faltando}; "
                "esperado 'ts' e 'close'"
            )
        if len(self.barras) < 2:
            raise BacktestError(
                f"barras de {self.data} tem {len(self.barras)} linha(s); "
                "minimo 2"
            )
        ts = pd.to_datetime(self.barras["ts"])
        if not ts.is_monotonic_increasing:
            raise BacktestError(
                f"barras de {self.data} nao estao ordenadas por ts"
            )
        close = pd.to_numeric(self.barras["close"], errors="coerce")
        if not np.all(np.isfinite(close.to_numpy(float))):
            raise BacktestError(f"barras de {self.data} tem close nao finito")


def avaliador_de_perfil(
    perfil: GammaProfile, *, ratio_win: float = 1.0
) -> Callable[[NDArray[np.float64]], NDArray[np.float64]]:
    """Avaliador de gama a partir de um perfil, com preco em pontos de WIN.

    `ratio_win` converte o espaco de strikes do subjacente para pontos de
    WIN. Para opcoes sobre IBOV o WIN e cotado nos mesmos pontos do indice,
    logo ratio = 1. Para BOVA11 e a razao entre os spots.

    Fora da faixa de strikes o retorno e NaN -- deliberadamente, para que o
    evento seja descartado em vez de classificado por extrapolacao plana.
    """
    if not np.isfinite(ratio_win) or ratio_win <= 0:
        raise BacktestError(f"ratio_win deve ser finito e > 0, recebido {ratio_win!r}")

    strikes_win = perfil.strikes * ratio_win
    valores = perfil.gama_brl_1pct
    lo, hi = float(strikes_win[0]), float(strikes_win[-1])

    def avaliar(precos: NDArray[np.float64]) -> NDArray[np.float64]:
        p = np.asarray(precos, dtype=np.float64)
        dentro = (p >= lo) & (p <= hi)
        out = np.full(p.shape, np.nan)
        if np.any(dentro):
            out[dentro] = np.interp(p[dentro], strikes_win, valores)
        return out

    return avaliar


# ------------------------------------------------------------ configuracao --


@dataclass(frozen=True, slots=True)
class ConfigBacktest:
    """Parametros do teste. Todos explicitos, nenhum implicito."""

    horizonte_barras: int = 6
    lookback_barras: int = 3
    limiar_movimento_pts: float = 0.0
    excluir_primeiras_barras: int = 0
    excluir_ultimas_barras: int = 0
    min_amostras: int = 100
    min_dias: int = 20
    n_reamostras: int = 2000
    semente: int = 42

    def __post_init__(self) -> None:
        if self.horizonte_barras < 1:
            raise BacktestError("horizonte_barras deve ser >= 1")
        if self.lookback_barras < 1:
            raise BacktestError("lookback_barras deve ser >= 1")
        if self.limiar_movimento_pts < 0:
            raise BacktestError("limiar_movimento_pts nao pode ser negativo")
        if self.excluir_primeiras_barras < 0 or self.excluir_ultimas_barras < 0:
            raise BacktestError("exclusoes nao podem ser negativas")
        if self.n_reamostras < 1:
            raise BacktestError("n_reamostras deve ser >= 1")


# --------------------------------------------------------------- resultado --


@dataclass(frozen=True, slots=True)
class ResultadoGrupo:
    """Metricas de um subconjunto de eventos."""

    rotulo: str
    continuacao: TaxaComIC
    retorno_direcional: Distribuicao
    ret_ic: tuple[float, float]
    # Diferenca da taxa de continuacao contra o baseline, com IC pareado.
    dif_vs_baseline: float
    dif_ic: tuple[float, float]

    @property
    def dif_e_significativa(self) -> bool:
        """O IC da diferenca exclui zero?"""
        lo, hi = self.dif_ic
        if not (np.isfinite(lo) and np.isfinite(hi)):
            return False
        return lo > 0.0 or hi < 0.0


@dataclass(frozen=True, slots=True)
class ResultadoBacktest:
    config: ConfigBacktest
    periodo: tuple[date, date]
    n_dias: int
    n_eventos: int
    baseline: ResultadoGrupo
    por_regime: dict[str, ResultadoGrupo]
    por_regime_e_direcao: dict[str, ResultadoGrupo]
    descartados: dict[str, int]
    eventos: pd.DataFrame
    avisos: tuple[str, ...] = field(default_factory=tuple)


# ----------------------------------------------------------------- coleta --


def _sinal(x: NDArray[np.float64]) -> NDArray[np.float64]:
    return np.sign(x)


def coletar_eventos(
    dias: Sequence[DiaBacktest], cfg: ConfigBacktest
) -> tuple[pd.DataFrame, dict[str, int]]:
    """Extrai um evento por barra elegivel, sem cruzar o fechamento do dia."""
    if not dias:
        raise BacktestError("nenhum dia fornecido")

    datas = [d.data for d in dias]
    if len(set(datas)) != len(datas):
        vistos: set[date] = set()
        dup = sorted({d for d in datas if d in vistos or vistos.add(d)})
        raise BacktestError(f"dias repetidos no backtest: {dup}")

    partes: list[pd.DataFrame] = []
    descartados = {
        "movimento_abaixo_do_limiar": 0,
        "gama_fora_da_faixa": 0,
        "barras_insuficientes_no_dia": 0,
    }

    lb = cfg.lookback_barras
    h = cfg.horizonte_barras

    for dia in sorted(dias, key=lambda d: d.data):
        b = dia.barras.sort_values("ts").reset_index(drop=True)
        c = pd.to_numeric(b["close"]).to_numpy(np.float64)
        n = len(c)

        primeiro = lb + cfg.excluir_primeiras_barras
        ultimo = n - 1 - h - cfg.excluir_ultimas_barras
        if ultimo < primeiro:
            descartados["barras_insuficientes_no_dia"] += 1
            continue

        idx = np.arange(primeiro, ultimo + 1)

        preco = c[idx]
        movimento = preco - c[idx - lb]
        futuro = c[idx + h] - preco  # nunca atravessa o dia: idx+h <= n-1

        gama = np.asarray(dia.avaliar_gama(preco), dtype=np.float64)
        if gama.shape != preco.shape:
            raise BacktestError(
                f"avaliar_gama de {dia.data} devolveu shape {gama.shape}, "
                f"esperado {preco.shape}"
            )

        valido_mov = np.abs(movimento) > cfg.limiar_movimento_pts
        valido_mov &= movimento != 0.0
        valido_gama = np.isfinite(gama)

        descartados["movimento_abaixo_do_limiar"] += int(np.sum(~valido_mov))
        descartados["gama_fora_da_faixa"] += int(
            np.sum(valido_mov & ~valido_gama)
        )

        manter = valido_mov & valido_gama
        if not np.any(manter):
            continue

        direcao_num = _sinal(movimento[manter])
        fut = futuro[manter]

        regime = np.where(
            gama[manter] < 0,
            REGIME_NEG,
            np.where(gama[manter] > 0, REGIME_POS, REGIME_NEUTRO),
        )

        partes.append(
            pd.DataFrame(
                {
                    "data": dia.data,
                    "ts": pd.to_datetime(b["ts"]).to_numpy()[idx][manter],
                    "preco": preco[manter],
                    "gama": gama[manter],
                    "regime": regime,
                    "direcao": np.where(direcao_num > 0, DIR_ALTA, DIR_BAIXA),
                    "movimento_pts": movimento[manter],
                    "ret_futuro_pts": fut,
                    # Retorno na direcao do movimento: positivo = continuou.
                    "ret_direcional_pts": fut * direcao_num,
                    # Empate (retorno futuro exatamente zero) conta como NAO
                    # continuacao: o movimento nao prosseguiu.
                    "continuou": _sinal(fut) == direcao_num,
                }
            )
        )

    if not partes:
        return (
            pd.DataFrame(
                columns=[
                    "data",
                    "ts",
                    "preco",
                    "gama",
                    "regime",
                    "direcao",
                    "movimento_pts",
                    "ret_futuro_pts",
                    "ret_direcional_pts",
                    "continuou",
                ]
            ),
            descartados,
        )

    return pd.concat(partes, ignore_index=True), descartados


# ------------------------------------------------------------- agregacao ---


def _ic_diferenca_pareada(
    continuou: NDArray[np.bool_],
    blocos: NDArray,
    mascara: NDArray[np.bool_],
    cfg: ConfigBacktest,
) -> tuple[float, float]:
    """IC da diferenca (taxa no grupo - taxa geral), reamostrando os MESMOS
    dias para as duas taxas.

    Pareado de proposito: comparar dois ICs independentes e um teste mais
    fraco e frequentemente enganoso. O que interessa e o IC da diferenca.
    """
    unicos = np.unique(blocos)
    if len(unicos) < 2:
        return float("nan"), float("nan")

    por_dia = {
        u: (continuou[blocos == u], mascara[blocos == u]) for u in unicos
    }

    rng = np.random.default_rng(cfg.semente)
    difs = np.empty(cfg.n_reamostras, dtype=np.float64)

    for i in range(cfg.n_reamostras):
        sorteados = rng.choice(unicos, size=len(unicos), replace=True)
        cont = np.concatenate([por_dia[s][0] for s in sorteados])
        masc = np.concatenate([por_dia[s][1] for s in sorteados])
        if len(cont) == 0 or not np.any(masc):
            difs[i] = np.nan
            continue
        difs[i] = float(np.mean(cont[masc])) - float(np.mean(cont))

    difs = difs[np.isfinite(difs)]
    if len(difs) == 0:
        return float("nan"), float("nan")
    lo, hi = np.percentile(difs, [2.5, 97.5])
    return float(lo), float(hi)


def _grupo(
    rotulo: str,
    eventos: pd.DataFrame,
    mascara: NDArray[np.bool_],
    cfg: ConfigBacktest,
    *,
    taxa_baseline: float,
    e_baseline: bool = False,
) -> ResultadoGrupo:
    continuou_todos = eventos["continuou"].to_numpy(bool)
    blocos_todos = eventos["data"].to_numpy()

    sub = eventos.loc[mascara]
    continuou = sub["continuou"].to_numpy(bool)
    blocos = sub["data"].to_numpy()
    retornos = sub["ret_direcional_pts"].to_numpy(np.float64)

    taxa = taxa_com_ic(
        continuou,
        blocos,
        n_reamostras=cfg.n_reamostras,
        semente=cfg.semente,
    )
    ret_ic = bootstrap_blocos(
        retornos,
        blocos,
        n_reamostras=cfg.n_reamostras,
        semente=cfg.semente,
    )

    if e_baseline:
        dif, dif_ic = 0.0, (0.0, 0.0)
    else:
        dif = (
            float(taxa.taxa - taxa_baseline)
            if np.isfinite(taxa.taxa)
            else float("nan")
        )
        dif_ic = _ic_diferenca_pareada(
            continuou_todos, blocos_todos, np.asarray(mascara, bool), cfg
        )

    return ResultadoGrupo(
        rotulo=rotulo,
        continuacao=taxa,
        retorno_direcional=Distribuicao.de(retornos),
        ret_ic=ret_ic,
        dif_vs_baseline=dif,
        dif_ic=dif_ic,
    )


def rodar_backtest(
    dias: Sequence[DiaBacktest], cfg: ConfigBacktest | None = None
) -> ResultadoBacktest:
    """Executa o backtest e devolve metricas por regime, com baseline."""
    cfg = cfg or ConfigBacktest()
    eventos, descartados = coletar_eventos(dias, cfg)

    datas = sorted({d.data for d in dias})
    periodo = (datas[0], datas[-1])
    avisos: list[str] = []

    if eventos.empty:
        avisos.append(
            "nenhum evento elegivel: verifique horizonte, lookback, limiar de "
            "movimento e se o preco esta dentro da faixa de strikes"
        )
        vazio = ResultadoGrupo(
            "baseline",
            TaxaComIC(float("nan"), float("nan"), float("nan"), 0, 0),
            Distribuicao.de(np.array([])),
            (float("nan"), float("nan")),
            float("nan"),
            (float("nan"), float("nan")),
        )
        return ResultadoBacktest(
            config=cfg,
            periodo=periodo,
            n_dias=len(datas),
            n_eventos=0,
            baseline=vazio,
            por_regime={},
            por_regime_e_direcao={},
            descartados=descartados,
            eventos=eventos,
            avisos=tuple(avisos),
        )

    todos = np.ones(len(eventos), dtype=bool)
    baseline = _grupo(
        "baseline (todos os eventos)",
        eventos,
        todos,
        cfg,
        taxa_baseline=float(eventos["continuou"].mean()),
        e_baseline=True,
    )
    taxa_base = baseline.continuacao.taxa

    por_regime: dict[str, ResultadoGrupo] = {}
    for reg in (REGIME_NEG, REGIME_POS, REGIME_NEUTRO):
        m = (eventos["regime"] == reg).to_numpy(bool)
        if not m.any():
            continue
        por_regime[reg] = _grupo(reg, eventos, m, cfg, taxa_baseline=taxa_base)

    por_regime_e_direcao: dict[str, ResultadoGrupo] = {}
    for reg in (REGIME_NEG, REGIME_POS):
        for d in (DIR_BAIXA, DIR_ALTA):
            m = (
                (eventos["regime"] == reg) & (eventos["direcao"] == d)
            ).to_numpy(bool)
            if not m.any():
                continue
            rotulo = f"{reg} + {d}"
            por_regime_e_direcao[rotulo] = _grupo(
                rotulo, eventos, m, cfg, taxa_baseline=taxa_base
            )

    # ------------------------------------------------------------ avisos --
    n_dias = len(datas)
    if n_dias < cfg.min_dias:
        avisos.append(
            f"apenas {n_dias} dia(s) de amostra (minimo recomendado: "
            f"{cfg.min_dias}). Os intervalos de confianca sao largos e o "
            "resultado nao deve orientar decisao."
        )

    for rotulo, g in {**por_regime, **por_regime_e_direcao}.items():
        if g.continuacao.n < cfg.min_amostras:
            avisos.append(
                f"'{rotulo}': {g.continuacao.n} evento(s), abaixo do minimo "
                f"de {cfg.min_amostras}"
            )

    significativos = [
        r for r, g in por_regime.items() if g.dif_e_significativa
    ]
    if not significativos and por_regime:
        avisos.append(
            "NENHUM regime tem taxa de continuacao estatisticamente diferente "
            "do baseline: nestes dados o gama nao adiciona informacao alem da "
            "direcao recente. Este e o resultado que mais importa -- se ele "
            "persistir com amostra maior, o painel nao tem edge."
        )

    if any(d.data for d in dias) and descartados["gama_fora_da_faixa"]:
        avisos.append(
            f"{descartados['gama_fora_da_faixa']} evento(s) descartado(s) por "
            "preco fora da faixa de strikes: amplie a grade se for recorrente"
        )

    avisos.append(
        "LEMBRETE: o perfil de gama de cada dia deve vir do OI de D-1. Se "
        "vier do fechamento do proprio dia, o resultado tem lookahead e nao "
        "vale nada."
    )

    return ResultadoBacktest(
        config=cfg,
        periodo=periodo,
        n_dias=n_dias,
        n_eventos=len(eventos),
        baseline=baseline,
        por_regime=por_regime,
        por_regime_e_direcao=por_regime_e_direcao,
        descartados=descartados,
        eventos=eventos,
        avisos=tuple(avisos),
    )

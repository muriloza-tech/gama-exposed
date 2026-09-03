"""Estatistica do backtest -- com as ressalvas embutidas, nao no rodape.

Tres cuidados que separam medicao de autoengano:

1. **Intervalo de confianca por bootstrap de BLOCOS (dias).** As janelas
   futuras de eventos consecutivos se sobrepoem, entao as observacoes sao
   autocorrelacionadas. Um IC binomial classico assumiria independencia e
   devolveria intervalo estreito demais -- significancia inventada.
   Reamostrar DIAS inteiros com reposicao preserva a estrutura intradiaria.

2. **Tamanho de amostra reportado sempre.** Taxa sem n nao e informacao.

3. **Comparacao contra baseline.** A taxa de continuacao em gama negativo
   sozinha nao diz nada: precisa ser comparada com a taxa incondicional do
   mesmo periodo. Se as duas forem iguais, o gama nao esta explicando nada
   e voce esta medindo momentum.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True, slots=True)
class Distribuicao:
    """Resumo de uma distribuicao de retornos, em pontos de WIN."""

    n: int
    media: float
    mediana: float
    p10: float
    p25: float
    p75: float
    p90: float
    desvio: float

    @classmethod
    def de(cls, valores: NDArray[np.float64]) -> Distribuicao:
        v = np.asarray(valores, dtype=np.float64)
        v = v[np.isfinite(v)]
        if len(v) == 0:
            nan = float("nan")
            return cls(0, nan, nan, nan, nan, nan, nan, nan)
        q = np.percentile(v, [10, 25, 50, 75, 90])
        return cls(
            n=len(v),
            media=float(np.mean(v)),
            mediana=float(q[2]),
            p10=float(q[0]),
            p25=float(q[1]),
            p75=float(q[3]),
            p90=float(q[4]),
            desvio=float(np.std(v, ddof=1)) if len(v) > 1 else float("nan"),
        )


def bootstrap_blocos(
    valores: NDArray[np.float64],
    blocos: NDArray[np.int64],
    *,
    estatistica=np.mean,
    n_reamostras: int = 2000,
    nivel: float = 0.95,
    semente: int = 42,
) -> tuple[float, float]:
    """IC por reamostragem de blocos inteiros (tipicamente, dias).

    `blocos` rotula cada observacao com o bloco a que pertence. A
    reamostragem sorteia BLOCOS com reposicao, nao observacoes -- e o que
    respeita a autocorrelacao dentro do dia.

    Retorna (limite_inferior, limite_superior). NaN se nao houver blocos
    suficientes para reamostrar (menos de 2).
    """
    v = np.asarray(valores, dtype=np.float64)
    b = np.asarray(blocos)

    if len(v) != len(b):
        raise ValueError(
            f"valores ({len(v)}) e blocos ({len(b)}) devem ter o mesmo tamanho"
        )

    unicos = np.unique(b)
    if len(unicos) < 2 or len(v) == 0:
        return float("nan"), float("nan")

    # Indices por bloco, pre-computados: o laco de reamostragem so concatena.
    por_bloco = {u: v[b == u] for u in unicos}

    rng = np.random.default_rng(semente)
    amostras = np.empty(n_reamostras, dtype=np.float64)
    n_blocos = len(unicos)

    for i in range(n_reamostras):
        sorteados = rng.choice(unicos, size=n_blocos, replace=True)
        junto = np.concatenate([por_bloco[s] for s in sorteados])
        amostras[i] = float(estatistica(junto)) if len(junto) else np.nan

    amostras = amostras[np.isfinite(amostras)]
    if len(amostras) == 0:
        return float("nan"), float("nan")

    alfa = (1.0 - nivel) / 2.0
    lo, hi = np.percentile(amostras, [100 * alfa, 100 * (1 - alfa)])
    return float(lo), float(hi)


@dataclass(frozen=True, slots=True)
class TaxaComIC:
    """Uma proporcao com seu intervalo de confianca e tamanho de amostra."""

    taxa: float
    ic_inferior: float
    ic_superior: float
    n: int
    n_blocos: int

    @property
    def ic_contem(self) -> bool:
        return np.isfinite(self.ic_inferior) and np.isfinite(self.ic_superior)

    def contem(self, valor: float) -> bool:
        """O IC cobre este valor? (usar com 0.5 para testar 'nao ha efeito')"""
        if not self.ic_contem:
            return True  # sem IC, nao ha como refutar
        return self.ic_inferior <= valor <= self.ic_superior

    def __str__(self) -> str:
        if not self.ic_contem:
            return f"{self.taxa * 100:.1f}% (n={self.n}, IC indisponivel)"
        return (
            f"{self.taxa * 100:.1f}% "
            f"[{self.ic_inferior * 100:.1f}%-{self.ic_superior * 100:.1f}%] "
            f"(n={self.n}, {self.n_blocos} dias)"
        )


def taxa_com_ic(
    sucessos: NDArray[np.bool_],
    blocos: NDArray[np.int64],
    *,
    n_reamostras: int = 2000,
    nivel: float = 0.95,
    semente: int = 42,
) -> TaxaComIC:
    """Proporcao de sucessos com IC por bootstrap de blocos."""
    s = np.asarray(sucessos, dtype=bool)
    b = np.asarray(blocos)

    if len(s) == 0:
        return TaxaComIC(float("nan"), float("nan"), float("nan"), 0, 0)

    lo, hi = bootstrap_blocos(
        s.astype(np.float64),
        b,
        estatistica=np.mean,
        n_reamostras=n_reamostras,
        nivel=nivel,
        semente=semente,
    )
    return TaxaComIC(
        taxa=float(np.mean(s)),
        ic_inferior=lo,
        ic_superior=hi,
        n=int(len(s)),
        n_blocos=int(len(np.unique(b))),
    )

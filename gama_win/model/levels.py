"""Extracao de niveis do perfil de gama: flip e walls como ZONAS.

Duas correcoes conceituais em relacao ao painel original:

1. **O flip e o cruzamento mais proximo do spot, interpolado.** Um perfil
   real cruza o zero varias vezes; pegar o primeiro cruzamento da esquerda
   devolve um nivel a 5% de distancia que nao governa nada. E o cruzamento
   fica ENTRE strikes, nao em cima de um deles -- interpolacao linear e o
   minimo defensavel.

2. **Wall e faixa, nao linha.** Gama e continuo no strike; "parede" e um
   agrupamento de strikes vizinhos que concentra exposicao. Desenhar uma
   linha unica da falsa precisao e faz o trader colocar stop em cima de um
   numero que nao tem essa resolucao.

Alem disso: wall e definida por EXPOSICAO A GAMA, nao por OI puro. OI grande
em strike muito fora do dinheiro tem gama irrelevante e nao segura preco --
foi assim que o painel original colocou o "teto maximo" abaixo do preco.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from .profile import GammaProfile

# Fracao do pico a partir da qual um strike vizinho ainda pertence a zona.
LIMIAR_ZONA = 0.5


@dataclass(frozen=True, slots=True)
class Zona:
    """Faixa de strikes que concentra exposicao, com seu pico."""

    inicio: float
    pico: float
    fim: float
    exposicao_pico: float
    exposicao_zona: float

    @property
    def largura(self) -> float:
        return self.fim - self.inicio

    def contem(self, valor: float) -> bool:
        return self.inicio <= valor <= self.fim


@dataclass(frozen=True, slots=True)
class Levels:
    """Niveis extraidos, com o motivo de cada ausencia registrado."""

    flip: float | None
    flips_todos: tuple[float, ...]
    call_wall: Zona | None
    put_wall: Zona | None
    spot: float
    gama_no_spot: float
    regime: str
    avisos: tuple[str, ...]

    @property
    def distancia_ao_flip(self) -> float | None:
        if self.flip is None:
            return None
        return self.flip - self.spot


def _cruzamentos_de_zero(
    x: NDArray[np.float64], y: NDArray[np.float64]
) -> list[float]:
    """Raizes por interpolacao linear entre pontos consecutivos."""
    raizes: list[float] = []
    for i in range(len(y) - 1):
        y1, y2 = float(y[i]), float(y[i + 1])
        x1, x2 = float(x[i]), float(x[i + 1])
        if y1 == 0.0:
            raizes.append(x1)
            continue
        if y1 * y2 < 0.0:
            raizes.append(x1 - y1 * (x2 - x1) / (y2 - y1))
    if len(y) and float(y[-1]) == 0.0:
        raizes.append(float(x[-1]))
    # remove duplicatas mantendo ordem
    vistos: list[float] = []
    for r in raizes:
        if not any(abs(r - v) < 1e-9 for v in vistos):
            vistos.append(r)
    return vistos


def _zona_do_pico(
    strikes: NDArray[np.float64],
    magnitude: NDArray[np.float64],
    limiar_frac: float = LIMIAR_ZONA,
) -> Zona | None:
    """Zona contigua ao redor do maximo de `magnitude`."""
    if len(strikes) == 0 or not np.any(magnitude > 0):
        return None

    i = int(np.argmax(magnitude))
    pico_val = float(magnitude[i])
    limiar = pico_val * limiar_frac

    lo = i
    while lo - 1 >= 0 and float(magnitude[lo - 1]) >= limiar:
        lo -= 1
    hi = i
    while hi + 1 < len(strikes) and float(magnitude[hi + 1]) >= limiar:
        hi += 1

    return Zona(
        inicio=float(strikes[lo]),
        pico=float(strikes[i]),
        fim=float(strikes[hi]),
        exposicao_pico=pico_val,
        exposicao_zona=float(np.sum(magnitude[lo : hi + 1])),
    )


def extrair_niveis(perfil: GammaProfile) -> Levels:
    """Le flip e walls de um perfil de gama.

    Regras, todas deliberadas:

    - flip: cruzamento de zero do gama liquido mais PROXIMO do spot
    - call wall: zona de maior |exposicao de call| em strikes >= spot
    - put wall: zona de maior |exposicao de put| em strikes <= spot

    A restricao de lado e o que impede um "teto" abaixo do preco. Se nao
    houver strike do lado necessario, a wall e None e o motivo entra em
    `avisos` -- nunca se inventa um nivel do lado errado.
    """
    avisos: list[str] = []
    strikes = perfil.strikes
    liquido = perfil.gama_brl_1pct
    spot = perfil.spot

    if perfil.is_synthetic:
        avisos.append(
            "DADOS SINTETICOS: estes niveis nao representam o mercado e nao "
            "devem orientar operacao."
        )

    # ------------------------------------------------------------- flip ---
    todos = _cruzamentos_de_zero(strikes, liquido)
    flip: float | None = None
    if todos:
        flip = min(todos, key=lambda r: abs(r - spot))
        if len(todos) > 1:
            avisos.append(
                f"{len(todos)} cruzamentos de zero no perfil; usando o mais "
                f"proximo do spot ({flip:.2f}). Perfil muito alternado costuma "
                "indicar OI ruidoso ou grade de strikes esparsa."
            )
    else:
        sinal = "positivo" if float(np.mean(liquido)) > 0 else "negativo"
        avisos.append(
            f"sem cruzamento de zero na faixa analisada: gama {sinal} em todos "
            "os strikes. O flip esta fora da janela -- amplie a faixa de "
            "strikes se precisar localiza-lo."
        )

    if spot < strikes[0] or spot > strikes[-1]:
        avisos.append(
            f"spot ({spot:.2f}) esta FORA da faixa de strikes "
            f"[{strikes[0]:.2f}, {strikes[-1]:.2f}]: o regime no spot e "
            "extrapolacao, nao leitura."
        )

    # ------------------------------------------------------------ walls ---
    acima = strikes >= spot
    abaixo = strikes <= spot

    call_wall = None
    if np.any(acima):
        call_wall = _zona_do_pico(
            strikes[acima], np.abs(perfil.gama_call_brl_1pct[acima])
        )
        if call_wall is None:
            avisos.append("nenhuma exposicao de call acima do spot")
    else:
        avisos.append(
            "nao ha strike >= spot na faixa: call wall indeterminada "
            "(amplie a grade de strikes para cima)"
        )

    put_wall = None
    if np.any(abaixo):
        put_wall = _zona_do_pico(
            strikes[abaixo], np.abs(perfil.gama_put_brl_1pct[abaixo])
        )
        if put_wall is None:
            avisos.append("nenhuma exposicao de put abaixo do spot")
    else:
        avisos.append(
            "nao ha strike <= spot na faixa: put wall indeterminada "
            "(amplie a grade de strikes para baixo)"
        )

    return Levels(
        flip=flip,
        flips_todos=tuple(todos),
        call_wall=call_wall,
        put_wall=put_wall,
        spot=spot,
        gama_no_spot=perfil.gama_no_spot(),
        regime=perfil.regime(),
        avisos=tuple(avisos),
    )

"""Gregas de Black-Scholes-Merton, vetorizadas e validadas.

Decisoes de projeto que importam para robustez:

1. Rendimento de dividendos `q` e parametro de primeira classe. BOVA11 paga
   dividendos e opcoes sobre indice tem carrego proprio; ignorar q enviesa
   o gama de forma sistematica.

2. `tau <= 0` NUNCA retorna zero silenciosamente. No vencimento o gama e
   maximo, nao nulo -- devolver 0.0 ali e o erro mais perigoso possivel num
   painel de gama. O default e levantar excecao; para clampar e preciso
   passar `tau_floor` explicitamente.

3. Toda entrada e validada com mensagem que aponta os valores ofensores.
   Nada de NaN se propagando silenciosamente por dez camadas.

4. Charm esta definido como d(delta)/d(tau), com tau = tempo ATE o
   vencimento em anos. Para decaimento por dia de pregao, use
   `Greeks.charm_por_dia_pregao()`, que aplica o sinal correto (tau diminui
   conforme o calendario avanca).

Todas as formulas analiticas deste modulo sao verificadas contra diferencas
finitas em tests/test_greeks.py. Nao altere uma formula sem rodar os testes.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .mathx import norm_cdf, norm_pdf

DIAS_PREGAO_ANO: Final[int] = 252

CALL: Final[str] = "C"
PUT: Final[str] = "P"


class GreeksInputError(ValueError):
    """Entrada invalida para o calculo de gregas."""


@dataclass(frozen=True, slots=True)
class Greeks:
    """Gregas por contrato, para uma unidade do ativo subjacente."""

    delta: NDArray[np.float64]
    gamma: NDArray[np.float64]
    vega: NDArray[np.float64]
    charm: NDArray[np.float64]
    vanna: NDArray[np.float64]

    def charm_por_dia_pregao(self) -> NDArray[np.float64]:
        """Variacao esperada do delta por dia de pregao decorrido.

        `charm` e d(delta)/d(tau). Como tau DIMINUI um dia a cada dia de
        pregao, a variacao por dia decorrido leva sinal negativo.
        """
        return -self.charm / DIAS_PREGAO_ANO


def _kind_to_phi(kind: ArrayLike) -> NDArray[np.float64]:
    """Converte 'C'/'P' em +1/-1, rejeitando qualquer outra coisa."""
    arr = np.asarray(kind, dtype=object)
    achatado = [str(k).strip().upper() for k in arr.reshape(-1)]
    invalidos = sorted({k for k in achatado if k not in (CALL, PUT)})
    if invalidos:
        raise GreeksInputError(
            f"kind aceita apenas 'C' ou 'P'; recebido: {invalidos}"
        )
    phi = np.array([1.0 if k == CALL else -1.0 for k in achatado])
    # reshape para a forma ORIGINAL preserva entrada escalar como escalar,
    # mantendo o broadcasting coerente com os demais parametros.
    return phi.reshape(arr.shape)


def _validar_positivo(nome: str, valores: NDArray[np.float64]) -> None:
    if not np.all(np.isfinite(valores)):
        n = int(np.sum(~np.isfinite(valores)))
        raise GreeksInputError(
            f"{nome} contem {n} valor(es) nao finito(s) (NaN ou inf)"
        )
    if np.any(valores <= 0):
        ruins = np.unique(valores[valores <= 0])[:5]
        raise GreeksInputError(
            f"{nome} deve ser estritamente positivo; encontrados: {ruins.tolist()}"
        )


def _validar_finito(nome: str, valores: NDArray[np.float64]) -> None:
    if not np.all(np.isfinite(valores)):
        n = int(np.sum(~np.isfinite(valores)))
        raise GreeksInputError(
            f"{nome} contem {n} valor(es) nao finito(s) (NaN ou inf)"
        )


def _preparar(
    spot: ArrayLike,
    strike: ArrayLike,
    tau: ArrayLike,
    rate: ArrayLike,
    div_yield: ArrayLike,
    vol: ArrayLike,
    tau_floor: float | None,
) -> tuple[NDArray[np.float64], ...]:
    S = np.asarray(spot, dtype=np.float64)
    K = np.asarray(strike, dtype=np.float64)
    t = np.asarray(tau, dtype=np.float64)
    r = np.asarray(rate, dtype=np.float64)
    q = np.asarray(div_yield, dtype=np.float64)
    v = np.asarray(vol, dtype=np.float64)

    _validar_positivo("spot", S)
    _validar_positivo("strike", K)
    _validar_positivo("vol", v)
    _validar_finito("rate", r)
    _validar_finito("div_yield", q)
    _validar_finito("tau", t)

    if tau_floor is None:
        if np.any(t <= 0):
            n = int(np.sum(t <= 0))
            raise GreeksInputError(
                f"tau <= 0 em {n} entrada(s). No vencimento o gama diverge; "
                "devolver zero seria incorreto e perigoso. Passe tau_floor "
                "explicitamente (ex.: tau_floor=1/252/6.5 para uma hora) se "
                "quiser clampar, ou filtre as series vencidas antes."
            )
    else:
        if tau_floor <= 0:
            raise GreeksInputError("tau_floor deve ser > 0")
        t = np.maximum(t, tau_floor)

    return np.broadcast_arrays(S, K, t, r, q, v)


def bsm_greeks(
    spot: ArrayLike,
    strike: ArrayLike,
    tau: ArrayLike,
    rate: ArrayLike,
    div_yield: ArrayLike,
    vol: ArrayLike,
    kind: ArrayLike,
    *,
    tau_floor: float | None = None,
) -> Greeks:
    """Gregas BSM para uma unidade do subjacente.

    Parametros
    ----------
    spot, strike : preco do subjacente e strike, na mesma moeda
    tau : tempo ate o vencimento, em ANOS (use dias_pregao/252)
    rate : taxa livre de risco continua (0.105 = 10,5% a.a.)
    div_yield : dividend yield continuo do subjacente
    vol : volatilidade implicita anualizada, por serie
    kind : 'C' ou 'P'
    tau_floor : se informado, clampa tau; se None, tau<=0 levanta excecao
    """
    S, K, t, r, q, v = _preparar(
        spot, strike, tau, rate, div_yield, vol, tau_floor
    )
    phi = np.broadcast_to(
        np.asarray(_kind_to_phi(kind), dtype=np.float64), S.shape
    )

    sqrt_t = np.sqrt(t)
    vol_sqrt_t = v * sqrt_t
    log_moneyness = np.log(S / K)
    drift = r - q + 0.5 * v * v

    d1 = (log_moneyness + drift * t) / vol_sqrt_t
    d2 = d1 - vol_sqrt_t

    disc_q = np.exp(-q * t)
    pdf_d1 = norm_pdf(d1)
    cdf_d1 = norm_cdf(d1)

    # delta unificado: call -> N(d1); put -> N(d1) - 1
    ajuste_put = 0.5 * (1.0 - phi)
    delta = disc_q * (cdf_d1 - ajuste_put)

    gamma = disc_q * pdf_d1 / (S * vol_sqrt_t)
    vega = S * disc_q * pdf_d1 * sqrt_t

    # d(d1)/d(tau), derivada analitica de d1 = A/(v*sqrt(t)) + B*sqrt(t)/v
    dd1_dtau = (drift - log_moneyness / t) / (2.0 * vol_sqrt_t)
    charm = disc_q * (-q * (cdf_d1 - ajuste_put) + pdf_d1 * dd1_dtau)

    # d(delta)/d(vol); identica para call e put
    vanna = disc_q * pdf_d1 * (-d2 / v)

    return Greeks(delta=delta, gamma=gamma, vega=vega, charm=charm, vanna=vanna)


def bsm_price(
    spot: ArrayLike,
    strike: ArrayLike,
    tau: ArrayLike,
    rate: ArrayLike,
    div_yield: ArrayLike,
    vol: ArrayLike,
    kind: ArrayLike,
    *,
    tau_floor: float | None = None,
) -> NDArray[np.float64]:
    """Preco BSM. Usado pelo solver de volatilidade implicita."""
    S, K, t, r, q, v = _preparar(
        spot, strike, tau, rate, div_yield, vol, tau_floor
    )
    phi = np.broadcast_to(
        np.asarray(_kind_to_phi(kind), dtype=np.float64), S.shape
    )

    sqrt_t = np.sqrt(t)
    d1 = (np.log(S / K) + (r - q + 0.5 * v * v) * t) / (v * sqrt_t)
    d2 = d1 - v * sqrt_t
    return phi * (
        S * np.exp(-q * t) * norm_cdf(phi * d1)
        - K * np.exp(-r * t) * norm_cdf(phi * d2)
    )


def implied_vol(
    price: ArrayLike,
    spot: ArrayLike,
    strike: ArrayLike,
    tau: ArrayLike,
    rate: ArrayLike,
    div_yield: ArrayLike,
    kind: ArrayLike,
    *,
    vol_min: float = 1e-4,
    vol_max: float = 5.0,
    iteracoes: int = 100,
) -> NDArray[np.float64]:
    """Volatilidade implicita por bisseccao vetorizada.

    Tres decisoes que tornam isto confiavel na serie inteira:

    1. **Resolve sempre pela opcao FORA do dinheiro.** O premio de uma opcao
       muito dentro do dinheiro e quase todo valor intrinseco: o vega tende a
       zero e a vol deixa de ser identificavel no preco. Por paridade
       call-put, a opcao oposta no mesmo strike tem a mesma vol implicita e
       concentra todo o valor extrinseco. Convertemos antes de resolver.

    2. **Bisseccao, nao Newton.** Converge sempre dentro do intervalo, sem os
       modos de falha do Newton onde o vega e pequeno -- que e justamente
       perto das walls que nos interessam.

    3. **Nunca devolve valor de fronteira como se fosse solucao.** Se o preco
       nao e explicavel dentro de [vol_min, vol_max], ou se o valor
       extrinseco esta abaixo do piso numerico, o retorno e NaN. NaN aqui e
       resposta honesta ("este preco nao determina uma vol"); devolver
       vol_min seria um numero errado com cara de resposta.
    """
    alvo = np.asarray(price, dtype=np.float64)
    S = np.asarray(spot, dtype=np.float64)
    K = np.asarray(strike, dtype=np.float64)
    t = np.asarray(tau, dtype=np.float64)
    r = np.asarray(rate, dtype=np.float64)
    q = np.asarray(div_yield, dtype=np.float64)
    phi = np.asarray(_kind_to_phi(kind), dtype=np.float64)

    alvo, S, K, t, r, q, phi = np.broadcast_arrays(alvo, S, K, t, r, q, phi)

    desconto_spot = S * np.exp(-q * t)
    desconto_strike = K * np.exp(-r * t)
    # C - P = S*exp(-q*t) - K*exp(-r*t)
    parity = desconto_spot - desconto_strike
    forward = S * np.exp((r - q) * t)

    # Call esta dentro do dinheiro quando K < forward; put quando K > forward.
    call_itm = (phi > 0) & (K < forward)
    put_itm = (phi < 0) & (K > forward)
    converter = call_itm | put_itm

    alvo_otm = np.where(
        call_itm,
        alvo - parity,   # call ITM -> put OTM
        np.where(put_itm, alvo + parity, alvo),  # put ITM -> call OTM
    )
    phi_otm = np.where(converter, -phi, phi)
    kind_otm = np.where(phi_otm > 0, CALL, PUT)

    def preco(v: NDArray[np.float64]) -> NDArray[np.float64]:
        return bsm_price(S, K, t, r, q, v, kind_otm)

    baixo = np.full(alvo_otm.shape, vol_min, dtype=np.float64)
    alto = np.full(alvo_otm.shape, vol_max, dtype=np.float64)
    p_baixo = preco(baixo)
    p_alto = preco(alto)

    # Piso numerico: abaixo disso o premio extrinseco e ruido de double e a
    # vol nao e identificavel, por mais bem condicionado que seja o solver.
    piso = np.maximum(1e-12, 1e-10 * S)
    sem_informacao = ~np.isfinite(alvo_otm) | (alvo_otm <= piso)
    fora_do_intervalo = (alvo_otm < p_baixo - piso) | (alvo_otm > p_alto + piso)

    for _ in range(iteracoes):
        meio = 0.5 * (baixo + alto)
        subir = preco(meio) < alvo_otm
        baixo = np.where(subir, meio, baixo)
        alto = np.where(subir, alto, meio)

    resultado = 0.5 * (baixo + alto)

    # Encostar na fronteira significa que a bisseccao nao encontrou raiz
    # interior -- reportamos ausencia de solucao, nao a fronteira.
    na_fronteira = (resultado <= vol_min * (1 + 1e-6)) | (
        resultado >= vol_max * (1 - 1e-6)
    )
    return np.where(
        sem_informacao | fora_do_intervalo | na_fronteira, np.nan, resultado
    )

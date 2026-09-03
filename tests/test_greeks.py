"""Prova numerica das gregas.

A estrategia aqui nao e "testar se o codigo roda": e verificar cada formula
analitica contra diferencas finitas centrais da grandeza que ela derivada
representa. Se uma formula tiver sinal invertido ou fator errado, estes
testes falham -- que e exatamente o tipo de erro que passou desapercebido
no painel original.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from gama_win.model.greeks import (
    GreeksInputError,
    bsm_greeks,
    bsm_price,
    implied_vol,
)
from gama_win.model.mathx import norm_cdf, norm_pdf

# Grade de cenarios: dentro, no e fora do dinheiro; curto e longo prazo;
# vol baixa e alta. Toda propriedade e verificada em todos eles.
SPOT = 100.0
STRIKES = np.array([80.0, 95.0, 100.0, 105.0, 120.0])
TAUS = np.array([1.0 / 252, 10.0 / 252, 0.5, 1.0])
VOLS = np.array([0.10, 0.22, 0.60])
RATE = 0.105
DIV = 0.035

CENARIOS = [
    (k, t, v, kind)
    for k in STRIKES
    for t in TAUS
    for v in VOLS
    for kind in ("C", "P")
]


def _greeks(strike, tau, vol, kind, spot=SPOT):
    return bsm_greeks(spot, strike, tau, RATE, DIV, vol, kind)


def _preco(strike, tau, vol, kind, spot=SPOT):
    return float(bsm_price(spot, strike, tau, RATE, DIV, vol, kind))


# ---------------------------------------------------------------- normal ---


def test_norm_cdf_valores_conhecidos():
    assert norm_cdf(0.0) == pytest.approx(0.5, abs=1e-15)
    assert norm_cdf(1.959963985) == pytest.approx(0.975, abs=1e-9)
    assert norm_cdf(-1.959963985) == pytest.approx(0.025, abs=1e-9)
    assert norm_cdf(8.0) == pytest.approx(1.0, abs=1e-14)


def test_norm_pdf_valores_conhecidos():
    assert norm_pdf(0.0) == pytest.approx(1.0 / math.sqrt(2 * math.pi), abs=1e-15)
    assert norm_pdf(1.0) == pytest.approx(0.24197072451914337, abs=1e-15)


def test_norm_cdf_vetorizada_preserva_forma():
    x = np.linspace(-3, 3, 13).reshape(13, 1)
    assert norm_cdf(x).shape == (13, 1)
    assert np.all(np.diff(norm_cdf(np.linspace(-5, 5, 200))) > 0)


# ------------------------------------------------- identidades analiticas ---


@pytest.mark.parametrize("strike,tau,vol", [(k, t, v) for k in STRIKES for t in TAUS for v in VOLS])
def test_gamma_identico_call_e_put(strike, tau, vol):
    """Gama de call e put no mesmo strike sao iguais. Se divergirem, ha erro
    de sinal em algum lugar do delta."""
    g_call = _greeks(strike, tau, vol, "C").gamma
    g_put = _greeks(strike, tau, vol, "P").gamma
    assert float(g_call) == pytest.approx(float(g_put), rel=1e-12)


@pytest.mark.parametrize("strike,tau,vol", [(k, t, v) for k in STRIKES for t in TAUS for v in VOLS])
def test_vanna_identica_call_e_put(strike, tau, vol):
    v_call = _greeks(strike, tau, vol, "C").vanna
    v_put = _greeks(strike, tau, vol, "P").vanna
    assert float(v_call) == pytest.approx(float(v_put), rel=1e-12)


@pytest.mark.parametrize("strike,tau,vol", [(k, t, v) for k in STRIKES for t in TAUS for v in VOLS])
def test_paridade_call_put(strike, tau, vol):
    """C - P = S*exp(-q*tau) - K*exp(-r*tau). Valida o precificador inteiro."""
    c = _preco(strike, tau, vol, "C")
    p = _preco(strike, tau, vol, "P")
    esperado = SPOT * math.exp(-DIV * tau) - strike * math.exp(-RATE * tau)
    assert c - p == pytest.approx(esperado, abs=1e-10, rel=1e-10)


@pytest.mark.parametrize("strike,tau,vol", [(k, t, v) for k in STRIKES for t in TAUS for v in VOLS])
def test_paridade_delta_call_put(strike, tau, vol):
    """delta_call - delta_put = exp(-q*tau)."""
    dc = float(_greeks(strike, tau, vol, "C").delta)
    dp = float(_greeks(strike, tau, vol, "P").delta)
    assert dc - dp == pytest.approx(math.exp(-DIV * tau), abs=1e-12)


# --------------------------------------------------- diferencas finitas ----


@pytest.mark.parametrize("strike,tau,vol,kind", CENARIOS)
def test_delta_bate_com_derivada_do_preco(strike, tau, vol, kind):
    h = SPOT * 1e-5
    fd = (
        _preco(strike, tau, vol, kind, SPOT + h)
        - _preco(strike, tau, vol, kind, SPOT - h)
    ) / (2 * h)
    analitico = float(_greeks(strike, tau, vol, kind).delta)
    assert analitico == pytest.approx(fd, abs=1e-6, rel=1e-5)


@pytest.mark.parametrize("strike,tau,vol,kind", CENARIOS)
def test_gamma_bate_com_derivada_do_delta(strike, tau, vol, kind):
    """gamma = d(delta)/d(spot)."""
    h = SPOT * 1e-5
    d_mais = float(_greeks(strike, tau, vol, kind, SPOT + h).delta)
    d_menos = float(_greeks(strike, tau, vol, kind, SPOT - h).delta)
    fd = (d_mais - d_menos) / (2 * h)
    analitico = float(_greeks(strike, tau, vol, kind).gamma)
    assert analitico == pytest.approx(fd, abs=1e-8, rel=1e-5)


@pytest.mark.parametrize("strike,tau,vol,kind", CENARIOS)
def test_vega_bate_com_derivada_do_preco(strike, tau, vol, kind):
    h = 1e-6
    fd = (
        _preco(strike, tau, vol + h, kind) - _preco(strike, tau, vol - h, kind)
    ) / (2 * h)
    analitico = float(_greeks(strike, tau, vol, kind).vega)
    assert analitico == pytest.approx(fd, abs=1e-5, rel=1e-5)


@pytest.mark.parametrize("strike,tau,vol,kind", CENARIOS)
def test_vanna_bate_com_derivada_do_delta_em_vol(strike, tau, vol, kind):
    """vanna = d(delta)/d(vol)."""
    h = 1e-6
    d_mais = float(_greeks(strike, tau, vol + h, kind).delta)
    d_menos = float(_greeks(strike, tau, vol - h, kind).delta)
    fd = (d_mais - d_menos) / (2 * h)
    analitico = float(_greeks(strike, tau, vol, kind).vanna)
    assert analitico == pytest.approx(fd, abs=1e-6, rel=1e-4)


@pytest.mark.parametrize(
    "strike,tau,vol,kind",
    [(k, t, v, kd) for k in STRIKES for t in TAUS[1:] for v in VOLS for kd in ("C", "P")],
)
def test_charm_bate_com_derivada_do_delta_em_tau(strike, tau, vol, kind):
    """charm = d(delta)/d(tau). Este e o teste que pega inversao de sinal --
    o erro classico em charm, porque metade da literatura define a derivada
    em relacao ao tempo de calendario, que tem sinal oposto."""
    h = tau * 1e-5
    d_mais = float(_greeks(strike, tau + h, vol, kind).delta)
    d_menos = float(_greeks(strike, tau - h, vol, kind).delta)
    fd = (d_mais - d_menos) / (2 * h)
    analitico = float(_greeks(strike, tau, vol, kind).charm)
    assert analitico == pytest.approx(fd, abs=1e-5, rel=1e-4)


def test_charm_por_dia_pregao_tem_sinal_de_tempo_decorrido():
    """Passar um dia significa tau menor. O helper precisa refletir isso:
    delta(tau - 1/252) - delta(tau) deve bater com charm_por_dia_pregao."""
    strike, tau, vol, kind = 105.0, 30.0 / 252, 0.22, "C"
    g = _greeks(strike, tau, vol, kind)
    passo = 1.0 / 252
    variacao_real = float(
        _greeks(strike, tau - passo, vol, kind).delta
    ) - float(g.delta)
    assert float(g.charm_por_dia_pregao()) == pytest.approx(
        variacao_real, rel=0.02
    )


# ------------------------------------------------ comportamento esperado ---


def test_gamma_maximo_perto_do_dinheiro():
    strikes = np.arange(70.0, 131.0, 1.0)
    g = bsm_greeks(SPOT, strikes, 30.0 / 252, RATE, DIV, 0.22, "C").gamma
    pico = strikes[int(np.argmax(g))]
    assert abs(pico - SPOT) <= 3.0


def test_gamma_cresce_conforme_vencimento_se_aproxima_no_dinheiro():
    taus = np.array([60.0, 30.0, 10.0, 2.0]) / 252
    g = bsm_greeks(SPOT, SPOT, taus, RATE, DIV, 0.22, "C").gamma
    assert np.all(np.diff(g) > 0), "gama no dinheiro deve crescer com tau caindo"


def test_broadcasting_de_grade_completa():
    strikes = np.arange(90.0, 111.0, 1.0)
    kinds = np.where(strikes >= SPOT, "C", "P")
    g = bsm_greeks(SPOT, strikes, 20.0 / 252, RATE, DIV, 0.25, kinds)
    assert g.gamma.shape == strikes.shape
    assert np.all(g.gamma > 0)


# ------------------------------------------------------------ validacao ----


def test_tau_zero_levanta_em_vez_de_retornar_zero():
    """O bug mais perigoso do painel original: gama 0 no vencimento."""
    with pytest.raises(GreeksInputError, match="tau <= 0"):
        bsm_greeks(SPOT, 100.0, 0.0, RATE, DIV, 0.22, "C")
    with pytest.raises(GreeksInputError, match="tau <= 0"):
        bsm_greeks(SPOT, 100.0, -0.5, RATE, DIV, 0.22, "C")


def test_tau_floor_clampa_quando_explicito():
    piso = 1.0 / 252 / 6.5
    g = bsm_greeks(SPOT, 100.0, 0.0, RATE, DIV, 0.22, "C", tau_floor=piso)
    esperado = bsm_greeks(SPOT, 100.0, piso, RATE, DIV, 0.22, "C")
    assert float(g.gamma) == pytest.approx(float(esperado.gamma), rel=1e-12)
    assert float(g.gamma) > 0.05, "gama no vencimento tem de ser grande"


def test_vol_nao_positiva_levanta():
    with pytest.raises(GreeksInputError, match="vol"):
        bsm_greeks(SPOT, 100.0, 0.1, RATE, DIV, 0.0, "C")
    with pytest.raises(GreeksInputError, match="vol"):
        bsm_greeks(SPOT, 100.0, 0.1, RATE, DIV, -0.22, "C")


def test_spot_e_strike_nao_positivos_levantam():
    with pytest.raises(GreeksInputError, match="spot"):
        bsm_greeks(0.0, 100.0, 0.1, RATE, DIV, 0.22, "C")
    with pytest.raises(GreeksInputError, match="strike"):
        bsm_greeks(SPOT, -1.0, 0.1, RATE, DIV, 0.22, "C")


def test_nan_levanta_em_vez_de_propagar():
    with pytest.raises(GreeksInputError, match="nao finito"):
        bsm_greeks(SPOT, np.nan, 0.1, RATE, DIV, 0.22, "C")
    with pytest.raises(GreeksInputError, match="nao finito"):
        bsm_greeks(SPOT, 100.0, 0.1, np.nan, DIV, 0.22, "C")


def test_kind_invalido_levanta():
    with pytest.raises(GreeksInputError, match="'C' ou 'P'"):
        bsm_greeks(SPOT, 100.0, 0.1, RATE, DIV, 0.22, "X")
    with pytest.raises(GreeksInputError, match="'C' ou 'P'"):
        bsm_greeks(SPOT, [100.0, 105.0], 0.1, RATE, DIV, 0.22, ["C", "CALL"])


def test_kind_aceita_minuscula_e_espaco():
    a = float(bsm_greeks(SPOT, 100.0, 0.1, RATE, DIV, 0.22, " c ").gamma)
    b = float(bsm_greeks(SPOT, 100.0, 0.1, RATE, DIV, 0.22, "C").gamma)
    assert a == pytest.approx(b, rel=1e-15)


# --------------------------------------------------------- vol implicita ---


def _valor_extrinseco(strike, tau, vol):
    """Premio da opcao FORA do dinheiro no strike -- o valor que carrega vega."""
    forward = SPOT * math.exp((RATE - DIV) * tau)
    kind_otm = "C" if strike >= forward else "P"
    return _preco(strike, tau, vol, kind_otm)


@pytest.mark.parametrize("strike,tau,vol,kind", CENARIOS)
def test_implied_vol_roundtrip(strike, tau, vol, kind):
    """Contrato: recupera a vol sempre que o valor extrinseco a determina, e
    devolve NaN quando nao determina. Nunca um numero errado no meio."""
    preco = _preco(strike, tau, vol, kind)
    recuperada = float(implied_vol(preco, SPOT, strike, tau, RATE, DIV, kind))

    if _valor_extrinseco(strike, tau, vol) > 1e-10 * SPOT:
        assert recuperada == pytest.approx(vol, abs=1e-6)
    else:
        assert math.isnan(recuperada), (
            "sem valor extrinseco a vol nao e identificavel; o retorno tem de "
            f"ser NaN e nao {recuperada}"
        )


def test_implied_vol_dentro_do_dinheiro_com_extrinseco_recupera():
    """Prova a conversao por paridade: call profundamente dentro do dinheiro
    mas com prazo longo tem extrinseco relevante e a vol tem de sair."""
    strike, tau, vol = 80.0, 1.0, 0.60
    preco = _preco(strike, tau, vol, "C")
    assert preco > SPOT - strike, "deve haver extrinseco neste cenario"
    rec = float(implied_vol(preco, SPOT, strike, tau, RATE, DIV, "C"))
    assert rec == pytest.approx(vol, abs=1e-6)


def test_implied_vol_sem_extrinseco_nao_devolve_fronteira():
    """Regressao: antes o solver devolvia vol_min (0.0001) como se fosse
    resposta em opcoes no intrinseco puro. Tem de ser NaN."""
    strike, tau, vol = 80.0, 1.0 / 252, 0.10
    preco = _preco(strike, tau, vol, "C")
    rec = float(implied_vol(preco, SPOT, strike, tau, RATE, DIV, "C"))
    assert math.isnan(rec)


def test_implied_vol_fora_do_intervalo_retorna_nan():
    intrinseco_impossivel = SPOT * 10
    r = implied_vol(intrinseco_impossivel, SPOT, 100.0, 0.5, RATE, DIV, "C")
    assert np.isnan(r)


def test_implied_vol_vetorizada():
    strikes = np.array([95.0, 100.0, 105.0])
    vols = np.array([0.20, 0.22, 0.25])
    precos = bsm_price(SPOT, strikes, 0.25, RATE, DIV, vols, "C")
    rec = implied_vol(precos, SPOT, strikes, 0.25, RATE, DIV, "C")
    assert np.allclose(rec, vols, atol=1e-6)

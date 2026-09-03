"""Relatorio do backtest e gerador de dias de demonstracao."""

from __future__ import annotations

import numpy as np
import pytest

from gama_win.backtest.demo import dias_demo
from gama_win.backtest.engine import (
    REGIME_NEG,
    REGIME_POS,
    ConfigBacktest,
    rodar_backtest,
)
from gama_win.view.relatorio_backtest import relatorio_backtest

CFG = ConfigBacktest(
    horizonte_barras=6, lookback_barras=3, n_reamostras=200, min_dias=5
)


@pytest.fixture(scope="module")
def resultado_demo():
    return rodar_backtest(dias_demo(n_dias=10, barras_por_dia=40), CFG)


# ---------------------------------------------------------------- demo ---


def test_demo_gera_dias_uteis_sem_repetir():
    dias = dias_demo(n_dias=12, barras_por_dia=20)
    assert len(dias) == 12
    datas = [d.data for d in dias]
    assert len(set(datas)) == 12
    assert all(d.weekday() < 5 for d in datas), "demo nao deve gerar fim de semana"
    assert datas == sorted(datas)


def test_demo_e_reprodutivel():
    a = dias_demo(n_dias=5, barras_por_dia=20, semente=11)
    b = dias_demo(n_dias=5, barras_por_dia=20, semente=11)
    for da, db in zip(a, b, strict=True):
        assert da.data == db.data
        assert np.allclose(da.barras["close"], db.barras["close"])


def test_demo_avalia_gama_dentro_da_faixa():
    dia = dias_demo(n_dias=1, barras_por_dia=20)[0]
    g = dia.avaliar_gama(dia.barras["close"].to_numpy(float))
    assert np.any(np.isfinite(g)), "o preco de abertura deve cair na faixa"


def test_demo_passeio_aleatorio_nao_produz_efeito(resultado_demo):
    """Se a demo mostrasse edge, seria bug no motor."""
    for reg in (REGIME_NEG, REGIME_POS):
        if reg in resultado_demo.por_regime:
            assert not resultado_demo.por_regime[reg].dif_e_significativa


# ------------------------------------------------------------ relatorio ---


def test_relatorio_tem_baseline_e_regimes(resultado_demo):
    txt = relatorio_backtest(resultado_demo)
    assert "BASELINE" in txt
    assert "POR REGIME DE GAMA" in txt
    assert "vs baseline" in txt


def test_relatorio_explica_a_metrica_principal(resultado_demo):
    txt = relatorio_backtest(resultado_demo)
    assert "nao esta explicando nada alem de momentum" in txt


def test_relatorio_marca_significancia(resultado_demo):
    txt = relatorio_backtest(resultado_demo)
    assert "nao significativa" in txt or "SIGNIFICATIVA" in txt


def test_relatorio_traz_avisos_e_lembrete_de_lookahead(resultado_demo):
    txt = relatorio_backtest(resultado_demo)
    assert "AVISOS" in txt
    assert "OI de D-1" in txt


def test_relatorio_reporta_n_e_dias(resultado_demo):
    txt = relatorio_backtest(resultado_demo)
    assert "n=" in txt
    assert "dia(s)" in txt


def test_relatorio_de_resultado_vazio_nao_quebra():
    cfg = ConfigBacktest(
        horizonte_barras=100, lookback_barras=100, n_reamostras=50, min_dias=1
    )
    r = rodar_backtest(dias_demo(n_dias=3, barras_por_dia=20), cfg)
    assert r.n_eventos == 0
    txt = relatorio_backtest(r)
    assert "n/d" in txt
    assert "nenhum evento elegivel" in txt


def test_relatorio_usa_formatacao_pt_br(resultado_demo):
    txt = relatorio_backtest(resultado_demo)
    # percentuais com virgula decimal, nao ponto
    assert "%" in txt
    linhas = [l for l in txt.splitlines() if "continuacao ....." in l]
    assert linhas
    assert any("," in l for l in linhas)

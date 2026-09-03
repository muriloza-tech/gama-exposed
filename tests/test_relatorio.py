"""Formatacao e relatorio.

Parece detalhe, mas numero mal formatado em painel de operacao e erro de
leitura: 'x1.048.00' foi um bug real deste arquivo, causado por trocar os
separadores em sequencia em vez de simultaneamente.
"""

from __future__ import annotations

from datetime import date

import pytest

from gama_win.data.sources.sintetica import FonteSintetica
from gama_win.model.levels import extrair_niveis
from gama_win.model.profile import construir_perfil
from gama_win.view.relatorio import brl, num, relatorio_texto

AS_OF = date(2026, 9, 3)


@pytest.mark.parametrize(
    "valor,esperado",
    [
        (1048.0, "1.048,00"),
        (176.5, "176,50"),
        (1_234_567.89, "1.234.567,89"),
        (0.0, "0,00"),
        (-1048.5, "-1.048,50"),
    ],
)
def test_num_formata_pt_br(valor, esperado):
    assert num(valor) == esperado


def test_num_sem_decimais():
    assert num(189731.0, 0) == "189.731"


@pytest.mark.parametrize(
    "valor,esperado",
    [
        (2_500_000_000.0, "R$ 2,50 bi"),
        (108_310_000.0, "R$ 108,31 mi"),
        (445_600.0, "R$ 445,6 mil"),
        (-1_684_000_000.0, "-R$ 1,68 bi"),
        (12.5, "R$ 12,50"),
    ],
)
def test_brl_escala_e_sinal(valor, esperado):
    assert brl(valor) == esperado


def _relatorio() -> str:
    chain = FonteSintetica(spot=176.5).buscar(AS_OF)
    perfil = construir_perfil(chain)
    niveis = extrair_niveis(perfil)
    return relatorio_texto(perfil, niveis, ratio_win=1048.0)


def test_relatorio_sintetico_traz_marca_dagua():
    txt = _relatorio()
    assert "DADOS SINTETICOS -- NAO OPERAR" in txt
    assert "nao devem orientar operacao" in txt


def test_relatorio_moldura_alinhada():
    """Toda linha da moldura de aviso tem a mesma largura."""
    linhas = [l for l in _relatorio().splitlines() if l.startswith("#")]
    assert linhas, "a moldura deve existir em relatorio sintetico"
    assert len({len(l) for l in linhas}) == 1


def test_relatorio_declara_convencao_e_procedencia():
    txt = _relatorio()
    assert "dealer long call / short put" in txt
    assert "SYNTHETIC" in txt
    assert "2026-09-03" in txt


def test_relatorio_traz_unidades():
    txt = _relatorio()
    assert "por 1% de movimento" in txt
    assert "por dia de pregao" in txt
    assert "por ponto de vol" in txt


def test_relatorio_converte_para_pontos_win():
    assert "[WIN" in _relatorio()


def test_relatorio_sem_ratio_nao_menciona_win():
    chain = FonteSintetica(spot=176.5).buscar(AS_OF)
    perfil = construir_perfil(chain)
    txt = relatorio_texto(perfil, extrair_niveis(perfil))
    assert "[WIN" not in txt
    assert "Razao -> WIN" not in txt


def test_fonte_sintetica_produz_perfil_com_um_flip_dominante():
    """Diferenca em relacao ao gerador de seno/cosseno: forma realista de OI
    gera poucos cruzamentos, nao seis."""
    chain = FonteSintetica(spot=176.5).buscar(AS_OF)
    niveis = extrair_niveis(construir_perfil(chain))
    assert len(niveis.flips_todos) <= 4
    assert niveis.flip is not None


def test_fonte_sintetica_wall_do_lado_certo():
    chain = FonteSintetica(spot=176.5).buscar(AS_OF)
    n = extrair_niveis(construir_perfil(chain))
    assert n.call_wall is not None and n.call_wall.pico >= 176.5
    assert n.put_wall is not None and n.put_wall.pico <= 176.5


def test_fonte_sintetica_e_reprodutivel():
    a = FonteSintetica(spot=176.5, semente=7).buscar(AS_OF)
    b = FonteSintetica(spot=176.5, semente=7).buscar(AS_OF)
    assert a.total_open_interest == b.total_open_interest

"""Niveis: flip mais proximo do spot, walls do lado certo, zonas em vez de
linhas. Os testes deste arquivo sao regressao direta dos defeitos do painel
original."""

from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd
import pytest

from gama_win.model.conventions import DealerConvention
from gama_win.model.levels import extrair_niveis
from gama_win.model.profile import GammaProfile

AS_OF = date(2026, 9, 3)


def _perfil(
    strikes,
    liquido,
    *,
    call=None,
    put=None,
    spot=100.0,
    is_synthetic=False,
) -> GammaProfile:
    """Fabrica um perfil com valores controlados, para testar `levels` isolado."""
    strikes = np.asarray(strikes, dtype=float)
    liquido = np.asarray(liquido, dtype=float)
    n = len(strikes)
    call = np.zeros(n) if call is None else np.asarray(call, dtype=float)
    put = np.zeros(n) if put is None else np.asarray(put, dtype=float)
    return GammaProfile(
        strikes=strikes,
        gama_brl_1pct=liquido,
        gama_call_brl_1pct=call,
        gama_put_brl_1pct=put,
        gama_bruto_brl_1pct=np.abs(call) + np.abs(put),
        charm_brl_dia=np.zeros(n),
        vanna_brl_por_ponto_vol=np.zeros(n),
        oi_call=np.zeros(n, dtype=np.int64),
        oi_put=np.zeros(n, dtype=np.int64),
        spot=spot,
        underlying="TESTE",
        as_of=AS_OF,
        source="teste",
        is_synthetic=is_synthetic,
        convention=DealerConvention.LONG_CALL_SHORT_PUT,
        expiries=(date(2026, 9, 18),),
        vol_media_ponderada=0.22,
        por_vencimento=pd.DataFrame(index=pd.Index(strikes, name="strike")),
    )


# ---------------------------------------------------------------- flip ---


def test_flip_interpolado_entre_strikes():
    """Cruzamento em 100.5, entre os strikes 100 e 101."""
    p = _perfil([99.0, 100.0, 101.0, 102.0], [2.0, 1.0, -1.0, -2.0])
    n = extrair_niveis(p)
    assert n.flip == pytest.approx(100.5)


def test_flip_escolhe_o_cruzamento_mais_proximo_do_spot():
    """REGRESSAO: o painel original pegava o primeiro cruzamento da esquerda,
    que ficava a 5% do preco e nao governava nada."""
    strikes = [90.0, 91.0, 95.0, 96.0, 100.0, 101.0]
    liquido = [1.0, -1.0, -1.0, 1.0, 1.0, -1.0]
    p = _perfil(strikes, liquido, spot=100.0)
    n = extrair_niveis(p)
    assert len(n.flips_todos) == 3
    assert n.flip == pytest.approx(100.5)
    assert any("mais proximo do spot" in a for a in n.avisos)


def test_flip_ausente_quando_nao_ha_cruzamento():
    p = _perfil([99.0, 100.0, 101.0], [1.0, 2.0, 3.0])
    n = extrair_niveis(p)
    assert n.flip is None
    assert n.distancia_ao_flip is None
    assert any("sem cruzamento" in a for a in n.avisos)


def test_flip_em_cima_de_strike_com_valor_zero():
    p = _perfil([99.0, 100.0, 101.0], [1.0, 0.0, -1.0])
    assert extrair_niveis(p).flip == pytest.approx(100.0)


def test_distancia_ao_flip_tem_sinal():
    p = _perfil([99.0, 100.0, 101.0, 102.0], [2.0, 1.0, -1.0, -2.0], spot=99.0)
    n = extrair_niveis(p)
    assert n.distancia_ao_flip == pytest.approx(1.5)
    p2 = _perfil([99.0, 100.0, 101.0, 102.0], [2.0, 1.0, -1.0, -2.0], spot=102.0)
    assert extrair_niveis(p2).distancia_ao_flip == pytest.approx(-1.5)


# --------------------------------------------------------------- walls ---


def test_call_wall_nunca_abaixo_do_spot():
    """REGRESSAO CRITICA: o painel original desenhou 'TETO MAXIMO' abaixo do
    preco atual. Mesmo com a maior exposicao de call embaixo, a call wall tem
    de sair do lado de cima."""
    strikes = [90.0, 95.0, 100.0, 105.0, 110.0]
    call = [900.0, 800.0, 10.0, 50.0, 20.0]  # pico artificial em 90
    p = _perfil(strikes, [1.0, 1.0, 1.0, -1.0, -1.0], call=call, spot=100.0)
    n = extrair_niveis(p)
    assert n.call_wall is not None
    assert n.call_wall.pico >= 100.0
    assert n.call_wall.pico == pytest.approx(105.0)


def test_put_wall_nunca_acima_do_spot():
    strikes = [90.0, 95.0, 100.0, 105.0, 110.0]
    put = [30.0, 60.0, 10.0, 900.0, 800.0]  # pico artificial em 105
    p = _perfil(strikes, [1.0] * 5, put=put, spot=100.0)
    n = extrair_niveis(p)
    assert n.put_wall is not None
    assert n.put_wall.pico <= 100.0
    assert n.put_wall.pico == pytest.approx(95.0)


def test_wall_e_zona_contigua_acima_do_limiar():
    """Pico em 106 com vizinho em 105 valendo 60% do pico: os dois entram."""
    strikes = [100.0, 105.0, 106.0, 107.0, 110.0]
    call = [1.0, 60.0, 100.0, 20.0, 5.0]
    p = _perfil(strikes, [1.0] * 5, call=call, spot=100.0)
    z = extrair_niveis(p).call_wall
    assert z.pico == pytest.approx(106.0)
    assert z.inicio == pytest.approx(105.0)
    assert z.fim == pytest.approx(106.0)
    assert z.largura == pytest.approx(1.0)
    assert z.exposicao_zona == pytest.approx(160.0)
    assert z.contem(105.5)
    assert not z.contem(107.0)


def test_wall_de_um_unico_strike_tem_largura_zero():
    strikes = [100.0, 105.0, 110.0]
    call = [1.0, 100.0, 1.0]
    z = extrair_niveis(_perfil(strikes, [1.0] * 3, call=call, spot=100.0)).call_wall
    assert z.inicio == z.fim == pytest.approx(105.0)
    assert z.largura == 0.0


def test_sem_strike_acima_do_spot_nao_inventa_call_wall():
    strikes = [90.0, 95.0]
    p = _perfil(strikes, [1.0, 1.0], call=[10.0, 20.0], spot=100.0)
    n = extrair_niveis(p)
    assert n.call_wall is None
    assert any("nao ha strike >= spot" in a for a in n.avisos)


def test_sem_exposicao_de_call_nao_inventa_wall():
    strikes = [100.0, 105.0]
    p = _perfil(strikes, [1.0, 1.0], call=[0.0, 0.0], put=[5.0, 5.0], spot=100.0)
    n = extrair_niveis(p)
    assert n.call_wall is None
    assert any("nenhuma exposicao de call" in a for a in n.avisos)


# --------------------------------------------------------- diagnostico ---


def test_spot_fora_da_faixa_avisa():
    p = _perfil([90.0, 95.0, 100.0], [1.0, 1.0, 1.0], spot=150.0)
    assert any("FORA da faixa" in a for a in extrair_niveis(p).avisos)


def test_dados_sinteticos_geram_aviso_de_nao_operar():
    p = _perfil([99.0, 100.0, 101.0], [1.0, 0.0, -1.0], is_synthetic=True)
    avisos = extrair_niveis(p).avisos
    assert any("SINTETICOS" in a and "nao devem orientar operacao" in a for a in avisos)


def test_perfil_muito_alternado_gera_aviso():
    """O perfil de seno/cosseno do painel original: 6 trocas de sinal."""
    strikes = np.arange(95.0, 106.0, 1.0)
    liquido = np.array([1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0])
    n = extrair_niveis(_perfil(strikes, liquido, spot=100.0))
    assert len(n.flips_todos) == 10
    assert any("alternado" in a for a in n.avisos)


def test_regime_e_gama_no_spot_sao_reportados():
    p = _perfil([99.0, 100.0, 101.0], [2.0, 1.0, -1.0], spot=100.0)
    n = extrair_niveis(p)
    assert n.gama_no_spot == pytest.approx(1.0)
    assert n.regime == "gama positivo"
    assert n.spot == 100.0

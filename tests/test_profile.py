"""Perfil de gama: unidades, convencao de sinal e ausencia de default oculto."""

from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd
import pytest

from gama_win.data.schema import OptionChain
from gama_win.model.calendario import tau_anos
from gama_win.model.conventions import DealerConvention
from gama_win.model.greeks import bsm_greeks
from gama_win.model.profile import ProfileError, construir_perfil

AS_OF = date(2026, 9, 3)
VENC = date(2026, 9, 18)
VENC2 = date(2026, 10, 16)
SPOT = 100.0
RATE = 0.105
DIV = 0.035
CS = 100.0


def _chain(rows: list[dict], **kwargs) -> OptionChain:
    params = dict(
        df=pd.DataFrame(rows),
        underlying="BOVA11",
        spot=SPOT,
        as_of=AS_OF,
        source="teste",
        rate=RATE,
        div_yield=DIV,
    )
    params.update(kwargs)
    return OptionChain(**params)


def _linha(strike, kind, oi, iv=0.22, expiry=VENC):
    return {
        "expiry": expiry,
        "strike": strike,
        "kind": kind,
        "open_interest": oi,
        "contract_size": CS,
        "implied_vol": iv,
    }


# ------------------------------------------------------------- unidades ---


def test_unidade_gama_brl_1pct_bate_com_a_formula():
    """gama_brl_1pct = gamma * OI * contract_size * spot^2 * 0.01.

    Este teste ancora a unidade. Se alguem mexer na formula, ele falha.
    """
    oi = 1000
    iv = 0.22
    chain = _chain([_linha(99.0, "C", 1), _linha(100.0, "C", oi, iv)])
    p = construir_perfil(chain)

    tau = tau_anos(AS_OF, VENC)
    g = bsm_greeks(SPOT, 100.0, tau, RATE, DIV, iv, "C")
    esperado = float(g.gamma) * oi * CS * SPOT**2 * 0.01

    i = int(np.where(p.strikes == 100.0)[0][0])
    assert p.gama_brl_1pct[i] == pytest.approx(esperado, rel=1e-12)


def test_unidade_charm_brl_dia_bate_com_a_formula():
    oi = 1000
    iv = 0.22
    chain = _chain([_linha(99.0, "C", 1), _linha(100.0, "C", oi, iv)])
    p = construir_perfil(chain)

    tau = tau_anos(AS_OF, VENC)
    g = bsm_greeks(SPOT, 100.0, tau, RATE, DIV, iv, "C")
    esperado = float(g.charm_por_dia_pregao()) * oi * CS * SPOT

    i = int(np.where(p.strikes == 100.0)[0][0])
    assert p.charm_brl_dia[i] == pytest.approx(esperado, rel=1e-12)


def test_unidade_vanna_brl_por_ponto_vol():
    oi = 1000
    iv = 0.22
    chain = _chain([_linha(99.0, "C", 1), _linha(105.0, "C", oi, iv)])
    p = construir_perfil(chain)

    tau = tau_anos(AS_OF, VENC)
    g = bsm_greeks(SPOT, 105.0, tau, RATE, DIV, iv, "C")
    esperado = float(g.vanna) * 0.01 * oi * CS * SPOT

    i = int(np.where(p.strikes == 105.0)[0][0])
    assert p.vanna_brl_por_ponto_vol[i] == pytest.approx(esperado, rel=1e-12)


# ---------------------------------------------------- convencao de sinal ---


def test_convencao_padrao_call_positiva_put_negativa():
    chain = _chain([_linha(100.0, "C", 1000), _linha(95.0, "P", 1000)])
    p = construir_perfil(chain, convention=DealerConvention.LONG_CALL_SHORT_PUT)
    assert p.gama_call_brl_1pct.sum() > 0
    assert p.gama_put_brl_1pct.sum() < 0


def test_convencao_invertida_espelha_o_sinal():
    chain = _chain([_linha(100.0, "C", 1000), _linha(95.0, "P", 1000)])
    padrao = construir_perfil(
        chain, convention=DealerConvention.LONG_CALL_SHORT_PUT
    )
    invertida = construir_perfil(
        chain, convention=DealerConvention.SHORT_CALL_LONG_PUT
    )
    assert np.allclose(padrao.gama_brl_1pct, -invertida.gama_brl_1pct)


def test_convencao_fica_registrada_no_resultado():
    """Nenhum resultado sai sem dizer qual convencao produziu."""
    chain = _chain([_linha(100.0, "C", 1000), _linha(95.0, "P", 1000)])
    p = construir_perfil(chain)
    assert p.convention is DealerConvention.LONG_CALL_SHORT_PUT
    assert "padrao de mercado" in p.convention.descricao


def test_gama_bruto_ignora_sinal():
    chain = _chain(
        [
            _linha(100.0, "C", 1000),
            _linha(100.0, "P", 1000),
            _linha(105.0, "C", 400),
        ]
    )
    p = construir_perfil(chain)
    assert np.all(p.gama_bruto_brl_1pct >= np.abs(p.gama_brl_1pct))
    # Call e put com o MESMO OI no mesmo strike se cancelam no liquido
    # (gama identico, sinais opostos) mas somam no bruto.
    i = int(np.where(p.strikes == 100.0)[0][0])
    assert p.gama_brl_1pct[i] == pytest.approx(0.0, abs=1e-6)
    assert p.gama_bruto_brl_1pct[i] > 0


# -------------------------------------------------- sem default oculto ----


def test_vol_ausente_sem_fallback_levanta_com_explicacao():
    """O painel original usava 22% plana para tudo, silenciosamente."""
    chain = _chain(
        [_linha(99.0, "C", 1000, iv=None), _linha(100.0, "C", 1000, iv=None)]
    )
    with pytest.raises(ProfileError, match="vol_fallback"):
        construir_perfil(chain)


def test_vol_fallback_explicito_funciona():
    chain = _chain(
        [_linha(99.0, "C", 1000, iv=None), _linha(100.0, "C", 1000, iv=None)]
    )
    p = construir_perfil(chain, vol_fallback=0.22)
    assert np.all(p.gama_brl_1pct > 0)
    assert p.vol_media_ponderada == pytest.approx(0.22)


def test_vol_fallback_invalido_levanta():
    chain = _chain(
        [_linha(99.0, "C", 1000, iv=None), _linha(100.0, "C", 1000, iv=None)]
    )
    with pytest.raises(ProfileError, match="vol_fallback deve ser"):
        construir_perfil(chain, vol_fallback=0.0)


def test_smile_e_respeitado_quando_presente():
    """Vol por strike deve produzir gama diferente de vol plana."""
    rows_smile = [_linha(95.0, "P", 1000, iv=0.30), _linha(105.0, "C", 1000, iv=0.18)]
    rows_plana = [_linha(95.0, "P", 1000, iv=0.22), _linha(105.0, "C", 1000, iv=0.22)]
    p_smile = construir_perfil(_chain(rows_smile))
    p_plana = construir_perfil(_chain(rows_plana))
    assert not np.allclose(p_smile.gama_brl_1pct, p_plana.gama_brl_1pct)


# ------------------------------------------------------- agregacao -------


def test_agrega_calls_e_puts_no_mesmo_strike():
    chain = _chain(
        [
            _linha(100.0, "C", 1000),
            _linha(100.0, "P", 1000),
            _linha(105.0, "C", 500),
        ]
    )
    p = construir_perfil(chain)
    assert list(p.strikes) == [100.0, 105.0]
    i = 0
    assert p.oi_call[i] == 1000 and p.oi_put[i] == 1000
    assert p.gama_brl_1pct[i] == pytest.approx(
        p.gama_call_brl_1pct[i] + p.gama_put_brl_1pct[i]
    )


def test_agrega_multiplos_vencimentos_e_guarda_a_quebra():
    chain = _chain(
        [
            _linha(100.0, "C", 1000, expiry=VENC),
            _linha(100.0, "C", 800, expiry=VENC2),
            _linha(105.0, "C", 300, expiry=VENC),
        ]
    )
    p = construir_perfil(chain)
    assert p.expiries == (VENC, VENC2)
    assert list(p.por_vencimento.columns) == [VENC, VENC2]
    # a soma da quebra por vencimento reproduz o liquido
    assert np.allclose(
        p.por_vencimento.sum(axis=1).to_numpy(), p.gama_brl_1pct
    )


def test_filtro_por_vencimento():
    chain = _chain(
        [
            _linha(100.0, "C", 1000, expiry=VENC),
            _linha(100.0, "C", 800, expiry=VENC2),
            _linha(105.0, "C", 300, expiry=VENC),
        ]
    )
    p = construir_perfil(chain, expiries=(VENC,))
    assert p.expiries == (VENC,)


def test_filtro_por_vencimento_inexistente_levanta_listando_disponiveis():
    chain = _chain([_linha(100.0, "C", 1000), _linha(105.0, "C", 300)])
    with pytest.raises(ProfileError, match="disponiveis"):
        construir_perfil(chain, expiries=(date(2027, 1, 15),))


def test_vencimento_curto_domina_o_gama():
    """Propriedade estrutural: mesmo OI, prazo curto gera mais gama."""
    curto = construir_perfil(
        _chain([_linha(99.0, "C", 1), _linha(100.0, "C", 1000, expiry=VENC)])
    )
    longo = construir_perfil(
        _chain([_linha(99.0, "C", 1), _linha(100.0, "C", 1000, expiry=VENC2)])
    )
    i_c = int(np.where(curto.strikes == 100.0)[0][0])
    i_l = int(np.where(longo.strikes == 100.0)[0][0])
    assert curto.gama_brl_1pct[i_c] > longo.gama_brl_1pct[i_l]


# ---------------------------------------------------------- regime -------


def test_gama_no_spot_e_interpolado_nao_do_strike_vizinho():
    """Com call em 105 e put em 95, o liquido no spot 100 vem da reta entre
    os dois pontos -- nao do valor de um deles."""
    chain = _chain([_linha(95.0, "P", 2000), _linha(105.0, "C", 2000)])
    p = construir_perfil(chain)
    interpolado = p.gama_no_spot()
    assert min(p.gama_brl_1pct) <= interpolado <= max(p.gama_brl_1pct)
    assert interpolado not in set(p.gama_brl_1pct.tolist())


def test_regime_positivo_com_calls_dominando():
    chain = _chain([_linha(99.0, "C", 5000), _linha(101.0, "C", 5000)])
    assert construir_perfil(chain).regime() == "gama positivo"


def test_regime_negativo_com_puts_dominando():
    chain = _chain([_linha(99.0, "P", 5000), _linha(101.0, "P", 5000)])
    assert construir_perfil(chain).regime() == "gama negativo"


def test_marca_sintetica_propaga_para_o_perfil():
    chain = _chain(
        [_linha(99.0, "C", 1000), _linha(100.0, "C", 1000)],
        source="SYNTHETIC",
        is_synthetic=True,
    )
    assert construir_perfil(chain).is_synthetic is True


def test_perfil_com_um_unico_strike_levanta():
    chain = _chain([_linha(100.0, "C", 1000), _linha(100.0, "P", 1000)])
    with pytest.raises(ProfileError, match="ao menos 2 strikes"):
        construir_perfil(chain)


def test_dataframe_de_saida_tem_todas_as_colunas():
    chain = _chain([_linha(99.0, "C", 1000), _linha(100.0, "C", 1000)])
    df = construir_perfil(chain).como_dataframe()
    assert list(df.columns) == [
        "strike",
        "gama_brl_1pct",
        "gama_call_brl_1pct",
        "gama_put_brl_1pct",
        "gama_bruto_brl_1pct",
        "charm_brl_dia",
        "vanna_brl_por_ponto_vol",
        "oi_call",
        "oi_put",
    ]

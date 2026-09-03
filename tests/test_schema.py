"""Contrato de dados: o que entra tem de estar limpo, e o erro tem de dizer
exatamente o que esta errado -- tudo de uma vez."""

from __future__ import annotations

from datetime import date

import pandas as pd
import pytest

from gama_win.data.schema import ChainValidationError, OptionChain

AS_OF = date(2026, 9, 3)
VENC = date(2026, 9, 18)


def _df(**overrides) -> pd.DataFrame:
    base = pd.DataFrame(
        {
            "expiry": [VENC, VENC, VENC, VENC],
            "strike": [95.0, 100.0, 95.0, 100.0],
            "kind": ["C", "C", "P", "P"],
            "open_interest": [1000, 2000, 1500, 2500],
            "contract_size": [100.0] * 4,
            "implied_vol": [0.24, 0.22, 0.26, 0.23],
        }
    )
    for k, v in overrides.items():
        base[k] = v
    return base


def _chain(df=None, **kwargs) -> OptionChain:
    params = dict(
        df=df if df is not None else _df(),
        underlying="BOVA11",
        spot=100.0,
        as_of=AS_OF,
        source="teste",
        rate=0.105,
        div_yield=0.035,
    )
    params.update(kwargs)
    return OptionChain(**params)


def test_serie_valida_constroi():
    c = _chain()
    assert c.total_open_interest == 7000
    assert c.expiries == (VENC,)
    assert not c.is_synthetic


def test_com_tau_calcula_tau_positivo():
    t = _chain().com_tau()
    assert (t["tau"] > 0).all()
    assert set(t["kind"]) == {"C", "P"}


def test_coluna_obrigatoria_ausente():
    df = _df().drop(columns=["open_interest"])
    with pytest.raises(ChainValidationError, match="open_interest"):
        _chain(df)


def test_serie_vazia():
    with pytest.raises(ChainValidationError, match="vazia"):
        _chain(_df().iloc[0:0])


def test_kind_invalido():
    with pytest.raises(ChainValidationError, match="kind fora de"):
        _chain(_df(kind=["C", "CALL", "P", "P"]))


def test_strike_nao_positivo():
    with pytest.raises(ChainValidationError, match="strike"):
        _chain(_df(strike=[95.0, 0.0, 95.0, 100.0]))


def test_open_interest_negativo():
    with pytest.raises(ChainValidationError, match="negativo"):
        _chain(_df(open_interest=[1000, -5, 1500, 2500]))


def test_open_interest_todo_zero():
    with pytest.raises(ChainValidationError, match="soma zero"):
        _chain(_df(open_interest=[0, 0, 0, 0]))


def test_duplicata_de_serie():
    df = _df()
    df.loc[1, "strike"] = 95.0  # duplica (VENC, 95, C)
    with pytest.raises(ChainValidationError, match="duplicadas"):
        _chain(df)


def test_vencimento_passado():
    with pytest.raises(ChainValidationError, match="anterior a as_of"):
        _chain(_df(expiry=[date(2026, 8, 1)] * 4))


def test_implied_vol_absurda():
    with pytest.raises(ChainValidationError, match="implied_vol fora"):
        _chain(_df(implied_vol=[0.24, 12.0, 0.26, 0.23]))


def test_implied_vol_nan_e_permitida():
    """NaN e permitido no schema: quem decide o que fazer com ela e o perfil,
    que exige vol_fallback explicito."""
    c = _chain(_df(implied_vol=[0.24, None, 0.26, None]))
    assert c.total_open_interest == 7000


def test_spot_invalido():
    with pytest.raises(ChainValidationError, match="spot"):
        _chain(spot=0.0)


def test_source_vazio():
    with pytest.raises(ChainValidationError, match="procedencia"):
        _chain(source="")


def test_erros_sao_acumulados_nao_um_por_vez():
    """Tres problemas de uma vez: a mensagem tem de listar os tres."""
    df = _df(kind=["C", "X", "P", "P"], open_interest=[1000, -1, 1500, 2500])
    with pytest.raises(ChainValidationError) as exc:
        _chain(df, spot=-1.0)
    msg = str(exc.value)
    assert "kind fora de" in msg
    assert "negativo" in msg
    assert "spot" in msg
    assert msg.startswith("3 problema") or msg.startswith("4 problema")


def test_marca_sintetica_e_visivel_na_procedencia():
    c = _chain(source="SYNTHETIC", is_synthetic=True)
    assert "SINTETICOS" in c.descricao_procedencia()


def test_procedencia_legivel():
    d = _chain().descricao_procedencia()
    assert "BOVA11" in d and "2026-09-03" in d and "teste" in d

"""Historico: amostragem do cadastro, snapshots e a premissa que os sustenta.

O ponto sensivel deste modulo e a otimizacao que evita baixar 1,8 GB de
cadastro: ela SO e valida porque strike/vencimento de uma serie nao mudam
apos a listagem. Se essa premissa cair, o backtest inteiro fica errado -- por
isso ela tem teste proprio, e o codigo levanta excecao em vez de escolher um
dos valores divergentes.
"""

from __future__ import annotations

import json
from datetime import date

import pandas as pd
import pytest

from gama_win.data import historico as H
from gama_win.data.historico import (
    HistoricoError,
    Snapshot,
    chain_do_snapshot,
    construir_catalogo,
    datas_de_amostragem,
)


# --------------------------------------------------------- amostragem ---


def test_amostragem_inclui_as_duas_pontas():
    d = datas_de_amostragem(date(2026, 6, 12), date(2026, 9, 3), passo_dias=10)
    assert d[0] == date(2026, 6, 12)
    assert d[-1] == date(2026, 9, 3)


def test_amostragem_so_devolve_dias_uteis():
    import numpy as np

    for d in datas_de_amostragem(date(2026, 6, 1), date(2026, 8, 31), passo_dias=10):
        assert np.is_busday(np.datetime64(d, "D")), f"{d} nao e dia util"


def test_amostragem_respeita_o_passo():
    from gama_win.model.calendario import dias_uteis_ate

    d = datas_de_amostragem(date(2026, 6, 1), date(2026, 8, 31), passo_dias=10)
    for a, b in zip(d, d[1:], strict=False):
        assert dias_uteis_ate(a, b) <= 12, f"salto grande demais: {a} -> {b}"


def test_amostragem_passo_menor_gera_mais_datas():
    curto = datas_de_amostragem(date(2026, 6, 1), date(2026, 8, 31), passo_dias=5)
    longo = datas_de_amostragem(date(2026, 6, 1), date(2026, 8, 31), passo_dias=20)
    assert len(curto) > len(longo)


def test_amostragem_intervalo_invertido_levanta():
    with pytest.raises(HistoricoError, match="anterior a inicio"):
        datas_de_amostragem(date(2026, 9, 1), date(2026, 6, 1))


# ---------------------------------------------------------- catalogo ---


def _inst(tickers: list[tuple[str, float, str, str]]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "ticker": [t[0] for t in tickers],
            "isin": ["BR" + t[0] + "0" for t in tickers],
            "asset": ["BOVA11"] * len(tickers),
            "kind": [t[2] for t in tickers],
            "strike": [t[1] for t in tickers],
            "expiry": pd.to_datetime([t[3] for t in tickers]),
            "style": ["EURO"] * len(tickers),
            "trading_end": pd.to_datetime([t[3] for t in tickers]),
            "distribution_id": ["120"] * len(tickers),
            "round_lot": ["1"] * len(tickers),
        }
    )


def test_catalogo_une_datas_diferentes(monkeypatch, tmp_path):
    por_data = {
        date(2026, 6, 12): _inst([("BOVAI170", 170.0, "C", "2026-06-19")]),
        date(2026, 6, 26): _inst([("BOVAI180", 180.0, "C", "2026-07-17")]),
    }
    monkeypatch.setattr(H, "baixar_arquivo", lambda *a, **k: tmp_path / "x")
    monkeypatch.setattr(
        H, "carregar_instrumentos", lambda c, ativo=None: por_data.pop(next(iter(por_data)))
    )

    cat = construir_catalogo(
        [date(2026, 6, 12), date(2026, 6, 26)], destino=tmp_path / "cat.json"
    )
    assert set(cat) == {"BRBOVAI1700", "BRBOVAI1800"}
    assert cat["BRBOVAI1700"]["strike"] == 170.0
    assert cat["BRBOVAI1700"]["expiry"] == "2026-06-19"
    assert cat["BRBOVAI1700"]["ticker"] == "BOVAI170"


def test_catalogo_grava_no_destino(monkeypatch, tmp_path):
    monkeypatch.setattr(H, "baixar_arquivo", lambda *a, **k: tmp_path / "x")
    monkeypatch.setattr(
        H, "carregar_instrumentos",
        lambda c, ativo=None: _inst([("BOVAI170", 170.0, "C", "2026-06-19")]),
    )
    destino = tmp_path / "cat.json"
    construir_catalogo([date(2026, 6, 12)], destino=destino)
    salvo = json.loads(destino.read_text(encoding="utf-8"))
    assert salvo["BRBOVAI1700"]["kind"] == "C"


def test_catalogo_levanta_se_isin_divergir(monkeypatch, tmp_path):
    """A PREMISSA. ISIN deve ser imutavel por serie; se o mesmo ISIN aparecer
    com strike diferente, a chave do catalogo tambem nao serve."""
    versoes = [
        _inst([("BOVAI170", 170.0, "C", "2026-06-19")]),
        _inst([("BOVAI170", 165.0, "C", "2026-06-19")]),  # strike mudou
    ]
    monkeypatch.setattr(H, "baixar_arquivo", lambda *a, **k: tmp_path / "x")
    monkeypatch.setattr(
        H, "carregar_instrumentos", lambda c, ativo=None: versoes.pop(0)
    )

    with pytest.raises(HistoricoError, match="cadastro divergente"):
        construir_catalogo(
            [date(2026, 6, 12), date(2026, 6, 26)], destino=tmp_path / "cat.json"
        )


def test_catalogo_aceita_ticker_reciclado_com_isin_diferente(monkeypatch, tmp_path):
    """DESCOBERTA REAL: a B3 recicla codigos. BOVAF162 aponta para a serie de
    junho/2026 e, depois que ela vence, para a de junho/2027. Chaveado por
    ISIN, as duas coexistem; chaveado por ticker, uma sobrescreve a outra."""
    v2026 = _inst([("BOVAF162", 162.0, "C", "2026-06-19")])
    v2027 = _inst([("BOVAF162", 162.0, "C", "2027-06-18")])
    v2027["isin"] = ["BRBOVAF162_2027"]
    versoes = [v2026, v2027]

    monkeypatch.setattr(H, "baixar_arquivo", lambda *a, **k: tmp_path / "x")
    monkeypatch.setattr(
        H, "carregar_instrumentos", lambda c, ativo=None: versoes.pop(0)
    )
    cat = construir_catalogo(
        [date(2026, 6, 12), date(2026, 7, 6)], destino=tmp_path / "cat.json"
    )
    assert len(cat) == 2, "as duas series com o mesmo ticker devem coexistir"
    venc = sorted(r["expiry"] for r in cat.values())
    assert venc == ["2026-06-19", "2027-06-18"]
    assert all(r["ticker"] == "BOVAF162" for r in cat.values())


def test_catalogo_pula_data_sem_arquivo(monkeypatch, tmp_path, capsys):
    from gama_win.data.sources.b3_arquivos import B3Error

    def baixar(nome, d, destino, **k):
        if d == date(2026, 6, 12):
            raise B3Error("sem pregao")
        return destino

    monkeypatch.setattr(H, "baixar_arquivo", baixar)
    monkeypatch.setattr(
        H, "carregar_instrumentos",
        lambda c, ativo=None: _inst([("BOVAI180", 180.0, "C", "2026-07-17")]),
    )
    cat = construir_catalogo(
        [date(2026, 6, 12), date(2026, 6, 26)], destino=tmp_path / "cat.json"
    )
    assert set(cat) == {"BRBOVAI1800"}
    assert "PULA" in capsys.readouterr().out.upper()


# ---------------------------------------------------------- snapshot ---


def _snap(series: list[dict], data=date(2026, 9, 2)) -> Snapshot:
    return Snapshot(
        data=data,
        ativo="BOVA11",
        spot=182.27,
        indice=184920.0,
        series=pd.DataFrame(series),
        cobertura_oi=1.0,
    )


def test_ratio_win_e_indice_sobre_spot():
    s = _snap([{"expiry": "2026-09-04", "strike": 182.0, "kind": "C",
               "open_interest": 100, "preco": 2.0}])
    assert s.ratio_win == pytest.approx(184920.0 / 182.27)


def test_chain_escolhe_o_vencimento_mais_proximo():
    s = _snap([
        {"expiry": "2026-09-18", "strike": 180.0, "kind": "C", "open_interest": 500, "preco": 6.0},
        {"expiry": "2026-09-18", "strike": 185.0, "kind": "P", "open_interest": 500, "preco": 5.0},
        {"expiry": "2026-09-04", "strike": 180.0, "kind": "C", "open_interest": 900, "preco": 3.0},
        {"expiry": "2026-09-04", "strike": 185.0, "kind": "P", "open_interest": 900, "preco": 2.5},
    ])
    ch = chain_do_snapshot(s, rate=0.105, div_yield=0.035)
    assert ch.expiries == (date(2026, 9, 4),)
    assert ch.total_open_interest == 1800


def test_chain_aceita_vencimento_explicito():
    s = _snap([
        {"expiry": "2026-09-18", "strike": 180.0, "kind": "C", "open_interest": 500, "preco": 6.0},
        {"expiry": "2026-09-18", "strike": 185.0, "kind": "P", "open_interest": 500, "preco": 5.0},
        {"expiry": "2026-09-04", "strike": 180.0, "kind": "C", "open_interest": 900, "preco": 3.0},
    ])
    ch = chain_do_snapshot(s, rate=0.105, div_yield=0.035, vencimento=date(2026, 9, 18))
    assert ch.expiries == (date(2026, 9, 18),)


def test_chain_descarta_vencimentos_passados():
    s = _snap([
        {"expiry": "2026-08-21", "strike": 180.0, "kind": "C", "open_interest": 900, "preco": 3.0},
        {"expiry": "2026-09-04", "strike": 180.0, "kind": "C", "open_interest": 500, "preco": 3.0},
        {"expiry": "2026-09-04", "strike": 185.0, "kind": "P", "open_interest": 500, "preco": 2.0},
    ])
    ch = chain_do_snapshot(s, rate=0.105, div_yield=0.035)
    assert ch.expiries == (date(2026, 9, 4),)


def test_chain_sem_vencimento_futuro_levanta():
    s = _snap([{"expiry": "2026-08-21", "strike": 180.0, "kind": "C",
                "open_interest": 900, "preco": 3.0}])
    with pytest.raises(HistoricoError, match="nenhum vencimento futuro"):
        chain_do_snapshot(s, rate=0.105, div_yield=0.035)


def test_chain_vencimento_inexistente_levanta():
    s = _snap([{"expiry": "2026-09-04", "strike": 180.0, "kind": "C",
                "open_interest": 900, "preco": 3.0},
               {"expiry": "2026-09-04", "strike": 185.0, "kind": "P",
                "open_interest": 900, "preco": 3.0}])
    with pytest.raises(HistoricoError, match="ausente no snapshot"):
        chain_do_snapshot(s, rate=0.105, div_yield=0.035, vencimento=date(2027, 1, 15))


def test_chain_soma_series_repetidas_e_pondera_o_preco():
    """Duas series no mesmo (venc, strike, tipo): OI soma, preco vira media
    ponderada pelo OI."""
    s = _snap([
        {"expiry": "2026-09-04", "strike": 180.0, "kind": "C", "open_interest": 300, "preco": 4.0},
        {"expiry": "2026-09-04", "strike": 180.0, "kind": "C", "open_interest": 100, "preco": 8.0},
        {"expiry": "2026-09-04", "strike": 185.0, "kind": "P", "open_interest": 200, "preco": 3.0},
    ])
    ch = chain_do_snapshot(s, rate=0.105, div_yield=0.035, resolver_vol=False)
    linha = ch.df[(ch.df["strike"] == 180.0) & (ch.df["kind"] == "C")].iloc[0]
    assert linha["open_interest"] == 400
    assert linha["settlement_price"] == pytest.approx((300 * 4.0 + 100 * 8.0) / 400)


def test_chain_resolve_vol_a_partir_do_preco():
    """Preco gerado por BSM a 25% deve devolver ~25% na vol implicita."""
    from gama_win.model.calendario import tau_anos
    from gama_win.model.greeks import bsm_price

    venc = date(2026, 9, 18)
    tau = tau_anos(date(2026, 9, 2), venc)
    p_call = float(bsm_price(182.27, 185.0, tau, 0.105, 0.035, 0.25, "C"))
    p_put = float(bsm_price(182.27, 175.0, tau, 0.105, 0.035, 0.25, "P"))

    s = _snap([
        {"expiry": venc.isoformat(), "strike": 185.0, "kind": "C",
         "open_interest": 100, "preco": p_call},
        {"expiry": venc.isoformat(), "strike": 175.0, "kind": "P",
         "open_interest": 100, "preco": p_put},
    ])
    ch = chain_do_snapshot(s, rate=0.105, div_yield=0.035)
    assert ch.df["implied_vol"].notna().all()
    assert ch.df["implied_vol"].to_numpy() == pytest.approx([0.25, 0.25], abs=1e-4)


def test_chain_nao_e_sintetico_e_registra_cobertura():
    s = _snap([{"expiry": "2026-09-04", "strike": 180.0, "kind": "C",
                "open_interest": 900, "preco": 3.0},
               {"expiry": "2026-09-04", "strike": 185.0, "kind": "P",
                "open_interest": 900, "preco": 3.0}])
    ch = chain_do_snapshot(s, rate=0.105, div_yield=0.035)
    assert ch.is_synthetic is False
    assert any("cobertura do OI" in n for n in ch.notas)
    assert any("contract_size" in n for n in ch.notas)

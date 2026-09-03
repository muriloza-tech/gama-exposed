"""Backtest: o motor mede o que afirma medir?

Um backtest errado e pior que nenhum, porque produz conviccao. Por isso os
testes centrais aqui sao dois, em direcoes opostas:

  - `test_detecta_efeito_injetado_forte`: com efeito plantado nos dados, o
    motor TEM de encontra-lo. Um motor incapaz de detectar edge nunca
    valida nada.
  - `test_nao_inventa_efeito_em_dados_sem_sinal`: com passeio aleatorio, o
    motor NAO pode reportar significancia. Falso positivo aqui custa
    dinheiro real.

O resto sao os vieses classicos de backtest intradiario, cada um com seu
teste nomeado.
"""

from __future__ import annotations

from datetime import date, datetime, timedelta

import numpy as np
import pandas as pd
import pytest

from gama_win.backtest.engine import (
    DIR_ALTA,
    DIR_BAIXA,
    REGIME_NEG,
    REGIME_POS,
    BacktestError,
    ConfigBacktest,
    DiaBacktest,
    avaliador_de_perfil,
    coletar_eventos,
    rodar_backtest,
)
from gama_win.backtest.stats import (
    Distribuicao,
    bootstrap_blocos,
    taxa_com_ic,
)
from gama_win.data.sources.sintetica import FonteSintetica
from gama_win.model.profile import construir_perfil

CFG_SIMPLES = ConfigBacktest(
    horizonte_barras=1,
    lookback_barras=1,
    min_amostras=1,
    min_dias=1,
    n_reamostras=400,
)


def _barras(closes, dia: date) -> pd.DataFrame:
    inicio = datetime(dia.year, dia.month, dia.day, 10, 0)
    return pd.DataFrame(
        {
            "ts": [inicio + timedelta(minutes=5 * i) for i in range(len(closes))],
            "close": np.asarray(closes, dtype=float),
        }
    )


def _gama_fixo(valor: float):
    def avaliar(precos):
        return np.full(np.shape(precos), valor, dtype=float)

    return avaliar


def _dia(closes, dia: date, gama: float) -> DiaBacktest:
    return DiaBacktest(data=dia, barras=_barras(closes, dia), avaliar_gama=_gama_fixo(gama))


# ------------------------------------------------------ contagem exata ----


def test_taxas_por_regime_sao_exatas_em_dados_construidos():
    """Dia de gama negativo com 3 de 4 continuacoes; dia de gama positivo
    com 0 de 4. Baseline = 3/8."""
    dias = [
        _dia([100, 101, 102, 103, 104, 103], date(2026, 9, 1), gama=-1.0),
        _dia([100, 101, 100, 101, 100, 101], date(2026, 9, 2), gama=+1.0),
    ]
    r = rodar_backtest(dias, CFG_SIMPLES)

    assert r.n_eventos == 8
    assert r.por_regime[REGIME_NEG].continuacao.n == 4
    assert r.por_regime[REGIME_POS].continuacao.n == 4
    assert r.por_regime[REGIME_NEG].continuacao.taxa == pytest.approx(0.75)
    assert r.por_regime[REGIME_POS].continuacao.taxa == pytest.approx(0.0)
    assert r.baseline.continuacao.taxa == pytest.approx(3 / 8)


def test_retorno_direcional_tem_sinal_de_continuacao():
    """Movimento de baixa que continua cai -> retorno direcional POSITIVO."""
    dias = [_dia([100, 99, 98], date(2026, 9, 1), gama=-1.0)]
    ev, _ = coletar_eventos(dias, CFG_SIMPLES)
    assert len(ev) == 1
    linha = ev.iloc[0]
    assert linha["direcao"] == DIR_BAIXA
    assert linha["ret_futuro_pts"] == pytest.approx(-1.0)
    assert linha["ret_direcional_pts"] == pytest.approx(+1.0)
    assert bool(linha["continuou"]) is True


def test_reversao_tem_retorno_direcional_negativo():
    dias = [_dia([100, 99, 101], date(2026, 9, 1), gama=-1.0)]
    ev, _ = coletar_eventos(dias, CFG_SIMPLES)
    linha = ev.iloc[0]
    assert linha["direcao"] == DIR_BAIXA
    assert linha["ret_direcional_pts"] == pytest.approx(-2.0)
    assert bool(linha["continuou"]) is False


def test_empate_conta_como_nao_continuacao():
    dias = [_dia([100, 101, 101], date(2026, 9, 1), gama=-1.0)]
    ev, _ = coletar_eventos(dias, CFG_SIMPLES)
    assert bool(ev.iloc[0]["continuou"]) is False


def test_direcao_alta_e_classificada():
    dias = [_dia([100, 101, 102], date(2026, 9, 1), gama=+1.0)]
    ev, _ = coletar_eventos(dias, CFG_SIMPLES)
    assert ev.iloc[0]["direcao"] == DIR_ALTA


# ------------------------------------------------------------- vieses ----


def test_nao_ha_lookahead_entre_dias():
    """VIES CLASSICO: janela futura atravessando o fechamento. O salto de
    104 para 200 entre os dias nunca pode aparecer num retorno."""
    dias = [
        _dia([100, 101, 102, 103, 104], date(2026, 9, 1), gama=-1.0),
        _dia([200, 201, 202, 203, 204], date(2026, 9, 2), gama=-1.0),
    ]
    ev, _ = coletar_eventos(dias, CFG_SIMPLES)
    assert len(ev) == 6  # 3 eventos por dia
    assert ev["ret_futuro_pts"].abs().max() == pytest.approx(1.0)
    assert set(ev["data"]) == {date(2026, 9, 1), date(2026, 9, 2)}


def test_gama_fora_da_faixa_e_descartado_nao_extrapolado():
    def avaliar(precos):
        p = np.asarray(precos, dtype=float)
        return np.where(p < 102, -1.0, np.nan)

    dia = DiaBacktest(
        data=date(2026, 9, 1),
        barras=_barras([100, 101, 102, 103, 104], date(2026, 9, 1)),
        avaliar_gama=avaliar,
    )
    ev, desc = coletar_eventos([dia], CFG_SIMPLES)
    assert len(ev) == 1  # so o evento em 101
    assert desc["gama_fora_da_faixa"] == 2


def test_limiar_de_movimento_descarta_ruido():
    cfg = ConfigBacktest(
        horizonte_barras=1, lookback_barras=1, limiar_movimento_pts=5.0,
        min_amostras=1, min_dias=1, n_reamostras=100,
    )
    dias = [_dia([100, 101, 102, 112, 122], date(2026, 9, 1), gama=-1.0)]
    ev, desc = coletar_eventos(dias, cfg)
    assert len(ev) == 1  # so o passo de +10
    assert desc["movimento_abaixo_do_limiar"] == 2


def test_movimento_zero_e_sempre_descartado():
    dias = [_dia([100, 100, 101, 102], date(2026, 9, 1), gama=-1.0)]
    ev, desc = coletar_eventos(dias, CFG_SIMPLES)
    assert desc["movimento_abaixo_do_limiar"] >= 1
    assert not (ev["movimento_pts"] == 0).any()


def test_exclusao_de_barras_de_abertura_e_fechamento():
    cfg = ConfigBacktest(
        horizonte_barras=1, lookback_barras=1,
        excluir_primeiras_barras=1, excluir_ultimas_barras=1,
        min_amostras=1, min_dias=1, n_reamostras=100,
    )
    dias = [_dia([100, 101, 102, 103, 104, 105], date(2026, 9, 1), gama=-1.0)]
    ev, _ = coletar_eventos(dias, cfg)
    # sem exclusao seriam 4 eventos (t=1..4); com 1+1 sobram t=2..3
    assert len(ev) == 2


def test_dia_com_barras_insuficientes_e_contabilizado():
    cfg = ConfigBacktest(
        horizonte_barras=3, lookback_barras=3, min_amostras=1, min_dias=1,
        n_reamostras=100,
    )
    dias = [_dia([100, 101, 102], date(2026, 9, 1), gama=-1.0)]
    ev, desc = coletar_eventos(dias, cfg)
    assert len(ev) == 0
    assert desc["barras_insuficientes_no_dia"] == 1


def test_dias_repetidos_levantam():
    dias = [
        _dia([100, 101, 102], date(2026, 9, 1), gama=-1.0),
        _dia([100, 101, 102], date(2026, 9, 1), gama=-1.0),
    ]
    with pytest.raises(BacktestError, match="dias repetidos"):
        coletar_eventos(dias, CFG_SIMPLES)


# -------------------------------------------- deteccao e falso positivo ---


def _dias_com_efeito(n_dias: int) -> list[DiaBacktest]:
    """Dias de gama negativo com tendencia limpa (continuacao total) e dias
    de gama positivo com zigue-zague (reversao total)."""
    dias = []
    d0 = date(2026, 3, 2)
    for i in range(n_dias):
        d = d0 + timedelta(days=i)
        if i % 2 == 0:
            closes = [100 + j for j in range(12)]
            dias.append(_dia(closes, d, gama=-1.0))
        else:
            closes = [100 + (j % 2) for j in range(12)]
            dias.append(_dia(closes, d, gama=+1.0))
    return dias


def test_detecta_efeito_injetado_forte():
    """Com efeito plantado, o motor tem de reportar diferenca significativa
    contra o baseline. Motor que nao detecta edge nunca valida nada."""
    r = rodar_backtest(_dias_com_efeito(30), CFG_SIMPLES)

    neg = r.por_regime[REGIME_NEG]
    pos = r.por_regime[REGIME_POS]

    assert neg.continuacao.taxa == pytest.approx(1.0)
    assert pos.continuacao.taxa == pytest.approx(0.0)
    assert neg.dif_vs_baseline > 0
    assert neg.dif_e_significativa, f"IC da diferenca: {neg.dif_ic}"
    assert pos.dif_e_significativa
    assert neg.dif_ic[0] > 0, "limite inferior do IC deve excluir zero"


def test_nao_inventa_efeito_em_dados_sem_sinal():
    """Passeio aleatorio com regime atribuido de forma nao informativa: o
    motor NAO pode reportar significancia, e tem de avisar isso."""
    rng = np.random.default_rng(20260903)
    dias = []
    d0 = date(2026, 3, 2)
    for i in range(40):
        d = d0 + timedelta(days=i)
        passos = rng.choice([-1.0, 1.0], size=60)
        closes = 100_000 + np.cumsum(passos) * 5
        # Regime alternado por barra, sem relacao com o preco futuro.
        def avaliar(precos, _i=i):
            p = np.asarray(precos, dtype=float)
            return np.where((np.arange(len(p)) + _i) % 2 == 0, -1.0, 1.0)

        dias.append(
            DiaBacktest(data=d, barras=_barras(closes, d), avaliar_gama=avaliar)
        )

    r = rodar_backtest(dias, ConfigBacktest(
        horizonte_barras=3, lookback_barras=2, min_amostras=50, min_dias=20,
        n_reamostras=500, semente=7,
    ))

    for reg in (REGIME_NEG, REGIME_POS):
        g = r.por_regime[reg]
        assert not g.dif_e_significativa, (
            f"falso positivo em {reg}: dif={g.dif_vs_baseline:.4f} "
            f"IC={g.dif_ic}"
        )
    assert any("NENHUM regime" in a for a in r.avisos)


def test_aviso_de_amostra_pequena():
    r = rodar_backtest(
        [_dia([100, 101, 102, 103], date(2026, 9, 1), gama=-1.0)],
        ConfigBacktest(
            horizonte_barras=1, lookback_barras=1, min_amostras=100,
            min_dias=20, n_reamostras=100,
        ),
    )
    assert any("abaixo do minimo" in a for a in r.avisos)
    assert any("dia(s) de amostra" in a for a in r.avisos)


def test_aviso_de_lookahead_de_oi_sempre_presente():
    """O lembrete de usar OI de D-1 nao e opcional."""
    r = rodar_backtest(
        [_dia([100, 101, 102, 103], date(2026, 9, 1), gama=-1.0)], CFG_SIMPLES
    )
    assert any("OI de D-1" in a for a in r.avisos)


def test_quebra_por_regime_e_direcao():
    dias = [
        _dia([100, 99, 98, 97], date(2026, 9, 1), gama=-1.0),
        _dia([100, 101, 102, 103], date(2026, 9, 2), gama=-1.0),
    ]
    r = rodar_backtest(dias, CFG_SIMPLES)
    rotulos = set(r.por_regime_e_direcao)
    assert f"{REGIME_NEG} + {DIR_BAIXA}" in rotulos
    assert f"{REGIME_NEG} + {DIR_ALTA}" in rotulos


def test_sem_eventos_nao_quebra():
    cfg = ConfigBacktest(
        horizonte_barras=10, lookback_barras=10, min_amostras=1, min_dias=1,
        n_reamostras=100,
    )
    r = rodar_backtest([_dia([100, 101, 102], date(2026, 9, 1), gama=-1.0)], cfg)
    assert r.n_eventos == 0
    assert any("nenhum evento elegivel" in a for a in r.avisos)


# --------------------------------------------------------- configuracao ---


@pytest.mark.parametrize(
    "kwargs,msg",
    [
        ({"horizonte_barras": 0}, "horizonte_barras"),
        ({"lookback_barras": 0}, "lookback_barras"),
        ({"limiar_movimento_pts": -1.0}, "limiar_movimento"),
        ({"excluir_primeiras_barras": -1}, "exclusoes"),
        ({"n_reamostras": 0}, "n_reamostras"),
    ],
)
def test_config_invalida_levanta(kwargs, msg):
    with pytest.raises(BacktestError, match=msg):
        ConfigBacktest(**kwargs)


def test_barras_sem_coluna_levanta():
    with pytest.raises(BacktestError, match="ts"):
        DiaBacktest(
            data=date(2026, 9, 1),
            barras=pd.DataFrame({"close": [1.0, 2.0]}),
            avaliar_gama=_gama_fixo(-1.0),
        )


def test_barras_desordenadas_levantam():
    b = _barras([100, 101, 102], date(2026, 9, 1)).iloc[::-1]
    with pytest.raises(BacktestError, match="ordenadas"):
        DiaBacktest(date(2026, 9, 1), b, _gama_fixo(-1.0))


def test_barras_com_close_nao_finito_levantam():
    b = _barras([100, np.nan, 102], date(2026, 9, 1))
    with pytest.raises(BacktestError, match="nao finito"):
        DiaBacktest(date(2026, 9, 1), b, _gama_fixo(-1.0))


# --------------------------------------------------------- avaliador -----


def test_avaliador_de_perfil_interpola_dentro_e_nan_fora():
    chain = FonteSintetica(spot=176.5).buscar(date(2026, 9, 3))
    perfil = construir_perfil(chain)
    avaliar = avaliador_de_perfil(perfil, ratio_win=1000.0)

    dentro = float(perfil.strikes[len(perfil.strikes) // 2]) * 1000.0
    fora_baixo = float(perfil.strikes[0]) * 1000.0 - 1.0
    fora_alto = float(perfil.strikes[-1]) * 1000.0 + 1.0

    r = avaliar(np.array([dentro, fora_baixo, fora_alto]))
    assert np.isfinite(r[0])
    assert np.isnan(r[1]) and np.isnan(r[2])


def test_avaliador_respeita_ratio():
    chain = FonteSintetica(spot=176.5).buscar(date(2026, 9, 3))
    perfil = construir_perfil(chain)
    k = float(perfil.strikes[5])
    a1 = avaliador_de_perfil(perfil, ratio_win=1.0)(np.array([k]))
    a2 = avaliador_de_perfil(perfil, ratio_win=1000.0)(np.array([k * 1000.0]))
    assert a1[0] == pytest.approx(a2[0])


def test_avaliador_ratio_invalido_levanta():
    chain = FonteSintetica(spot=176.5).buscar(date(2026, 9, 3))
    perfil = construir_perfil(chain)
    with pytest.raises(BacktestError, match="ratio_win"):
        avaliador_de_perfil(perfil, ratio_win=0.0)


# ------------------------------------------------------------- stats -----


def test_distribuicao_de_valores_conhecidos():
    d = Distribuicao.de(np.array([1.0, 2.0, 3.0, 4.0, 5.0]))
    assert d.n == 5
    assert d.media == pytest.approx(3.0)
    assert d.mediana == pytest.approx(3.0)
    assert d.p25 == pytest.approx(2.0)
    assert d.p75 == pytest.approx(4.0)


def test_distribuicao_vazia_nao_quebra():
    d = Distribuicao.de(np.array([]))
    assert d.n == 0
    assert np.isnan(d.media)


def test_distribuicao_ignora_nan():
    d = Distribuicao.de(np.array([1.0, np.nan, 3.0]))
    assert d.n == 2
    assert d.media == pytest.approx(2.0)


def test_bootstrap_blocos_precisa_de_dois_blocos():
    lo, hi = bootstrap_blocos(
        np.array([1.0, 2.0]), np.array(["a", "a"]), n_reamostras=50
    )
    assert np.isnan(lo) and np.isnan(hi)


def test_bootstrap_blocos_ic_contem_a_media():
    rng = np.random.default_rng(1)
    v = rng.normal(10.0, 1.0, size=500)
    blocos = np.repeat(np.arange(25), 20)
    lo, hi = bootstrap_blocos(v, blocos, n_reamostras=500, semente=1)
    assert lo < float(np.mean(v)) < hi


def test_bootstrap_valores_e_blocos_de_tamanhos_diferentes_levanta():
    with pytest.raises(ValueError, match="mesmo tamanho"):
        bootstrap_blocos(np.array([1.0, 2.0]), np.array(["a"]))


def test_taxa_com_ic_reporta_n_e_blocos():
    s = np.array([True, False, True, True])
    b = np.array(["d1", "d1", "d2", "d2"])
    t = taxa_com_ic(s, b, n_reamostras=200)
    assert t.n == 4
    assert t.n_blocos == 2
    assert t.taxa == pytest.approx(0.75)
    assert "n=4" in str(t)


def test_taxa_com_ic_vazia():
    t = taxa_com_ic(np.array([], dtype=bool), np.array([]))
    assert t.n == 0
    assert np.isnan(t.taxa)


def test_contem_detecta_ausencia_de_efeito():
    rng = np.random.default_rng(3)
    s = rng.random(600) < 0.5
    b = np.repeat(np.arange(30), 20)
    t = taxa_com_ic(s, b, n_reamostras=500, semente=3)
    assert t.contem(0.5)

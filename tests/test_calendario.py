"""Calendario: datas conhecidas e propriedades de contagem."""

from __future__ import annotations

from datetime import date, timedelta

import pytest

from gama_win.model.calendario import (
    dias_uteis_ate,
    feriados_do_ano,
    pascoa,
    tau_anos,
)


@pytest.mark.parametrize(
    "ano,esperado",
    [
        (2024, date(2024, 3, 31)),
        (2025, date(2025, 4, 20)),
        (2026, date(2026, 4, 5)),
        (2027, date(2027, 3, 28)),
        (2030, date(2030, 4, 21)),
    ],
)
def test_pascoa_datas_conhecidas(ano, esperado):
    assert pascoa(ano) == esperado


def test_carnaval_2025():
    """Carnaval 2025 caiu em 3 e 4 de marco."""
    f = feriados_do_ano(2025)
    assert date(2025, 3, 3) in f
    assert date(2025, 3, 4) in f


def test_corpus_christi_2025():
    assert date(2025, 6, 19) in feriados_do_ano(2025)


def test_sexta_santa_2026():
    assert date(2026, 4, 3) in feriados_do_ano(2026)


def test_consciencia_negra_somente_de_2024_em_diante():
    assert date(2023, 11, 20) not in feriados_do_ano(2023)
    assert date(2024, 11, 20) in feriados_do_ano(2024)
    assert date(2026, 11, 20) in feriados_do_ano(2026)


def test_natal_e_ano_novo_sempre_presentes():
    for ano in (2024, 2025, 2026, 2027):
        f = feriados_do_ano(ano)
        assert date(ano, 1, 1) in f
        assert date(ano, 12, 25) in f


def test_mesmo_dia_util_conta_uma_sessao():
    """Terca-feira comum: a propria sessao conta."""
    assert dias_uteis_ate(date(2026, 9, 8), date(2026, 9, 8)) == 1


def test_mesmo_dia_em_feriado_conta_zero():
    assert dias_uteis_ate(date(2026, 12, 25), date(2026, 12, 25)) == 0


def test_fim_antes_do_inicio_retorna_zero():
    assert dias_uteis_ate(date(2026, 9, 10), date(2026, 9, 1)) == 0


def test_fim_de_semana_nao_conta():
    # sexta 04/09/2026 -> segunda 07/09/2026 (07/09 e feriado)
    assert dias_uteis_ate(date(2026, 9, 4), date(2026, 9, 4)) == 1
    # segunda 07/09 e Independencia: nao ha sessao
    assert dias_uteis_ate(date(2026, 9, 7), date(2026, 9, 7)) == 0


def test_feriado_no_meio_reduz_contagem():
    """Semana com 07/09 (Independencia, segunda) tem 4 sessoes."""
    sem_feriado = dias_uteis_ate(date(2026, 9, 14), date(2026, 9, 18))
    com_feriado = dias_uteis_ate(date(2026, 9, 7), date(2026, 9, 11))
    assert sem_feriado == 5
    assert com_feriado == 4


def test_contagem_e_monotonica_nao_decrescente():
    """Empurrar o vencimento para frente nunca reduz o numero de sessoes."""
    inicio = date(2026, 9, 3)
    anterior = 0
    for offset in range(0, 60):
        fim = inicio + timedelta(days=offset)
        atual = dias_uteis_ate(inicio, fim)
        assert atual >= anterior, f"caiu de {anterior} para {atual} em {fim}"
        anterior = atual
    assert anterior > 30, "60 dias corridos devem conter mais de 30 sessoes"


def test_tau_anos_positivo_e_consistente():
    t = tau_anos(date(2026, 9, 3), date(2026, 9, 18))
    assert t > 0
    assert t == pytest.approx(dias_uteis_ate(date(2026, 9, 3), date(2026, 9, 18)) / 252)


def test_tau_no_proprio_vencimento_nao_e_zero():
    """Garantia estrutural: nunca entregamos tau=0 as gregas."""
    t = tau_anos(date(2026, 9, 18), date(2026, 9, 18))
    assert t == pytest.approx(1 / 252)


# ---------------------------------------------- vencimento do indice ------


def test_vencimento_indice_e_sempre_quarta_ou_dia_util_seguinte():
    """A regra e 'quarta mais proxima do dia 15'; so escapa da quarta quando
    ela cai em feriado e o vencimento anda para o dia util seguinte."""
    import numpy as np

    from gama_win.model.calendario import _feriados_np, vencimento_indice

    for ano in (2025, 2026, 2027, 2028):
        for mes in (2, 4, 6, 8, 10, 12):
            v = vencimento_indice(ano, mes)
            assert v.month == mes and v.year == ano
            fer = _feriados_np(ano, ano + 1)
            assert np.is_busday(np.datetime64(v, "D"), holidays=fer), v
            # quarta, ou empurrado para frente por feriado
            assert v.weekday() >= 2, v


def test_vencimento_indice_fica_perto_do_dia_15():
    from gama_win.model.calendario import vencimento_indice

    for ano in (2025, 2026, 2027):
        for mes in (2, 4, 6, 8, 10, 12):
            v = vencimento_indice(ano, mes)
            assert abs(v.day - 15) <= 4, v


@pytest.mark.parametrize(
    "ano,mes,esperado",
    [
        (2026, 2, date(2026, 2, 18)),
        (2026, 4, date(2026, 4, 15)),
        (2026, 8, date(2026, 8, 12)),
        (2026, 10, date(2026, 10, 14)),
        (2026, 12, date(2026, 12, 16)),
    ],
)
def test_vencimento_indice_valores_calculados(ano, mes, esperado):
    """Fixa o resultado da regra. Se alguem mexer na implementacao, quebra."""
    from gama_win.model.calendario import vencimento_indice

    assert vencimento_indice(ano, mes) == esperado


def test_vencimento_indice_rejeita_mes_impar():
    from gama_win.model.calendario import vencimento_indice

    for mes in (1, 3, 5, 7, 9, 11):
        with pytest.raises(ValueError, match="nao e mes de vencimento"):
            vencimento_indice(2026, mes)


def test_proximos_vencimentos_indice_sao_crescentes_e_futuros():
    from gama_win.model.calendario import proximos_vencimentos_indice

    ref = date(2026, 9, 3)
    vs = proximos_vencimentos_indice(ref, 5)
    assert len(vs) == 5
    assert all(v >= ref for v in vs)
    assert vs == sorted(vs)
    assert all(v.month in (2, 4, 6, 8, 10, 12) for v in vs)


def test_proximos_vencimentos_inclui_o_do_proprio_dia():
    from gama_win.model.calendario import proximos_vencimentos_indice, vencimento_indice

    v = vencimento_indice(2026, 10)
    assert proximos_vencimentos_indice(v, 1)[0] == v


def test_proximos_vencimentos_atravessa_o_ano():
    from gama_win.model.calendario import proximos_vencimentos_indice

    vs = proximos_vencimentos_indice(date(2026, 11, 20), 3)
    assert vs[0] == date(2026, 12, 16)
    assert vs[1].year == 2027


def test_proximos_vencimentos_quantos_invalido():
    from gama_win.model.calendario import proximos_vencimentos_indice

    with pytest.raises(ValueError, match="quantos"):
        proximos_vencimentos_indice(date(2026, 9, 3), 0)

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

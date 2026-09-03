"""Calendario de pregao da B3 para contagem de dias uteis.

Por que isto existe: `tau` em anos e o insumo mais sensivel do gama de curto
prazo. Contar dias corridos, ou dias de semana ignorando feriados, distorce o
gama justamente na semana de vencimento -- quando ele mais importa para day
trade.

IMPORTANTE: os feriados moveis (Carnaval, Sexta-feira Santa, Corpus Christi)
sao derivados da Pascoa por algoritmo, portanto exatos. Os fixos estao
listados abaixo. Ainda assim, o calendario oficial da B3 e a unica fonte
autoritativa e pode conter fechamentos extraordinarios. Use o arquivo de
override (`data/feriados_b3.csv`, uma data ISO por linha) para acrescentar ou
corrigir datas sem mexer no codigo -- `gama_win doctor` mostra o calendario
efetivo do ano para conferencia visual.
"""

from __future__ import annotations

import csv
from datetime import date, timedelta
from functools import lru_cache
from pathlib import Path

import numpy as np

DIAS_PREGAO_ANO = 252

# Feriados nacionais com data fixa em que a B3 nao opera.
# 20/11 (Consciencia Negra) tornou-se feriado nacional a partir de 2024.
_FIXOS: tuple[tuple[int, int, str], ...] = (
    (1, 1, "Confraternizacao Universal"),
    (4, 21, "Tiradentes"),
    (5, 1, "Dia do Trabalho"),
    (9, 7, "Independencia"),
    (10, 12, "Nossa Senhora Aparecida"),
    (11, 2, "Finados"),
    (11, 15, "Proclamacao da Republica"),
    (12, 25, "Natal"),
)

_FIXOS_DESDE_2024: tuple[tuple[int, int, str], ...] = (
    (11, 20, "Consciencia Negra"),
)


def pascoa(ano: int) -> date:
    """Domingo de Pascoa pelo algoritmo gregoriano anonimo (exato)."""
    a = ano % 19
    b = ano // 100
    c = ano % 100
    d = b // 4
    e = b % 4
    f = (b + 8) // 25
    g = (b - f + 1) // 3
    h = (19 * a + b - d - g + 15) % 30
    i = c // 4
    k = c % 4
    l = (32 + 2 * e + 2 * i - h - k) % 7
    m = (a + 11 * h + 22 * l) // 451
    mes = (h + l - 7 * m + 114) // 31
    dia = ((h + l - 7 * m + 114) % 31) + 1
    return date(ano, mes, dia)


def feriados_do_ano(ano: int) -> dict[date, str]:
    """Feriados de pregao do ano, com nome, ordenados."""
    p = pascoa(ano)
    moveis = {
        p - timedelta(days=48): "Carnaval (segunda)",
        p - timedelta(days=47): "Carnaval (terca)",
        p - timedelta(days=2): "Sexta-feira Santa",
        p + timedelta(days=60): "Corpus Christi",
    }
    fixos = {date(ano, m, d): nome for m, d, nome in _FIXOS}
    if ano >= 2024:
        fixos.update({date(ano, m, d): nome for m, d, nome in _FIXOS_DESDE_2024})

    todos = {**fixos, **moveis}
    return dict(sorted(todos.items()))


def _caminho_override() -> Path:
    return Path(__file__).resolve().parents[2] / "data" / "feriados_b3.csv"


def carregar_override() -> set[date]:
    """Datas extras de fechamento, uma ISO (AAAA-MM-DD) por linha.

    Linhas vazias e comecadas por '#' sao ignoradas. Ausencia do arquivo nao
    e erro -- e o caso normal.
    """
    caminho = _caminho_override()
    if not caminho.exists():
        return set()
    extras: set[date] = set()
    with caminho.open("r", encoding="utf-8", newline="") as fh:
        for linha_num, linha in enumerate(csv.reader(fh), start=1):
            if not linha:
                continue
            bruto = linha[0].strip()
            if not bruto or bruto.startswith("#"):
                continue
            try:
                extras.add(date.fromisoformat(bruto))
            except ValueError as exc:
                raise ValueError(
                    f"{caminho}:{linha_num}: '{bruto}' nao e uma data ISO "
                    "valida (esperado AAAA-MM-DD)"
                ) from exc
    return extras


@lru_cache(maxsize=32)
def _feriados_np(ano_inicio: int, ano_fim: int) -> np.ndarray:
    datas: set[date] = set()
    for ano in range(ano_inicio, ano_fim + 1):
        datas.update(feriados_do_ano(ano))
    datas.update(carregar_override())
    return np.array(sorted(datas), dtype="datetime64[D]")


def dias_uteis_ate(inicio: date, fim: date) -> int:
    """Sessoes de pregao entre `inicio` e `fim`, INCLUINDO a de `fim`.

    Convencao deliberada: incluir a sessao de vencimento evita tau = 0 no
    proprio dia do vencimento, que faria o gama divergir ou zerar dependendo
    do tratamento. O refinamento intradiario (fracao da sessao ainda por
    correr) e responsabilidade da camada intraday, nao desta.

    Retorna 0 se `fim` for anterior a `inicio`.
    """
    if fim < inicio:
        return 0
    feriados = _feriados_np(min(inicio.year, fim.year), max(inicio.year, fim.year))
    contagem = int(
        np.busday_count(
            np.datetime64(inicio, "D"), np.datetime64(fim, "D"), holidays=feriados
        )
    )
    # busday_count exclui o dia final; somamos a sessao de vencimento se ela
    # for de fato dia de pregao.
    fim_e_pregao = bool(
        np.is_busday(np.datetime64(fim, "D"), holidays=feriados)
    )
    return max(contagem + (1 if fim_e_pregao else 0), 0)


def tau_anos(inicio: date, fim: date) -> float:
    """Tempo ate o vencimento em anos de pregao (base 252)."""
    return dias_uteis_ate(inicio, fim) / DIAS_PREGAO_ANO


# ------------------------------------------------- vencimento do indice ---

# Contratos futuros de Ibovespa (IND e WIN) vencem em meses PARES.
MESES_VENCIMENTO_INDICE: tuple[int, ...] = (2, 4, 6, 8, 10, 12)


def vencimento_indice(ano: int, mes: int) -> date:
    """Vencimento do futuro de Ibovespa (IND/WIN) no mes informado.

    Regra da B3: quarta-feira mais proxima do dia 15 do mes de vencimento;
    se nao houver pregao nesse dia, o dia util seguinte.

    Nao ha empate possivel: quartas distam 7 dias, entao duas nunca ficam a
    igual distancia de um dia 15 inteiro.

    Levanta para mes impar -- nao existe contrato.
    """
    if mes not in MESES_VENCIMENTO_INDICE:
        raise ValueError(
            f"mes {mes} nao e mes de vencimento do indice; "
            f"validos: {MESES_VENCIMENTO_INDICE}"
        )

    quartas = []
    d = date(ano, mes, 1)
    while d.month == mes:
        if d.weekday() == 2:  # quarta-feira
            quartas.append(d)
        d += timedelta(days=1)

    alvo = min(quartas, key=lambda q: abs(q.day - 15))

    feriados = _feriados_np(ano, ano + 1)
    while not np.is_busday(np.datetime64(alvo, "D"), holidays=feriados):
        alvo += timedelta(days=1)
    return alvo


def proximos_vencimentos_indice(
    a_partir_de: date, quantos: int = 3
) -> list[date]:
    """Proximos vencimentos do indice, incluindo o do proprio dia."""
    if quantos < 1:
        raise ValueError("quantos deve ser >= 1")

    saida: list[date] = []
    ano, mes = a_partir_de.year, a_partir_de.month
    if mes not in MESES_VENCIMENTO_INDICE:
        mes += 1
    if mes > 12:
        ano, mes = ano + 1, 2

    while len(saida) < quantos:
        if mes in MESES_VENCIMENTO_INDICE:
            v = vencimento_indice(ano, mes)
            if v >= a_partir_de:
                saida.append(v)
        mes += 2
        if mes > 12:
            ano, mes = ano + 1, 2
    return saida

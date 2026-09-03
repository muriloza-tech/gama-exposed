"""Relatorio de texto dos niveis -- a saida minima que ja e util.

Deliberadamente texto antes de grafico: um numero com unidade e procedencia
vale mais que uma linha colorida sem legenda. O grafico vem depois, em cima
disto.
"""

from __future__ import annotations

from gama_win.model.levels import Levels, Zona
from gama_win.model.profile import GammaProfile


def num(valor: float, decimais: int = 2) -> str:
    """Numero em notacao pt-BR: milhar com ponto, decimal com virgula.

    A troca precisa ser simultanea; substituir um separador depois do outro
    produz '1.048.00' em vez de '1.048,00'.
    """
    s = f"{valor:,.{decimais}f}"
    return s.replace(",", "\x00").replace(".", ",").replace("\x00", ".")


def brl(valor: float) -> str:
    """Formata em reais, notacao pt-BR, com escala automatica."""
    sinal = "-" if valor < 0 else ""
    v = abs(valor)
    if v >= 1e9:
        return f"{sinal}R$ {v / 1e9:.2f} bi".replace(".", ",")
    if v >= 1e6:
        return f"{sinal}R$ {v / 1e6:.2f} mi".replace(".", ",")
    if v >= 1e3:
        return f"{sinal}R$ {v / 1e3:.1f} mil".replace(".", ",")
    return f"{sinal}R$ {v:.2f}".replace(".", ",")


def _zona_txt(z: Zona | None, ratio_win: float | None) -> str:
    if z is None:
        return "indeterminada"
    if z.inicio == z.fim:
        faixa = f"{z.pico:.2f}"
    else:
        faixa = f"{z.inicio:.2f} a {z.fim:.2f} (pico {z.pico:.2f})"
    win = ""
    if ratio_win:
        if z.inicio == z.fim:
            win = f"  [WIN ~{z.pico * ratio_win:,.0f}]".replace(",", ".")
        else:
            win = (
                f"  [WIN {z.inicio * ratio_win:,.0f} a "
                f"{z.fim * ratio_win:,.0f}]".replace(",", ".")
            )
    return f"{faixa}{win} | exposicao na zona: {brl(z.exposicao_zona)}"


def _no_spot(perfil: GammaProfile, serie) -> str:
    """Valor da serie no strike mais proximo do spot, formatado."""
    i = int(abs(perfil.strikes - perfil.spot).argmin())
    return brl(float(serie[i]))


def relatorio_texto(
    perfil: GammaProfile,
    niveis: Levels,
    *,
    ratio_win: float | None = None,
) -> str:
    """Relatorio completo, com procedencia, unidades e avisos."""
    linhas: list[str] = []
    larg = 74

    if perfil.is_synthetic:
        aviso = (
            "DADOS SINTETICOS -- NAO OPERAR",
            "Os numeros abaixo foram gerados localmente e nao",
            "representam posicao real de mercado.",
        )
        linhas.append("#" * larg)
        for texto in aviso:
            # "#  " (3) + texto (larg-4) + "#" (1) = larg
            linhas.append("#  " + texto.ljust(larg - 4) + "#")
        linhas.append("#" * larg)
        linhas.append("")

    linhas += [
        "=" * larg,
        "  PERFIL DE GAMA",
        "=" * larg,
        f"  Procedencia .... {perfil.underlying} @ {perfil.as_of.isoformat()} "
        f"(fonte: {perfil.source})",
        f"  Spot ........... {num(perfil.spot)}",
        f"  Convencao ...... {perfil.convention.descricao}",
        f"  Vencimentos .... {', '.join(d.isoformat() for d in perfil.expiries)}",
        f"  Strikes ........ {len(perfil.strikes)} de "
        f"{num(float(perfil.strikes[0]))} a {num(float(perfil.strikes[-1]))}",
        f"  Vol media ...... {num(perfil.vol_media_ponderada * 100, 1)}% "
        "(ponderada por vega)",
        f"  OI total ....... "
        f"{int(perfil.oi_call.sum() + perfil.oi_put.sum()):,}".replace(",", "."),
    ]

    if ratio_win:
        linhas.append(f"  Razao -> WIN ... x{num(ratio_win)}")

    linhas += [
        "",
        "-" * larg,
        "  REGIME NO SPOT",
        "-" * larg,
        f"  Gama liquido ... {brl(niveis.gama_no_spot)} por 1% de movimento",
        f"  Regime ......... {niveis.regime.upper()}",
        "",
        "  Leitura: gama positivo -> hedge do dealer e contraciclico (compra na",
        "  queda, vende na alta), o que tende a FREAR o movimento. Gama negativo",
        "  -> hedge prociclico, que tende a ACELERAR o movimento. Efeito",
        "  probabilistico, nao determinismo.",
        "",
        "-" * larg,
        "  NIVEIS",
        "-" * larg,
    ]

    if niveis.flip is None:
        linhas.append("  Gamma flip ..... nao encontrado na faixa analisada")
    else:
        dist = niveis.distancia_ao_flip or 0.0
        lado = "acima" if dist > 0 else "abaixo"
        win = (
            f"  [WIN ~{niveis.flip * ratio_win:,.0f}]".replace(",", ".")
            if ratio_win
            else ""
        )
        linhas.append(
            f"  Gamma flip ..... {niveis.flip:.2f}{win} "
            f"({abs(dist):.2f} {lado} do spot)"
        )
        if len(niveis.flips_todos) > 1:
            outros = ", ".join(f"{f:.2f}" for f in niveis.flips_todos)
            linhas.append(f"                   cruzamentos no perfil: {outros}")

    linhas += [
        f"  Call wall ...... {_zona_txt(niveis.call_wall, ratio_win)}",
        f"  Put wall ....... {_zona_txt(niveis.put_wall, ratio_win)}",
        "",
        "-" * larg,
        "  DINAMICA (o que muda sozinho, sem o preco andar)",
        "-" * larg,
        f"  Charm no spot .. {_no_spot(perfil, perfil.charm_brl_dia)} "
        "de delta por dia de pregao",
        f"  Vanna no spot .. "
        f"{_no_spot(perfil, perfil.vanna_brl_por_ponto_vol)} "
        "de delta por ponto de vol",
    ]

    if niveis.avisos:
        linhas += ["", "-" * larg, "  AVISOS", "-" * larg]
        linhas += [f"  ! {a}" for a in niveis.avisos]

    linhas.append("=" * larg)
    return "\n".join(linhas)

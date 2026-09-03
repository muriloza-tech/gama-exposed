"""CLI do gama-win.

    python -m gama_win doctor     verifica ambiente, calendario e a matematica
    python -m gama_win perfil     calcula e imprime o perfil de gama

`doctor` e o primeiro comando do onboarding: ele diz exatamente o que esta
pronto, o que falta e o que ainda depende de dado externo, sem que voce tenha
de ler codigo.
"""

from __future__ import annotations

import argparse
import math
import sys
from datetime import date

import numpy as np

from .data.sources import base as fontes
from .data.sources import sintetica  # noqa: F401  (registra a fonte)
from .model.calendario import feriados_do_ano, tau_anos
from .model.greeks import bsm_greeks, bsm_price, implied_vol
from .model.levels import extrair_niveis
from .model.profile import construir_perfil
from .view.relatorio import relatorio_texto

OK = "  [ok]  "
FALHA = "  [FALHA]"
AVISO = "  [aviso]"


def _autoteste_matematica() -> list[tuple[bool, str]]:
    """Invariantes verificados em runtime, sem depender do pytest.

    Se algum destes falhar na maquina do usuario, o problema e de ambiente
    (numpy/plataforma), nao de logica -- e melhor descobrir aqui do que num
    numero errado no meio do pregao.
    """
    r: list[tuple[bool, str]] = []

    S, K, tau, rate, div, vol = 100.0, 100.0, 10 / 252, 0.105, 0.035, 0.22

    gc = float(bsm_greeks(S, K, tau, rate, div, vol, "C").gamma)
    gp = float(bsm_greeks(S, K, tau, rate, div, vol, "P").gamma)
    r.append((abs(gc - gp) < 1e-12, "gama de call e put identicos no strike"))

    h = S * 1e-5
    d_mais = float(bsm_greeks(S + h, K, tau, rate, div, vol, "C").delta)
    d_menos = float(bsm_greeks(S - h, K, tau, rate, div, vol, "C").delta)
    fd = (d_mais - d_menos) / (2 * h)
    r.append((abs(fd - gc) < 1e-6, "gama = derivada do delta (diferenca finita)"))

    c = float(bsm_price(S, K, tau, rate, div, vol, "C"))
    p = float(bsm_price(S, K, tau, rate, div, vol, "P"))
    parity = S * math.exp(-div * tau) - K * math.exp(-rate * tau)
    r.append((abs((c - p) - parity) < 1e-10, "paridade call-put no precificador"))

    rec = float(implied_vol(c, S, K, tau, rate, div, "C"))
    r.append((abs(rec - vol) < 1e-6, "volatilidade implicita recuperada"))

    t_venc = tau_anos(date(2026, 9, 18), date(2026, 9, 18))
    r.append((t_venc > 0, "tau no dia do vencimento nao e zero"))

    try:
        bsm_greeks(S, K, 0.0, rate, div, vol, "C")
        r.append((False, "tau<=0 deveria levantar excecao"))
    except ValueError:
        r.append((True, "tau<=0 levanta excecao em vez de devolver zero"))

    return r


def comando_doctor(_args: argparse.Namespace) -> int:
    print()
    print("=" * 74)
    print("  GAMA-WIN :: DIAGNOSTICO")
    print("=" * 74)

    falhas = 0

    print("\n-- Ambiente")
    v = sys.version_info
    ok_py = v >= (3, 11)
    print(f"{OK if ok_py else FALHA} Python {v.major}.{v.minor}.{v.micro}")
    falhas += 0 if ok_py else 1
    for mod in ("numpy", "pandas", "plotly"):
        try:
            m = __import__(mod)
            print(f"{OK} {mod} {getattr(m, '__version__', '?')}")
        except ImportError:
            print(f"{FALHA} {mod} nao instalado")
            falhas += 1
    print(f"{OK} scipy nao e necessario (normal implementada em model/mathx.py)")

    print("\n-- Matematica (autoteste)")
    for ok, desc in _autoteste_matematica():
        print(f"{OK if ok else FALHA} {desc}")
        falhas += 0 if ok else 1

    ano = date.today().year
    print(f"\n-- Calendario de pregao {ano} (confira contra o oficial da B3)")
    for d, nome in feriados_do_ano(ano).items():
        print(f"         {d.isoformat()}  {nome}")
    print(
        f"{AVISO} fechamentos extraordinarios: acrescente em "
        "data/feriados_b3.csv"
    )

    print("\n-- Fontes de dados registradas")
    for nome, fonte in sorted(fontes.listar().items()):
        disp, motivo = fonte.disponivel()
        print(f"{OK if disp else AVISO} {nome}: {motivo}")

    print("\n-- Pendencias conhecidas")
    print(f"{AVISO} OI real: nenhuma fonte de mercado conectada ainda.")
    print("         Sem ela, so a fonte 'sintetica' funciona, e ela nunca")
    print("         produz saida operavel (marca d'agua obrigatoria).")
    print(f"{AVISO} Backtest: motor pronto e testado, mas sem dado historico.")
    print("         Rode 'backtest --demo' para ver a forma do relatorio.")
    print("         Para valer, precisa de OI de D-1 + barras do WIN por dia.")
    print(f"{AVISO} Grafico: view/ tem apenas relatorio de texto.")

    print("\n" + "=" * 74)
    if falhas:
        print(f"  {falhas} FALHA(S). Nao use antes de resolver.")
    else:
        print("  Tudo verificado. O motor esta consistente.")
    print("=" * 74 + "\n")
    return 1 if falhas else 0


def comando_perfil(args: argparse.Namespace) -> int:
    fonte = fontes.obter(args.fonte)

    disp, motivo = fonte.disponivel()
    if not disp:
        print(f"Fonte '{args.fonte}' indisponivel: {motivo}", file=sys.stderr)
        return 1

    if args.spot is not None and hasattr(fonte, "spot"):
        fonte.spot = args.spot

    chain = fonte.buscar(args.data or date.today())
    print("\n" + chain.descricao_procedencia())

    perfil = construir_perfil(chain, vol_fallback=args.vol_fallback)
    niveis = extrair_niveis(perfil)
    print()
    print(relatorio_texto(perfil, niveis, ratio_win=args.ratio_win))
    print()

    if args.csv:
        perfil.como_dataframe().to_csv(args.csv, index=False)
        print(f"Perfil gravado em {args.csv}\n")

    return 0


def comando_backtest(args: argparse.Namespace) -> int:
    if not args.demo:
        print(
            "Backtest com dados reais ainda nao tem fonte conectada.\n"
            "\n"
            "Para rodar, sao necessarios dois insumos por dia de pregao:\n"
            "  1. o perfil de gama do dia, montado com o OI de D-1\n"
            "  2. as barras intradiarias do WIN (ts, close)\n"
            "\n"
            "Monte uma lista de DiaBacktest e chame rodar_backtest(). O\n"
            "contrato esta em gama_win/backtest/engine.py.\n"
            "\n"
            "Para ver a forma do relatorio agora: --demo\n"
            "(dados sinteticos em passeio aleatorio; o resultado esperado e\n"
            "'nenhum efeito significativo' -- se aparecer edge, e bug)",
            file=sys.stderr,
        )
        return 2

    from .backtest.demo import dias_demo
    from .backtest.engine import ConfigBacktest, rodar_backtest
    from .view.relatorio_backtest import relatorio_backtest

    print("\n" + "#" * 78)
    print("#  DADOS SINTETICOS -- passeio aleatorio. Serve para conferir o")
    print("#  pipeline e a forma do relatorio, nunca para concluir nada.")
    print("#" * 78)

    dias = dias_demo(n_dias=args.dias)
    cfg = ConfigBacktest(
        horizonte_barras=args.horizonte,
        lookback_barras=args.lookback,
        limiar_movimento_pts=args.limiar,
        n_reamostras=args.reamostras,
    )
    resultado = rodar_backtest(dias, cfg)

    print()
    print(relatorio_backtest(resultado))
    print()

    if args.csv_eventos:
        resultado.eventos.to_csv(args.csv_eventos, index=False)
        print(f"Eventos gravados em {args.csv_eventos}\n")

    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="gama_win",
        description="Motor de perfil de gama para o mini indice (WIN).",
    )
    sub = parser.add_subparsers(dest="comando", required=True)

    p_doctor = sub.add_parser(
        "doctor", help="verifica ambiente, calendario e a matematica"
    )
    p_doctor.set_defaults(func=comando_doctor)

    p_perfil = sub.add_parser("perfil", help="calcula e imprime o perfil de gama")
    p_perfil.add_argument("--fonte", default="sintetica", help="nome da fonte")
    p_perfil.add_argument(
        "--data", type=date.fromisoformat, default=None, help="AAAA-MM-DD"
    )
    p_perfil.add_argument(
        "--spot", type=float, default=None, help="sobrescreve o spot da fonte"
    )
    p_perfil.add_argument(
        "--vol-fallback",
        dest="vol_fallback",
        type=float,
        default=None,
        help="vol a usar onde faltar implied_vol (explicito, sem default)",
    )
    p_perfil.add_argument(
        "--ratio-win",
        dest="ratio_win",
        type=float,
        default=None,
        help="razao para converter strikes em pontos de WIN",
    )
    p_perfil.add_argument("--csv", default=None, help="grava o perfil em CSV")
    p_perfil.set_defaults(func=comando_perfil)

    p_bt = sub.add_parser(
        "backtest",
        help="testa se o regime de gama informa algo alem da direcao recente",
    )
    p_bt.add_argument(
        "--demo",
        action="store_true",
        help="roda com dias sinteticos (passeio aleatorio); esperado: sem efeito",
    )
    p_bt.add_argument("--dias", type=int, default=40, help="dias na amostra demo")
    p_bt.add_argument(
        "--horizonte", type=int, default=6, help="barras a frente (6 = 30min em 5min)"
    )
    p_bt.add_argument(
        "--lookback", type=int, default=3, help="barras para medir a direcao recente"
    )
    p_bt.add_argument(
        "--limiar",
        type=float,
        default=0.0,
        help="movimento minimo em pontos para o evento contar",
    )
    p_bt.add_argument(
        "--reamostras", type=int, default=2000, help="reamostras do bootstrap"
    )
    p_bt.add_argument(
        "--csv-eventos",
        dest="csv_eventos",
        default=None,
        help="grava os eventos individuais em CSV para inspecao",
    )
    p_bt.set_defaults(func=comando_backtest)

    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())

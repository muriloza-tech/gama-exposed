"""Gera a pagina do dia a partir dos arquivos publicos da B3.

Este e o script que o job diario executa. Ele e deliberadamente estrito: se
qualquer insumo faltar, ele FALHA em vez de publicar uma pagina com dado
velho ou incompleto. Pagina publica consultada por terceiros nao pode exibir
numero errado com aparencia de atual.

Uso:
    python scripts/gerar_pagina.py [--data AAAA-MM-DD] [--vencimentos 3]
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import date, timedelta
from pathlib import Path

import numpy as np

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ))

# Em maquina com TLS interceptado (antivirus/proxy) o PEM do sistema resolve.
# No runner do CI o arquivo nao existe e o certifi padrao funciona.
_pem = RAIZ / "data" / "ca-bundle.pem"
if _pem.exists():
    for var in ("CURL_CA_BUNDLE", "SSL_CERT_FILE", "REQUESTS_CA_BUNDLE"):
        os.environ.setdefault(var, str(_pem))

from gama_win.data.sources.b3_arquivos import (  # noqa: E402
    FILE_INSTRUMENTOS,
    FILE_NEGOCIOS,
    FILE_POSICOES,
    B3Error,
    baixar_arquivo,
    montar_chain,
    preco_de_fechamento,
)
from gama_win.view.site import gerar_site, montar_payload  # noqa: E402

ATIVO = "BOVA11"
TICKER_INDICE = "IBOV11"
RATE = 0.105
DIV = 0.035
VOL_FALLBACK = 0.22

DIR_CACHE = RAIZ / "data" / "cache"
SAIDA = RAIZ / "docs" / "index.html"


def log(m: str) -> None:
    print(m, flush=True)


def ultima_data_disponivel(a_partir_de: date, tentativas: int = 6) -> tuple[date, dict]:
    """Ultimo dia de pregao cujos TRES arquivos estao publicados.

    Anda para tras porque os dados de D saem depois do fechamento: rodando de
    manha, o mais recente e D-1. Feriados e dias sem publicacao sao pulados.
    """
    d = a_partir_de
    ultimo_erro = ""
    for _ in range(tentativas):
        if not np.is_busday(np.datetime64(d, "D")):
            d -= timedelta(days=1)
            continue
        caminhos = {
            "posicoes": DIR_CACHE / f"DerivativesOpenPosition_{d.isoformat()}.csv",
            "instrumentos": DIR_CACHE / f"Instruments_{d.isoformat()}.csv",
            "negocios": DIR_CACHE / f"TradeInformation_{d.isoformat()}.csv",
        }
        try:
            baixar_arquivo(FILE_POSICOES, d, caminhos["posicoes"])
            baixar_arquivo(FILE_INSTRUMENTOS, d, caminhos["instrumentos"])
            baixar_arquivo(FILE_NEGOCIOS, d, caminhos["negocios"])
            log(f"arquivos de {d.isoformat()} obtidos")
            return d, caminhos
        except B3Error as exc:
            ultimo_erro = str(exc)[:160]
            log(f"  {d.isoformat()} indisponivel: {ultimo_erro}")
            d -= timedelta(days=1)

    raise SystemExit(
        f"nenhuma data com os tres arquivos publicados nas ultimas "
        f"{tentativas} tentativas a partir de {a_partir_de}. "
        f"Ultimo erro: {ultimo_erro}"
    )


def rotulo_do_vencimento(v: date, referencia: date) -> str:
    """Nomeia o vencimento como o mercado o chama: semanal ou mensal.

    Mensal na B3 e a terceira sexta-feira; as demais sextas sao semanais.
    """
    terceira_sexta = None
    d = v.replace(day=1)
    sextas = []
    while d.month == v.month:
        if d.weekday() == 4:
            sextas.append(d)
        d += timedelta(days=1)
    if len(sextas) >= 3:
        terceira_sexta = sextas[2]

    tipo = "mensal" if v == terceira_sexta else "semanal"
    return f"{tipo} {v.strftime('%d/%m')}"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=date.fromisoformat, default=None,
                    help="data de referencia (default: mais recente disponivel)")
    ap.add_argument("--vencimentos", type=int, default=3,
                    help="quantos vencimentos exibir")
    args = ap.parse_args()

    inicio = args.data or date.today()
    ref, caminhos = ultima_data_disponivel(inicio)

    spot = preco_de_fechamento(caminhos["negocios"], ATIVO)
    indice = preco_de_fechamento(caminhos["negocios"], TICKER_INDICE)
    log(f"{ATIVO} {spot:.2f} | {TICKER_INDICE} {indice:.0f} | "
        f"razao {indice / spot:.4f}")

    completo = montar_chain(
        caminhos["posicoes"], caminhos["instrumentos"],
        ativo=ATIVO, rate=RATE, div_yield=DIV,
        caminho_negocios=caminhos["negocios"],
    )
    proximos = [v for v in completo.expiries if v >= ref][: args.vencimentos]
    if not proximos:
        raise SystemExit(f"nenhum vencimento futuro em {ref}")
    log(f"vencimentos: {[v.isoformat() for v in proximos]}")

    chains = {}
    for v in proximos:
        chains[rotulo_do_vencimento(v, ref)] = montar_chain(
            caminhos["posicoes"], caminhos["instrumentos"],
            ativo=ATIVO, rate=RATE, div_yield=DIV,
            caminho_negocios=caminhos["negocios"], expiries=(v,),
        )

    payload = montar_payload(
        chains, ratio_win=indice / spot, vol_fallback=VOL_FALLBACK
    )
    destino = gerar_site(payload, SAIDA)
    log(f"pagina gerada: {destino} ({destino.stat().st_size} bytes)")

    # Marcador de versao: a pagina consulta este arquivo de tempos em tempos
    # e avisa quando ha publicacao nova. Sem isso o visitante fica com a
    # copia em cache do GitHub Pages (10 min) sem saber que ela envelheceu.
    versao = SAIDA.parent / "versao.json"
    versao.write_text(
        json.dumps({"gerado_em": payload["gerado_em"], "as_of": payload["as_of"]}),
        encoding="utf-8",
    )
    log(f"marcador de versao: {versao.name}")

    for v in payload["vencimentos"]:
        log(f"  {v['rotulo']}: gama {v['gama_no_spot'] / 1e6:+.2f} mi "
            f"({v['regime']}) | flip {v['flip']} | "
            f"vol {v['vol_media']:.1%} | iv {v['iv_resolvidas']}/{v['iv_total']}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

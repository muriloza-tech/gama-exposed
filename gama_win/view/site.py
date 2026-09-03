"""Gerador da pagina estatica.

O fluxo e deliberadamente de mao unica: Python calcula tudo e injeta um JSON
no template; a pagina nao faz conta nenhuma, so desenha. Isso mantem a
matematica no lado que tem 1.100 testes e evita reimplementar gregas em
JavaScript.

O mesmo `gerar_site` roda na maquina local e no job diario do GitHub Actions.
"""

from __future__ import annotations

import json
from datetime import date, datetime
from pathlib import Path

import numpy as np

from ..data.schema import OptionChain
from ..model.levels import Zona, extrair_niveis
from ..model.profile import construir_perfil

MARCADOR = "/*__DADOS__*/ null"
CAMINHO_VALIDACAO = Path(__file__).resolve().parents[2] / "validacao.json"
CAMINHO_CONFIG = Path(__file__).resolve().parents[2] / "config.json"
TEMPLATE_PADRAO = Path(__file__).resolve().parents[2] / "site" / "template.html"


class SiteError(RuntimeError):
    """Falha ao gerar a pagina."""


def _zona_json(z: Zona | None) -> dict | None:
    if z is None:
        return None
    return {
        "inicio": round(z.inicio, 2),
        "pico": round(z.pico, 2),
        "fim": round(z.fim, 2),
        "exposicao": round(z.exposicao_zona, 0),
    }


def montar_payload(
    chains: dict[str, OptionChain],
    *,
    ratio_win: float,
    vol_fallback: float | None = None,
    faixa_relativa: float = 0.10,
) -> dict:
    """Monta o JSON da pagina a partir de um OptionChain por vencimento.

    `chains` mapeia rotulo -> chain ja filtrado por vencimento.

    `faixa_relativa` recorta os strikes exibidos a +/- essa fracao do spot.
    Series muito longe do dinheiro tem gama irrelevante e so achatam a escala
    do grafico; o recorte e de EXIBICAO e fica registrado no payload.
    """
    if not chains:
        raise SiteError("nenhum vencimento fornecido")

    spots = {c.spot for c in chains.values()}
    if len(spots) != 1:
        raise SiteError(f"chains com spots diferentes: {sorted(spots)}")
    spot = float(spots.pop())

    datas = {c.as_of for c in chains.values()}
    if len(datas) != 1:
        raise SiteError(f"chains com as_of diferentes: {sorted(datas)}")
    as_of: date = datas.pop()

    if any(c.is_synthetic for c in chains.values()):
        raise SiteError(
            "ha chain sintetico entre os vencimentos. A pagina e publica e "
            "consultada por terceiros: nao publicamos dado fabricado nela."
        )

    vencimentos: list[dict] = []
    for rotulo, chain in chains.items():
        perfil = construir_perfil(chain, vol_fallback=vol_fallback)
        niveis = extrair_niveis(perfil)

        m = (perfil.strikes >= spot * (1 - faixa_relativa)) & (
            perfil.strikes <= spot * (1 + faixa_relativa)
        )
        if not np.any(m):
            raise SiteError(
                f"{rotulo}: nenhum strike dentro de +/-{faixa_relativa:.0%} "
                f"do spot {spot}"
            )

        if len(chain.expiries) != 1:
            raise SiteError(
                f"{rotulo}: esperado um unico vencimento, recebido "
                f"{[d.isoformat() for d in chain.expiries]}"
            )

        vencimentos.append(
            {
                "data": chain.expiries[0].isoformat(),
                "rotulo": rotulo,
                "oi_total": int(perfil.oi_call.sum() + perfil.oi_put.sum()),
                "gama_no_spot": round(perfil.gama_no_spot(), 0),
                "regime": perfil.regime(),
                "vol_media": round(float(perfil.vol_media_ponderada), 4),
                "iv_resolvidas": int(chain.df["implied_vol"].notna().sum()),
                "iv_total": int(len(chain.df)),
                "flip": round(niveis.flip, 2) if niveis.flip is not None else None,
                "call_wall": _zona_json(niveis.call_wall),
                "put_wall": _zona_json(niveis.put_wall),
                "strikes": [round(float(s), 2) for s in perfil.strikes[m]],
                "gama": [round(float(v), 0) for v in perfil.gama_brl_1pct[m]],
                "gama_call": [
                    round(float(v), 0) for v in perfil.gama_call_brl_1pct[m]
                ],
                "gama_put": [
                    round(float(v), 0) for v in perfil.gama_put_brl_1pct[m]
                ],
                "oi_call": [int(v) for v in perfil.oi_call[m]],
                "oi_put": [int(v) for v in perfil.oi_put[m]],
                "avisos": list(niveis.avisos),
            }
        )

    vencimentos.sort(key=lambda v: v["data"])

    # Resultado do backtest, quando existir. A pagina exibe o que houver --
    # inclusive resultado nulo. Omitir uma validacao desfavoravel seria
    # apresentar o painel como melhor do que a evidencia sustenta.
    validacao = None
    if CAMINHO_VALIDACAO.exists():
        validacao = json.loads(CAMINHO_VALIDACAO.read_text(encoding="utf-8"))

    # URL do proxy de cotacao. Ausente ou vazia desliga o preco ao vivo, e a
    # pagina cai para o fechamento -- degradar para o dado certo e mais
    # seguro do que exibir preco velho como se fosse atual.
    cotacao_url = ""
    if CAMINHO_CONFIG.exists():
        cotacao_url = str(
            json.loads(CAMINHO_CONFIG.read_text(encoding="utf-8")).get(
                "cotacao_url", ""
            )
        ).strip()

    return {
        "spot": round(spot, 2),
        "ratio_win": round(float(ratio_win), 2),
        "as_of": as_of.isoformat(),
        "gerado_em": datetime.now().strftime("%d/%m/%Y às %H:%M"),
        "faixa_exibida": faixa_relativa,
        "vol_fallback": vol_fallback,
        "vencimentos": vencimentos,
        "validacao": validacao,
        "cotacao_url": cotacao_url,
    }


def gerar_site(
    payload: dict,
    destino: Path | str,
    *,
    template: Path | str | None = None,
) -> Path:
    """Injeta o payload no template e grava a pagina."""
    template = Path(template or TEMPLATE_PADRAO)
    if not template.exists():
        raise SiteError(f"template nao encontrado: {template}")

    html = template.read_text(encoding="utf-8")
    if MARCADOR not in html:
        raise SiteError(
            f"marcador {MARCADOR!r} ausente em {template}: o template precisa "
            "do ponto de injecao dos dados"
        )

    dados = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    # Fecha a porta para quebra do bloco <script> por conteudo dos dados.
    dados = dados.replace("</", "<\\/")

    destino = Path(destino)
    destino.parent.mkdir(parents=True, exist_ok=True)
    destino.write_text(html.replace(MARCADOR, dados), encoding="utf-8")
    return destino

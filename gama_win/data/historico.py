"""Historico diario: snapshots compactos a partir dos arquivos da B3.

Problema de volume. Um backtest de 60 dias de pregao precisaria, ingenuamente,
de 60 x (4 MB de posicoes + 30 MB de instrumentos + 8,5 MB de negocios) =
2,5 GB de download e a mesma coisa em disco.

Duas decisoes cortam isso:

1. **O cadastro nao precisa ser diario.** Strike, vencimento e tipo de uma
   serie sao fixos desde a listagem. Basta AMOSTRAR o
   InstrumentsConsolidatedFile a cada poucos dias e unir num catalogo
   ISIN -> (strike, vencimento, tipo). Uma serie que nasce e morre
   inteiramente entre duas amostras seria perdida -- por isso o passo de
   amostragem tem de ser menor que a vida de uma semanal, e a COBERTURA e
   medida e reportada, nunca assumida.

   A chave e o ISIN, NAO o ticker: a B3 recicla codigos de opcao. BOVAF162
   (call de junho, strike 162) aponta para a serie de junho/2026 e, depois
   que ela vence, para a de junho/2027. Chavear por ticker num historico de
   60 dias produz strike e vencimento errados sem nenhum sintoma visivel.
   Descoberto porque a validacao de divergencia levantou.

2. **Guardamos o extrato, nao o bruto.** De cada dia sobra um JSON de ~80 KB
   com o que o modelo usa: spot, indice e a lista de series com OI e preco.
   Os arquivos brutos sao apagados depois da extracao (`manter_brutos=True`
   preserva, para auditoria).

Resultado: ~930 MB de trafego para 60 dias e ~5 MB em disco.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from ..model.calendario import dias_uteis_ate, tau_anos
from .schema import OptionChain
from .sources.b3_arquivos import (
    CONTRACT_SIZE_OPCAO_ACAO,
    FILE_INSTRUMENTOS,
    FILE_NEGOCIOS,
    FILE_POSICOES,
    B3Error,
    baixar_arquivo,
    carregar_instrumentos,
    carregar_negocios,
    carregar_posicoes_abertas,
)

RAIZ = Path(__file__).resolve().parents[2]
DIR_CACHE = RAIZ / "data" / "cache"
DIR_SNAPSHOTS = RAIZ / "data" / "snapshots"
CAMINHO_CATALOGO = RAIZ / "data" / "catalogo_series.json"

TICKER_INDICE = "IBOV11"


class HistoricoError(RuntimeError):
    """Falha ao montar o historico."""


# ------------------------------------------------------------ catalogo ---


def datas_de_amostragem(inicio: date, fim: date, passo_dias: int = 10) -> list[date]:
    """Datas para amostrar o cadastro, de `passo_dias` em `passo_dias` uteis.

    Inclui sempre as duas pontas: o cadastro do fim cobre as series ainda
    vivas, e o do inicio cobre as que ja venceram no meio da janela.
    """
    if fim < inicio:
        raise HistoricoError(f"fim ({fim}) anterior a inicio ({inicio})")

    datas: list[date] = []
    d = inicio
    while d <= fim:
        if np.is_busday(np.datetime64(d, "D")):
            if not datas or dias_uteis_ate(datas[-1], d) > passo_dias:
                datas.append(d)
        d += timedelta(days=1)

    if not datas:
        raise HistoricoError(f"nenhum dia util entre {inicio} e {fim}")
    if datas[-1] != fim and np.is_busday(np.datetime64(fim, "D")):
        datas.append(fim)
    return datas


def construir_catalogo(
    datas: list[date],
    *,
    ativo: str = "BOVA11",
    manter_brutos: bool = False,
    destino: Path | None = None,
) -> dict[str, dict]:
    """Une o cadastro de varias datas num unico dicionario por ISIN.

    A chave e o ISIN porque a B3 RECICLA codigos de opcao: um mesmo ticker
    aponta para series de anos diferentes conforme as antigas vencem. ISIN e
    imutavel por serie.

    Se o mesmo ISIN aparecer com strike/vencimento divergentes, levanta -- ai
    nem o ISIN serviria como chave, e o backtest inteiro depende disso.
    """
    destino = destino or CAMINHO_CATALOGO
    catalogo: dict[str, dict] = {}

    for d in datas:
        bruto = DIR_CACHE / f"Instruments_{d.isoformat()}.csv"
        try:
            baixar_arquivo(FILE_INSTRUMENTOS, d, bruto)
        except B3Error as exc:
            print(f"  [pula] cadastro de {d}: {exc}")
            continue

        inst = carregar_instrumentos(bruto, ativo=ativo)
        novos = 0
        for r in inst.itertuples(index=False):
            reg = {
                "strike": float(r.strike),
                "expiry": r.expiry.date().isoformat(),
                "kind": r.kind,
                "ticker": r.ticker,
            }
            antigo = catalogo.get(r.isin)
            if antigo is None:
                catalogo[r.isin] = reg
                novos += 1
            elif {k: antigo[k] for k in ("strike", "expiry", "kind")} != {
                k: reg[k] for k in ("strike", "expiry", "kind")
            }:
                raise HistoricoError(
                    f"ISIN {r.isin} com cadastro divergente entre datas: "
                    f"{antigo} vs {reg} (visto em {d}). ISIN deveria ser "
                    "imutavel por serie -- se isto disparar, a chave do "
                    "catalogo tambem nao serve."
                )
        print(f"  cadastro {d}: {len(inst)} series, {novos} novas "
              f"(catalogo: {len(catalogo)})")

        if not manter_brutos:
            bruto.unlink(missing_ok=True)

    destino.parent.mkdir(parents=True, exist_ok=True)
    destino.write_text(json.dumps(catalogo), encoding="utf-8")
    return catalogo


def carregar_catalogo(caminho: Path | None = None) -> dict[str, dict]:
    caminho = caminho or CAMINHO_CATALOGO
    if not caminho.exists():
        raise HistoricoError(
            f"catalogo ausente em {caminho}. Rode construir_catalogo() antes."
        )
    return json.loads(caminho.read_text(encoding="utf-8"))


# ------------------------------------------------------------ snapshot ---


@dataclass(frozen=True, slots=True)
class Snapshot:
    """Extrato de um dia: o minimo que o modelo consome."""

    data: date
    ativo: str
    spot: float
    indice: float
    series: pd.DataFrame  # expiry, strike, kind, open_interest, preco
    cobertura_oi: float   # fracao do OI que casou com o catalogo

    @property
    def ratio_win(self) -> float:
        return self.indice / self.spot


def _caminho_snapshot(data: date, ativo: str) -> Path:
    return DIR_SNAPSHOTS / f"{ativo}_{data.isoformat()}.json"


def construir_snapshot(
    data: date,
    catalogo: dict[str, dict],
    *,
    ativo: str = "BOVA11",
    manter_brutos: bool = False,
    reusar: bool = True,
) -> Snapshot:
    """Baixa, extrai e grava o snapshot de um dia. Reusa se ja existir."""
    destino = _caminho_snapshot(data, ativo)
    if reusar and destino.exists():
        return _ler_snapshot(destino)

    pos_bruto = DIR_CACHE / f"DerivativesOpenPosition_{data.isoformat()}.csv"
    neg_bruto = DIR_CACHE / f"TradeInformation_{data.isoformat()}.csv"
    baixar_arquivo(FILE_POSICOES, data, pos_bruto)
    baixar_arquivo(FILE_NEGOCIOS, data, neg_bruto)

    raiz = ativo[:4].upper()
    pos = carregar_posicoes_abertas(pos_bruto, ativo=raiz)
    neg = carregar_negocios(neg_bruto)

    from .sources.b3_arquivos import preco_de_fechamento

    spot = preco_de_fechamento(neg_bruto, ativo, negocios=neg)
    indice = preco_de_fechamento(neg_bruto, TICKER_INDICE, negocios=neg)

    preco = neg["referencia"].where(neg["referencia"].notna(), neg["ultimo"])
    precos = dict(zip(neg["ticker"], preco, strict=True))

    linhas = []
    oi_total = 0
    oi_casado = 0
    for r in pos.itertuples(index=False):
        oi_total += int(r.open_interest)
        reg = catalogo.get(r.isin)
        if reg is None:
            continue
        oi_casado += int(r.open_interest)
        linhas.append(
            {
                "expiry": reg["expiry"],
                "strike": reg["strike"],
                "kind": reg["kind"],
                "open_interest": int(r.open_interest),
                "preco": float(precos.get(r.ticker, float("nan"))),
            }
        )

    if not linhas:
        raise HistoricoError(
            f"{data}: nenhuma serie de {ativo} casou com o catalogo"
        )

    cobertura = oi_casado / oi_total if oi_total else 0.0

    payload = {
        "data": data.isoformat(),
        "ativo": ativo,
        "spot": spot,
        "indice": indice,
        "cobertura_oi": cobertura,
        "series": linhas,
    }
    destino.parent.mkdir(parents=True, exist_ok=True)
    destino.write_text(json.dumps(payload, separators=(",", ":")), encoding="utf-8")

    if not manter_brutos:
        pos_bruto.unlink(missing_ok=True)
        neg_bruto.unlink(missing_ok=True)

    return _ler_snapshot(destino)


def _ler_snapshot(caminho: Path) -> Snapshot:
    p = json.loads(caminho.read_text(encoding="utf-8"))
    df = pd.DataFrame(p["series"])
    return Snapshot(
        data=date.fromisoformat(p["data"]),
        ativo=p["ativo"],
        spot=float(p["spot"]),
        indice=float(p["indice"]),
        series=df,
        cobertura_oi=float(p["cobertura_oi"]),
    )


def snapshots_disponiveis(ativo: str = "BOVA11") -> list[date]:
    if not DIR_SNAPSHOTS.exists():
        return []
    saida = []
    for f in DIR_SNAPSHOTS.glob(f"{ativo}_*.json"):
        try:
            saida.append(date.fromisoformat(f.stem.split("_")[-1]))
        except ValueError:
            continue
    return sorted(saida)


def carregar_snapshot(data: date, ativo: str = "BOVA11") -> Snapshot:
    c = _caminho_snapshot(data, ativo)
    if not c.exists():
        raise HistoricoError(f"snapshot ausente: {c}")
    return _ler_snapshot(c)


# --------------------------------------------------------------- chain ---


def chain_do_snapshot(
    snap: Snapshot,
    *,
    rate: float,
    div_yield: float,
    vencimento: date | None = None,
    contract_size: float = CONTRACT_SIZE_OPCAO_ACAO,
    resolver_vol: bool = True,
) -> OptionChain:
    """Converte um snapshot num OptionChain de um vencimento.

    `vencimento=None` escolhe o PRIMEIRO vencimento a partir da data do
    snapshot -- o que domina o gama de curto prazo.
    """
    df = snap.series.copy()
    df["expiry_date"] = df["expiry"].map(date.fromisoformat)
    df = df[df["expiry_date"] >= snap.data]
    if df.empty:
        raise HistoricoError(f"{snap.data}: nenhum vencimento futuro no snapshot")

    if vencimento is None:
        vencimento = min(df["expiry_date"])
    df = df[df["expiry_date"] == vencimento]
    if df.empty:
        raise HistoricoError(
            f"{snap.data}: vencimento {vencimento} ausente no snapshot"
        )

    df["preco_x_oi"] = df["preco"] * df["open_interest"]
    ag = df.groupby(["expiry_date", "strike", "kind"], as_index=False).agg(
        open_interest=("open_interest", "sum"),
        preco_x_oi=("preco_x_oi", "sum"),
        oi=("open_interest", "sum"),
    )
    ag["preco"] = ag["preco_x_oi"] / ag["oi"]

    saida = pd.DataFrame(
        {
            "expiry": pd.to_datetime(ag["expiry_date"]),
            "strike": ag["strike"].astype(float),
            "kind": ag["kind"],
            "open_interest": ag["open_interest"].astype("int64"),
            "contract_size": float(contract_size),
            "implied_vol": float("nan"),
            "settlement_price": ag["preco"].astype(float),
        }
    )

    if resolver_vol:
        from ..model.greeks import implied_vol

        tau = tau_anos(snap.data, vencimento)
        precos = saida["settlement_price"].to_numpy(float)
        ok = np.isfinite(precos) & (precos > 0)
        vols = np.full(len(saida), np.nan)
        if np.any(ok) and tau > 0:
            vols[ok] = implied_vol(
                precos[ok],
                snap.spot,
                saida["strike"].to_numpy(float)[ok],
                tau,
                rate,
                div_yield,
                saida["kind"].to_numpy(object)[ok],
            )
        saida["implied_vol"] = np.where(
            np.isfinite(vols) & (vols > 0) & (vols <= 5.0), vols, np.nan
        )

    return OptionChain(
        df=saida,
        underlying=snap.ativo,
        spot=snap.spot,
        as_of=snap.data,
        source=f"B3/snapshot@{snap.data.isoformat()}",
        rate=rate,
        div_yield=div_yield,
        is_synthetic=False,
        notas=(
            f"cobertura do OI pelo catalogo: {snap.cobertura_oi:.1%}",
            f"contract_size={contract_size} e premissa de modelagem",
        ),
    )

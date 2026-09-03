"""Perfil de exposicao a gama por strike, em unidades economicas reais.

A unidade importa. "Net GEX" sem unidade e o erro mais comum em paineis de
gama: o numero fica grande, parece significativo, e ninguem sabe do que ele
e feito. Aqui cada grandeza tem unidade declarada e formula documentada:

    gama_brl_1pct
        Variacao da posicao de DELTA do dealer, em REAIS, para cada 1% de
        movimento do subjacente.
        formula: gamma * OI * contract_size * spot**2 * 0.01
        derivacao: gamma*OI*cs = delta (em unidades) por R$1 de movimento;
                   x spot -> reais; x (0.01*spot) -> por 1% de movimento.

    charm_brl_dia
        Variacao da posicao de delta, em reais, por dia de pregao decorrido,
        com o resto constante. Explica por que os niveis "vazam" ao longo do
        dia e por que a semana de vencimento se comporta diferente.
        formula: charm_por_dia_pregao * OI * contract_size * spot

    vanna_brl_por_ponto_vol
        Variacao da posicao de delta, em reais, por 1 ponto percentual de
        volatilidade implicita. Explica movimentos que furam a wall sem
        volume, no puro recuo de vol.
        formula: vanna * 0.01 * OI * contract_size * spot

Nenhum parametro tem default escondido. Se a serie nao traz volatilidade
implicita, e obrigatorio informar `vol_fallback` explicitamente -- nao existe
"22% porque sim".
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from ..data.schema import CALL, PUT, OptionChain
from .conventions import DEFAULT_CONVENTION, DealerConvention
from .greeks import bsm_greeks


class ProfileError(ValueError):
    """Nao foi possivel construir o perfil com os insumos dados."""


@dataclass(frozen=True, slots=True)
class GammaProfile:
    """Exposicao agregada por strike, no espaco de strikes do subjacente."""

    strikes: NDArray[np.float64]

    gama_brl_1pct: NDArray[np.float64]      # liquido, com sinal da convencao
    gama_call_brl_1pct: NDArray[np.float64]  # componente call, com sinal
    gama_put_brl_1pct: NDArray[np.float64]   # componente put, com sinal
    gama_bruto_brl_1pct: NDArray[np.float64]  # |call| + |put|, sem sinal

    charm_brl_dia: NDArray[np.float64]
    vanna_brl_por_ponto_vol: NDArray[np.float64]

    oi_call: NDArray[np.int64]
    oi_put: NDArray[np.int64]

    spot: float
    underlying: str
    as_of: date
    source: str
    is_synthetic: bool
    convention: DealerConvention
    expiries: tuple[date, ...]
    vol_media_ponderada: float
    por_vencimento: pd.DataFrame

    def __post_init__(self) -> None:
        n = len(self.strikes)
        for nome in (
            "gama_brl_1pct",
            "gama_call_brl_1pct",
            "gama_put_brl_1pct",
            "gama_bruto_brl_1pct",
            "charm_brl_dia",
            "vanna_brl_por_ponto_vol",
            "oi_call",
            "oi_put",
        ):
            if len(getattr(self, nome)) != n:
                raise ProfileError(
                    f"{nome} tem {len(getattr(self, nome))} elementos, "
                    f"esperado {n} (mesmo tamanho de strikes)"
                )
        if n < 2:
            raise ProfileError(
                f"perfil precisa de ao menos 2 strikes, recebido {n}"
            )
        if not np.all(np.diff(self.strikes) > 0):
            raise ProfileError("strikes devem estar ordenados e sem repeticao")

    @property
    def strike_mais_proximo_do_spot(self) -> float:
        return float(self.strikes[int(np.argmin(np.abs(self.strikes - self.spot)))])

    def gama_no_spot(self) -> float:
        """Gama liquido interpolado NO spot -- nao o do strike vizinho.

        Interpolar importa: perto do flip, ler o strike vizinho faz o
        semaforo de regime piscar sem que nada tenha mudado no mercado.
        """
        return float(np.interp(self.spot, self.strikes, self.gama_brl_1pct))

    def regime(self) -> str:
        g = self.gama_no_spot()
        if g > 0:
            return "gama positivo"
        if g < 0:
            return "gama negativo"
        return "neutro"

    def como_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "strike": self.strikes,
                "gama_brl_1pct": self.gama_brl_1pct,
                "gama_call_brl_1pct": self.gama_call_brl_1pct,
                "gama_put_brl_1pct": self.gama_put_brl_1pct,
                "gama_bruto_brl_1pct": self.gama_bruto_brl_1pct,
                "charm_brl_dia": self.charm_brl_dia,
                "vanna_brl_por_ponto_vol": self.vanna_brl_por_ponto_vol,
                "oi_call": self.oi_call,
                "oi_put": self.oi_put,
            }
        )


def construir_perfil(
    chain: OptionChain,
    *,
    convention: DealerConvention = DEFAULT_CONVENTION,
    vol_fallback: float | None = None,
    expiries: tuple[date, ...] | None = None,
) -> GammaProfile:
    """Agrega a serie de opcoes num perfil de exposicao por strike.

    Parametros
    ----------
    chain : serie validada
    convention : convencao de sinal do dealer, carregada no resultado
    vol_fallback : vol a usar onde `implied_vol` for NaN. Obrigatorio se
        houver qualquer NaN -- nao ha default escondido.
    expiries : se informado, restringe aos vencimentos listados. Util para
        isolar o vencimento curto, que domina o gama de day trade.
    """
    df = chain.com_tau()

    if expiries is not None:
        alvo = {d for d in expiries}
        df = df[df["expiry"].dt.date.isin(alvo)]
        if df.empty:
            disponiveis = [d.isoformat() for d in chain.expiries]
            raise ProfileError(
                f"nenhuma serie nos vencimentos {sorted(d.isoformat() for d in alvo)}; "
                f"disponiveis: {disponiveis}"
            )

    if (df["tau"] <= 0).any():
        n = int((df["tau"] <= 0).sum())
        raise ProfileError(
            f"{n} serie(s) com tau <= 0 apos a contagem de dias uteis. "
            "Filtre vencimentos passados na fonte."
        )

    # --- volatilidade: explicita ou fallback declarado -------------------
    if "implied_vol" in df.columns:
        vol = pd.to_numeric(df["implied_vol"], errors="coerce").to_numpy(float)
    else:
        vol = np.full(len(df), np.nan)

    faltantes = int(np.sum(~np.isfinite(vol)))
    if faltantes:
        if vol_fallback is None:
            raise ProfileError(
                f"{faltantes} de {len(df)} serie(s) sem implied_vol e nenhum "
                "vol_fallback informado. Passe vol_fallback=<valor> de forma "
                "explicita (ex.: 0.22) assumindo o vies de vol plana, ou "
                "traga a vol implicita da fonte. Nao ha default implicito: "
                "vol plana distorce o gama exatamente nas asas, onde ficam "
                "as walls."
            )
        if not np.isfinite(vol_fallback) or vol_fallback <= 0:
            raise ProfileError(
                f"vol_fallback deve ser finito e > 0, recebido {vol_fallback!r}"
            )
        vol = np.where(np.isfinite(vol), vol, float(vol_fallback))

    # --- gregas por serie ------------------------------------------------
    g = bsm_greeks(
        chain.spot,
        df["strike"].to_numpy(float),
        df["tau"].to_numpy(float),
        chain.rate,
        chain.div_yield,
        vol,
        df["kind"].to_numpy(object),
    )

    oi = df["open_interest"].to_numpy(float)
    cs = df["contract_size"].to_numpy(float)
    S = float(chain.spot)

    nocional = oi * cs
    gama_brl_1pct = g.gamma * nocional * S * S * 0.01
    charm_brl_dia = g.charm_por_dia_pregao() * nocional * S
    vanna_brl_pt = g.vanna * 0.01 * nocional * S

    e_call = df["kind"].to_numpy(object) == CALL
    sinal = np.where(e_call, convention.call_sign, convention.put_sign)

    trabalho = pd.DataFrame(
        {
            "strike": df["strike"].to_numpy(float),
            "expiry": df["expiry"].dt.date.to_numpy(),
            "e_call": e_call,
            "oi": oi.astype(np.int64),
            "gama_assinado": gama_brl_1pct * sinal,
            "gama_bruto": np.abs(gama_brl_1pct),
            "charm_assinado": charm_brl_dia * sinal,
            "vanna_assinado": vanna_brl_pt * sinal,
            "vega_peso": g.vega * nocional,
            "vol": vol,
        }
    )

    strikes = np.sort(trabalho["strike"].unique())
    idx = pd.Index(strikes, name="strike")

    def _soma(mascara: np.ndarray, coluna: str) -> NDArray[np.float64]:
        s = (
            trabalho.loc[mascara]
            .groupby("strike")[coluna]
            .sum()
            .reindex(idx, fill_value=0.0)
        )
        return s.to_numpy(float)

    todos = np.ones(len(trabalho), dtype=bool)
    calls = trabalho["e_call"].to_numpy(bool)
    puts = ~calls

    # Vol media ponderada por vega: a media que faz sentido economico, ja que
    # vega e o quanto cada serie responde a vol.
    peso = trabalho["vega_peso"].to_numpy(float)
    vol_media = (
        float(np.sum(peso * vol) / np.sum(peso)) if np.sum(peso) > 0 else float("nan")
    )

    por_vencimento = (
        trabalho.groupby(["expiry", "strike"])["gama_assinado"]
        .sum()
        .unstack("expiry")
        .reindex(idx)
        .fillna(0.0)
    )

    return GammaProfile(
        strikes=strikes,
        gama_brl_1pct=_soma(todos, "gama_assinado"),
        gama_call_brl_1pct=_soma(calls, "gama_assinado"),
        gama_put_brl_1pct=_soma(puts, "gama_assinado"),
        gama_bruto_brl_1pct=_soma(todos, "gama_bruto"),
        charm_brl_dia=_soma(todos, "charm_assinado"),
        vanna_brl_por_ponto_vol=_soma(todos, "vanna_assinado"),
        oi_call=_soma(calls, "oi").astype(np.int64),
        oi_put=_soma(puts, "oi").astype(np.int64),
        spot=S,
        underlying=chain.underlying,
        as_of=chain.as_of,
        source=chain.source,
        is_synthetic=chain.is_synthetic,
        convention=convention,
        expiries=tuple(sorted({d for d in trabalho["expiry"]})),
        vol_media_ponderada=vol_media,
        por_vencimento=por_vencimento,
    )

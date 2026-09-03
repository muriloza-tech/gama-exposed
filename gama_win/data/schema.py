"""Contrato de dados da serie de opcoes -- a fronteira onde tudo e validado.

Filosofia: qualquer fonte de dados (B3, corretora, CSV, sintetico) precisa
produzir um `OptionChain` valido. A validacao acontece UMA vez, aqui, e
acumula TODOS os problemas antes de falhar -- corrigir dez erros um a um,
rodando de novo a cada vez, e o que faz gente desistir de validar.

Depois desta fronteira, o resto do sistema pode assumir que os dados estao
limpos. Nenhum modulo a jusante checa NaN de novo.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date

import numpy as np
import pandas as pd

from ..model.calendario import tau_anos

CALL = "C"
PUT = "P"

COLUNAS_OBRIGATORIAS: tuple[str, ...] = (
    "expiry",
    "strike",
    "kind",
    "open_interest",
    "contract_size",
)

COLUNAS_OPCIONAIS: tuple[str, ...] = (
    "implied_vol",
    "settlement_price",
    "symbol",
)

FONTE_SINTETICA = "SYNTHETIC"


class ChainValidationError(ValueError):
    """A serie de opcoes violou o contrato. A mensagem lista tudo."""


@dataclass(frozen=True, slots=True)
class OptionChain:
    """Serie de opcoes validada, com procedencia rastreavel.

    Atributos
    ---------
    df : DataFrame com COLUNAS_OBRIGATORIAS (+ opcionais quando houver)
    underlying : ticker do subjacente ('BOVA11', 'IBOV', ...)
    spot : preco do subjacente no momento `as_of`
    as_of : data de referencia dos dados
    source : identificador da fonte, para auditoria
    is_synthetic : True marca dados fabricados. Propaga-se ate a marca
        d'agua do grafico; nenhuma saida pode parecer operavel sem ser.
    rate : taxa livre de risco continua
    div_yield : dividend yield continuo do subjacente
    """

    df: pd.DataFrame
    underlying: str
    spot: float
    as_of: date
    source: str
    rate: float
    div_yield: float
    is_synthetic: bool = False
    notas: tuple[str, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        problemas = self._validar()
        if problemas:
            cabecalho = (
                f"{len(problemas)} problema(s) na serie de opcoes "
                f"[{self.underlying} @ {self.as_of} via {self.source}]:"
            )
            raise ChainValidationError(
                cabecalho + "\n" + "\n".join(f"  - {p}" for p in problemas)
            )

    # ------------------------------------------------------------ validacao --

    def _validar(self) -> list[str]:
        p: list[str] = []
        df = self.df

        if not isinstance(df, pd.DataFrame):
            return [f"df deve ser DataFrame, recebido {type(df).__name__}"]

        faltando = [c for c in COLUNAS_OBRIGATORIAS if c not in df.columns]
        if faltando:
            p.append(f"colunas obrigatorias ausentes: {faltando}")
            return p  # sem colunas nao ha o que validar adiante

        if df.empty:
            p.append("serie vazia (0 linhas)")
            return p

        if not np.isfinite(self.spot) or self.spot <= 0:
            p.append(f"spot deve ser finito e > 0, recebido {self.spot!r}")
        if not np.isfinite(self.rate):
            p.append(f"rate deve ser finito, recebido {self.rate!r}")
        if not np.isfinite(self.div_yield):
            p.append(f"div_yield deve ser finito, recebido {self.div_yield!r}")
        if not self.source:
            p.append("source vazio: procedencia dos dados e obrigatoria")

        # kind
        kinds = df["kind"].astype("string").str.strip().str.upper()
        invalidos = sorted(set(kinds.dropna()) - {CALL, PUT})
        if invalidos:
            p.append(f"kind fora de {{'C','P'}}: {invalidos}")
        if kinds.isna().any():
            p.append(f"kind nulo em {int(kinds.isna().sum())} linha(s)")

        # strike
        strike = pd.to_numeric(df["strike"], errors="coerce")
        if strike.isna().any():
            p.append(
                f"strike nao numerico/nulo em {int(strike.isna().sum())} linha(s)"
            )
        ruins = strike[(strike <= 0) | ~np.isfinite(strike)]
        if len(ruins):
            p.append(
                f"strike <= 0 ou nao finito em {len(ruins)} linha(s): "
                f"{ruins.head(3).tolist()}"
            )

        # open_interest
        oi = pd.to_numeric(df["open_interest"], errors="coerce")
        if oi.isna().any():
            p.append(
                f"open_interest nao numerico/nulo em {int(oi.isna().sum())} linha(s)"
            )
        if (oi < 0).any():
            p.append(f"open_interest negativo em {int((oi < 0).sum())} linha(s)")
        if oi.fillna(0).sum() <= 0:
            p.append("open_interest soma zero: nao ha posicao a analisar")

        # contract_size
        cs = pd.to_numeric(df["contract_size"], errors="coerce")
        if cs.isna().any() or (cs <= 0).any():
            p.append("contract_size deve ser numerico e > 0 em todas as linhas")

        # expiry
        try:
            expiry = pd.to_datetime(df["expiry"], errors="coerce")
        except (ValueError, TypeError) as exc:
            expiry = None
            p.append(f"expiry nao convertivel para data: {exc}")
        if expiry is not None:
            if expiry.isna().any():
                p.append(
                    f"expiry invalido/nulo em {int(expiry.isna().sum())} linha(s)"
                )
            vencidos = expiry[expiry.dt.date < self.as_of]
            if len(vencidos):
                p.append(
                    f"{len(vencidos)} linha(s) com expiry anterior a as_of "
                    f"({self.as_of}): filtre series vencidas na fonte. "
                    f"Exemplos: {sorted({d.date().isoformat() for d in vencidos})[:3]}"
                )

        # duplicatas
        if expiry is not None and not expiry.isna().any():
            chave = pd.DataFrame(
                {"expiry": expiry.dt.date, "strike": strike, "kind": kinds}
            )
            dup = chave.duplicated(keep=False)
            if dup.any():
                exemplos = chave[dup].head(3).to_dict("records")
                p.append(
                    f"{int(dup.sum())} linha(s) duplicadas em "
                    f"(expiry, strike, kind). Exemplos: {exemplos}"
                )

        # implied_vol -- NaN e permitido (sera estimado), valor absurdo nao
        if "implied_vol" in df.columns:
            iv = pd.to_numeric(df["implied_vol"], errors="coerce")
            presente = iv.notna()
            if presente.any():
                fora = iv[presente & ((iv <= 0) | (iv > 5.0))]
                if len(fora):
                    p.append(
                        f"implied_vol fora de (0, 5] em {len(fora)} linha(s): "
                        f"{fora.head(3).tolist()}"
                    )

        return p

    # -------------------------------------------------------------- derivados --

    @property
    def expiries(self) -> tuple[date, ...]:
        return tuple(
            sorted({d.date() for d in pd.to_datetime(self.df["expiry"])})
        )

    @property
    def total_open_interest(self) -> int:
        return int(pd.to_numeric(self.df["open_interest"]).sum())

    def com_tau(self) -> pd.DataFrame:
        """Copia do df com `tau` (anos de pregao) e `kind` normalizado.

        Calculado aqui para que todas as camadas usem a MESMA contagem de
        dias uteis -- e nao cada uma a sua.
        """
        out = self.df.copy()
        out["kind"] = out["kind"].astype("string").str.strip().str.upper()
        out["strike"] = pd.to_numeric(out["strike"])
        out["open_interest"] = pd.to_numeric(out["open_interest"])
        out["contract_size"] = pd.to_numeric(out["contract_size"])
        expiry = pd.to_datetime(out["expiry"])
        out["expiry"] = expiry
        out["tau"] = [tau_anos(self.as_of, d.date()) for d in expiry]
        return out

    def descricao_procedencia(self) -> str:
        marca = " [DADOS SINTETICOS]" if self.is_synthetic else ""
        return (
            f"{self.underlying} @ {self.as_of.isoformat()} | fonte: "
            f"{self.source}{marca} | {len(self.df)} series | "
            f"OI total: {self.total_open_interest:,}".replace(",", ".")
        )

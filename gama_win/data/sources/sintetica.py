"""Fonte SINTETICA -- estrutura realista, dados inventados, sempre marcados.

Existe para dois usos legitimos:

  1. Desenvolver e testar o pipeline sem depender de dados de mercado.
  2. Demonstrar a ferramenta.

E NAO existe para operar. Todo `OptionChain` que sai daqui tem
`is_synthetic=True`, o que propaga aviso ate os niveis e a marca d'agua do
grafico. Nao ha como uma saida sintetica se passar por real.

Diferenca em relacao ao gerador de seno/cosseno do painel original: a forma
do OI aqui imita a de uma serie de verdade -- concentracao perto do dinheiro,
decaimento nas asas, acumulacao em strikes redondos e smile de volatilidade
com skew de put. Isso faz o perfil ter UM flip dominante em vez de seis
trocas de sinal artificiais.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta

import numpy as np
import pandas as pd

from ..schema import FONTE_SINTETICA, OptionChain
from .base import registrar


@dataclass
class FonteSintetica:
    """Gera uma serie plausivel em torno de um spot informado."""

    nome: str = "sintetica"
    underlying: str = "SINTETICO"
    spot: float = 100.0
    passo_strike: float = 1.0
    n_strikes_cada_lado: int = 15
    contract_size: float = 100.0
    rate: float = 0.105
    div_yield: float = 0.035
    vol_atm: float = 0.22
    skew: float = -0.35      # put mais caro que call (skew tipico de indice)
    curvatura: float = 0.90
    oi_base: int = 12_000
    largura_oi: float = 0.045  # em log-moneyness
    dias_ate_vencimento: int = 11
    semente: int = 42

    def disponivel(self) -> tuple[bool, str]:
        return True, "gerador local; nao depende de rede"

    # -------------------------------------------------------------------

    def buscar(self, as_of: date) -> OptionChain:
        rng = np.random.default_rng(self.semente)

        centro = round(self.spot / self.passo_strike) * self.passo_strike
        n = self.n_strikes_cada_lado
        strikes = centro + self.passo_strike * np.arange(-n, n + 1)
        strikes = strikes[strikes > 0]

        vencimento = as_of + timedelta(days=self.dias_ate_vencimento)

        m = np.log(strikes / self.spot)  # log-moneyness
        forma = np.exp(-0.5 * (m / self.largura_oi) ** 2)

        # Strikes redondos concentram posicao -- comportamento real de mercado.
        redondo = np.where(np.isclose(strikes % 5.0, 0.0), 1.8, 1.0)

        # Call acumula acima do dinheiro; put acumula abaixo.
        peso_call = forma * redondo * (1.0 + 0.9 * np.clip(m, 0, None) / 0.05)
        peso_put = forma * redondo * (1.0 + 1.3 * np.clip(-m, 0, None) / 0.05)

        ruido_c = rng.uniform(0.75, 1.25, size=len(strikes))
        ruido_p = rng.uniform(0.75, 1.25, size=len(strikes))

        oi_call = np.maximum((self.oi_base * peso_call * ruido_c).astype(int), 0)
        oi_put = np.maximum((self.oi_base * peso_put * ruido_p).astype(int), 0)

        # Smile: ATM + skew linear + curvatura quadratica em log-moneyness.
        iv = self.vol_atm + self.skew * m + self.curvatura * m**2
        iv = np.clip(iv, 0.05, 1.50)

        df = pd.DataFrame(
            {
                "expiry": list(np.repeat(vencimento, len(strikes) * 2)),
                "strike": np.concatenate([strikes, strikes]),
                "kind": ["C"] * len(strikes) + ["P"] * len(strikes),
                "open_interest": np.concatenate([oi_call, oi_put]),
                "contract_size": self.contract_size,
                "implied_vol": np.concatenate([iv, iv]),
            }
        )
        df = df[df["open_interest"] > 0].reset_index(drop=True)

        return OptionChain(
            df=df,
            underlying=self.underlying,
            spot=self.spot,
            as_of=as_of,
            source=FONTE_SINTETICA,
            rate=self.rate,
            div_yield=self.div_yield,
            is_synthetic=True,
            notas=(
                "Serie gerada localmente. Forma imita uma serie real, mas os "
                "numeros sao inventados e nao representam posicao de mercado.",
            ),
        )


registrar(FonteSintetica())

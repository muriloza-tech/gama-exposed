"""Convencao de sinal do dealer -- explicita, nunca implicita.

O sinal do gama agregado nao e um detalhe de implementacao: e a diferenca
entre "mercado travado" e "mercado acelerando". Este modulo forca a escolha
a ser nomeada e carregada junto com o resultado, para que nenhum grafico
saia sem dizer qual convencao usou.
"""

from __future__ import annotations

from enum import Enum


class DealerConvention(str, Enum):
    """Quem se assume estar do outro lado do fluxo.

    LONG_CALL_SHORT_PUT
        Convencao padrao de mercado. Assume que o dealer esta comprado em
        call e vendido em put (o varejo compra call e compra put como
        protecao, o dealer fica do outro lado da call e vende a put).
        Resultado: call contribui gama POSITIVO, put contribui NEGATIVO.
        Leitura: gama positivo = hedge contraciclico = reversao a media;
        gama negativo = hedge prociclico = movimento acelera.

    SHORT_CALL_LONG_PUT
        Convencao invertida. Existe aqui apenas para reproduzir paineis que
        a usam e para teste de sensibilidade. NAO e a leitura padrao.
    """

    LONG_CALL_SHORT_PUT = "long_call_short_put"
    SHORT_CALL_LONG_PUT = "short_call_long_put"

    @property
    def call_sign(self) -> float:
        return 1.0 if self is DealerConvention.LONG_CALL_SHORT_PUT else -1.0

    @property
    def put_sign(self) -> float:
        return -1.0 if self is DealerConvention.LONG_CALL_SHORT_PUT else 1.0

    @property
    def descricao(self) -> str:
        if self is DealerConvention.LONG_CALL_SHORT_PUT:
            return "dealer long call / short put (padrao de mercado)"
        return "dealer short call / long put (invertida)"


DEFAULT_CONVENTION = DealerConvention.LONG_CALL_SHORT_PUT

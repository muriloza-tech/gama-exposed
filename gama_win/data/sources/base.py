"""Contrato de fonte de dados -- trocar de fonte deve ser trocar um arquivo.

Toda fonte responde duas perguntas:

    disponivel() -> (bool, motivo)
        Pode ser consultada AGORA? Se nao, por que? Isto alimenta o
        `doctor` e evita a falha silenciosa em que o painel abre com dados
        de outro dia sem ninguem perceber.

    buscar(as_of) -> OptionChain
        Devolve serie JA validada. A fonte e responsavel por normalizar
        para o schema; nenhuma camada a jusante adivinha formato.
"""

from __future__ import annotations

from datetime import date
from typing import Protocol, runtime_checkable

from ..schema import OptionChain


class SourceUnavailableError(RuntimeError):
    """A fonte nao pode ser consultada. A mensagem diz o que fazer."""


@runtime_checkable
class ChainSource(Protocol):
    """Fonte de series de opcoes."""

    nome: str

    def disponivel(self) -> tuple[bool, str]:
        """(pode_consultar, motivo_legivel)."""
        ...

    def buscar(self, as_of: date) -> OptionChain:
        """Serie validada para a data. Levanta SourceUnavailableError se nao
        houver dados -- NUNCA devolve dados de outra data como substituto."""
        ...


_REGISTRO: dict[str, ChainSource] = {}


def registrar(fonte: ChainSource) -> ChainSource:
    """Registra uma fonte pelo nome, para selecao por configuracao/CLI."""
    if fonte.nome in _REGISTRO:
        raise ValueError(f"fonte '{fonte.nome}' ja registrada")
    _REGISTRO[fonte.nome] = fonte
    return fonte


def obter(nome: str) -> ChainSource:
    if nome not in _REGISTRO:
        disponiveis = sorted(_REGISTRO)
        raise KeyError(
            f"fonte '{nome}' desconhecida. Registradas: {disponiveis}"
        )
    return _REGISTRO[nome]


def listar() -> dict[str, ChainSource]:
    return dict(_REGISTRO)

"""Parser dos arquivos publicos da B3.

A fixture `b3_posicoes_amostra.csv` contem LINHAS REAIS do
DerivativesOpenPositionFile de 2026-09-02 (BOVA, PETR, um future do WIN),
mais duas linhas construidas para exercitar os caminhos de erro. Testar
contra o layout real e o que impede uma mudanca silenciosa da B3 de passar
como numero valido.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path

import pandas as pd
import pytest

from gama_win.data.sources.b3_arquivos import (
    BASE_URL,
    CONTRACT_SIZE_OPCAO_ACAO,
    FILE_INSTRUMENTOS,
    FILE_POSICOES,
    B3Error,
    baixar_arquivo,
    carregar_posicoes_abertas,
    resumo_posicoes,
    url_download,
    url_requestname,
)

FIXTURE = Path(__file__).parent / "fixtures" / "b3_posicoes_amostra.csv"
DATA = date(2026, 9, 2)


# ------------------------------------------------------------------ URLs ---


def test_url_requestname():
    u = url_requestname(FILE_POSICOES, DATA)
    assert u == (
        f"{BASE_URL}/api/download/requestname"
        f"?fileName=DerivativesOpenPositionFile&date=2026-09-02"
    )


def test_url_download_usa_rota_de_api_nao_da_spa():
    """REGRESSAO: /download?token= devolve o HTML da SPA com challenge do
    Cloudflare. So /api/download?token= devolve o arquivo."""
    u = url_download("abc123")
    assert u == f"{BASE_URL}/api/download?token=abc123"
    assert "/api/download?" in u


# -------------------------------------------------------------- download ---


def test_baixar_usa_dois_passos_e_grava(tmp_path):
    chamadas: list[str] = []

    def falso_obter(url: str) -> bytes:
        chamadas.append(url)
        if "requestname" in url:
            return b'{"redirectUrl":"~/download?token=TK","token":"TK"}'
        return b"RptDt;TckrSymb\n" + b"x" * 2000

    destino = tmp_path / "arq.csv"
    r = baixar_arquivo(FILE_POSICOES, DATA, destino, obter=falso_obter)

    assert r == destino
    assert destino.exists() and destino.stat().st_size > 1024
    assert len(chamadas) == 2
    assert "requestname" in chamadas[0]
    assert chamadas[1] == url_download("TK")


def test_baixar_reusa_cache_sem_rede(tmp_path):
    destino = tmp_path / "arq.csv"
    destino.write_bytes(b"y" * 5000)

    def nao_deve_ser_chamado(url: str) -> bytes:
        raise AssertionError("cache deveria ter evitado a rede")

    r = baixar_arquivo(FILE_POSICOES, DATA, destino, obter=nao_deve_ser_chamado)
    assert r == destino


def test_baixar_ignora_cache_pequeno_demais(tmp_path):
    destino = tmp_path / "arq.csv"
    destino.write_bytes(b"erro")  # 4 bytes: pagina de erro antiga

    def falso_obter(url: str) -> bytes:
        if "requestname" in url:
            return b'{"token":"TK"}'
        return b"z" * 3000

    baixar_arquivo(FILE_POSICOES, DATA, destino, obter=falso_obter)
    assert destino.stat().st_size == 3000


def test_baixar_sem_token_levanta_com_diagnostico(tmp_path):
    def falso_obter(url: str) -> bytes:
        return b'{"title":"Bad Request","status":400}'

    with pytest.raises(B3Error, match="nao devolveu token"):
        baixar_arquivo(
            "NomeInvalido", DATA, tmp_path / "x.csv", obter=falso_obter
        )


def test_baixar_resposta_nao_json_levanta(tmp_path):
    def falso_obter(url: str) -> bytes:
        return b"<html>manutencao</html>"

    with pytest.raises(B3Error, match="nao e JSON"):
        baixar_arquivo(FILE_POSICOES, DATA, tmp_path / "x.csv", obter=falso_obter)


def test_baixar_html_da_spa_nao_e_gravado(tmp_path):
    """Modo de falha silenciosa: gravar a SPA como se fosse dado."""
    def falso_obter(url: str) -> bytes:
        if "requestname" in url:
            return b'{"token":"TK"}'
        return b"<!DOCTYPE html>\n<html>" + b" " * 3000

    destino = tmp_path / "x.csv"
    with pytest.raises(B3Error, match="veio como HTML"):
        baixar_arquivo(FILE_POSICOES, DATA, destino, obter=falso_obter)
    assert not destino.exists(), "nada deve ser gravado quando o download falha"


def test_baixar_arquivo_curto_demais_nao_e_gravado(tmp_path):
    def falso_obter(url: str) -> bytes:
        if "requestname" in url:
            return b'{"token":"TK"}'
        return b"vazio"

    destino = tmp_path / "x.csv"
    with pytest.raises(B3Error, match="apenas 5 bytes"):
        baixar_arquivo(FILE_POSICOES, DATA, destino, obter=falso_obter)
    assert not destino.exists()


# ---------------------------------------------------------------- parser ---


def test_fixture_existe():
    assert FIXTURE.exists(), "fixture com linhas reais da B3 e obrigatoria"


def test_carrega_apenas_opcoes():
    df = carregar_posicoes_abertas(FIXTURE, validar_soma=False)
    assert set(df["kind"]) <= {"C", "P"}
    # EQUITY FORWARD (BOVA11T) e FINANCIAL (WINV26) devem ficar de fora
    assert "BOVA11T" not in set(df["ticker"])
    assert "WINV26" not in set(df["ticker"])


def test_open_interest_vem_de_ttlpos_nao_de_opnintrst():
    """A ARMADILHA. OpnIntrst esta vazio nas opcoes; ler o nome do campo
    daria zero em tudo."""
    df = carregar_posicoes_abertas(FIXTURE, ativo="BOVA", validar_soma=False)
    linha = df[df["ticker"] == "BOVAI176W4"].iloc[0]
    assert linha["open_interest"] == 11_000_100

    bruto = pd.read_csv(FIXTURE, sep=";", dtype=str, keep_default_na=False)
    crua = bruto[bruto["TckrSymb"] == "BOVAI176W4"].iloc[0]
    assert crua["OpnIntrst"] == "", "OpnIntrst e vazio nas opcoes de acao"
    assert crua["TtlPos"] == "11000100"


def test_kind_derivado_do_segmento():
    df = carregar_posicoes_abertas(FIXTURE, ativo="BOVA", validar_soma=False)
    assert df[df["ticker"] == "BOVAI176W4"]["kind"].iloc[0] == "C"
    assert df[df["ticker"] == "BOVAU176W4"]["kind"].iloc[0] == "P"


def test_series_com_oi_zero_sao_removidas():
    df = carregar_posicoes_abertas(FIXTURE, validar_soma=False)
    assert "ZERO0000" not in set(df["ticker"])
    assert (df["open_interest"] > 0).all()


def test_filtro_por_ativo():
    df = carregar_posicoes_abertas(FIXTURE, ativo="bova", validar_soma=False)
    assert set(df["asset"]) == {"BOVA"}
    assert len(df) == 6  # 3 calls + 3 puts com OI > 0 na fixture
    assert int((df["kind"] == "C").sum()) == 3
    assert int((df["kind"] == "P").sum()) == 3


def test_ativo_inexistente_lista_alternativas():
    with pytest.raises(B3Error, match="Ativos com mais series"):
        carregar_posicoes_abertas(FIXTURE, ativo="NAOEXISTE", validar_soma=False)


def test_validacao_de_soma_pega_layout_alterado():
    """A linha BADSUM01 tem 10+20+30=60 mas TtlPos=999."""
    with pytest.raises(B3Error, match="TtlPos"):
        carregar_posicoes_abertas(FIXTURE, validar_soma=True)


def test_validacao_de_soma_passa_no_ativo_real():
    """BOVA e PETR da fixture (dados reais) batem a soma."""
    for ativo in ("BOVA", "PETR"):
        df = carregar_posicoes_abertas(FIXTURE, ativo=ativo, validar_soma=True)
        assert len(df) > 0


def test_report_date_e_data_real():
    df = carregar_posicoes_abertas(FIXTURE, ativo="BOVA", validar_soma=False)
    assert set(df["report_date"]) == {DATA}


def test_colunas_de_saida():
    df = carregar_posicoes_abertas(FIXTURE, ativo="BOVA", validar_soma=False)
    assert list(df.columns) == [
        "report_date",
        "ticker",
        "isin",
        "asset",
        "kind",
        "open_interest",
        "distribution_id",
    ]
    assert df["open_interest"].dtype == "int64"


def test_arquivo_inexistente_levanta():
    with pytest.raises(B3Error, match="nao encontrado"):
        carregar_posicoes_abertas("/caminho/que/nao/existe.csv")


def test_layout_sem_coluna_obrigatoria_levanta(tmp_path):
    ruim = tmp_path / "ruim.csv"
    ruim.write_text("RptDt;TckrSymb;SgmtNm\n2026-09-02;X;EQUITY CALL\n")
    with pytest.raises(B3Error, match="colunas ausentes"):
        carregar_posicoes_abertas(ruim)


def test_resumo_legivel():
    df = carregar_posicoes_abertas(FIXTURE, ativo="BOVA", validar_soma=False)
    r = resumo_posicoes(df)
    assert "series" in r and "calls" in r and "puts" in r
    assert "2026-09-02" in r


def test_resumo_nao_destroi_virgulas_do_texto():
    """REGRESSAO: o replace de separador de milhar aplicado na frase inteira
    transformava 'calls, puts' em 'calls. puts'."""
    df = carregar_posicoes_abertas(FIXTURE, ativo="BOVA", validar_soma=False)
    r = resumo_posicoes(df)
    assert "calls, " in r, f"virgula da frase foi perdida: {r}"


def test_resumo_formata_milhar_com_ponto():
    df = carregar_posicoes_abertas(FIXTURE, ativo="BOVA", validar_soma=False)
    r = resumo_posicoes(df)
    # OI total da fixture BOVA passa de 20 milhoes
    assert "OI total: 2" in r
    trecho = r.split("OI total: ")[1].split(" |")[0]
    assert "." in trecho and "," not in trecho


def test_resumo_de_dataframe_vazio():
    assert "nenhuma serie" in resumo_posicoes(pd.DataFrame())


# ------------------------------------------- arquivo real, se disponivel ---

REAL = (
    Path(__file__).parents[1]
    / "data"
    / "cache"
    / "DerivativesOpenPosition_2026-09-02.bin"
)


@pytest.mark.skipif(not REAL.exists(), reason="arquivo real nao baixado")
def test_arquivo_real_da_b3_bate_a_soma_em_todas_as_series():
    """Valida a premissa central contra o arquivo inteiro (~51 mil linhas)."""
    df = carregar_posicoes_abertas(REAL, validar_soma=True)
    assert len(df) > 20_000
    assert (df["open_interest"] > 0).all()
    assert set(df["kind"]) == {"C", "P"}


@pytest.mark.skipif(not REAL.exists(), reason="arquivo real nao baixado")
def test_bova_no_arquivo_real_tem_muitas_series():
    df = carregar_posicoes_abertas(REAL, ativo="BOVA", validar_soma=True)
    assert len(df) > 1_000, "BOVA deve ter mais de mil series com OI > 0"
    assert df["open_interest"].max() > 1_000_000


# --------------------------------------------------- tamanho do contrato ---


def test_contract_size_e_uma_cota():
    """Na B3, 1 contrato de opcao sobre ETF = 1 cota. O 'x 100' e a convencao
    americana. Errar isto multiplica toda a exposicao em reais por 100 -- sem
    mudar strike, virada de sinal nem paredes, o que torna o erro dificil de
    perceber olhando o grafico."""
    assert CONTRACT_SIZE_OPCAO_ACAO == 1.0


def test_contract_size_e_o_default_de_montar_chain():
    """A constante so protege se for de fato o default usado."""
    import inspect

    from gama_win.data.sources.b3_arquivos import montar_chain

    padrao = inspect.signature(montar_chain).parameters["contract_size"].default
    assert padrao == CONTRACT_SIZE_OPCAO_ACAO == 1.0

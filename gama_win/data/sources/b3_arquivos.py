"""Arquivos publicos da B3 (portal UP2DATA / arquivos.b3.com.br).

Download em DOIS passos, descoberto na tentativa: a URL que o site usa
(`/download?token=`) devolve o shell JavaScript da SPA com challenge do
Cloudflare, nao o arquivo. O arquivo sai em `/api/download?token=`.

    GET /api/download/requestname?fileName=<nome>&date=<AAAA-MM-DD>
        -> {"redirectUrl": "~/download?token=...", "token": "..."}
    GET /api/download?token=<token>
        -> o arquivo (CSV com ';', decimal com virgula)

ARMADILHA DOCUMENTADA: no arquivo de posicoes em aberto, a coluna chamada
`OpnIntrst` esta VAZIA para os segmentos EQUITY CALL e EQUITY PUT -- ela so
e preenchida em FINANCIAL, AGRIBUSINESS e EQUITY FORWARD. O interesse em
aberto das opcoes de acao/ETF esta em `TtlPos`, que e a soma de `CvrdQty`
(coberta), `TtlBlckdPos` (travada) e `UcvrdQty` (descoberta). Confirmado nas
2.101 series de BOVA de 2026-09-02: soma bate em 100% das linhas.

Quem ler `OpnIntrst` por causa do nome vai obter zero em todas as opcoes.

SEGUNDA ARMADILHA: o numero no ticker NAO e o strike. Os codigos sao
atribuidos na listagem e nao acompanham ajustes por proventos -- BOVAJ210
tinha 8,1 milhoes de contratos abertos com o ETF em ~176. Strike e
vencimento reais so vem do InstrumentsConsolidatedFile, por TckrSymb.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

BASE_URL = "https://arquivos.b3.com.br"
USER_AGENT = "Mozilla/5.0 (compatible; gama-win/0.1)"

FILE_POSICOES = "DerivativesOpenPositionFile"
FILE_INSTRUMENTOS = "InstrumentsConsolidatedFile"
FILE_NEGOCIOS = "TradeInformationConsolidatedFile"

# Segmentos que sao opcoes de acao/ETF -- os que nos interessam e os unicos
# em que TtlPos (nao OpnIntrst) carrega o interesse em aberto.
SEGMENTO_CALL = "EQUITY CALL"
SEGMENTO_PUT = "EQUITY PUT"
SEGMENTOS_OPCAO = (SEGMENTO_CALL, SEGMENTO_PUT)

_KIND_POR_SEGMENTO = {SEGMENTO_CALL: "C", SEGMENTO_PUT: "P"}

COLUNAS_POSICOES = (
    "RptDt",
    "TckrSymb",
    "ISIN",
    "Asst",
    "XprtnCd",
    "SgmtNm",
    "OpnIntrst",
    "VartnOpnIntrst",
    "DstrbtnId",
    "CvrdQty",
    "TtlBlckdPos",
    "UcvrdQty",
    "TtlPos",
    "BrrwrQty",
    "LndrQty",
    "CurQty",
    "FwdPric",
)


class B3Error(RuntimeError):
    """Falha ao obter ou interpretar um arquivo publico da B3."""


# ------------------------------------------------------------- download ---


_trust_store_pronto = False


def _garantir_trust_store() -> None:
    """Faz o Python usar a loja de certificados do SISTEMA.

    Por que: em maquinas com antivirus ou proxy corporativo que inspeciona
    TLS, o certificado apresentado e assinado por uma raiz propria que esta
    instalada no Windows mas NAO no bundle do certifi que o `requests` usa.
    O sintoma e CERTIFICATE_VERIFY_FAILED apenas no Python, enquanto curl e
    o navegador funcionam.

    `truststore` resolve usando a loja do sistema. A alternativa preguicosa
    -- verify=False -- desligaria a verificacao de certificado e abriria a
    porta para man-in-the-middle. Nao e uma opcao aqui.

    Se o pacote nao estiver instalado, seguimos com o comportamento padrao:
    em maquina sem interceptacao funciona igual.
    """
    global _trust_store_pronto
    if _trust_store_pronto:
        return
    try:
        import truststore

        truststore.inject_into_ssl()
    except ImportError:
        pass
    _trust_store_pronto = True


def _obter_http(url: str, *, timeout: int = 180) -> bytes:
    """Busca uma URL. Isolado para permitir injecao nos testes."""
    import requests  # import local: o parser funciona sem rede

    _garantir_trust_store()

    try:
        resp = requests.get(
            url,
            headers={"User-Agent": USER_AGENT, "Referer": f"{BASE_URL}/"},
            timeout=timeout,
        )
    except requests.exceptions.SSLError as exc:
        raise B3Error(
            "falha de verificacao de certificado TLS ao acessar a B3.\n"
            "Causa tipica: antivirus ou proxy corporativo inspecionando "
            "HTTPS com uma raiz propria, que existe na loja do Windows mas "
            "nao no bundle do certifi.\n"
            "Correcao: instale o pacote `truststore` (pip install truststore) "
            "para o Python usar a loja de certificados do sistema.\n"
            "NAO use verify=False: isso desliga a verificacao e permite "
            f"man-in-the-middle.\nErro original: {exc}"
        ) from exc
    except requests.exceptions.RequestException as exc:
        raise B3Error(f"falha de rede ao acessar {url}: {exc}") from exc

    if resp.status_code != 200:
        raise B3Error(f"HTTP {resp.status_code} em {url}")
    return resp.content


def url_requestname(file_name: str, data: date) -> str:
    return (
        f"{BASE_URL}/api/download/requestname"
        f"?fileName={file_name}&date={data.isoformat()}"
    )


def url_download(token: str) -> str:
    """A rota de API, nao a da SPA. Ver docstring do modulo."""
    return f"{BASE_URL}/api/download?token={token}"


def baixar_arquivo(
    file_name: str,
    data: date,
    destino: Path | str,
    *,
    obter: Callable[[str], bytes] | None = None,
    reusar_cache: bool = True,
    minimo_bytes: int = 1024,
) -> Path:
    """Baixa um arquivo publico da B3 para `destino`.

    `reusar_cache=True` (default) devolve o arquivo local se ele ja existir
    com tamanho plausivel, sem tocar na rede: os arquivos da B3 sao
    imutaveis por data, entao rebaixar e desperdicio.

    `minimo_bytes` protege contra o modo de falha silenciosa mais comum:
    receber a pagina de erro HTML (~2 KB) e grava-la como se fosse dado.
    """
    obter = obter or _obter_http
    destino = Path(destino)

    if reusar_cache and destino.exists() and destino.stat().st_size >= minimo_bytes:
        return destino

    bruto = obter(url_requestname(file_name, data))
    try:
        payload = json.loads(bruto.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise B3Error(
            f"resposta de requestname para {file_name} @ {data} nao e JSON: "
            f"{bruto[:200]!r}"
        ) from exc

    token = payload.get("token")
    if not token:
        raise B3Error(
            f"B3 nao devolveu token para {file_name} @ {data.isoformat()}. "
            f"Resposta: {payload}. Causas comuns: nome de arquivo invalido, "
            "data sem pregao, ou arquivo ainda nao publicado (os dados de "
            "D saem depois do fechamento)."
        )

    conteudo = obter(url_download(token))

    if len(conteudo) < minimo_bytes:
        raise B3Error(
            f"arquivo {file_name} @ {data.isoformat()} veio com apenas "
            f"{len(conteudo)} bytes -- provavelmente pagina de erro em vez de "
            "dado. Nada foi gravado."
        )
    if conteudo.lstrip()[:15].lower().startswith(b"<!doctype html"):
        raise B3Error(
            f"arquivo {file_name} @ {data.isoformat()} veio como HTML. Use a "
            "rota /api/download?token=, nao /download?token=. Nada gravado."
        )

    destino.parent.mkdir(parents=True, exist_ok=True)
    destino.write_bytes(conteudo)
    return destino


# --------------------------------------------------------------- parsers ---


def _achar_linha_do_cabecalho(caminho: Path, limite: int = 5) -> int:
    """Indice da linha que contem o cabecalho real.

    Os arquivos da B3 nao sao uniformes: o DerivativesOpenPositionFile
    comeca direto no cabecalho, enquanto o InstrumentsConsolidatedFile tem
    uma linha de preambulo ("Status do Arquivo: Final") antes dele. Detectar
    e mais robusto que cravar skiprows por arquivo.
    """
    with caminho.open("r", encoding="latin-1", newline="") as fh:
        for i, linha in enumerate(fh):
            if i >= limite:
                break
            if linha.startswith("RptDt;"):
                return i
    raise B3Error(
        f"{caminho}: cabecalho iniciado por 'RptDt;' nao encontrado nas "
        f"primeiras {limite} linhas. Layout da B3 mudou?"
    )


def _ler_csv_b3(caminho: Path | str) -> pd.DataFrame:
    """CSV da B3: separador ';', decimal ',', acentos em latin-1."""
    caminho = Path(caminho)
    if not caminho.exists():
        raise B3Error(f"arquivo nao encontrado: {caminho}")
    pular = _achar_linha_do_cabecalho(caminho)
    try:
        return pd.read_csv(
            caminho,
            sep=";",
            decimal=",",
            dtype=str,
            keep_default_na=False,
            encoding="latin-1",
            skiprows=pular,
            low_memory=False,
        )
    except Exception as exc:  # pragma: no cover - depende do arquivo
        raise B3Error(f"falha ao ler {caminho}: {exc}") from exc


def _para_int(serie: pd.Series) -> pd.Series:
    limpo = serie.astype(str).str.strip().replace({"": "0", "nan": "0"})
    return pd.to_numeric(limpo, errors="coerce").fillna(0).astype("int64")


def carregar_posicoes_abertas(
    caminho: Path | str,
    *,
    ativo: str | None = None,
    validar_soma: bool = True,
) -> pd.DataFrame:
    """Le o DerivativesOpenPositionFile e devolve as OPCOES normalizadas.

    Colunas de saida: report_date, ticker, isin, asset, kind,
    open_interest, distribution_id.

    `open_interest` vem de TtlPos -- ver a armadilha na docstring do modulo.

    Com `validar_soma=True` (default), confere que
    CvrdQty + TtlBlckdPos + UcvrdQty == TtlPos e levanta se divergir: uma
    divergencia significa que a B3 mudou o layout e o campo passou a
    significar outra coisa. Falhar aqui e melhor que calcular gama errado.
    """
    df = _ler_csv_b3(caminho)

    faltando = [c for c in COLUNAS_POSICOES if c not in df.columns]
    if faltando:
        raise B3Error(
            f"{caminho}: colunas ausentes {faltando}. Layout da B3 mudou? "
            f"Colunas encontradas: {list(df.columns)}"
        )

    df["SgmtNm"] = df["SgmtNm"].str.strip().str.upper()
    opcoes = df[df["SgmtNm"].isin(SEGMENTOS_OPCAO)].copy()

    if ativo is not None:
        alvo = ativo.strip().upper()
        opcoes = opcoes[opcoes["Asst"].str.strip().str.upper() == alvo].copy()
        if opcoes.empty:
            disponiveis = (
                df.loc[df["SgmtNm"].isin(SEGMENTOS_OPCAO), "Asst"]
                .str.strip()
                .str.upper()
                .value_counts()
                .head(10)
                .index.tolist()
            )
            raise B3Error(
                f"nenhuma opcao do ativo '{alvo}' em {caminho}. "
                f"Ativos com mais series: {disponiveis}"
            )

    if opcoes.empty:
        raise B3Error(f"{caminho}: nenhuma linha de EQUITY CALL/PUT")

    for col in ("CvrdQty", "TtlBlckdPos", "UcvrdQty", "TtlPos"):
        opcoes[col] = _para_int(opcoes[col])

    if validar_soma:
        soma = (
            opcoes["CvrdQty"] + opcoes["TtlBlckdPos"] + opcoes["UcvrdQty"]
        )
        divergentes = opcoes[soma != opcoes["TtlPos"]]
        if len(divergentes):
            exemplos = divergentes[
                ["TckrSymb", "CvrdQty", "TtlBlckdPos", "UcvrdQty", "TtlPos"]
            ].head(3)
            raise B3Error(
                f"{caminho}: {len(divergentes)} linha(s) onde "
                "CvrdQty+TtlBlckdPos+UcvrdQty != TtlPos. O significado de "
                "TtlPos pode ter mudado -- NAO use este arquivo sem revisar "
                f"o layout.\n{exemplos.to_string(index=False)}"
            )

    saida = pd.DataFrame(
        {
            "report_date": pd.to_datetime(opcoes["RptDt"]).dt.date,
            "ticker": opcoes["TckrSymb"].str.strip(),
            "isin": opcoes["ISIN"].str.strip(),
            "asset": opcoes["Asst"].str.strip().str.upper(),
            "kind": opcoes["SgmtNm"].map(_KIND_POR_SEGMENTO),
            "open_interest": opcoes["TtlPos"],
            "distribution_id": opcoes["DstrbtnId"].str.strip(),
        }
    ).reset_index(drop=True)

    # Series com posicao zerada nao contribuem gama e so poluem a grade.
    return saida[saida["open_interest"] > 0].reset_index(drop=True)


CATEGORIA_OPCAO_ACAO = "OPTION ON EQUITIES"

_KIND_POR_OPTNTP = {"CALL": "C", "PUT": "P"}

# Tamanho do contrato das opcoes de acao/ETF na B3.
#
# CONFIRMADO (03/09/2026): na B3, 1 contrato de opcao sobre BOVA11 equivale a
# 1 COTA de BOVA11. O premio e cotado por cota e o interesse em aberto e
# contado em contratos de uma cota. O "x 100" que aparece em muito material
# e a convencao AMERICANA (1 contrato = 100 acoes) -- nao vale aqui.
#
# O arquivo da B3 nao ajuda a descobrir isso: nos 120.683 registros de
# OPTION ON EQUITIES, `CtrctMltplr` e `AsstQtnQty` estao vazios, e
# `AllcnRndLot` (1, 10 ou 100) e o lote de NEGOCIACAO -- quantas opcoes se
# compra de uma vez --, nao o tamanho do contrato. Confundir os dois e o
# caminho natural para o erro.
#
# Teste de plausibilidade que apontou na mesma direcao antes da confirmacao:
# a maior serie de BOVA11 em 02/09 tinha 11.007.308 contratos abertos. A 1
# cota por contrato, sao 11 milhoes de cotas (R$ 2,0 bi, ~2 dias de volume do
# ETF). A 100, seriam 1,1 bilhao de cotas (R$ 200 bi, 217 dias de volume
# integral do fundo numa unica serie) -- implausivel.
#
# Errar isto multiplica toda a exposicao em reais por 100. Nao muda strikes,
# virada de sinal nem paredes: e constante em todos os strikes.
CONTRACT_SIZE_OPCAO_ACAO = 1.0


def carregar_instrumentos(
    caminho: Path | str, *, ativo: str | None = None
) -> pd.DataFrame:
    """Le o InstrumentsConsolidatedFile e devolve o cadastro das OPCOES.

    E aqui que estao strike e vencimento reais. Nao tente deduzi-los do
    ticker: os codigos sao atribuidos na listagem e nao acompanham ajustes
    por proventos.

    Colunas de saida: ticker, asset, kind, strike, expiry, style,
    trading_end, distribution_id, round_lot.
    """
    df = _ler_csv_b3(caminho)

    obrigatorias = (
        "TckrSymb",
        "ISIN",
        "Asst",
        "SctyCtgyNm",
        "XprtnDt",
        "OptnTp",
        "ExrcPric",
    )
    faltando = [c for c in obrigatorias if c not in df.columns]
    if faltando:
        raise B3Error(
            f"{caminho}: colunas ausentes {faltando}. Layout da B3 mudou?"
        )

    df["SctyCtgyNm"] = df["SctyCtgyNm"].str.strip().str.upper()
    opc = df[df["SctyCtgyNm"] == CATEGORIA_OPCAO_ACAO].copy()

    if ativo is not None:
        alvo = ativo.strip().upper()
        opc = opc[opc["Asst"].str.strip().str.upper() == alvo].copy()
        if opc.empty:
            candidatos = (
                df.loc[df["SctyCtgyNm"] == CATEGORIA_OPCAO_ACAO, "Asst"]
                .str.strip()
                .str.upper()
                .value_counts()
                .head(10)
                .index.tolist()
            )
            raise B3Error(
                f"nenhuma opcao do ativo '{alvo}' no cadastro. Note que este "
                "arquivo usa o ticker completo do subjacente (ex.: 'BOVA11'), "
                "diferente do arquivo de posicoes, que usa a raiz de 4 letras "
                f"('BOVA'). Ativos com mais series: {candidatos}"
            )

    if opc.empty:
        raise B3Error(f"{caminho}: nenhuma linha de {CATEGORIA_OPCAO_ACAO}")

    kind = opc["OptnTp"].str.strip().str.upper().map(_KIND_POR_OPTNTP)
    strike = pd.to_numeric(
        opc["ExrcPric"].str.strip().str.replace(",", ".", regex=False),
        errors="coerce",
    )
    expiry = pd.to_datetime(opc["XprtnDt"].str.strip(), errors="coerce")

    saida = pd.DataFrame(
        {
            "ticker": opc["TckrSymb"].str.strip(),
            "isin": opc["ISIN"].str.strip(),
            "asset": opc["Asst"].str.strip().str.upper(),
            "kind": kind,
            "strike": strike,
            "expiry": expiry,
            "style": opc["OptnStyle"].str.strip()
            if "OptnStyle" in opc.columns
            else "",
            "trading_end": pd.to_datetime(
                opc["TradgEndDt"].str.strip(), errors="coerce"
            )
            if "TradgEndDt" in opc.columns
            else pd.NaT,
            "distribution_id": opc["DstrbtnId"].str.strip()
            if "DstrbtnId" in opc.columns
            else "",
            "round_lot": opc["AllcnRndLot"].str.strip()
            if "AllcnRndLot" in opc.columns
            else "",
        }
    )

    ruins = saida[saida["kind"].isna() | saida["strike"].isna() | saida["expiry"].isna()]
    if len(ruins):
        raise B3Error(
            f"{caminho}: {len(ruins)} serie(s) com kind/strike/expiry "
            f"invalidos. Exemplos:\n{ruins.head(3).to_string(index=False)}"
        )
    if (saida["strike"] <= 0).any():
        n = int((saida["strike"] <= 0).sum())
        raise B3Error(f"{caminho}: {n} serie(s) com strike <= 0")

    return saida.reset_index(drop=True)


def resumo_posicoes(df: pd.DataFrame) -> str:
    """Resumo legivel de um DataFrame de posicoes, para o doctor/CLI."""
    if df.empty:
        return "nenhuma serie com posicao aberta"
    datas = sorted({d.isoformat() for d in df["report_date"]})
    calls = int((df["kind"] == "C").sum())
    puts = int((df["kind"] == "P").sum())
    oi = int(df["open_interest"].sum())
    ativos = df["asset"].nunique()
    # O replace de milhar tem de ficar restrito ao numero: aplicado na frase
    # inteira ele destroi as virgulas do texto.
    oi_fmt = f"{oi:,}".replace(",", ".")
    return (
        f"{len(df)} series ({calls} calls, {puts} puts) em {ativos} ativo(s) | "
        f"OI total: {oi_fmt} | data: {', '.join(datas)}"
    )


def montar_chain(
    caminho_posicoes: Path | str,
    caminho_instrumentos: Path | str,
    *,
    ativo: str,
    rate: float,
    div_yield: float,
    spot: float | None = None,
    caminho_negocios: Path | str | None = None,
    contract_size: float = CONTRACT_SIZE_OPCAO_ACAO,
    as_of: date | None = None,
    expiries: tuple[date, ...] | None = None,
    descartar_vencidos: bool = True,
):
    """Junta posicoes + cadastro (+ precos) e devolve um OptionChain validado.

    `ativo` usa o ticker completo do subjacente ('BOVA11'). O join e feito
    por TckrSymb, que e inequivoco, entao a diferenca de nomenclatura entre
    os dois arquivos nao importa.

    Series com o mesmo (vencimento, strike, tipo) e DstrbtnId diferente sao
    SOMADAS: economicamente sao a mesma serie, e o schema exige unicidade. O
    preco dessas series e a media ponderada pelo interesse em aberto.

    Com `caminho_negocios` informado:
      - `spot` sai do fechamento oficial do subjacente, se nao for passado
      - a volatilidade implicita e resolvida serie por serie a partir do
        RefPric da B3, ficando NaN onde o preco nao determina uma vol

    Sem ele, `spot` e obrigatorio e a vol sai toda NaN -- e `construir_perfil`
    vai exigir `vol_fallback` explicito, por desenho.
    """
    from ..schema import OptionChain

    pos = carregar_posicoes_abertas(caminho_posicoes)
    inst = carregar_instrumentos(caminho_instrumentos, ativo=ativo)

    if as_of is None:
        datas = sorted({d for d in pos["report_date"]})
        if len(datas) != 1:
            raise B3Error(
                f"arquivo de posicoes cobre varias datas ({datas}); informe "
                "as_of explicitamente"
            )
        as_of = datas[0]

    if spot is None:
        if caminho_negocios is None:
            raise B3Error(
                "informe `spot` ou `caminho_negocios`. Sem um dos dois nao ha "
                "preco do subjacente, e estimar o spot por outra via foi o que "
                "produziu um erro de 3% num painel anterior."
            )
        spot = preco_de_fechamento(caminho_negocios, ativo)

    juntos = pos.merge(
        inst[["ticker", "kind", "strike", "expiry"]],
        on="ticker",
        how="inner",
        suffixes=("_pos", "_inst"),
    )
    if juntos.empty:
        raise B3Error(
            f"nenhuma serie de '{ativo}' com posicao aberta casou com o "
            "cadastro. Verifique se os dois arquivos sao da MESMA data."
        )

    divergentes = juntos[juntos["kind_pos"] != juntos["kind_inst"]]
    if len(divergentes):
        raise B3Error(
            f"{len(divergentes)} serie(s) com tipo divergente entre posicoes "
            f"(SgmtNm) e cadastro (OptnTp). Exemplos:\n"
            f"{divergentes[['ticker', 'kind_pos', 'kind_inst']].head(3).to_string(index=False)}"
        )

    juntos = juntos.rename(columns={"kind_pos": "kind"}).drop(columns=["kind_inst"])
    juntos["expiry_date"] = juntos["expiry"].dt.date

    notas: list[str] = []

    if descartar_vencidos:
        vencidas = juntos["expiry_date"] < as_of
        if vencidas.any():
            notas.append(
                f"{int(vencidas.sum())} serie(s) vencida(s) antes de "
                f"{as_of.isoformat()} descartada(s)"
            )
            juntos = juntos[~vencidas]

    if expiries is not None:
        alvo = set(expiries)
        disponiveis = sorted({d for d in juntos["expiry_date"]})
        juntos = juntos[juntos["expiry_date"].isin(alvo)]
        if juntos.empty:
            raise B3Error(
                f"nenhuma serie nos vencimentos {sorted(d.isoformat() for d in alvo)}. "
                f"Disponiveis: {[d.isoformat() for d in disponiveis]}"
            )

    # ------------------------------------------------ precos, se houver
    if caminho_negocios is not None:
        neg = carregar_negocios(caminho_negocios)
        preco = neg["referencia"].where(neg["referencia"].notna(), neg["ultimo"])
        juntos = juntos.merge(
            pd.DataFrame({"ticker": neg["ticker"], "preco": preco}),
            on="ticker",
            how="left",
        )
    else:
        juntos["preco"] = float("nan")

    juntos["preco_x_oi"] = juntos["preco"] * juntos["open_interest"]

    antes = len(juntos)
    agregado = juntos.groupby(
        ["expiry_date", "strike", "kind"], as_index=False
    ).agg(
        open_interest=("open_interest", "sum"),
        symbol=("ticker", "first"),
        preco_x_oi=("preco_x_oi", "sum"),
        oi_com_preco=("open_interest", "sum"),
    )
    if antes != len(agregado):
        notas.append(
            f"{antes - len(agregado)} serie(s) somada(s) por coincidirem em "
            "(vencimento, strike, tipo) com DstrbtnId diferente"
        )

    # Media do preco ponderada pelo interesse em aberto.
    agregado["preco"] = agregado["preco_x_oi"] / agregado["oi_com_preco"].replace(
        0, pd.NA
    )

    df = pd.DataFrame(
        {
            "expiry": pd.to_datetime(agregado["expiry_date"]),
            "strike": agregado["strike"].astype(float),
            "kind": agregado["kind"],
            "open_interest": agregado["open_interest"].astype("int64"),
            "contract_size": float(contract_size),
            "implied_vol": float("nan"),
            "settlement_price": pd.to_numeric(agregado["preco"], errors="coerce"),
            "symbol": agregado["symbol"],
        }
    )

    # ------------------------------------ volatilidade implicita real
    if caminho_negocios is not None:
        from ...model.calendario import tau_anos
        from ...model.greeks import implied_vol

        tau = np.array(
            [tau_anos(as_of, d.date()) for d in df["expiry"]], dtype=float
        )
        preco_arr = df["settlement_price"].to_numpy(float)
        com_preco = np.isfinite(preco_arr) & (preco_arr > 0) & (tau > 0)

        vols = np.full(len(df), np.nan)
        if np.any(com_preco):
            vols[com_preco] = implied_vol(
                preco_arr[com_preco],
                float(spot),
                df["strike"].to_numpy(float)[com_preco],
                tau[com_preco],
                float(rate),
                float(div_yield),
                df["kind"].to_numpy(object)[com_preco],
            )
        # O schema aceita NaN, mas rejeita vol fora de (0, 5].
        vols = np.where(np.isfinite(vols) & (vols > 0) & (vols <= 5.0), vols, np.nan)
        df["implied_vol"] = vols

        n_ok = int(np.sum(np.isfinite(vols)))
        notas.append(
            f"volatilidade implicita resolvida em {n_ok}/{len(df)} serie(s) a "
            "partir do RefPric da B3 (preco de referencia da bolsa, nao "
            "necessariamente negociado); o resto fica NaN e depende de "
            "vol_fallback explicito"
        )

    notas.append(
        f"contract_size={contract_size}: 1 contrato de opcao sobre "
        f"{ativo} equivale a 1 cota (confirmado; o 'x 100' e convencao "
        "americana). A B3 nao declara o multiplicador no arquivo."
    )

    return OptionChain(
        df=df,
        underlying=ativo,
        spot=float(spot),
        as_of=as_of,
        source=f"B3/{FILE_POSICOES}+{FILE_INSTRUMENTOS}@{as_of.isoformat()}",
        rate=float(rate),
        div_yield=float(div_yield),
        is_synthetic=False,
        notas=tuple(notas),
    )


# ------------------------------------------------------- precos e vol ---

SEGMENTO_CAIXA = "CASH"


def carregar_negocios(
    caminho: Path | str, *, apenas_opcoes: bool = False
) -> pd.DataFrame:
    """Le o TradeInformationConsolidatedFile.

    Colunas de saida: ticker, segment, minimo, maximo, medio, ultimo,
    referencia, quantidade, volume_financeiro.

    Sobre `referencia` (RefPric): e o preco de referencia calculado pela
    PROPRIA B3, publicado para toda serie -- inclusive as que nao negociaram.
    No arquivo de 2026-09-02, as 3.330 series de BOVA tem RefPric, mas apenas
    1.000 tem LastPric. Sem RefPric a volatilidade implicita existiria em 30%
    da grade, e as asas -- onde ficam as paredes -- ficariam de fora.

    Nao e preco negociado: e a marcacao da bolsa. A distincao esta registrada
    aqui porque muda a interpretacao.
    """
    df = _ler_csv_b3(caminho)

    obrigatorias = ("TckrSymb", "SgmtNm", "LastPric", "RefPric")
    faltando = [c for c in obrigatorias if c not in df.columns]
    if faltando:
        raise B3Error(f"{caminho}: colunas ausentes {faltando}")

    def num(col: str) -> pd.Series:
        return pd.to_numeric(
            df[col].astype(str).str.strip().str.replace(",", ".", regex=False),
            errors="coerce",
        )

    saida = pd.DataFrame(
        {
            "ticker": df["TckrSymb"].str.strip(),
            "segment": df["SgmtNm"].str.strip().str.upper(),
            "minimo": num("MinPric"),
            "maximo": num("MaxPric"),
            "medio": num("TradAvrgPric"),
            "ultimo": num("LastPric"),
            "referencia": num("RefPric"),
            "quantidade": num("TradQty"),
            "volume_financeiro": num("NtlFinVol"),
        }
    )

    if apenas_opcoes:
        saida = saida[saida["segment"].isin(SEGMENTOS_OPCAO)]

    return saida.reset_index(drop=True)


def preco_de_fechamento(
    caminho: Path | str, ticker: str, *, negocios: pd.DataFrame | None = None
) -> float:
    """Fechamento oficial de um instrumento a vista (LastPric).

    Levanta se o ticker nao existir ou nao tiver preco -- devolver um numero
    aproximado silenciosamente foi exatamente o erro que fez o painel abrir
    com spot 3% errado.
    """
    df = carregar_negocios(caminho) if negocios is None else negocios
    alvo = ticker.strip().upper()
    linha = df[df["ticker"].str.upper() == alvo]
    if linha.empty:
        raise B3Error(f"ticker '{alvo}' ausente em {caminho}")

    for col in ("ultimo", "medio", "referencia"):
        v = linha.iloc[0][col]
        if pd.notna(v) and v > 0:
            return float(v)

    raise B3Error(
        f"ticker '{alvo}' existe em {caminho} mas nao tem preco utilizavel "
        "(ultimo, medio e referencia todos vazios)"
    )

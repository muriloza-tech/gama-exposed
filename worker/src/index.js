/**
 * Proxy de cotação para o painel Gama Exposed.
 *
 * Por que existe: a página é estática e roda no navegador, onde a política
 * de CORS bloqueia a chamada direta à API de cotações. Um Worker é servidor
 * — CORS não se aplica à sua chamada de saída —, então ele busca o dado e
 * devolve ao navegador com o cabeçalho que o libera.
 *
 * Não guarda segredo nenhum: a fonte usada não exige token. O Worker existe
 * puramente para atravessar o CORS e para colocar um cache curto na frente
 * da origem, evitando uma chamada por visitante a cada 30 segundos.
 *
 * Contrato de saída (sempre este formato, mesmo em erro):
 *   { ok: bool, ts: <epoch ms>, cotacoes: { BOVA11: n|null, IBOV: n|null },
 *     erro: string|null }
 */

const FONTES = {
  IBOV: "https://query1.finance.yahoo.com/v8/finance/chart/%5EBVSP?interval=1m&range=1d",
  BOVA11: "https://query1.finance.yahoo.com/v8/finance/chart/BOVA11.SA?interval=1m&range=1d",
};

// Cache curto: o dado da fonte gratuita tem atraso proprio, entao 20s de
// cache nao piora a informacao e derruba muito a carga na origem.
const CACHE_SEGUNDOS = 20;

const ORIGENS_PERMITIDAS = [
  "https://muriloza-tech.github.io",
  "http://localhost:8000",
  "http://127.0.0.1:8000",
];

function cabecalhosCors(origem) {
  const permitida = ORIGENS_PERMITIDAS.includes(origem) ? origem : ORIGENS_PERMITIDAS[0];
  return {
    "Access-Control-Allow-Origin": permitida,
    "Access-Control-Allow-Methods": "GET, OPTIONS",
    "Access-Control-Max-Age": "86400",
    "Cache-Control": `public, max-age=${CACHE_SEGUNDOS}`,
    "Content-Type": "application/json; charset=utf-8",
  };
}

async function buscarPreco(url) {
  const resp = await fetch(url, {
    headers: { "User-Agent": "Mozilla/5.0 (compatible; gama-exposed/1.0)" },
    cf: { cacheTtl: CACHE_SEGUNDOS, cacheEverything: true },
  });
  if (!resp.ok) throw new Error(`HTTP ${resp.status}`);

  const dados = await resp.json();
  const meta = dados?.chart?.result?.[0]?.meta;
  const preco = meta?.regularMarketPrice;

  if (typeof preco !== "number" || !isFinite(preco) || preco <= 0) {
    throw new Error("resposta sem regularMarketPrice utilizavel");
  }
  return { preco, ts: (meta.regularMarketTime ?? 0) * 1000 };
}

export default {
  async fetch(request) {
    const origem = request.headers.get("Origin") ?? "";
    const cors = cabecalhosCors(origem);

    if (request.method === "OPTIONS") {
      return new Response(null, { status: 204, headers: cors });
    }
    if (request.method !== "GET") {
      return new Response(
        JSON.stringify({ ok: false, erro: "use GET", cotacoes: {} }),
        { status: 405, headers: cors },
      );
    }

    const chaves = Object.keys(FONTES);
    const resultados = await Promise.allSettled(
      chaves.map((k) => buscarPreco(FONTES[k])),
    );

    const cotacoes = {};
    const erros = [];
    let tsFonte = 0;

    resultados.forEach((r, i) => {
      const chave = chaves[i];
      if (r.status === "fulfilled") {
        cotacoes[chave] = r.value.preco;
        tsFonte = Math.max(tsFonte, r.value.ts);
      } else {
        // Falha parcial nao derruba a resposta: a pagina sabe lidar com null
        // e continua exibindo o fechamento para o que faltou.
        cotacoes[chave] = null;
        erros.push(`${chave}: ${r.reason?.message ?? "falha"}`);
      }
    });

    const algum = Object.values(cotacoes).some((v) => v !== null);

    return new Response(
      JSON.stringify({
        ok: algum,
        ts: Date.now(),
        ts_fonte: tsFonte || null,
        cotacoes,
        erro: erros.length ? erros.join("; ") : null,
      }),
      { status: algum ? 200 : 502, headers: cors },
    );
  },
};

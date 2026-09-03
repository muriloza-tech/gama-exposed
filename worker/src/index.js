/**
 * Proxy do painel Gama Exposed. Duas rotas, isoladas uma da outra:
 *
 *   GET /              cotação (BOVA11 e Ibovespa)  — alimenta o marcador de preço
 *   GET /?r=agenda     calendário da SEMANA         — alimenta a aba Calendário
 *
 * Por que existe: a página é estática e roda no navegador, onde a política
 * de CORS bloqueia a chamada direta. Um Worker é servidor — CORS não se
 * aplica à sua chamada de saída —, então ele busca o dado e devolve com o
 * cabeçalho que o libera. Não guarda segredo: nenhuma das fontes exige token.
 *
 * As rotas são deliberadamente independentes. Se a agenda cair, a cotação
 * continua; se a cotação cair, a agenda continua. E o painel de gama não
 * depende de nenhuma das duas: ele já vem calculado no HTML.
 *
 * Contrato de saída (sempre este formato, inclusive em erro):
 *   cotação: { ok, ts, cotacoes: { BOVA11: n|null, IBOV: n|null }, erro }
 *   agenda:  { ok, ts, dia, eventos: [...], erro }
 */

const FONTES_COTACAO = {
  IBOV: "https://query1.finance.yahoo.com/v8/finance/chart/%5EBVSP?interval=1m&range=1d",
  BOVA11: "https://query1.finance.yahoo.com/v8/finance/chart/BOVA11.SA?interval=1m&range=1d",
};

// Endpoint que alimenta o widget publico de calendario do TradingView.
// NAO e documentado: pode mudar ou bloquear sem aviso. Por isso a aba de
// agenda degrada sozinha e nada no painel depende dela.
const URL_AGENDA = "https://economic-calendar.tradingview.com/events";
const PAISES = "BR,US,EU,GB,DE,CN,JP";

const CACHE_COTACAO = 20;   // segundos
const CACHE_AGENDA = 300;   // a agenda muda devagar; o "efetivo" chega ao longo do dia

const ORIGENS_PERMITIDAS = [
  "https://muriloza-tech.github.io",
  "http://localhost:8000",
  "http://127.0.0.1:8000",
];

const UA = "Mozilla/5.0 (compatible; gama-exposed/1.0)";

function cors(origem, cacheSeg) {
  const permitida = ORIGENS_PERMITIDAS.includes(origem) ? origem : ORIGENS_PERMITIDAS[0];
  return {
    "Access-Control-Allow-Origin": permitida,
    "Access-Control-Allow-Methods": "GET, OPTIONS",
    "Access-Control-Max-Age": "86400",
    "Cache-Control": `public, max-age=${cacheSeg}`,
    "Content-Type": "application/json; charset=utf-8",
  };
}

const json = (corpo, status, cab) =>
  new Response(JSON.stringify(corpo), { status, headers: cab });

/* ============================================================= cotacao == */

async function buscarPreco(url) {
  const resp = await fetch(url, {
    headers: { "User-Agent": UA },
    cf: { cacheTtl: CACHE_COTACAO, cacheEverything: true },
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

async function rotaCotacao(cab) {
  const chaves = Object.keys(FONTES_COTACAO);
  const rs = await Promise.allSettled(chaves.map((k) => buscarPreco(FONTES_COTACAO[k])));

  const cotacoes = {};
  const erros = [];
  let tsFonte = 0;

  rs.forEach((r, i) => {
    const chave = chaves[i];
    if (r.status === "fulfilled") {
      cotacoes[chave] = r.value.preco;
      tsFonte = Math.max(tsFonte, r.value.ts);
    } else {
      // Falha parcial nao derruba a resposta: a pagina lida com null e
      // continua exibindo o fechamento no que faltou.
      cotacoes[chave] = null;
      erros.push(`${chave}: ${r.reason?.message ?? "falha"}`);
    }
  });

  const algum = Object.values(cotacoes).some((v) => v !== null);
  return json(
    {
      ok: algum,
      ts: Date.now(),
      ts_fonte: tsFonte || null,
      cotacoes,
      erro: erros.length ? erros.join("; ") : null,
    },
    algum ? 200 : 502,
    cab,
  );
}

/* ============================================================== agenda == */

/** Semana corrente (segunda a domingo) no fuso de Brasilia, em ISO UTC.
 *
 *  Brasilia e UTC-3 o ano todo desde o fim do horario de verao, entao a
 *  meia-noite local e sempre 03:00Z. */
function janelaDaSemana() {
  const brt = new Date(new Date().toLocaleString("en-US", { timeZone: "America/Sao_Paulo" }));
  const iso = (d) =>
    `${d.getFullYear()}-${String(d.getMonth() + 1).padStart(2, "0")}-${String(d.getDate()).padStart(2, "0")}`;

  const hoje = iso(brt);
  const diaSemana = (brt.getDay() + 6) % 7;            // 0 = segunda
  const segunda = new Date(brt);
  segunda.setDate(brt.getDate() - diaSemana);

  const de = `${iso(segunda)}T03:00:00.000Z`;
  return {
    hoje,
    inicio: iso(segunda),
    de,
    ate: new Date(Date.parse(de) + 7 * 86400000).toISOString(),
  };
}

function normalizar(e) {
  return {
    id: String(e.id ?? ""),
    hora: e.date ?? null,          // ISO UTC; a pagina converte para Brasilia
    pais: e.country ?? "",
    titulo: e.title ?? e.indicator ?? "",
    periodo: e.period ?? "",
    relevancia: typeof e.importance === "number" ? e.importance : -1,
    efetivo: e.actual ?? null,
    consenso: e.forecast ?? null,
    anterior: e.previous ?? null,
    // Versoes numericas: a pagina compara efetivo com consenso para marcar
    // surpresa, e comparar string formatada daria errado.
    efetivo_n: typeof e.actualRaw === "number" ? e.actualRaw : null,
    consenso_n: typeof e.forecastRaw === "number" ? e.forecastRaw : null,
    anterior_n: typeof e.previousRaw === "number" ? e.previousRaw : null,
  };
}

async function rotaAgenda(cab) {
  const { hoje, inicio, de, ate } = janelaDaSemana();
  const url = `${URL_AGENDA}?from=${de}&to=${ate}&countries=${PAISES}`;

  try {
    const resp = await fetch(url, {
      headers: { "User-Agent": UA, Origin: "https://www.tradingview.com" },
      cf: { cacheTtl: CACHE_AGENDA, cacheEverything: true },
    });
    if (!resp.ok) throw new Error(`HTTP ${resp.status}`);

    const dados = await resp.json();
    const bruto = Array.isArray(dados?.result) ? dados.result : [];

    const eventos = bruto
      .map(normalizar)
      .filter((e) => e.titulo && e.hora)
      .sort((a, b) => a.hora.localeCompare(b.hora));

    return json({ ok: true, ts: Date.now(), hoje, inicio, eventos, erro: null }, 200, cab);
  } catch (err) {
    // Devolve o contrato completo mesmo em falha, para a pagina so precisar
    // olhar `ok` e mostrar o aviso -- nunca quebrar.
    return json(
      { ok: false, ts: Date.now(), hoje, inicio, eventos: [], erro: String(err?.message ?? err) },
      502,
      cab,
    );
  }
}

/* =============================================================== fetch == */

export default {
  async fetch(request) {
    const origem = request.headers.get("Origin") ?? "";
    const url = new URL(request.url);
    const recurso = url.searchParams.get("r") ?? "cotacao";
    const cab = cors(origem, recurso === "agenda" ? CACHE_AGENDA : CACHE_COTACAO);

    if (request.method === "OPTIONS") {
      return new Response(null, { status: 204, headers: cab });
    }
    if (request.method !== "GET") {
      return json({ ok: false, erro: "use GET" }, 405, cab);
    }

    if (recurso === "agenda") return rotaAgenda(cab);
    if (recurso === "cotacao") return rotaCotacao(cab);
    return json({ ok: false, erro: `recurso desconhecido: ${recurso}` }, 400, cab);
  },
};

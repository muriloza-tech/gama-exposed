# gama-win

Motor de perfil de gama para operar o mini índice (WIN) na B3.

Não é um port do painel de Colab. É um motor novo, com a matemática provada
por testes, unidades declaradas e nenhum valor padrão escondido. O painel
antigo serviu de mapa do que construir — e de catálogo do que não repetir.

---

## Começando (3 comandos)

```bash
python -m venv .venv
./.venv/Scripts/python.exe -m pip install numpy pandas plotly requests pytest
./.venv/Scripts/python.exe -m gama_win doctor
```

O `doctor` é o primeiro comando a rodar sempre. Ele verifica o ambiente,
roda um autoteste da matemática, imprime o calendário de pregão que será
usado e lista o que ainda está pendente. Se ele não fechar limpo, não use o
resto.

Para ver o motor funcionando:

```bash
./.venv/Scripts/python.exe -m gama_win perfil --spot 176.5 --ratio-win 1048
```

Isso roda com a fonte **sintética** — dados inventados, marcados com marca
d'água obrigatória. É o estado atual: o motor está pronto, a fonte de dados
reais não está conectada.

---

## O que está pronto

| Camada | Módulo | Estado |
|---|---|---|
| Normal padrão sem scipy | `model/mathx.py` | pronto |
| Gregas BSM (delta, gama, vega, charm, vanna) | `model/greeks.py` | pronto, provado |
| Volatilidade implícita | `model/greeks.py` | pronto |
| Calendário de pregão B3 | `model/calendario.py` | pronto |
| Convenção de sinal do dealer | `model/conventions.py` | pronto |
| Contrato/validação de dados | `data/schema.py` | pronto |
| Perfil de exposição por strike | `model/profile.py` | pronto |
| Extração de níveis (flip, walls) | `model/levels.py` | pronto |
| Relatório de texto | `view/relatorio.py` | pronto |
| Fonte sintética | `data/sources/sintetica.py` | pronto |
| Motor de backtest | `backtest/engine.py` | pronto, provado |
| Estatística (bootstrap de blocos) | `backtest/stats.py` | pronto |
| Relatório do backtest | `view/relatorio_backtest.py` | pronto |
| **Fonte de OI real** | `data/sources/` | **PENDENTE** |
| **Barras históricas do WIN** | — | **PENDENTE** |
| Gráfico | `view/` | só texto |

`python -m gama_win doctor` sempre mostra esta lista atualizada.

---

## Decisões de projeto (e o porquê)

**Convenção de sinal explícita.** `DealerConvention.LONG_CALL_SHORT_PUT` é o
padrão de mercado: call contribui gama positivo, put negativo. A convenção
viaja dentro do resultado e aparece em todo relatório. Nenhuma saída sai sem
dizer qual convenção produziu o número — porque o sinal invertido é a
diferença entre "mercado travado" e "mercado acelerando".

**Unidades declaradas.** Não existe "Net GEX" adimensional aqui.

- `gama_brl_1pct` — variação da posição de delta do dealer, **em reais**, por
  **1% de movimento** do subjacente.
  Fórmula: `gamma · OI · contract_size · spot² · 0,01`
- `charm_brl_dia` — variação de delta em reais por **dia de pregão**
  decorrido, com o resto constante.
- `vanna_brl_por_ponto_vol` — variação de delta em reais por **1 ponto
  percentual** de vol implícita.

**Nenhum default escondido.** Série sem `implied_vol` não vira "22% porque
sim": é obrigatório passar `vol_fallback` de forma explícita, e o erro
explica que vol plana distorce o gama justamente nas asas, onde ficam as
walls.

**`tau ≤ 0` levanta exceção.** No vencimento o gama é máximo, não zero.
Devolver `0.0` ali é o erro mais perigoso possível num painel de gama. Para
clampar é preciso pedir `tau_floor` explicitamente. Na prática o calendário
já garante ao menos a sessão do vencimento, então isso não acontece por
acidente.

**Volatilidade implícita resolvida pela opção fora do dinheiro.** Opção
muito dentro do dinheiro é quase toda valor intrínseco: o vega tende a zero
e a vol não é identificável. Convertemos por paridade call-put antes de
resolver, e quando o preço genuinamente não determina uma vol o retorno é
`NaN` — nunca um valor de fronteira disfarçado de resposta.

**Validação numa única fronteira.** `OptionChain` valida tudo em
`data/schema.py` e acumula **todos** os problemas antes de falhar. Depois
dessa fronteira nenhum módulo checa `NaN` de novo.

**Flip é o cruzamento mais próximo do spot, interpolado.** Um perfil real
cruza o zero várias vezes; pegar o primeiro da esquerda devolve um nível a
5% de distância que não governa nada. E o cruzamento fica entre strikes.

**Wall é zona, não linha.** Gama é contínuo no strike. E a wall é definida
por **exposição a gama**, não por OI puro — OI grande muito fora do dinheiro
tem gama irrelevante e não segura preço.

**Wall nunca sai do lado errado.** Call wall só é procurada em strikes
`≥ spot`, put wall em `≤ spot`. Se não houver strike do lado necessário, a
wall é `None` e o motivo entra em `avisos`. Nunca se inventa um teto abaixo
do preço.

**Dado sintético não passa por real.** `is_synthetic` propaga da fonte até
os avisos dos níveis e até a marca d'água do relatório.

---

## Testes

```bash
./.venv/Scripts/python.exe -m pytest -q
```

1020+ testes. A estratégia não é "testar se roda": cada fórmula analítica é
verificada contra **diferenças finitas centrais** da grandeza que ela
representa, numa grade de moneyness × prazo × vol. Gama contra a derivada do
delta, charm contra a derivada em `tau`, vanna contra a derivada em vol,
vega e delta contra as derivadas do preço, mais paridade call-put.

Se alguém inverter um sinal ou errar um fator, os testes falham. Foi
exatamente esse tipo de erro que passou desapercebido no painel original.

Há também testes de regressão nomeados para os defeitos concretos daquele
painel — por exemplo `test_call_wall_nunca_abaixo_do_spot` e
`test_flip_escolhe_o_cruzamento_mais_proximo_do_spot`.

---

## O backtest

```bash
./.venv/Scripts/python.exe -m gama_win backtest --demo
```

A pergunta, na forma exata em que ela é falsificável:

> Quando o preço vinha caindo e o gama líquido no preço estava negativo, com
> que frequência a queda continuou no horizonte H — e essa frequência é
> **diferente da frequência incondicional do mesmo período**?

A segunda metade é a que quase todo painel de gama ignora. Taxa de
continuação de 62% em gama negativo não significa nada se a incondicional
também for 62%: nesse caso mediu-se momentum, não gama. Por isso a métrica
principal do relatório não é a taxa — é a **diferença contra o baseline, com
IC pareado**. Se o intervalo não exclui zero, não há evidência de edge.

**Cuidados anti-viés embutidos:**

- **Sem lookahead entre dias.** O laço é por dia; janela futura nunca
  atravessa o fechamento. É o erro clássico de backtest intradiário, e tem
  teste nomeado (`test_nao_ha_lookahead_entre_dias`).
- **Sem lookahead no OI.** O perfil de cada dia deve vir do OI de D-1. O
  relatório imprime esse lembrete em toda execução.
- **Sem extrapolação de regime.** Preço fora da faixa de strikes → evento
  descartado e contabilizado, não classificado por extrapolação plana.
- **IC por bootstrap de blocos de dia.** Janelas consecutivas se sobrepõem,
  então as observações são autocorrelacionadas; um IC binomial clássico
  devolveria intervalo estreito demais — significância inventada.
- **Empate conta como não-continuação.** Retorno futuro exatamente zero
  significa que o movimento não prosseguiu. Isso deprime todas as taxas de
  forma uniforme (num grid discreto como o WIN empates são comuns), mas não
  afeta a diferença contra o baseline — outra razão para a diferença ser a
  métrica de destaque.

**O motor é testado nas duas direções**, o que é o essencial num backtest:
`test_detecta_efeito_injetado_forte` planta um efeito nos dados e exige que
ele seja encontrado; `test_nao_inventa_efeito_em_dados_sem_sinal` roda
passeio aleatório e exige que nada seja reportado como significativo. Um
motor que não detecta edge nunca valida nada; um que inventa edge custa
dinheiro.

Para rodar com dados reais, monte uma lista de `DiaBacktest` (data, barras
do WIN com `ts`/`close`, e um avaliador de gama do dia) e chame
`rodar_backtest()`. O contrato está em `backtest/engine.py`.

---

## O que falta, em ordem de prioridade

**1. Fonte de OI real.** É o único bloqueador de verdade. Enquanto não
entrar, tudo aqui é estrutura. Implementar uma fonte significa escrever uma
classe com dois métodos (`disponivel()` e `buscar()`) que devolva um
`OptionChain` válido — o contrato está em `data/sources/base.py`.

**2. Barras históricas do WIN.** O outro insumo do backtest. Bastam `ts` e
`close` intradiários por dia de pregão.

**3. Painel gráfico.** Depois do backtest, para que cada regime exiba a
estatística que o acompanha — o que nenhum painel gratuito faz.

**4. Intraday.** Charm recalculando os níveis ao longo do dia, confluência
com VWAP e abertura, alertas de aproximação.

---

## Limitação honesta do método

Open interest **não** informa quem está comprado e quem está vendido.
"Dealer long call / short put" é uma **heurística** sobre o outro lado do
fluxo, não um dado — é a suposição em que todo painel de GEX se apoia,
inclusive os pagos. E o efeito é probabilístico: gama negativo aumenta a
chance de continuação do movimento, não a garante.

É por isso que o backtest vem antes do painel bonito.

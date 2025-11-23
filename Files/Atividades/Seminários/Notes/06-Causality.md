# Causalidade

- LLMs sabem o que acontece, mas não por que acontece.
- Correlação != Causalidade.

## SCM (Structural Causal Model)

## Causal Discovery

LLM ajuda a inferir o grafo causal

## Fine Tuning

Treinar o modelo com um dataset de causalidades

## Causal Chain-of-Thoughts (Causal CoT)

## Benchmarking robuto

## (Me ausentei)

## SCM e Contrafactuais

Associação, Intervenção, Contrafactual

## Testes Causar Comportamentais (Black Box)

Estudar como pequenas mudanças alteram a saída do LLM

## Testes Causais Intrínsecos (White Box)

Analiza a relação do input com o mecanismo interno.

## Abstração...?

## Abstração Causal e Representação

## Pesquisas futuras

## Referências

## Dúvidas

Anísios: não importa só o dado, importa como o dado foi gerado.

Grafo de causalidade: uma função que modela o mundo.

## Anotações pra prova

- **Correlação:** Duas variáveis alteram juntas, mas não necessariamente uma causa a outra.
- **Causalidade:** Uma variável influencia diretamente a outra.
- Causal Parroting: modelo repete relações causais sem entendê-las.
- Structural Causal Model (SCM): modelo matemático que representa relações causais entre variáveis usando grafos direcionados. A representação em grafos auxilia nos cenários contrafactuais.
- O objetivo então é encontrar o grafo de causalidade entre as variáveis. Para isso a LLM pode ser utilizada.
- Uma abordagem é executar o fine-tuning para que o modelo aprenda relações causais específicas, não apenas repetir palavras estatisticamente relevantes.
- Causal Chain-of-Thoughts (Causal CoT): técnica de prompting que incentiva o modelo a gerar raciocínios passo a passo focados em relações causais.
- Domain Knowledge Retrieval: incorporar conhecimento especializado para melhorar a compreensão causal do modelo.
- Critérios de Benchmarking: COS
  - Causal/Interventional
  - Open-Ended
  - Scalable
- Evolução do raciocínio nas LLMs:
  1. Geração direta e simples baseada na sequência de palavras.
  2. RAG: Geração aumentada por RI. Busca e integra informações relevantes.
  3. Futuro Causal: Raciocínio estruturado e com inferência formal.
- Causalidade vs interpretabilidade: se conseguirmos garantir que o modelo entende causalidade, ele se torna mais interpretável.
  - Confiança: modelos transparentes são mais confiáveis.
  - Justiça: auditoria de viés
  - Aperfeiçoamento: diagnóstico de erros
  - Regulamentação: conformidade com normas de explicabilidade
- Causalidades:
  1. Associação: observar correlações
  2. Intervenção: manipular variáveis e observar efeitos
  3. Contrafactual: imaginar cenários alternativos
- Se uma LLM não responde bem à contrafactuais, ela não entende causalidade.
- Testes de causalidade:
  - Black Box: testes causais comportamentais. Analisar como pequenas mudanças no input afetam a saída.
    - Focado no input-output;
    - Variação no prompt e análise de mudanças na resposta.
    - Coin-flipping game: mudar detalhes irrelevantes e ver se a resposta muda.
  - White Box: testes causais intrínsecos.
    - Investiga de que forma o input afeta o mecanismo interno do modelo.
    - Deseja entender como funciona a caixa-preta
    - Análise de ativação de features; e mechanistic interpretability.

### Abstração Causal e Representação (2)

- Considera-se que a Abstração Causal é a capacidade de um modelo de representar e raciocinar sobre relações causais em diferentes níveis de detalhe automaticamente pelos valores pré-treinados.

### Pesquisas futuras (2)

- É esperado que a causal se torne parte fundamental da estrutura dos LLMs
- O Grafo causal pode ser aplicar para diversas tarefas
- Conversão em representações formais para que um solucionador externo resolva

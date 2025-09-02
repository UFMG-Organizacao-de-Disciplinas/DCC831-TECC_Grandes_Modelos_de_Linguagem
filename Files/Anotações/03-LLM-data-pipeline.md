# Aula 03

## Data fundamentally limits model performance

- Para treinar os modelos é necessário ter muitos dados. E de qualidade.

| Tabela |
| ------ |
|        |

---

- Data-centric AI
  - ...
- Model-centric AI
  - ...

Passaremos pela implementação completa do GPT.

## The Importance of Data Quality Over Quantity (alone)

Instruções melhores são melhores do que centenas de milhares de exemplos ruins.

---

- Llama 3 team...
  - Llama 3's...

Dados gerados por humanos são barulhentos e inconsistentes. Eles então focaram em IA para auxiliar nas anotações dos dados.

(Minha ideia de usar SCHEMAS pode ser um caminho interessante)

## The importance of Data Diversity

- Table 8-1

| X                           | Pre-training | Supervised Fine-tuning | Preference Finetuning |
| --------------------------- | ------------ | ---------------------- | --------------------- |
| General Knowledge (English) |              |                        |                       |

## Roadmap

Primeiro iremos aprender o bottom-up da ingestão desse tipo de sistema.

- Parte 1: The Atoms of Language - Text Representation
- Parte 2: Data Augmentation and Synthesis
- Parte 3: The Research Frontier

### Parte 1: The Atoms of Language - Text Representation

Figure 2.1

```mermaid

```

---

#### 1.1 - Tokenização - The Fundamental Unit

- THe First Step:
- Vocabulary and Token IDs:
- The Modern Solution - BPE: We will focus on Byte-Pair Encoding (BPE), ...
- A Comparative Look: ...

Primeiro tokenizamos, subdividindo o texto em pedaços menores. De início eles eram convertidos diretamente ao vocabulário e as palavras desconhecidas eram substituídas por um token especial. Depois surgiu o BPE, que consegue lidar melhor com palavras desconhecidas.

#### 1.2 - Vectorization - Turning Tokens into Tensors

Tensores são vetores de n dimensões.

- Token Embeddings: ...
- The Positional Problem: ...
- Positional Embeddings: ...

#### 1.3 - Preparing Data for Model Training

- The Autoregressive Objective
- The Sliding Window:

---

```mermaid

```

- Dúvida: Por que se chama tensor e não simplesmente um ponto?

## Understanding Embeddings

Mapeia de objetos discretos para pontos num espaço contínuo.

- ...

---

- Figure 2.2

---

- Figure 2.3

Idealmente esperamos que palavras com significados semelhantes estejam próximas no espaço vetorial.

- Dúvida: Então um dado de qualidade, com anotações boas, tendem a levar os tokens a estarem mais próximos no espaço vetorial?
- Resposta: Sim, é uma forma de pensar. Porém a qualidade do dado não é apenas pela anotação. Deve-se também levar em conta a diversidade, cobertura, etc.

## Working with Text Data (Mostrando o código do Notebook do Capítulo 2)

Simple Tokenizer V1:

Encode: converte texto passado em ids de tokens.

Decode: converte a lista de Ids em uma string.

Tokens especiais: Modelos como o BERT usam tokens como <|unk|> e <|EOF|>. O End Of File (EOF) é usado para indicar o fim do texto, assim auxilizando o modelo a entender quando parar de entender como um contexto sequencial.

Ele adiciona os tokens especiais na lista de tokens para evitar erros por token desconhecido.

- Dúvida: Por que não adicionar as novas palavras não vistas antes ao vocabulário?
- Resposta: Porque seria necessário re-treinar o modelo. E isso é caro. Além disso ele é certamente infrequente.

- Dúvida: Mas e se for uma palavra frequente e que por algum motivo não ocorreu no dataset de treinamento?
- Resposta: ...

Testar em casa: usar o tokenizer do GPT2 sem o allowed_special="..."

- Dúvida: O que é um texto regressivo?
- Resposta (Copilot): É um texto que é previsto com base no texto anterior. Ou seja, o modelo tenta prever a próxima palavra com base nas anteriores.

- Dúvida: o que são os parâmetros do PyTorch? (batch size, shuffle, drop_last)

Mudou para que o Stride seja do mesmo tamanho da janela. Assim, não há sobreposição nos dados...
Mas e no próximo Lote?

É esperado que não haja sobreposição para evitar overfitting. Como há dado em abundância, é preferível diversificar os dados do que reutilizá-los.

## Tokenization Algorithms - Two Components

- The Training Algorithm: ...
- The tokenization Algorithm: ...

- [Imagem]

## Parte 2 (?)

### The Role of Synthetic Data

- Data Augmentation
- Data Synthesis

### Papers

### Why Programmatically Generate Data?

- E.g., instruction Data Synthesis
- Increase Data Quantity
- Increase Data Coverage
- Increase Data Quality
- Mitigate Privacy Concerns
- To distill models

### Challenges of Full Programmatic Data Generations

- Superficial Imitation
- Quality Control

### Parte 3: The Research Frontier

- Model Collapse

Consideram que a área de síntese de dataset é promissora justamente pelo Model Collapse por causa das pessoas usando muito a geração de texto por IA.

#### The causes of Model Collapse

AI models tend to...

Over multiple iterations...

##### Mitigation Strategies

- Human-in-the-loop.
- Livros de autores que não usaram IA.

## Recap

...

## Próxima aula

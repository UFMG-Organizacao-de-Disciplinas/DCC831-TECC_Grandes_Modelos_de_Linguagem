# Aula 4 -

## Introdução

### ...? [Cheguei Tarde]

## Fundamentals of Sequence Modeling

### Temporal Dependencies and Memory

- Several Dependencies
  - Short-range dependencies
  - Long-range dependencies
  - Hierarchical dependencies

## Fundamentals of Sequence Modeling (2)

### The Alignment Problem

I love you = Te amo
...

---

- Traditional approaches...
- This creates...
- Alignment reflects...

## Language Models

### Markov models and n-grams for language modeling

Num bigrama, ele olha só para a palavra anterior.

Num trigrama, ele olha para as duas palavras anteriores.

Nesse contexto, só seria necessário uma quantidade fixa de memória.

#### The Problem of Short Memory

- What if the context you need is furher back?
- ...

#### The Exponential Cost of a Better Memory

O número de combinações seria exponencial

---

$|V|^n$

E ainda nesses casos de muitos parâmetros a quantidade de informação recebida não é suficiente para resolver o problema do contexto.

- Increasing n is computationally infeasible
  - Storage - ...
  - Data Sparsity - ...

## Neurons with Recurrence (MIT 6.S191) (MIT Introduction to Deep Learning, 1/8/24)

### The Perceptron: Forward Propagation

```mermaid

```

---

...

#### Importance of Activation Functions

#### Perceptron Revisited

#### Feed-Forward Network Revisited

```mermaid

```

#### Handling Individual Time Steps [19min30]

```mermaid

```

#### Neurons with Recurrence

```mermaid

```

### Recurrence [Não sei oq] [21min45]

#### Working with Sequences

- 1-to-1
  - Vanilla model of p...
- 1-to-many
  - e.g., Geração de vídeo por prompt (várias imagens); Descrição de imagem (várias palavras);
  - JV: Não seria decomposição em fatores primos porque não precisa de uma ordem específica.
- many-to-1
- many-to-many (synchronized)
- many-to-many (unsynchronized)

#### The Architectural Innovation [30min30]

...

A inovação é a memória da rede a medida em que a rede progride, isso servindo como memória.

Definição de variável latente: uma variável que não é observada diretamente, mas inferida a partir de outras variáveis que são observadas (medidas diretamente).

...

Essa é a parte mais importante da aula

#### Recurrent Neural Networks (RNNs) [31min30]

#### RNN Intuition [33min]

A predição ser correta ou errada impacta na mudança do hidden state

#### RNN State Update and Output [37min10]

$h_t = tanh(W_{hh}^{T} h_{t-1} + W_{xh}^{T} x_t)$

- Dúvida: O que significa cada um desses termos?

Aparentemente tanh é tangente hiperbólica

#### RNNs: Computational Graph Across Time [39min10]

Estamos querendo encontrar os melhores parâmetros para um modelo.

Para otimizar essa função, queremos encontrar os parâmetros que minimizam a função de perda.

O Anísio recomenda assistir a aula anterior a essa que explica sobre Loss Functions.

"A Loss é tipo o aluno aprendendo"

#### RNNs from Scratch [43min]

#### RNN Implementation in TensorFlow [44min30]

Ele usou um livro chamado "Deep dive into deep learning"

### A Sequence Modeling Problem: ... [45min20]

#### A Sequence Modeling Problem: Predict the Next Word

#### Encoding Language for a Neural Network

1. Vocabulary
2. Indexing
3. One-hot encoding

#### Sequence Modeling: Design Criteria

...

#### Model Long-Term Dependencies [49min]

#### Capture Differences in Sequence Order [50min]

### Backpropagation Through Time (BPTT) [50min40]

#### Loss Optimization

$f(x^{(i)}; W) = \hat{y}$

#### Gradient Descent

- Dúvida: O que faz para não cair em mínimo local?
- Resposta: Vai testando várias inicializações diferentes. Se for paper, encontra o baseline e explica. Se for indústria, faz benchmarks, quando for bom o bastante, libera.

---

#### Recall: Backpropagation in Feed Forward Models [58min10]

"Machine Learning é muito empírico, então muita coisa não tem resposta"

##### Forward Pass

...

Agora queremos encontrar pesos melhores do que os que encontramos anteriormente.

##### Backward Pass

#### Machine Translation [67min]

Converter uma palavra em duas ou mais palavras.

O conceito de encoder e decoder são importantes assim como os embeddings.

Ele tá explicando as ideias ao longo da história porque às vezes juntar ideias novas com antigas gera inovações úteis.

##### E.g., "cat" $\to$ "o gato"

### RNN Tricks [72min]

#### Standard RNN Gradient Flow: Vanishing Gradients

...

#### Trick #2: Parameter...

#### Trick #3: Gated Cells

#### Goal of Sequence MOdeling [75min]

Limitado quanto a paralelização

- Desejados:
  - Contínuo
  - Paralelizável
  - Memória Longa

### Applications

- Language Modeling
- Speech Recognition

### The Legacy of RNNs [78min]

- Sequential Processing Paradigm
- Weight Sharing
- Hidden State Representation
- ...

## Resumo

## Referências

## Próxima Aula: Attention Mechanisms

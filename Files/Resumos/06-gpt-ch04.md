# Chapter 4: Implementing a GPT model from Scratch To Generate Text

Nessa etapa, vamos utilizar os conhecimentos adquiridos até agora para implementar a arquitetura do GPT. Para isso passamos primeiro pela preparação dos dados, seguido da implementação do componente de atenção.

## 4.1 Coding an LLM architecture

Os modelos como GPT e Llama geram palavras sequencialmente, isso se baseando na parte _decoder_ da arquitetura _transformer_.

DÚVIDA: Mas e o que seria um _transformer_?
RESPOSTA COPILOT: Um _transformer_ é uma arquitetura de rede neural que utiliza mecanismos de atenção para processar dados sequenciais, como texto ou áudio. Ele é composto por camadas de atenção e camadas feedforward, permitindo que o modelo capture relações de longo alcance em dados sequenciais de maneira eficiente.

DÚVIDA: Então o _transformer_ é basicamente o esqueleto de uma LLM antes de ter sido treinada com muitos dados?
RESPOSTA COPILOT: Sim, o _transformer_ é a arquitetura básica que serve como esqueleto para modelos de linguagem como GPT.

Então, meu entendimento é que primeiro iremos programar um transformer, que usa o mecanismo de atenção. Depois, com os dados preparados iremos treinar esse transformer para que ele se torne um modelo de linguagem como o GPT.

---

GEMINI

A arquitetura de um LLM como o GPT é baseada no **Decoder-Only Transformer**. Ele é construído empilhando-se múltiplos blocos idênticos chamados **Transformer Blocks**.

Os principais componentes estruturais adicionados neste capítulo para completar a arquitetura são:

1. **Layer Normalization** (Normalização de Camada).
2. **Feed Forward Network (FFN)** com a função de ativação **GELU**.
3. **Shortcut Connections** (Conexões Residuais).

## 4.2 Normalizing activations with layer normalization --- GEMINI

A **Layer Normalization** (Normalização de Camada) é um mecanismo introduzido para **estabilizar o treinamento** de redes neurais profundas.

- **Finalidade:** Garante que as ativações de cada camada mantenham uma **média consistente (zero) e desvio padrão consistente (unitário)** ao longo da dimensão dos recursos (o vetor de _embedding_).
- **Mecanismo:** A normalização é aplicada **horizontalmente** (através da dimensão do _embedding_) em cada token, independentemente do _batch size_.
- **Implementação:** É implementada pela classe `LayerNorm` e inclui dois parâmetros treináveis por _feature_ ($\gamma$ e $\beta$) para aprender a escala e o deslocamento ideais após a normalização.

## 4.3 Implementing a feed forward network with GELU activations --- GEMINI

A **Feed Forward Network (FFN)** é o segundo sub-bloco dentro do Transformer Block, vindo logo após o módulo de atenção.

- **Estrutura:** Consiste em duas camadas lineares sequenciais, com uma função de ativação não linear no meio.
- **Expansão:** O FFN normalmente **expande** a dimensão do _embedding_ em um fator de 4 antes de revertê-la ao tamanho original (ex: $D \to 4D \to D$), permitindo que a rede processe as informações enriquecidas pelo mecanismo de atenção.
- **Ativação GELU:** O GPT utiliza a função **Gaussian Error Linear Unit (GELU)**, uma função de ativação que pesa as entradas por seu valor (em vez de usar um limite rígido como a ReLU). O código a implementa com a função `torch.nn.GELU()`.

## 4.4 Adding shortcut connections --- GEMINI

As **Shortcut Connections** (também conhecidas como **Residual Connections**) são um elemento arquitetônico fundamental em redes neurais profundas.

- **Finalidade:** Ajudam a **mitigar o problema do _vanishing gradient_** (gradiente nulo) em redes muito profundas, garantindo que os gradientes possam fluir mais facilmente para camadas anteriores durante o _backpropagation_.
- **Mecanismo:** A saída de uma camada é adicionada **diretamente** à entrada de uma camada mais profunda (pulando uma ou mais camadas).
- **Padrão GPT:** No Transformer Block do GPT, as conexões residuais são aplicadas **após cada sub-bloco** (Multi-Head Attention e FFN) e seguidas pela **Layer Normalization** (padrão Post-Layer Norm).

## 4.5 Connecting attention and linear layers in a transformer block --- GEMINI

O **Transformer Block** (bloco de transformação) é a unidade fundamental que é replicada $N$ vezes para formar a rede neural do GPT.

Cada bloco (implementado pela classe `TransformerBlock`) consiste em dois sub-blocos principais, cada um com uma conexão residual e uma normalização de camada:

1. **Sub-bloco de Atenção:**
   - _Entrada_ $\to$ **Multi-Head Attention Causal** (MHA).
   - _Saída do MHA_ $\to$ **Shortcut Connection** (adição com a entrada) $\to$ **Layer Normalization**.
2. **Sub-bloco Feed Forward:**
   - _Saída do Layer Norm_ $\to$ **Feed Forward Network (FFN)** (Linear $\to$ GELU $\to$ Linear).
   - _Saída do FFN_ $\to$ **Shortcut Connection** (adição com a entrada) $\to$ **Layer Normalization**.

## 4.6 Coding the GPT model --- GEMINI

A classe `GPTModel` é a arquitetura completa que orquestra todos os componentes, desde a entrada até a camada de saída:

1. **Camada de Entrada (Embeddings):** Combina os **Token Embeddings** e os **Positional Embeddings** (aprendidos durante o treinamento).
2. **Stack de Blocos:** O núcleo do modelo é uma sequência de **Transformer Blocks** idênticos, empilhados verticalmente.
3. **Camada de Saída (Projection Head):** Após o último bloco Transformer, um **Layer Normalization final** é aplicado, seguido por uma **camada linear de projeção**. Esta camada mapeia a representação interna do modelo de volta para a dimensão do vocabulário, produzindo o _logits_ (scores brutos de previsão) para cada token possível.
4. **Tamanho do Modelo:** O mesmo código (`GPTModel`) pode ser usado para criar modelos de vários tamanhos (ex: 124 milhões a 1.5 bilhões de parâmetros), ajustando-se apenas a profundidade (número de blocos) e a largura (dimensão do _embedding_).

## 4.7 Generating text --- GEMINI

A capacidade de gerar texto é implementada no método `generate` da classe `GPTModel`.

- **Processo de Geração (Autoregressivo):** O modelo é **autoregressivo**, o que significa que ele gera o texto **um token de cada vez**. A saída da etapa anterior torna-se a entrada para a próxima.
- **Loop:** O método entra em _loop_ pelo número de _tokens_ que devem ser gerados. Em cada passo, ele:

  1. Recebe a sequência atual de entrada (_context_).
  2. Faz a passagem _forward_ pelo modelo, obtendo os _logits_ (scores brutos) para o próximo _token_.
  3. Aplica a função **Softmax** para obter as probabilidades.
  4. **Amostra** (seleciona) o próximo _token_ com base nessas probabilidades (ex: usando `torch.multinomial` para amostragem ponderada).
  5. Anexa o novo _token_ gerado à sequência de entrada para a próxima iteração.

- **Saída:** Sem treinamento, o modelo gera texto incoerente (aleatório), sublinhando a importância da etapa de pré-treinamento (Capítulo 5).
  O Capítulo 4 marcou a conclusão da arquitetura do GPT, reunindo todos os blocos fundamentais para criar um modelo capaz de gerar texto.

Aqui está o resumo dos principais pontos e aprendizados do capítulo:

## Summary and takeaways (Resumo e Pontos Principais) --- GEMINI

- **Estrutura Central (Transformer Block):** O coração da arquitetura GPT é a repetição de **Transformer Blocks**. Cada bloco combina o módulo de atenção multi-cabeças mascarado com a rede _Feed Forward_ (totalmente conectada).
- **Estabilidade (Layer Normalization):** A **Layer Normalization** (Normalização de Camada) é introduzida para estabilizar o treinamento. Ela garante que as saídas de cada camada mantenham uma média e variância consistentes.
- **Conexões Residuais (Shortcut Connections):** **Shortcut Connections** (Conexões de Atalho) são adicionadas para ignorar uma ou mais camadas e alimentar a saída de uma camada diretamente a uma camada mais profunda. Essa técnica é vital para mitigar o problema do _vanishing gradient_ (gradiente nulo) ao treinar redes neurais profundas, como os LLMs.
- **Rede Feed Forward (FFN):** A FFN utiliza a função de ativação **GELU (Gaussian Error Linear Unit)**, que é o padrão em modelos GPT.
- **Escalabilidade do Modelo:** O código da classe `GPTModel` (implementado em `06-gpt-ch04.ipynb`) pode ser usado para construir modelos GPT de vários tamanhos, variando de milhões a bilhões de parâmetros (por exemplo, 124M a 1.542M parâmetros).
- **Geração de Texto (Autoregressiva):** A capacidade de geração de texto é baseada na **decodificação** dos tensores de saída em texto legível. Isso é feito sequencialmente, prevendo um _token_ por vez com base no contexto de entrada fornecido.
- **Necessidade de Treinamento:** Um modelo GPT recém-implementado (sem o estágio de pré-treinamento) gera texto incoerente. Isso enfatiza a importância crítica do treinamento do modelo (que será o foco do próximo capítulo) para gerar texto coeso e de qualidade.

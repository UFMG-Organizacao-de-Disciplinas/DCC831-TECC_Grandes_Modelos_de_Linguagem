# Chapter 5: Pretraining on Unlabeled Data

O Capítulo 5 inicia a fase de pré-treinamento, focando primeiramente em como avaliar a performance de um LLM, que é crucial antes de iniciar o _training loop_.

## 5.1 Evaluating generative text models --- GEMINI

### 5.1.1 Using GPT to generate text --- GEMINI

A capacidade de geração do GPT é baseada no processo **autoregressivo** (um _token_ por vez) que foi implementado no método `generate` (do Capítulo 4).

- **Processo de Geração:** O modelo recebe um contexto inicial, produz _scores_ brutos (_logits_), converte esses _scores_ em probabilidades.
- **Seleção de Tokens:** O próximo _token_ é escolhido com base nessas probabilidades. Por padrão, isso é feito por **decodificação gulosa (_greedy decoding_)** (selecionando o _token_ com a maior probabilidade), mas pode ser modificado com **amostragem probabilística** e **escalonamento de temperatura** para introduzir diversidade.
- **Finalidade:** Este é o resultado final que se busca avaliar: a qualidade do texto gerado.

### 5.1.2 Calculating the text generation loss: cross-entropy and perplexity --- GEMINI

Para quantificar a qualidade das previsões do modelo, utiliza-se uma função de perda (_loss function_).

- **Cross-Entropy Loss (Entropia Cruzada):** É a função de perda padrão para a tarefa de predição da próxima palavra. Ela mede a diferença entre a distribuição de probabilidade prevista pelo modelo (os _logits_ da camada de saída) e a distribuição de probabilidade alvo (o _token_ verdadeiro, `y`). O objetivo do treinamento é **minimizar** essa perda.
- **Perplexity (PPL):** Perplexidade é uma métrica de avaliação que é diretamente derivada da perda de entropia cruzada.
  - **Fórmula:** $PPL = e^{\text{Cross-Entropy Loss}}$.
  - **Interpretação:** É interpretada como o número médio de _tokens_ para os quais o modelo deve escolher a próxima palavra. **Quanto menor a perplexidade**, melhor é a capacidade do modelo de prever o texto.

### 5.1.3 Calculating the training and validation set losses --- GEMINI

O cálculo da perda é expandido para avaliar o modelo em conjuntos de dados inteiros, fornecendo uma visão macro de sua performance.

- **Propósito:** Calcular as perdas do conjunto de treinamento e do conjunto de validação é o método primário para **avaliar a qualidade** do texto gerado pelo LLM durante o treinamento.
- **Overfitting:** A comparação entre as duas perdas é crucial: se a perda de treinamento for significativamente menor que a perda de validação, isso indica **overfitting** (o modelo decorou o conjunto de treinamento, mas não generaliza).
- **Implementação (no Notebook `07-pretraining-ch05.ipynb`):** No código, isso é implementado através de uma função auxiliar que itera sobre o `DataLoader` (do Capítulo 2), realiza a passagem _forward_ do modelo para cada lote (_batch_) e acumula as perdas médias de cada lote.
  - O modelo é colocado em modo de avaliação (`model.eval()`) para calcular essas perdas, desativando mecanismos como o Dropout para garantir que as perdas sejam determinísticas.

## 5.2 Training an LLM --- GEMINI

A fase de pré-treinamento envolve a atualização dos pesos da LLM para minimizar a perda de treinamento, ensinando-a a prever o próximo _token_.

- **Processo Padrão:** O _training loop_ em si é um procedimento padrão no _deep learning_.
- **Função de Perda e Otimizador:** O treinamento utiliza a **Cross-Entropy Loss** como função de perda e o **AdamW optimizer** para ajustar os pesos. O AdamW é uma variante do Adam que aprimora o tratamento do _weight decay_ (decaimento de peso).
- **Alternativa de Pré-treinamento:** Como pré-treinar uma LLM em um grande _corpus_ é dispendioso em termos de tempo e recursos, é comum a prática de carregar **pesos pré-treinados abertamente disponíveis** (como os da OpenAI). Isso fornece um ponto de partida sólido para as fases subsequentes de _fine-tuning_.

## 5.3 Decoding strategies to control randomness --- GEMINI

Embora o modelo possa gerar texto por padrão usando **decodificação gulosa (_greedy decoding_)** (selecionando o _token_ de maior probabilidade), essa abordagem frequentemente resulta em texto repetitivo. Para adicionar diversidade e controlar a coerência, são utilizadas estratégias de amostragem probabilística.

### 5.3.1 Temperature scaling --- GEMINI

A temperatura é um hiperparâmetro que **escala os _logits_** (scores brutos de previsão) antes da função Softmax, modificando a distribuição de probabilidade dos _tokens_.

- **Baixa Temperatura (coerência):** Uma temperatura **mais baixa** (ex: < 1.0) torna a distribuição **mais nítida**, aumentando a probabilidade do _token_ de maior _score_. Isso gera texto **mais determinístico e focado**.
- **Alta Temperatura (diversidade):** Uma temperatura **mais alta** (ex: > 1.0) torna a distribuição **mais plana**, distribuindo a probabilidade de forma mais uniforme. Isso resulta em texto **mais aleatório e criativo**.

### 5.3.2 Top-k sampling --- GEMINI

O _Top-k sampling_ é uma estratégia para **restringir o vocabulário** de onde o próximo _token_ será amostrado.

- **Mecanismo:** Em vez de considerar todos os _tokens_, o modelo seleciona apenas os **k _tokens_ mais prováveis** (com os _scores_ mais altos).
- **Re-normalização:** As probabilidades desse subconjunto de _tokens_ são **re-normalizadas** para que somem 1.
- **Amostragem:** O próximo _token_ é então amostrado apenas dentro deste conjunto restrito.

### 5.3.3 Modifying the text generation function --- GEMINI

As estratégias de decodificação são incorporadas à função `generate` do modelo. O _notebook_ (`07-pretraining-ch05.ipynb`) demonstra como a função é modificada para:

1. Aplicar o **escalonamento de temperatura** aos _logits_.
2. Aplicar o **filtro Top-k** aos _logits_ (opcionalmente).
3. Usar `torch.multinomial` para realizar a amostragem probabilística, escolhendo o próximo _token_ com base na distribuição de probabilidades ajustada.

O final do Capítulo 5 concentra-se nas práticas de engenharia e otimização do fluxo de trabalho, essenciais para gerenciar modelos grandes e aproveitar o trabalho da comunidade científica.

## 5.4 Loading and saving model weights in PyTorch --- GEMINI

Gerenciar os parâmetros treináveis (pesos) de um LLM é feito de maneira padronizada pelo PyTorch:

- **`state_dict`:** O método padrão para salvar os pesos de um modelo é extrair seu **dicionário de estado** (`state_dict`). O `state_dict` é um dicionário Python que mapeia o nome de cada camada aos seus tensores de parâmetro correspondentes (pesos e _biases_).
- **Salvando Pesos:** O comando `torch.save(model.state_dict(), PATH)` salva esse dicionário em um arquivo.
- **Carregando Pesos:** Para restaurar um modelo, a função `torch.load(PATH)` carrega o dicionário de volta, e ele é injetado no modelo com `model.load_state_dict()`. Essa funcionalidade é crucial para retomar o treinamento ou para usar o modelo em uma fase subsequente de _fine-tuning_.

## 5.5 Loading pretrained weights from OpenAI --- GEMINI

O pré-treinamento de um LLM em um _corpus_ massivo de texto é uma tarefa extremamente intensiva em tempo e recursos. Por isso, carregar pesos pré-treinados disponíveis publicamente (como os do GPT-2 original da OpenAI) é uma alternativa prática.

- **Vantagem:** Permite que o modelo implementado (nos Capítulos 2-4) comece com um **sólido conhecimento de linguagem**, servindo como um "modelo fundação" pronto para o _fine-tuning_ (Capítulos 6 e 7).
- **Mapeamento de Pesos:** Para carregar os pesos pré-treinados em uma arquitetura de modelo implementada do zero (como a `GPTModel`), é necessário um **mapeamento de pesos**. Isso se deve a diferenças nas convenções de nomenclatura e na ordem dos parâmetros entre o modelo original da OpenAI e a implementação customizada.
- **Função de Conversão:** O notebook `07-pretraining-ch05.ipynb` implementa uma função (`load_and_map_weights`) para lidar com essa complexa conversão, garantindo que cada tensor de peso pré-treinado seja alocado corretamente para o parâmetro correspondente no modelo customizado.

## Summary and takeaways --- GEMINI

- **Métricas de Avaliação:** A qualidade do texto gerado e o progresso do treinamento são medidos pela **Cross-Entropy Loss** (que o treinamento visa minimizar) e pela **Perplexity (PPL)**, uma métrica derivada da perda.
- **Loop de Treinamento Padrão:** O pré-treinamento usa um _training loop_ convencional de _deep learning_, com a **Cross-Entropy Loss** e o otimizador **AdamW**.
- **Estratégias de Decodificação:** Para gerar texto mais criativo e menos repetitivo, são usadas estratégias de amostragem probabilística, como o **escalonamento de temperatura** (para ajustar a aleatoriedade da distribuição de _tokens_) e o **Top-k Sampling** (para restringir a amostragem aos _k_ _tokens_ mais prováveis).
- **Prática Comum:** Devido à intensidade de recursos, carregar **pesos pré-treinados abertos** é o caminho mais prático para se obter um LLM funcional e poderoso para tarefas de _fine-tuning_.

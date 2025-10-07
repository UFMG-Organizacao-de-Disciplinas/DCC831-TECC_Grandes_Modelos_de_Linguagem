# Chapter 6: Finetuning for Text Classification

## 6.1 Different categories of finetuning

O fine-tuning (ajuste fino) de Large Language Models (LLMs) é a adaptação de um modelo pré-treinado para uma tarefa específica. As duas formas mais comuns de fine-tuning são o _Instruction Fine-tuning_ e o _Classification Fine-tuning_.

- **Instruction Fine-tuning (Ajuste Fino por Instruções):** Envolve treinar o modelo em um conjunto de tarefas usando **instruções específicas** para melhorar sua capacidade de entender e executar tarefas descritas em _prompts_ de linguagem natural.
  - É ideal para modelos que precisam lidar com uma **variedade de tarefas** e melhorar a flexibilidade e a qualidade da interação.
  - Modelos ajustados por instrução, como o FLAN, são mais versáteis, mas exigem conjuntos de dados maiores e mais recursos computacionais.
- **Classification Fine-tuning (Ajuste Fino por Classificação):** Envolve treinar o modelo para reconhecer um **conjunto específico de rótulos de classe**.
  - É ideal para projetos que exigem **categorização precisa** de dados em classes predefinidas, como análise de sentimentos ou detecção de spam.
  - O modelo ajustado por classificação é **altamente especializado**; ele só pode prever as classes que encontrou durante seu treinamento (por exemplo, "spam" ou "not spam") e não pode dizer mais nada sobre o texto de entrada.
  - Em contraste com o Instruction Fine-tuning, requer **menos dados** e poder computacional.
  - Neste capítulo, o exemplo concreto examinado é a classificação de mensagens de texto como "**spam**" ou "**not spam**".

<!--
Como treinar um modelo do zero é dispendioso, e retreinar um modelo grande pode ser inviável, o fine-tuning de LLMs pré-treinados é uma abordagem prática e eficiente. Ele geralmente resulta em duas categorias de modelos: os voltados a instrução e os voltados a classificação.

O primeiro é treinado para seguir instruções em linguagem natural, tornando-o versátil para várias tarefas, e para isso, precisa de um grande conjunto de dados e poder computacional. Já o segundo é especializado em categorizar dados em classes predefinidas, como "spam" ou "not spam", exigindo menos dados e recursos computacionais, mas sendo limitado às classes vistas durante o treinamento.
-->

## 6.2 Preparing the dataset

A preparação do conjunto de dados é o primeiro passo no processo de _Classification Fine-tuning_.

1. **Download e Carregamento:** O processo começa com o download de um conjunto de dados de mensagens de texto que consiste em mensagens "spam" e "não spam" (ham). O dataset original continha 5.572 linhas.
2. **Análise e Balanceamento de Classes:** O dataset original estava desbalanceado, contendo "ham" (4825) muito mais frequentemente do que "spam" (747).
   - Para simplificar e facilitar um fine-tuning mais rápido do LLM, é realizado um **undersampling** (subamostragem) para criar um conjunto de dados **balanceado** com 747 instâncias de cada classe.
3. **Conversão de Rótulos:** Os rótulos de classe de _string_ ("ham" e "spam") são convertidos em **rótulos de classe inteiros** (0 e 1, respectivamente).
4. **Divisão do Dataset:** O dataset balanceado é dividido aleatoriamente em três partes para: **treinamento (70%)**, **validação (10%)** e **teste (20%)**. Os subconjuntos são então salvos como arquivos CSV.

<!--
Definimos qual classificação desejamos fazer, e com isso escolhe-se um dataset adequado. No exemplo dado, foi usado um dataset que classifica mensagens de texto como "spam" ou "not spam" ("ham"). Primeiro as categorias precisam ser balanceadas, seja por undersampling (removendo itens das classes maiores) ou oversampling (duplicando os itens das classes menores). Os rótulos então são convertidos em valores inteiros (0 e 1), e o dataset é dividido em conjuntos de treino, validação e teste.
-->

## 6.3 Creating data loaders

Esta seção detalha a criação dos _data loaders_ do PyTorch para preparar os dados tokenizados para o treinamento em _batches_.

- **Padronização do Comprimento da Sequência:** Mensagens de texto têm comprimentos variados, mas o modelo requer que todos os exemplos em um _batch_ tenham o mesmo tamanho. A abordagem adotada é o **padding** (preenchimento): todas as mensagens mais curtas são estendidas para o comprimento da mensagem mais longa no conjunto de dados de treinamento.
- **Token de Padding:** O token ID **50256** (que corresponde a `<|endoftext|>`) é usado como o _padding token_. Este ID é adicionado ao final das sequências mais curtas até que elas atinjam o comprimento máximo do treinamento (que é de **120 tokens** para este conjunto de dados).
- **A classe `SpamDataset`** gerencia a tokenização, determina o comprimento máximo da sequência e aplica o _padding_ ou o _truncamento_ (se uma sequência for maior que o `max_length` definido).
- **Data Loaders:** São criados _Data Loaders_ para os conjuntos de treinamento, validação e teste, que carregam os dados em _batches_ (e.g., tamanho 8), onde cada _batch_ consiste em oito sequências de 120 tokens e seus respectivos rótulos de classe (0 ou 1).

<!--
Os dataloaders criam os batches onde cada um de seus exemplos de treino devem ter a mesma quantidade de tokens. Para isso, é necessário definir um tamanho máximo (max_length) e então aplicar padding (com o token `<|endoftext|>` que tem ID 50256) nas sequências menores que esse tamanho, ou truncar as sequências maiores. A tokenização é feita com o tiktoken usando o BPE. Lembrando que os batches são compostos por pares (input, target), onde o input é a sequência de tokens da mensagem a ser classificada e o target é o rótulo da classe (0 ou 1).
-->

## 6.4 Initializing a model with pretrained weights

Nesta etapa, o modelo pré-treinado é configurado para ser modificado e usado na tarefa de classificação.

- **Configurações e Carregamento:** O modelo GPT-like é inicializado usando as mesmas configurações usadas durante o pré-treinamento (e.g., `gpt2-small` com 12 camadas, 12 cabeças de atenção e dimensão de _embedding_ de 768). Os **pesos pré-treinados** são então carregados na arquitetura do LLM.
- **Verificação de Coerência:** É realizada uma verificação de geração de texto simples para garantir que os pesos foram carregados corretamente.
- **Tentativa de _Prompting_:** O modelo pré-treinado é testado com instruções (e.g., "O texto a seguir é 'spam'? Responda com 'sim' ou 'não'") para verificar se ele pode classificar o spam **sem fine-tuning**. O modelo **falha** em seguir a instrução e em fornecer uma resposta adequada, o que é esperado, pois ele só passou pelo pré-treinamento e **carece de _Instruction Fine-tuning_**. Esta falha confirma a necessidade de prepará-lo para o **Classification Fine-tuning**.
  Aqui está o resumo das seções 6.5 e 6.6, seguindo o fluxo do Capítulo 6:

<!--
Como explicado no capítulo anterior, para evitar retrabalho e gasto desnecessário de recursos, é comum carregar pesos pré-treinados abertamente disponíveis. Assim, o modelo GPT-like é inicializado com as mesmas configurações do pré-treinamento e os pesos são carregados. Para garantir que os pesos foram carregados corretamente, uma verificação simples de geração de texto é feita. Essa verificação falha, o que comprova a necessidade de fine-tuning, já que o modelo não foi ajustado para seguir instruções ou classificar textos.
-->

## 6.5 Adding a classification head

Esta seção foca na modificação da arquitetura do LLM pré-treinado para a tarefa de classificação:

- **Substituição da Camada de Saída:** O modelo GPT-like original possui uma **camada de saída linear** (`out_head`) que mapeia as representações ocultas (e.g., 768 unidades) para o **tamanho total do vocabulário** (50.257 tokens). Para classificação binária (spam/não spam), essa camada é **substituída** por uma nova camada de saída linear que mapeia as 768 unidades ocultas para **apenas duas unidades de saída** (classes 0 e 1).
- **Congelamento de Camadas:** Inicialmente, **todas as camadas** do LLM são tornadas **não-treináveis (congeladas)** para preservar o conhecimento pré-treinado e otimizar a eficiência computacional.
- **Descongelamento Seletivo:** A nova camada de saída (`model.out_head`) é treinável por padrão. Para melhorar o desempenho, o modelo também é configurado para treinar o **último bloco Transformer** (`model.trf_blocks[-1]`) e a **camada `LayerNorm` final** que se conecta à saída, enquanto as camadas anteriores permanecem congeladas.
- **Foco no Último Token:** O fine-tuning concentra-se exclusivamente na saída correspondente ao **último token** de entrada. Isso se deve ao **mecanismo de atenção causal** (usado em modelos GPT-like), que garante que o último token tenha acumulado a informação de **todos os tokens anteriores** na sequência, tornando-o o vetor de contexto mais informativo para a classificação.

<!--
Para adaptar o modelo pré-treinado à tarefa de classificação, precisamos modificar sua arquitetura. A camada de saída original, que mapeia para o tamanho do vocabulário, é substituída por uma nova camada que mapeia para duas classes ("spam" e "not spam" - 0 e 1), como usual, ela é iniciada com valores aleatórios que serão treinados. Como retreinar todas as camadas simultaneamente seria também muito custoso, todas as camadas são congeladas (definidas como não-treináveis) exceto a nova camada de saída, o último bloco Transformer e a camada LayerNorm final.

Por causa da atenção causal (Causal Attention - mascaramento dos tokens futuros), o último token de entrada é o que contém a informação de todos os tokens anteriores, então, para a classificação, apenas a saída referente ao último token é usada.
-->

## 6.6 Calculating the classification loss and accuracy

Esta seção implementa as utilidades de avaliação necessárias para o processo de fine-tuning:

- **Previsão de Rótulos:** Para converter os _logits_ (saídas) de 2 dimensões do último token em uma previsão de classe, usa-se a função **argmax**. A função `argmax` retorna o índice da saída com o maior valor, que corresponde ao rótulo da classe prevista (0 para "não spam", 1 para "spam").
- **Acurácia de Classificação:** A acurácia (medida no `calc_accuracy_loader`) mede a **porcentagem de previsões corretas** no conjunto de dados.
  - As acurácias iniciais antes do fine-tuning são próximas a **50%** (e.g., 46.25% no treinamento), indicando um desempenho aleatório, o que confirma a necessidade do fine-tuning.
- **Função de Perda:** A **acurácia não é uma função diferenciável**. Portanto, o treinamento minimiza a **perda de entropia cruzada (_cross-entropy loss_)** como um _proxy_ para maximizar a acurácia.
  - A função `calc_loss_batch` é ajustada para calcular a perda apenas com base nos _logits_ do **último token** de saída (`model(input_batch)[:, -1, :]`), alinhado com a estratégia de classificação.
  - Os valores de perda iniciais antes do treinamento (e.g., Perda de Treinamento 2.453 ) também confirmam que o modelo precisa ser fine-tuned.
    Aqui está o resumo final das seções 6.7, 6.8 e o sumário do Capítulo 6.

<!--
Para avaliarmos a performance do modelo, a partir dos _logits_ de 2 dimensões do último token, usamos a função argmax para obter o rótulo da classe prevista (0 ou 1). A acurácia é então calculada como a porcentagem de previsões corretas. Porém, apesar de querermos maximizar a acurácia, ela não é uma função diferenciável, então usamos a perda de entropia cruzada (cross-entropy loss) como um proxy para minimizar a perda e assim maximizar a acurácia. A função de cálculo de perda é ajustada para considerar apenas os _logits_ do último token.

Ao executar o teste de acurácia antes do fine-tuning, o valor resultante de acurácia é próximo de 50%, o que indica que o modelo está fazendo previsões aleatórias, confirmando a necessidade do fine-tuning. O valor da perda também é alto, reforçando essa necessidade.
-->

## 6.7 Finetuning the model on supervised data

Esta seção descreve a execução do **fine-tuning supervisionado** para melhorar a acurácia de classificação do modelo.

- **Função de Treinamento:** A função `train_classifier_simple` é utilizada, sendo muito similar à função de pré-treinamento, mas com duas modificações principais: ela rastreia o número de **exemplos vistos** (em vez de tokens) e calcula a **acurácia** (em vez de gerar texto de amostra) após cada época.
- **Otimizador e Treinamento:** Um otimizador (e.g., AdamW com $\text{lr}=5 \times 10^{-5}$) é inicializado, e o treinamento é executado por um número definido de épocas (e.g., 5). O processo envolve o _forward pass_, o cálculo da perda (entropia cruzada no último token) e o _backward pass_ para atualizar os pesos das camadas não-congeladas.
- **Resultados:** Ao longo das 5 épocas, a **perda (loss)** tanto de treinamento quanto de validação declina acentuadamente e a **acurácia** aumenta, atingindo pontuações elevadas (e.g., mais de 97% na validação). A proximidade das curvas de perda e acurácia de treinamento e validação indica que o modelo **não sofreu _overfitting_ significativo**.
- **Avaliação Completa:** A acurácia é calculada sobre os **conjuntos de dados completos** (treinamento, validação e teste), resultando em alta performance (e.g., Acurácia de Teste de **95.67%**) e confirmando o sucesso do fine-tuning.

<!--
Durante o fine-tuning, a função de treinamento considera a quantidade de exemplos vistos ao invés da quantidade de tokens vistos. A acurácia é calculada ao final de cada época. O otimizador AdamW é usado para atualizar os pesos das camadas não congeladas. Após 5 épocas, a perda de treinamento e validação diminui significativamente, enquanto a acurácia aumenta, atingindo mais de 97% na validação. A acurácia final no conjunto de teste é de 95.67%, indicando que o fine-tuning foi bem-sucedido.
-->

## 6.8 Using the LLM as a spam classifier

Com o modelo ajustado, a etapa final é utilizá-lo para a classificação de novos dados.

- **Função de Classificação:** A função `classify_review` é implementada. Ela pega um novo texto de entrada, aplica os mesmos passos de pré-processamento (tokenização, truncamento e _padding_).
- **Previsão:** O texto tokenizado é passado ao modelo para inferência (sem cálculo de gradiente). O modelo gera os _logits_ do último token, e o **argmax** é usado para obter o rótulo da classe prevista (0 ou 1).
- **Validação em Exemplos:** O modelo fine-tuned classifica corretamente mensagens de teste como "spam" ou "not spam".
- **Salvamento do Modelo:** Os pesos finais do modelo (`state_dict`) são salvos (e.g., em um arquivo `.pth`) para que o modelo possa ser reutilizado posteriormente sem a necessidade de um novo treinamento.

<!--
Por fim, com o modelo treinado, podemos usá-lo para classificar novos textos. A função de classificação tokeniza o texto de entrada, aplica truncamento e padding, e então passa o texto pelo modelo para obter os _logits_ do último token. O argmax desses _logits_ fornece o rótulo da classe prevista (0 ou 1). Testes com mensagens de exemplo confirmam que o modelo classifica corretamente "spam" e "not spam". Os pesos finais do modelo são salvos para uso futuro.
-->

## Summary and takeaways

- Existem diferentes estratégias para fine-tuning de LLMs, incluindo o **Classification Fine-tuning** e o **Instruction Fine-tuning**.
- O Classification Fine-tuning adapta o LLM **substituindo sua camada de saída** (que antes tinha o tamanho do vocabulário, 50.257) por uma **pequena camada de classificação** (e.g., 2 nós de saída para "spam" ou "not spam").
- O modelo é treinado para prever um **rótulo de classe correto** (em vez do próximo token).
- O treinamento utiliza o texto convertido em IDs de token, de forma similar ao pré-treinamento, mas com a adição de _padding_ para uniformizar o tamanho das sequências.
- O **modelo pré-treinado** é carregado como modelo base antes do fine-tuning.
- A **acurácia de classificação** é a métrica principal de avaliação, e a **perda de entropia cruzada** é usada como a função de perda a ser minimizada durante o treinamento.

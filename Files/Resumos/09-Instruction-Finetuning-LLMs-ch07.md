# Chapter 7: Finetuning To Follow Instructions

O fluxo para o **Instruction Fine-tuning (IF)**, também conhecido como **Supervised Fine-tuning (SFT)**, começa definindo a meta do processo e, em seguida, preparando cuidadosamente o _dataset_ para garantir que o modelo aprenda a formatar e responder corretamente às instruções.

## 7.1 Introduction to Instruction Fine-tuning

O **Instruction Fine-tuning** é um processo crucial que transforma um Large Language Model (LLM) pré-treinado, que é primariamente um **modelo de complementação de texto**, em um modelo capaz de **seguir instruções humanas**.

- **O Problema do LLM Pré-treinado:** LLMs pré-treinados (como o GPT-2 base) aprendem a **prever o próximo token** e são bons em **completar sentenças ou escrever parágrafos**. No entanto, eles tipicamente **não conseguem seguir instruções específicas** como "Converta esta frase para voz passiva" ou "Corrija a gramática".
- **O Objetivo do IF:** O objetivo é treinar o LLM para aceitar uma **instrução como _input_** e gerar uma **resposta desejada**, atuando como um **chatbot** ou **assistente pessoal**.
- **O Processo:** O IF é classificado como _supervisionado_ porque o treinamento ocorre em um _dataset_ onde os pares **(instrução, resposta)** são fornecidos explicitamente.

## 7.2 Preparing a Dataset for Supervised Instruction Fine-tuning

A preparação do _dataset_ é um aspecto chave do fine-tuning. O processo envolve formatar os dados de forma que o LLM entenda o limite entre a instrução e a resposta esperada.

### Estrutura do Dataset e Formatação

1. **Aquisição de Dados:** O processo começa com o download de um _dataset_ em **formato JSON**, que consiste em **1.100 pares** de instrução-resposta.
2. **Estrutura de Entrada:** Cada entrada é um objeto que contém as chaves **`instruction`**, **`input`** (campo opcional que pode estar vazio) e **`output`** (a resposta desejada).
3. **Prompt Styles (Estilos de Prompt):** Para que o LLM entenda o papel de cada parte da entrada, o dado é formatado usando um _prompt style_ específico.

   - **Estilo Alpaca:** Este estilo estruturado, que é adotado no capítulo, utiliza delimitadores de _markdown_ para definir as seções:

     ```text
         Below is an instruction that describes a task. Write a response that appropriately completes the request.

         ### Instruction:
         [Conteúdo da Instrução]
         ### Input:
         [Conteúdo do Input, se houver]
         ### Response:
         [Conteúdo do Output Desejado]
     ```

   - **Estilo Phi-3:** Emprega um formato mais simples com _tokens_ designados: `<|user|>` e `<|assistant|>`.

4. **Lógica da Formatação:** Uma função `format_input` é usada para gerar a _string_ formatada, e ela é projetada para **ignorar a seção `### Input:`** caso o campo `input` original do _dataset_ esteja vazio.

### Particionamento do Dataset

O _dataset_ total de 1.100 entradas é dividido em três subconjuntos para garantir que o modelo seja treinado e avaliado em dados não vistos:

| Conjunto        | Porcentagem | Tamanho (Entradas) | Propósito                                                                          |
| :-------------- | :---------: | :----------------: | :--------------------------------------------------------------------------------- |
| **Treinamento** |   $85\%$    |       $935$        | Onde o modelo aprende ativamente as instruções.                                    |
| **Teste**       |   $10\%$    |       $110$        | Usado para a avaliação final e quantificação da _performance_.                     |
| **Validação**   |    $5\%$    |        $55$        | Usado para monitorar a _performance_ durante o treinamento e evitar _overfitting_. |

O fluxo para o **Instruction Fine-tuning (IF)**, também conhecido como **Supervised Fine-tuning (SFT)**, começa definindo a meta do processo e, em seguida, preparando cuidadosamente o _dataset_ para garantir que o modelo aprenda a formatar e responder corretamente às instruções.
You've covered the foundation of Instruction Fine-tuning (IF) and dataset preparation. The next steps involve transforming the prepared data into batches for efficient training, setting up the data loaders, and finally, loading the foundational large model.

## 7.3 Organizing Data into Training Batches

The process of organizing data involves crucial steps to convert formatted text entries into numerical, uniform batches that the LLM can process.

### The Five Substeps of Batching

Since LLM instruction fine-tuning requires specialized handling, a **custom collate function** is used instead of the standard PyTorch one. This function performs five key substeps:

1. **Format and Tokenize (Steps 2.1 & 2.2):** The instruction-response entry is first structured using a prompt template (e.g., Alpaca style) and then converted into a sequence of **token IDs**. This is typically done within an `InstructionDataset` class.
2. **Padding (Step 2.3):** Sequences within a batch are padded to the length of the longest sequence in that batch to ensure uniformity. The **end-of-text token** (`<|endoftext|>`), corresponding to token ID **50256**, is used as the **padding token**.
3. **Create Target IDs (Step 2.4):** A corresponding sequence of **target token IDs** is created. These targets are the **inputs shifted one position to the right**, allowing the LLM to learn to predict the next token, similar to pre-training.
4. **Mask Padding Tokens (Step 2.5):** All but the first instance of the padding token (ID 50256) in the **target sequence** are replaced with the value **$-100$**.
   - **Why $-100$?** The PyTorch cross-entropy loss function is configured by default to ignore any targets labeled with **$-100$** (`ignore_index=-100`).
   - **The Purpose:** This masking ensures that the loss calculation **excludes padding tokens**, while keeping the first `50256` token so the model learns when to generate the **end-of-text** token, indicating a complete response.
   - _(Note: Masking the instruction tokens as well is optional and sometimes debated, but generally not applied here.)_

## 7.4 Creating Data Loaders for an Instruction Dataset

The custom batching function is now integrated into PyTorch DataLoaders to automate the feeding of data to the model.

1. **Device Preparation:** The code determines the optimal computational device (e.g., `cuda`, `cpu`, or `mps`). The custom collate function is enhanced to **move data to the target device during the batching process**, which allows this transfer to run as a background process, improving efficiency.
2. **Partial Function Application:** The `custom_collate_fn` is "pre-filled" with arguments like the target `device` and an `allowed_max_length` (e.g., **1024**, matching the GPT-2 context size) using `functools.partial`.
3. **Data Loader Instantiation:** The training, validation, and test sets are wrapped in `InstructionDataset` objects, and then loaded into `DataLoader`s, supplying the `customized_collate_fn`.
4. **Verification:** The resulting batches from the data loader have a uniform batch size (e.g., 8) but correctly demonstrate **variable sequence lengths** across different batches, avoiding unnecessary padding for the entire dataset.

## 7.5 Loading a Pretrained LLM

The final step before starting the fine-tuning training loop is loading the foundational model.

1. **Model Selection:** Instead of the smallest $124$-million-parameter model used in previous chapters, the **GPT-2 medium model** with **355 million parameters** is selected. This is because the smaller model **lacks the capacity** to effectively learn and retain the nuanced behaviors required for high-quality instruction following.
2. **Model Loading:** The necessary configuration (vocabulary size of 50,257, context length of 1024) and weights are loaded, a process similar to pre-training and classification fine-tuning. The medium model requires about **1.42 GB** of storage.
3. **Baseline Assessment:** The **pretrained model performs poorly** on instruction-following tasks. When given an instruction (e.g., convert a sentence to passive voice), it often simply repeats the original input sentence or parts of the instruction, failing to generate the desired response. This establishes a clear **baseline** to measure the effect of the upcoming fine-tuning process.

## 7.6 Fine-tuning the LLM on Instruction Data

Esta seção foca na execução do treinamento do modelo pré-treinado GPT-2 medium (355M) usando o _dataset_ de instruções preparado.

### O Processo de Treinamento

O fine-tuning em si **reutiliza as funções de perda e treinamento** implementadas em capítulos anteriores (como as funções `calc_loss_loader` e `train_model_simple`).

- **Loss Inicial:** O _loss_ inicial de treinamento e validação é alto (cerca de $3.8$), o que é esperado antes de o modelo começar a aprender a seguir instruções.
- **Hiperparâmetros:** É utilizado o otimizador **AdamW** com uma taxa de aprendizado ($lr$) de $0.00005$ e `weight_decay` de $0.1$.
- **Execução:** O modelo é treinado por **duas épocas**.
- **Monitoramento:** A **perda de treinamento e validação diminui consistentemente** ao longo das épocas, indicando que o modelo está aprendendo de forma eficaz. O _loss_ cai abruptamente no início da primeira época e estabiliza, sugerindo que o _fine-tuning_ foi eficiente e mais épocas poderiam levar a um _overfitting_.
- **Teste em Contexto:** Durante o treinamento, a resposta gerada para uma tarefa de validação (ex.: converter uma frase para voz passiva) muda de uma **repetição do _input_** (antes do _fine-tuning_) para uma **resposta correta**, confirmando que o modelo adquiriu a capacidade de seguir a instrução.

## 7.7 Extracting and Saving Responses

Após o treinamento, o modelo é usado para gerar respostas para todo o _test set_ (_held-out_) para posterior avaliação.

- **Geração de Respostas:** O modelo gera respostas para cada entrada do _test set_ usando a função `generate`.
- **Extração da Resposta:** A função `generate` retorna o texto de _input_ e _output_ combinados. Para isolar a resposta, o texto de entrada (`input_text`) é subtraído do texto gerado (`generated_text`), e a _string_ "### Response:" é removida.
- **Qualidade Qualitativa:** Em exemplos visuais, o modelo se mostra **relativamente bom**, fornecendo respostas corretas ou muito próximas, como a conversão de sentenças ou perguntas de conhecimento.
- **Salvamento de Dados:** As respostas geradas pelo modelo são adicionadas ao dicionário `test_data` e salvas no arquivo **"instruction-data-with-response.json"** para registro e análise futura.
- **Salvamento do Modelo:** O modelo _fine-tuned_ é salvo como **gpt2-medium355M-sft.pth** para ser reutilizado.

## 7.8 Evaluating the Finetuned LLM

A avaliação de um LLM _fine-tuned_ para instruções é mais complexa do que uma simples métrica de acurácia (como na classificação de spam). Esta seção implementa uma **avaliação automatizada** usando um modelo de linguagem mais capaz como juiz.

### Métodos de Avaliação de LLMs (Contexto)

O modelo pode ser avaliado por:

- **Benchmarks de Múltipla Escolha/Resposta Curta:** Como MMLU (_Measuring Massive Multitask Language Understanding_), que testam o conhecimento geral do modelo.
- **Comparação Humana:** Como LMSYS chatbot arena, que compara a preferência humana entre respostas de diferentes LLMs.
- **Benchmarks Conversacionais Automatizados:** Onde um LLM maior (como GPT-4 ou Llama) atua como avaliador (ex.: AlpacaEval).

### Avaliação Automatizada com Ollama e Llama 3

Adota-se uma abordagem similar ao AlpacaEval, usando um LLM localmente executável como juiz.

1. **Ferramentas:** O processo utiliza o **Ollama** (um _wrapper_ eficiente para rodar LLMs) para interagir com o modelo **Llama 3** (8 bilhões de parâmetros), que atua como avaliador.
2. **Mecanismo de Scoring:** O LLM avaliador é _promptado_ com a **instrução**, a **resposta correta do _dataset_**, e a **resposta do modelo fine-tuned**. Ele é instruído a **atribuir uma pontuação de 0 a 100** (onde 100 é o melhor).
3. **Resultado e Métrica:** Ao processar todas as $110$ entradas do _test set_, o modelo _fine-tuned_ alcança um **score médio** quantitativo.
4. **Conclusão da Avaliação:** O modelo _fine-tuned_ GPT-2 medium (355M) atinge um **score médio de $50.32$**. Este valor serve como um _benchmark_ para comparar com outros modelos ou futuras modificações no treinamento. - (Para referência, o modelo Llama 3 8B base atinge $58.51$, e o Llama 3 8B _instruct_ atinge $82.65$ no mesmo _test set_).

Aqui está a continuação do fluxo lógico para o processo de **Fine-tuning para Seguir Instruções**, abordando o treinamento, a extração de respostas e a avaliação final.

A jornada pelo Capítulo 7 conclui o ciclo essencial de desenvolvimento de um LLM: desde a arquitetura básica até o fine-tuning para tarefas específicas.

## 7.9 Conclusions

### 7.9.1 What's Next

O **Instruction Fine-tuning** para criar um assistente pessoal marca o fim dos estágios principais do desenvolvimento de um LLM. No entanto, existe uma **etapa opcional** que pode ser seguida para aprimorar ainda mais o modelo: o **Preference Fine-tuning**.

- **Objetivo:** O Preference Fine-tuning é particularmente útil para **personalizar um modelo** para que ele se alinhe melhor com **preferências específicas do usuário** ou normas sociais.
- **Implementação:** Essa técnica ajusta o modelo para que ele aprenda quais respostas são **"melhores"** ou **"preferidas"** em um par de opções, o que é um passo além de apenas fornecer uma resposta correta. Um exemplo comum dessa abordagem é o **Direct Preference Optimization (DPO)**.

### 7.9.2 Staying Up to Date in a Fast-Moving Field

O campo de pesquisa em LLMs e IA está em constante e rápida evolução. Para se manter atualizado, você pode:

- **Acompanhar Pesquisas:** Explorar artigos recentes no **arXiv**, especialmente na seção `cs.LG` (Machine Learning).
- **Engajar com a Comunidade:** Muitos pesquisadores e profissionais compartilham e debatem os últimos desenvolvimentos em plataformas de mídia social, como o **Reddit** (sub-reddit **r/LocalLLAMA**) e **X** (antigo Twitter).
- **Ferramentas Populares:** Para aplicações no mundo real com modelos mais poderosos, considere ferramentas de _fine-tuning_ conhecidas, como **Axolotl** ou **LitGPT**.

### 7.9.3 Final Words

Implementar um LLM "do zero" é a maneira mais eficaz de obter uma **compreensão profunda de como eles funcionam**.

- O aprendizado forneceu uma base sólida em todo o ciclo de desenvolvimento: desde a implementação da arquitetura, passando pelo pré-treinamento e terminando no _fine-tuning_ para seguir instruções.
- Embora este livro tenha fins educacionais, o conhecimento adquirido permite a transição para a utilização de LLMs mais robustos para aplicações práticas e futuras inovações em IA.

## Summary and Takeaways

O Capítulo 7 detalha o processo de **Instruction Fine-tuning (IF)**, adaptando um LLM pré-treinado para se tornar um assistente capaz de seguir instruções.

### Processo e Metodologia

- **Finalidade do IF:** O processo adapta um LLM pré-treinado para **seguir instruções humanas** e gerar respostas desejadas, transformando-o em um **assistente pessoal**.
- **Preparo do Dataset:** A base é um _dataset_ de pares instrução-resposta, que é formatado (ex.: estilo **Alpaca**) e dividido em conjuntos de treino, validação e teste.
- **Lógica de Batching:** _Batches_ de treino são construídos usando uma **função _custom collate_** que realiza três tarefas principais:
  - **Preenchimento (_Padding_):** Garante que as sequências no _batch_ tenham o mesmo comprimento.
  - **Criação de Targets:** Gera os IDs de _target_ (deslocados em $+1$ do _input_).
  - **Mascaramento (_Masking_):** Substitui os tokens de _padding_ adicionais por **$-100$** no _target_, excluindo-os do cálculo do _loss_ (com o `ignore_index` do PyTorch).
- **Modelo Base:** Um **GPT-2 medium (355 milhões de parâmetros)** é carregado como ponto de partida, pois o modelo _small_ (124M) carece da capacidade necessária para obter resultados satisfatórios.
- **Treinamento:** O modelo é _fine-tuned_ no _dataset_ de instruções usando um _training loop_ padrão, com o _loss_ diminuindo significativamente ao longo de **duas épocas**.
- **Avaliação:** A _performance_ é quantificada de forma automatizada. Um LLM mais capaz (ex.: **Llama 3 8B instruct via Ollama**) é usado para **pontuar as respostas** do modelo _fine-tuned_ no _test set_, resultando em um score médio de referência (ex.: $\sim 50.32$ para o modelo treinado).

## What's Next?

O _Instruction Fine-tuning_ completa o ciclo essencial de implementação e treinamento de um LLM. O próximo passo lógico na cadeia de desenvolvimento para aprimorar a utilidade do modelo é o **Preference Fine-tuning**.

- **Preference Fine-tuning:** Esta é uma etapa opcional que refina o LLM para se **alinhar com as preferências humanas** ou normas sociais, indo além da simples correção e buscando a "melhor" ou "preferida" resposta.
- **Tópicos Avançados:** Outras áreas para exploração incluem:
  - **LoRA (Low-Rank Adaptation):** Uma técnica de **Parameter Efficient Fine-tuning (PEFT)** que permite treinar modelos grandes de forma mais eficiente, atualizando apenas um pequeno subconjunto de parâmetros.
  - **Modelos e Arquiteturas Avançadas:** Explorar implementações de modelos mais recentes e poderosos, como o **Llama 3.2**.
  - **Multi-Modalidade:** A aplicação do _fine-tuning_ de instruções a dados que combinam modalidades (como imagem e texto).

# Aula 09 - Professor: Anísio

## Slide

### Language Models as multitask assistants?

### Today (17h21 - 06min40seg)

- ...
- Lora

### How do humans learn?

### Humanos vs ML

- Humanos: instruções detalhadas, poucos exemplos
- ML: Sem instruções, muitos dados de treino

> Como que instruções ajudam no aprendizado?

### O que são essas instruções?

- Amazon Mturk Guidelines: ?
- LLM-oriented...?

### Como resolvemos o problema de análise de sentimentos?

- Podemos considerar que é um problema de classificação
  Podemos então prever o próximo token apenas com uma limitada quantidade de tokens.

### Pattern-Exploiting Training (PET) (Schick and Schütze, 2020/2021)

Pares de (Pattern, Verbalizer).

### Human-oriented instructions

- Task Instructions
  - Definition
  - Positive Examples
  - Negative Examples

Natural-instructions (Mishra et al., 2022; Wang et al., 2022)

### Instruções

- Transformer Architecture
  - Infill masked words
    - Enconders, masked LMs
  - Auto-regressivavamente predizem texto
    - Decoders, causal LMs
    - Encoder-decoders, conditional LMs

(17h30min - 12min)

Gerar complemento de texto é diferente de responder a uma instrução. A aula de hoje é sobre as instruções; A próxima aula é sobre o ajuste de normas sociais.

### Nessa aula (17h32)

...

### Language modeling $\neq$ following instructions

Usar o GPT sem instruções, ele vai só completar texto com alguma coerência, mas que não segue a instrução colocada.

Por que ele não respondeu bem?

- Porque ele não sabe entender as instruções, apenas sabe completar texto. Ele inclusive não sabe exatamente quando parar após já ter concluído a tarefa.

### Instruction tuning

- Pretrain-finetune (BERT, T5)
- Prompting (GPT-3)
- Instruction Tuning (FLAN)

A ideia é seguir esse último que parte de uma treinada e conseguir fazer algo para a qual não foi treinada antes.

### T5

- Text-to-text

### Instructions finetuning

devemos coletar perguntas/tarefas e testar para outras não vistas.

### Benchmarks for multitask LMs

Massive Multitask Language Understanding (MMLU) (Hendrycks et al., 2021)

"Quanto mais diverso...?"

### Some intuition: examples from MMLU (17h42min)

### Natural instructions

(17h44): pergunta de aluno

Existem vários datasets para finetuning

### The FLAN collection

Medir o teste durante o treino não ajuda muito.

### Instruction Tuning: Example

Depois do finetuning, como chegamos a uma resposta de instrução?

### Scaling Instruction-Tuning

Rouge-L vê se tem interseção...

Quanto maior os treinos, melhor o desempenho.

O mesmo ocorre com a quantidade de parâmetros do modelo.

Porém aumentar a quantidade de instâncias por treino não é tão significativo;

Dúvida (17h49): Será que ele pode acabar fazendo overfit por tarefa?

### Natural Instructions

Dúvidas (17h51min)

### SUpernatural Instructions

### Scaling Instruction Tuning (17h54)

Diferentes tamanhos de modelos, ao aumentar a quantidade de tarefas de finetune, melhores ficam.

### What have we learned from this?

Como fizeram o Alpaca 7B?

Fizeram um dataset sintético.

- Generate instructions, input and output from a LM
- You don't need many samples to instructions tune

### Let's build from scratch (17h58)

Revisão importante.

- Pegamos texto
- Tokenizamos
- ...
- Mecanismo de atenção
- Precisamos entender o que é o forward e backward
- Alguma coisa de Loss

Queremos mudar nosso modelo. Ele ainda deve ter entrada e saída, mas precisamos mudar algo.

pra isso...

---

1. Precisamos de dataset para extrairmos padrões dele.
2. .
3. .
4. .
5. .
6. .
7. .
8. .
9. Dar score pras respostas

Minha dúvida (18h05)

A loss é uma forma matemática de avaliar os pesos. Há uma distância entre a avaliação real da resposta e a avaliação feita pela LOSS.

### Listings 7.1 Downloading the dataset (18h07)

Formatações diferentes de prompts iniciais.

### Listings 7.2 ? (18h09)

### Listings 7.3 Partitioning the dataset (18h10)

### Tokenizador (18h11)

Converte o json em um formato markdown. Depois converte em embeddings.

---

Aqui iremos usar tokens especiais que ajudarão a definir o final da resposta. E todos os batches devem ter o mesmo tamanho, então serão preenchidos com tokens de padding.

(Minha dúvida 18h20)

Em algum momento comparamos a loss com o Y predito.

Usando o -100 ele aprende a parar, embora eu não tenha entendido bem como.

Para alguns Ys, não queremos atualizar algo.

---

(18h26) Já foram feitos testes para verificar se mascarar as instruções afeta o desempenho?

E não há consenso sobre.

---

Cada batch pode ter quantidade de tokens diferentes, mas dentro dele, cada um dos exemplos deve ter a mesma quantidade de tokens.

Análise: que diferença faz usar batches em que as instruções estão agregadas em batches de tamanhos diferentes?

### 5. Finetuning (18h28)

### 6. (18min31)

Como avaliar e quantificar a qualidade da resposta?

### 7. (18min32)

Fazer um comparativo entre a resposta certa e a resposta dada pode ter uma interseção grande mas ser bem diferente.

### Model Evaluation is harder than completion fine-tuning (18h36)

- Arenas de LLMs
- Usar um LLM mais poderoso pra avaliar

No livro ele pega um modelo mais poderoso, carrega, e deixa esse modelo avaliar o modelo que foi treinado.

---

Minha dúvida (18h39): mas e se o modelo mais poderoso também errar? E se ele alucinar e começar a avaliar errado?

Resposta: Ainda não se sabe. Essa questão está em aberto.

### Other Data and Models

## Resumo

## Referências

## Próxima Aula

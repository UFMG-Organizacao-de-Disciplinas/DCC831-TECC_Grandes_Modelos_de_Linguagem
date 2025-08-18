# Aula 02 - XXX

## A

---

Pra quem tá começando na área, ele sugere brincar com os modelos abertos. Não é muito científico, mas é uma boa prática na área

- Módulos distintos:
  - Reasoning
  - Múltiplos agentes

## Developing an LLM

- Stage 1: Building
  - Initial model
- Stage 2: Pre-training
  - Foundation Model
- Stage 3: Post-Training
  - Adapted Model
- Stage 4: More Specialization
  - Web search LLMs
  - RAGs
  - Topic chatbots
  - Multimodal LLMs
  - Code assistants
  - reasoning models
  - Agents
  - Distilled models

Ele hoje focará na pipeline em azul (os subtópicos)

### Stage 1: Building

O modelo inicial consiste em um código com vários pesos

- The model is simply...

Os LLMs apenas preveem a próxima palavra. E essa tarefa é treinada usando a Sliding Window alguma coisa que funciona como um aprendizado supervisionado. Ele passa a ser autosupervisionado porque os labels já estão prontos em textos presentes na internet.

#### Batch

Dúvida: por que usar esses segmentos ao invés de ter apenas uma lista?
Resposta: paralelismo. Poder colocar esses batches em processos diferentes é útil.

Dúvida: não seria mais denso enviar todo o texto e definir os batches por índices?
Reposta: ter segmentos com intercessão podem gerar overfitting. E ele pretende falar mais osbre adiante.

#### Tokenization

Converter conjuntos de caracteres a IDs.

##### Subword-level tokenization

Veremos mais pra frente como e porquê isso é feito.

Um dos motivos é permitir que as LLMs lidem com tokens nunca vistos antes.

#### GPT-3 was trained on 0.5T tokens

| Dataset | Quantity (tokens) | ... | ... |
| ------- | ----------------- | --- | --- |

...

#### Llama 1 was trained on 1.4T tokens

| Dataset | Quantity (tokens) | ... | ... |
| ------- | ----------------- | --- | --- |

...

#### Llama 2 was trained on ... T tokens

#### Llama 3 was trained on 15T tokens

Llama, Gemma e GPToss são modelos com pesos abertos.

Podemos alterá-los, mas não se sabe como que esses pesos foram gerados.

#### Llama 4 was trained on 30T tokens

Se treinar em quaisquer dados, o aprendizado pode não ser bom mesmo com grande volume. Uma alternativa é filtrar apenas dados de boa qualidade e boa proveniência.

#### Quantity vs quality

Um dos métodos é usar o Textbook learning que se baseiam em textos mais didáticos.

Modelos nano conseguem ser rodados no notebook. O modelo alguma-coisa-3M roda até em celular.

#### LLM Architecture

Focaremos na arquitetura do

A palavra vira um número, o número vira um embedding, ele é marcado para saber a sua posição na entrada, essa entrada é então aprimorada pelo Transformer.

- Dúvidas:
  - 1: ?
  - 2: É melhor pegar um pré-treinado ou a gente treinar do zero?
    - R: ?
  - 3: O positional embedding era estática. Num curso que ele fez, isso se tornou parametrizável... [Me perdi]
  - 4: Como faz para definir o erro da previsão do modelo?
    - R: o certo era o que concentrava a massa na palavra correta.

---

- GPT2
- Llama 2
- Llama 3
- Llama 3.1
- Llama 3.2

[Ele falou algo sobre rotação mas eu me perdi: 41min]

Geralmente usam RELU como função de ativação. GPT usa GELU

Usam métodos de normalização que no geral querem Média 0, desvio 1.

Não necessariamente melhoram o modelo, mas são mais baratas. E como são usadas em grande escala, qualquer desconto vale.

Ao invés da atenção densa de todos pra todos, pode-se usar a atenção esparsa onde apenas alguns são usados.

---

Passaram a usar um ensemble de reforço (?) [45min]

---

Espera-se que até o final do semestre "estejamos craques" no GPT-OSS 120B

Implementar um Transformer no Excel :eyes:

#### Takeaway: Not "that much" has changed

### Stage 2: Pre-training (Foundation Model)

#### Pretraining at a high level

- Dúvida: por que treinam com GPU e não CPU?

#### LLM are deep neural networks

- ...

---

Pretty standard deep learning training loop

Labels are the inputs shifted by +1

[A, B, C] -> [B, C, D]
Por que isso e não "[A, B, C, D] + function(lista, shift=1)"?

- Poderiam botar mais dados de trinamento

---

- Training for ~1-2 epochs is usually a good sweet spot

[Imagem: Loss x Epochs] Quanto mais treino, mais overfitting.

GPT-2 é bom para fins didáticos.

Nvidia divulgou GPUs poderosas quase como um HD externo para esse tipo de processamento.

#### What makes it hard, then?

Talvez estejam comprando mais do que a produtora de GPU consegue prover.

[Ele falou algo que foge do escopo por falta de material. Oq? 56min]

#### Scaling choices

- Goal: maximize model performance
  - tempo de treinamento, tamanho do dataset e quantidade de parâmetros

#### Compute-optimal models

- [Imagens de modelos]
  - Geralmente, ou com o tamanho do dataset ele não tinha um modelo grande o bastante; ou o oposto.

...

#### Memory needs beyond parameters

---

Deepseek: tiro na Nvidia.

Desejo de exploração na disciplina: quais os ganhos alcançáveis em otimizações?

[Ideia: tipos diferentes de promps serem mais eficientes]

#### Increasing price and complexity due to

- Larger models
- Larger datasets
- Multi-stage training

---

Llama 3 Herd of Models.

Talvez falemos sobre instabilidade no processo de [?] [1h08min]

#### Should you pretrain models? Nope

- É caro e não trivial. Provavelmente alguém torraria grana com isso antes de conseguir render algo bom o bastante.

#### Loading pretrained weigths

Llama não permite retreinamento

litgpt

### Stage 3: Post-Training (Adapted model)

#### Instruction tuning [1h12min]

```json
{
  "instruction": "...",
  "input": "...",
  "output": "..."
}
```

---

---

Still a next-token prediction task?

- Dúvida: por que foi necessário reformatar?
- Resposta: não tem certeza, mas chuta que possa ser pela forma como o tokenizador funciona.

---

#### Preference tuning (aka alignment)

Ajuste fino para se adequar a preferências humanas.

##### Example: alignment via RLHF (Reinforced Learning [alguma coisa] Human Feedback) [1h18min]

- Step 1

---

- Step 2: Rank responses

---

- Step 3: Train reward model

- podem usar uma outra LLM que ranqueia. Tipo um modelo de ranking pairwise.

---

- Step 4: Prefere-tune the LLM

- Usa um algoritmo de aprendizado por reforço para calcular o quão bem o modelo tá performando, assim atualizando os pesos da LLM para que depois ela responda de forma mais alinhada.

- Dúvida: isso seria destilação?
- Resposta: sim, e o que tem de diferente é a ideia da aprendizagem por reforço.

- Dúvida: Como as métricas são calculadas? [1h24min]
- Resposta: basicamente é um regressor. E ele aprendeu de forma supervisionada.

Deepseek usou GRPO.

#### Stage 4: More specialization

Próximos slides: como interagir com LLMs prontas?

Ele tentará anexar o teórico com o prático.

## Summary [1h26]

- Pretraining from scratch: almost never necessary
- Continued pretraining: expand knowledge
- Finetuning: special use case, follow instructions
- Alignment: improve helpfulness + safety

## Referências

- Raschka (2024)
- Raschka (2025)

## Dúvidas finais

- Alguma coisa sobre multimodalidade

## Próxima aula: [Prompting][LinkProximaAula]

[LinkProximaAula]: <>

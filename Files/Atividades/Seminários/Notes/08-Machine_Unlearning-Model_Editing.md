# Model Unlearning - 04/11/2025

## ?

## Motivação

- Objetivo: LGPD: direito de apagar dados pessoais
- Ou então Copyright

---

Por que Model Unlearning?

## Abordagens

### Unlearning Exato (SISA)

Reduz o custo de retreinamento em 1/N.

Primeiro subdivide os dados de treino, vê onde está o dado a ser esquecido, depois retreina apenas esse bloco.

O Algoritmo é a prova do unlearning. O dado nunca será usado para inferências.

Difícil de escalar.

### Unlearning Aproximado (DP)

- Privacidade Diferencial: dificuldade de diferenciar dois registros na base de dados. Não conseguindo diferenciar, não conseguiríamos inferir o dado a ser esquecido.
- DP-SGD
- Drawbacks: Não é possível provar o esquecimento.
- Unlearning Empírico com exemplos em espaço conhecido.
- A interpolação de conceitos associados a um tema pode acabar fazendo esquecer mais coisas do que o desejado.

### Unlearning Aproximado (EEX)

- Descida em gradiente.
- Drawbacks: Precisa de acesso a todo dataset. É difícil separar o que deve ser desaprendido. Como definir o que é "tóxico"?

---

Como esquecer o máximo que puder?

### Unlearning Aproximado (ENE)

- Ainda não conseguem fazer de forma eficiente.

### Unlearning Aproximado (ICU)

- PEdir para as LLMs esquecerem.
- In-Context Unlearning: são poderosas o suficiente para fingir que esqueceram.
- Drawbacks: a pessoa pode fazer um jailbreak e retorna o modelo a fazer o que foi solicitado a não fazer.

## Como avaliar?

- Eficiência
- Utilidade: tarefas ortongonais são danificadas?
- Qualidade de esquecimento: Realmente foram esquecidos?
- Benchmarks: TOFU e WMDP
  - Treina com autoros fictícios, depois pede para esquecer um autor.

## Perspectivas Futuras

- O que é fácil ou difícil de esquecer?

## Dúvidas

- E de que forma a Interpretabilidade se conecta com o Unlearning? A Interpretabilidade Mecanística pode ajudar a entender o que o modelo aprendeu e, assim, facilitar o processo de unlearning

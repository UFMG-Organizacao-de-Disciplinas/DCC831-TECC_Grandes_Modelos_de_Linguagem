# Grupo 14: Interpretability

## Motivações

## Dictiorany Learning

### Decompor palavras em fatores

#### Factores

- Cada coluna é uma coluna do embedding
- Cada um dos pesos tá associado a cada fator.
- O número de fatores é maior que a dimensão dos embeddings. Assim, mais de um fatores são associados a uma mesma dimensão.

#### Spectral Clustering

#### Decomposing Words

#### Vector + Factor

- $V_{man} - V_{woman} = V_{king} - V_{queen}$
- $V_{|Kingle} - V_{Book} = V_{iPad}$

## Space Autoencoders (SAE)

...

---

...

## Evaluation 1: "Autointerpretability"

- Testar se os modelos conseguiriam prever qual eram os fatores associados a cada palavra.

---

- Space Coding
- ICA
- Identity ReLU
- PCA
- Random

## Evaluation 2: Causality bia Activation Patching

Checar se os logits de saída são parecidos

## Outro artigo

### Motivation

Attribution Pathing (AtP)

### Activation Patching

- Compara um prompt correto com um prompt correto, ativando um nó baseado no outro.

Problema: Brute-force AP is expensive

Estimativa de ativação de nós

#### Limitações: Non-liinearities

QK Fix: recomputa o Attention Block

#### Diagnostics - How much are we missing?

### Key Results

- AtP\*: Reduz falso negativos
- QK fix
- Grad Drop

## Dúvidas

Há algum consenso do que pode ser usado para interpretabilidade

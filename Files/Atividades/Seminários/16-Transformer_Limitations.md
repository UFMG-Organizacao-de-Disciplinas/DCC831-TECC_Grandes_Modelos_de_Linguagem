# 06/11/2025

## Introdução

- Transformers dominaram. BERT e GPT principalmente.

## Artigos

### On Limitations of the Transformer Architecture

- Ideias lógicas

#### Causa

#### Chain of Thought

#### Memória

### Representational Strengths and Limitations of Transformers

- Quais os pontos fortes dos Transformers?

---

qSparse Averaging (qSA)

Análogo ao que é o score de atenção

---

Transformers (Eficiência Logarítmica)

MLPs (Crescimento Polinomial)

Recorrentes (RNNs)

---

Papel crítico da dimensão dos embeddings. Se ele for grande o bastante, ele escala bem.

---

Match3(X)

---

- Priorize Agregação Flexível e Longa DistÂncia
- Dimensione o Embedding pela Complexidade
- Use a Estrutura para Superar Limitações

### Theoretical Limitations of Self-Attention in Neural Sequence Models

- Será que os transformers de self-attention são capazes de reconhecer as limitações?
- Resposta: não completamente.

---

- O que o artigo testa:
  - Parity
  - 2dyck

---

- Mesmo transformer sendo poderoso, ele não tem memória infinita nem lê passo a passo, então perde capacidade de entender ordem e capacidade.
- Se limitados, como performam bem? Na prática frases resumidas representam bem.

---

- Comparação entre arquiteturas

---

Síntese: São bons mas não perfeitos. Não entendem hierarquias profundas e estruturas complexas/aninhadas

### Transformers Learn Shortcuts to Automata

- Semiautômato
  - Conjunto de estados
  - Alfabeto
  - Função de transição

---

RNN x Transformers

- RNN: O(T)
- Transformer: O(L) processa em parelelo nas L camadas de atenção.

---

Resultados principais

- Simulam autômato com profundidade menor que T usando paralelismo
- Alta precisão
- Alta variância

---

Resultados principais

## Lição principais

Eles são bons mas limitados. Não adianta só aumentar o tamanho do modelo. É importante tentar fazer estruturas mais inteligentes, como Chain of Thought ou subestruturas.

## Dúvida

- (On Limitations... Architecture) Em alguns casos ele gera código python, roda, e mostra o resultado. Talvez escrever em linguagem de programação lógica não resolveria esse problema?
  - R: A parte lógica é uma representação do problema, não necessariamente como ele ocorre, então rodar linguagem de programação lógica não necessariamente ajudaria.
- (Representational Strengths...) Foi comentado sobre tarefas em que o transformer performa bem. Foi visto algo sobre tarefas de alta complexidade polinomial? Eles performariam melhor?

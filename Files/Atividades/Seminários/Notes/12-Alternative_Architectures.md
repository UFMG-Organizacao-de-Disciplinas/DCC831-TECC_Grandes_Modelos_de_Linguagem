# Arquiteturas Alternativas - 30/10/2025

## 1. Differential Transformers

- Problema a ser resolvido: o mecanismo de atenção (e o uso do softmax) do transformer gera muito ruído e em contextos longos o sinal correto se dilui.
- Objetivo: cancelar ruido subtraindo duas atenções softmax.
- Resultados: cancelando ruído eles conseguem melhorar a eficiência do modelo, chegando nos mesmos resultados necessitando de menos parâmetros.

Usam um raciocínio similar ao de cancelamento de ruídos.

## 2. MAMBA

- Motivação: Transformers têm complexidade quadrática em tempo e memória. O MAMBA propõe uma arquitetura linear.
- O MAMBA comprime os dados de entrada, mantendo apenas as informações mais relevantes.
- Treino paralelizável.

---

- CNN: paralelizável mas não guarda input de longo prazo.
- RNN: teoricamente guarda os inputs
  - Difícil treinamento
  - Não paralelizável
- Linear RNN: lineares recorrentes alternadas com redes neurais
  - Paralelizável
  - inicialização estável
  - RNN que lembra contexto de maior distâncias
- SMM (State Space Models)
- MAMBA: SSM
  - Perto de 1 lembra, perto de 0 esquece
  - Aumenta dimensionalidade do output
- Selective SSM: Mamba
  - Variações: NLP, Vision, Tabular (MambaTab), Time Series, Graph, Audio and Speech, Medical

### Performance Evaluation

- Timeseries: Mamba ganha
- Visão: acurácia competitiva com menos FLOPs
- NLP: menor log-perplexity, mas perde nas outras métricas

### Limitações

- Linear mas não utiliza menos GPU
- Desempenho incerto em modelos grandes e sequências longas
- ...

## 3. Vision-RWKV

- RWKV: se propõe a fazer uma atenção linear e paralelizável.
  - Acumula os pesos recursivamente.
  - Complexidade linear, paralelizável.
  - Assim como o MAMBA, é uma alternativa aos transformers sem utilizar o mecanismo de atenção.
- VWRKV: alternativa ao Vit
  - Avaliação de imagens de alta resolução

---

A diferença do RWKV pro Mamba é que o RWKV é uma rede recorrente com escalares que armazenam o estado, enquanto o Mamba é baseado em State Space Models (SSM) que utilizam matrizes para capturar dependências de longo prazo. Aparentemente o MAMBA tem mais dimensões (?)

### Mudanças-chave

- Bi-WKV
- Q-Shift
- Estabilidade ao escalar

### Resultados

- Acurácia similar

## ?

### Ideia Central: Dualidade

- Paralela//Recorrente

### Como funciona na Prática?

- Multi-Scale Retention (MSR)
- Estabilidade
- Chunkwise (treino longo)

### Qualidade (LM) e Ablations

- Melhor que transformers após 2B parâmetros

### Limitações e praticalidades

- Se teoricamente superam os transformers, por que não são usadas?
  - Os ambientes geralmente são usados considerando transformers, essa migração acaba sendo não sendo utilizada.

## Dúvidas

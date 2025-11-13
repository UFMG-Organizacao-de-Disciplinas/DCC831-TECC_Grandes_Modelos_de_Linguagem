# Arquiteturas Alternativas - 30/10/2025

## 1. ??

Usam um raciocínio similar ao de cancelamento de ruídos.

### Experimentos

Usa 65% dos recursos de um transformer usual. Menos parâmetros e menos tokens.

## 2. MAMBA

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

## Vision-RWKV

- RWKV: se propõe a fazer uma atenção linear e paralelizável.
  - Acumula os pesos recursivamente.
- VWRKV: alternativa ao Vit
  - Avaliação de imagens de alta resolução

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

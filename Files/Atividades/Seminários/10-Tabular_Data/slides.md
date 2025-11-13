# Slides

## LLMs and Tabular Data

### Características dos Dados Tabulares

- Estrutura: linhas e colunas, dados heterogêneos (numéricos, categóricos, textuais)
- Esparsidade e valores ausentes
- Interdependência entre atributos
- Ordem invariante (não sequencial)
- Falta de conhecimento estrutural prévio
- Tipos: relacionais, hierárquicos, layout/visuais
  > (Fang et al., 2024; Sui et al., 2024; Dong, 2024)

### Técnicas Tradicionais

- Modelos baseados em árvores: GBDT, XGBoost, LightGBM, CatBoost
- Redes neurais: Wide&Deep, DeepFM, TabTransformer, TabNet, SAINT
- Modelos generativos: Copulas, VAEs, GANs
- Limitações: engenharia de features, generalização, dados esparsos
  > (Fang et al., 2024)

### Aplicações usuais das LLMs

- Compreensão e geração de texto
- Aprendizado em contexto (ICL), Chain-of-Thought
- Instrução-tuning e multitarefas
- Agentes autônomos e ferramentas externas
  > (Fang et al., 2024; Dong, 2024)

### LLMs com Dados Tabulares

- Serialização: JSON, Markdown, HTML, CSV
- HTML/XML → melhor desempenho por exposição prévia
- Prompt engineering e few-shot examples
- Fine-tuning / Instruction-tuning: QA, Text2SQL, Data Generation
- Modelos: TABERT, TAPAS, TableGPT, TableLlama, TableLLM
  > (Sui et al., 2024; Fang et al., 2024; Dong, 2024)

### Estado da Arte

- Benchmarks: SUC Benchmark > (Sui et al., 2024), TableLLM, TableGPT3
- Desempenho: GPT-4 > GPT-3.5, mas ainda limitado
- Abordagens:
  - Self-augmentation
  - Retrieval-Augmented Generation (RAG)
  - Multimodalidade (texto + imagem)
    > (Sui et al., 2024; Dong, 2024)

### Desafios

- Limite de contexto e custo computacional
- Alucinação e inconsistência estrutural
- Serialização sensível ao formato
- Falta de benchmarks padronizados
- Raciocínio quantitativo deficiente
- Escalabilidade e interpretabilidade
  > (Fang et al., 2024; Sui et al., 2024; Dong, 2024)

### Trabalhos Futuros

- Modelos híbridos (LLM + GBDT)
- Representações numéricas aprimoradas
- Benchmarks padronizados e datasets abertos
- LLM-agents com raciocínio iterativo
- Multimodalidade (texto, imagem, tabela)
- Self-verification e auto-raciocínio
  > (Fang et al., 2024; Sui et al., 2024; Dong, 2024)

### Conclusão

- LLMs ampliam o potencial sobre dados estruturados
- Persistem limitações em precisão e compreensão estrutural
- Tendência: integração multimodal, interpretabilidade, padronização
  > (Fang et al., 2024; Sui et al., 2024; Dong, 2024)

# Notes

## Large Language Models for Tabular Data - Progresses and Future Directions

- fórmulas não participam do entendimento semântico
- prompting vs training
- reasoning, generation
- representações das tabelas
- compressões
- benchmarks
- structured (SQL), semi-structure (word), image (PDF)
- serialization: CSV, MD, JSON, XML, HTML
- foreign key 👍, explicações 👎
- DBCopilot: text-to-sql
- 1000+ tokens 👎
- DB-GPT
- Tabular Language Models (TaLMs)
- TaLMs atuais escolhem não manter o mesmo modelo
- Table-GPT, TableLLM, Table Llama
- Imagem: Multimodal. Ainda pouco explorado
- Knowledge in table: column type prediction, entity linking, relation identification, table retrieval.
- benchmarks
- agents

## [Table Meets LLM - Can Large Language Models Understand Structured Table Data - A Benchmark and Empirical Study][Artigo 2]

[Artigo 2]: <./Table Meets LLM - Can Large Language Models Understand Structured Table Data - A Benchmark and Empirical Study.pdf>

- [GitHub](http://github.com/microsoft/TableProvider)

### Abstract

- Melhorias em Few-shot
- Falta de estudo em verificação de entendimento da serialização tabular
- Serialização vs Tabela
- Benchmark para avaliar a SUC (Structural Understanding Capability)
- 7 Tasks
  1. Cell Lookup
  2. Row Retrieval
  3. Size Detection
- Tests on GPT-3.5, GPT-4
- Performance depended on: Table input format, content order, role prompting, partition marks (?)
- Proposes self-augmentation for structural prompting (Aparentemente é tipo o Chain-of-Thoughts)
- Tabular Tasks:
  - TabFact
  - HybridQA
  - SQA
  - Feverous
  - ToTTo

### 1. Introduction

- Structured data: plan text blocks organized by predefined structures to compress recurring information
  - Ex: tables
- Sem estudos prévios que examinaram a habilidade dos LLMs em entender dados tabulares estruturados ou suas limitações.
  - Eles tentam corrigir isso.
- Dificuldades: tabelas diferentes estruturam os dados de modos distintos, nem sempre tendo conversão direta pra forma sequencial
  - (Ideia JV: usar algum esquema de embedding de posição que é repetido para os itens das linhas)
- A serialização é flexível e não há consenso.
  - TaPEx: Tokens for headers and rows;
  - Tabbie: serializa linhas e colunas
  - Table-GPT: Concatena células de forma textual (ex: "Name: John" seria "Name is John")
- Pergunta de pesquisa
  - Quais designs de input e escolhas são as mais efetivas em permitir que LLMs entendam dados tabulares?
- Proposed Structural Understanding Capability (SUC) benchmark
  - Prompt variantes:
    - Input format
    - format explanation
    - Role prompting
    - Partition Mark
    - Zero-shot
    - One-shot
  - self-augmentation prompting: motiva LLM a gerar conhecimento intermediário.
- Conclusões:
  1. LLMs entendem de forma básica a estrutura mas não perfeitamente mesmo em tarefas simples como detectar o tamanho da tabela
  2. Escolher corretamente o formato de input pode aumentar significativamente o entendimento
  3. O Self-Augmented Prompting (SAP) é simples e aprimora o entendimento estrutural.
  4. Propõem usar linguagem de marcação como HTML com explicação de formatos e marcação de partições (ex: linhas, colunas) em conjunto com o SAP.
- Contribuições:
  1. Proposta do SUC benchmark
  2. Experimentos compreensivos
  3. SAP em 5 datasets

### 2. Preliminaries

- 2.1 Table Structure
  - (41) Flexibilidade das tabelas
    - Relational Tables
    - Entity Tables
    - Matrix Tables
    - Layout Tables
    - ...
    - Flat vs Hierarchical
    - Formatação: texto, número, data, hora, fórmula
    - Meta informações: cabeçalhos, notas, legendas
    - Número: relações matemáticas como soma ou proporção
  - Distanciamento das tabelas pra linguagem natural.
- 2.2 Table Serial & Splitting
  - Tabela $\to$ Texto sequencial
  - Serialização linha a linha: TaPas, MATE, TableFormer, TUTA, TURL
  - Tokens especiais (`<head>`, `<row>`): TaPEx
  - serialização por linhas e serialização por colunas: Tabbie
  - Pares chave-valor: Table-GPT
  - Self-attention:
    - Limites do mecanismo
      - complexidade quadrática
    - Soluções Antigas:
      - Truncagem
    - Problema:
      - perda de informação estrutural
    - Soluções Novas:
      - Random row sampling
      - 1-shot example based on remaining tokens

### 3. SUC Benchmark

#### 3.1 Structural Understanding Capabilities (SUC)

| Stages              | Capabilities                     | Tasks                        |
| ------------------- | -------------------------------- | ---------------------------- |
| Partition & Parsing | Structural Description/Detection | Table Partition              |
| Partition & Parsing | Format Understanding             | Table Size Detection         |
| Partition & Parsing | Format Understanding             | Hierarchy Detection          |
| Search & Retrieval  | Grounding/Location               | Cell Lookup & Reverse Lookup |
| Search & Retrieval  | Operation Reasoning              | Column & Row Retrieval       |

1. Partition & Parsing
   - Usos de informações extras:
     - Informações textuais: HybridQA
     - Anotações humanas: TabFact, FEVEROUS
     - Imagens: MultiModalQA
   - Formatos diferentes comprimem textos de formas diferentes:
     - CSV: vírgulas
     - XML: Tags
   - LLMS devem primeiro entender a estrutura.
   - Input Designs
     - Partition Mark
     - Role Prompting
     - Order Permutation
     - Format Explanation
     - Serialization
       - Natural Language + Separators
       - Markup Language (HTML, XML, Markdown, ...)
2. Search & Retrieval
   - É importante que a LLM entenda quais são as informações relevantes e que devem ser utilizadas para que a resposta seja correta.
   - Distinguir o processo de busca dos demais auxilia a entender o processo de aprendizado das LLMs sobre as tabelas.

#### 3.2 Task Design

- Table Partition: definir as extremidades (boundaries) da tabela informando os tokens de início e fim.
- Table Size Detection: definir os valores $m$ e $n$ de uma tabela com $m$ linhas e $n$ colunas.
- Merged Cell Detection: estrutura especial presente em tabelas, usualmente hierárquicas. Deve retornar o índice de linha e coluna onde há células mescladas.
- Cell Lookup & Reverse Lookup:
- Column & Row Retrieval:

## Large Language Models (LLMs) on Tabular Data - Prediction, Generation, and Understanding - A Survey

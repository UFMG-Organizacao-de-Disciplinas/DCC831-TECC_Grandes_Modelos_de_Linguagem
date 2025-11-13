# Alucinação - 04/11/2025

- Extrínseco: Não verificável
- Intrínseco: Verificável

## Taxonomia

- Alucinação por factualidade
- Alucinação ...

### Alucinação por factualidade

#### COntradição Factual

#### Contradição Fabricada

O que seria isso de "não ser verificável?"

### Faithfulness Hallucination

#### Inconsistência de Instrução

- Não responde a instrução, mas uma sub pergunta.

#### Context Inconsistency

## Relevância

## Causas

- Proveniente dos Dados

  - Memorização de falsidades
  - Vieses sociais
  - Conhecimento de cauda longa
  - Conhecimento desatualizados
  - Conhecimento Restrito

- Proveniente do treinamento

  - Viés de exposição: erro inicial se propaga após geração de token inicial incorreto.
  - Incapacidade de admitir não saber
  - Excesso de instrução
  - Sycophancy: proriza agradar, não a verdade.

- Proveniente de Inferência
  - Alta temperatura de amostragem: respostas menos precisas
  - Foco Local: Incoerência factual
  - Maldição da reversão: erra relações lógicas inversas

## Detecção

- Alucinações Factuais
  - Verificação de fatos: fontes confiáveis e verificação interna
  - Estimação de Incerteza: Usar outra LLM ou medir entropia
- Alucinação de fidelidade
  - Verificar se a resposta é consistente com o contexto

## Mitigar Alucinações

- Mitigar Alucinações + RAG (Limites e Boas Práticas)
- Mitigação no nível de dados
  - O que fazer?
    - Atualização factual
  - Quando usar?
- Mitigação no Treinamento/Alinhamento
  - Boas Práticas
  - Efeito Esperado: Menos invenção
- Mitigação na Inferência/Decodificação
  - Técnicas

## Por que o RAG ainda alucina?

- Fonte vs memória
- Esquecimento de instrução
- Contexto ruim
- Decodificação criativa

---

- Estratégias de RAG
  - One-time Retrieval
  - Iterative Retrieval
  - Post-hoc Retrieval

---

- Boas práticas
  - Recuperar melhor: reescrever consulta, rerankers, BM25 + denso
  - Gerar com base: Citação obrigatória...

## Limitaçõess

- Falta de Benchmarks generalistas
- Limites do conehcimento dos LLMs: não se sabe onde ela tá geometricamente alucinando.
- LLMs não distinguem entre crenças e fatos
- Falta método para identificar mentiras.
- A alucinação não é só em texto.

## Dúvidas

Diferença entre fact checking e hallucination detection?

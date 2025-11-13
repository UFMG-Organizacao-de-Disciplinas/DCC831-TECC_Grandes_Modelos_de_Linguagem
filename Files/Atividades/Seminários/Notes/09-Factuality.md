# Grupo: ? - Factuality

## Introdução

## O que é factualidade?

- A capacidade de um modelo gerar conteúdo que adere a conteúdo verificável e verdadeiro.

## Por que isso é relevante?

- Riscos sociais: Efeito Halo (confiança excessiva na expertise do modelo)
- Undersourcing
- Usos maliciosos

## O que é alucinação?

Desvio de conhecimento anterior e algo já pré estabelecido

## Factualidade vs Alucinação

### O que isso pode gerar?

Mesmo ele falando coisas que façam sentido, o contexto tá errado.

### Exemplo de grau bom de factualidade

### Exemplo de grau ruim de factualidade

## Como avaliar factualidade?

1. Problemas de Múltipla escolha
2. Sim ou Não
3. Respostas Curtas
4. Geração Aberta

---

Avaliar sentenças atômicas com bancos de dados.

## Como os LLMs "guardam" fatos?

- Camada de feed-forward
- Encodam conhecimento conceitual similar ao humano
- Causal Tracing

### Causal Tracing

1. Forwarding de 2 frases:
   - Factual
   - Contra factual
2. Mede ativações em ponto específico
3. Injeta factual na contrafactual e vê se o logit do correto aumenta.

---

Ajuda na explicabilidade e auditoria. Não retreinar o modelo inteiro.

## Causas de erros de factualidade

- Arquitetura do modelo
- Recuperação de informação
- Inferência incorreta

## Como melhorar factualidade?

- Melhorar pré-treino: dados melhores, generalização melhor
- Pré-treino contínuo: iterativo, especializado por domínio

---

- RAG (Retrieval Augmented Generation)

---

- Recuperação baseada em CoT (Chain of Thought)
- Adição de etapas de reasoning
- Recuperação baseada em agentes

---

- Fine-tuning: injeção direta de informações
  - Grafo de Conhecimento (KG): Predição de partes ocultas da tripla
  - Aprendizado por contraste: distinguir fatos verdadeiros e falsos
  - Aprendizado por feedback humano (RLHF)

## Maiores desafios

- As LLMs preveem o próximo token, não a verdade.

## Perspectivas Futuras

- Entender como LLM armazenam e acessam conhecimento

## Dúvidas

- Não factual vs alucinada:
  - Não factual: informação não verificável
  - Alucinada: informação baseada no que aprendeu

Modelo Cut-Off 2022: pedir pra ele prever o presidente dos EUA em 2025.

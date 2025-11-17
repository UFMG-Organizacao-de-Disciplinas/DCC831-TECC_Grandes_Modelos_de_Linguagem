# 06/11/2025

## Domínios Verticais

Áreas específicas com termos próprios e problemas específicos. Ex.: Medicina, direito, finanças, etc.

## LLMs Generalistas

Apesar de performarem razoavelmente bem, são passíveis de alucinação. Em situações mais específicas onde o resultado é crítico, é necessário um modelo especializado para amenizar riscos.

## LLMs Especialistas

Modelos de linguagem treinados em dados específicos de um domínio vertical, visando melhorar a precisão e a relevância das respostas em contextos especializados.

### Vantagens da Especialização

- **Conformidade Regulatória:** Tende a responder baseado nas normas esperadas do domínio.
- **Vocabulário Adequado:** Utiliza termos técnicos corretos, assim melhorando a comunicação.
- **Menos "achismo" e mais confiabilidade:** Reduz a probabilidade de alucinações e respostas incorretas, visto que pode se alçar em casos já vistos no treinamento.

### Desafios para especialização

- **Dados:**
  - Escassez, enviesados, desatualizados
  - Risco de aprender padrões indesejados
  - Não generalização, performando mal em casos novos
  - Atualização constante para não ficar obsoleto
- **Custo:**
  - Computacional
  - Profissionais capacitados

### Estratégias de adaptação de LLMs para especialização

- **Transfer Learning:** reaproveita modelos já treinados como base
- **LoRA Adapters:** adiciona camadas treinadas
- **Prompt Engineering:** ajusta o comportamento via prompts
- **RAG:** conecta o modelo à bases de conhecimento externas
- **Fine-Tuning:** refina parâmetros para dados específicos

#### Especialização via Promp Engineering

- Técnicas avançadas de Prompting:
  - **Chain of Thought (CoT):** ajuda o modelo a gerar raciocínios passo a passo, performando melhor em tarefas complexas.
  - **Self-Consistency:** gera múltiplas respostas e escolhe a mais comum, aumentando a confiabilidade.

---

- Direcionando o comportamento do modelo com contexto e Estrutura
  - Instruções claras para guiar o modelo a performar conforme esperado.
  - Usam zero/few-shot learning para fornecer exemplos específicos do domínio.
  - Sem treinamento adicional
  - Limitado a tarefas simples e ainda tendo lacunas.
  - Não há alteração de pesos/Não é vertical

#### Especiaização via RAG (Retrieval-Augmented Generation)

- Combina LLMs com sistemas de recuperação de informação.
- Pode retornar dados errôneos caso a consulta seja mal formulada.
- Não há alteração de pesos/Não é vertical

#### Especialização via Transfer Learning

- Usando o conhecimento geral da LLM para domínios especializados
  - Usa modelos já treinados como base
  - Performa melhor à medida em que os parâmetros aumentam

#### Especialização via Fine-Tuning

- Começa com modelo pré-treinado e ajusta os pesos com dados específicos do domínio.
- Usa perguntas/comandos/descrições de tarefas e suas respostas.

#### Especialização via Adapters/LoRA

(Ignorando o que tá no slide)

- Os adapters são camadas intermediárias pequenas inseridas no modelo pré-treinado para que elas sejam treinadas com dados específicos do domínio, mantendo os pesos originais do modelo fixos.
- Já o LoRA (Low-Rank Adaptation) é uma técnica que modifica o modelo na parte das projeções $W_q$, $W_k$ e $W_v$ das camadas de atenção, permitindo uma adaptação eficiente com menos parâmetros.

## Avaliação e Alinhamento com Especialistas

- Quem deve avaliar o modelo são especialistas do domínio, não usuários.

## Estudo de caso: LLMs para Medicina

- Domínio de alto risco

### Large Language Models Encode Clinical Knowledge (Google Research, DeepMind)

- State-of-the-art (SOTA) Benchmarks via Instruction Tuning
  - Flan-PaLM 540B, usando CoT e Self-Consistency prompting atingiu SOTA
  - Med-PaLM: intruction prompt tuning para alinhar o Flan-PaLM para o domínio médico
    - Aumento de acerto no consenso científico e redução de respostas danosas

### Performance of ChatGPT on USMLE: Potential for AI-Assisted Medical Education Using Large Language Models

- Modelagem de prompts
- GPT4 > Med-PaLM; GPT4 > GPT3.5
- GPT4 é bom com multimodalidade no contexto da medicina
- Calibração é fundamental: confiança do modelo em acertar deve estar alinhada com a probabilidade real de acerto
- Raciocínio Qualitativo:
  - Explica bem o raciocínio médico
  - Metacognição/Empatia: supõe possíveis erros e retorna feedback encorajador
  - Contrafactuais: responde bem a perguntas hipotéticas

---

- Limitações e Direções
  - CoT não resultou em melhorias significativas
  - Few-shot não teve grande impacto
  - Possibilidade de memorização dos dados

### Semantic Clinical Artificial Intelligence (SCAI-RAG) vs Native LLM Performance on the USMLE

- Baseado em LLaMA

(Cansei)

---

...

---

1. X
2. X
3. Análise da Pergunta
4. Recuperação Direcionada (RAG)
5. Geração da Resposta por ?

---

Desempenho do USMLE

---

Pontos críticos e desafios

- Dependência da Fonte de Conhecimento
- Risco de Conhecimento Desatualizado
- Custo Computacional do RAG
- Exclusão de Questões Visuais
- Generalização do Método

## Cenário Atual LLMs para Medicina: Cuidado e Vigilância

Riscos e Benefícios:

- Riscos de Erro
- Riscos de Viés
- Questões sociais e profissionais
- Potencial a Longo Prazo

## Futuro dos LLMs verticais

- Melhores
- Mais seguras
- Mais baratas

## Conclusão

## Dúvidas

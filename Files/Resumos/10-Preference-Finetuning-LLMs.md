# Resumo: Preference Finetuning para LLMs (RLHF e DPO)

O resumo a seguir aborda os principais pontos sobre o **Preference Finetuning** para Large Language Models (LLMs), com foco nas técnicas de **RLHF** e **DPO**.

## O que é Preference Finetuning?

**Preference Finetuning** (Ajuste Fino por Preferência) é o processo de adaptar LLMs para que eles sigam melhor as **preferências humanas** em vez de apenas o padrão dos dados brutos de pré-treinamento.

- Vai além da simples previsão da próxima palavra.
- O modelo aprende o que os humanos preferem ler ou usar.
- Utiliza **sinais de feedback** para guiar a geração do modelo, como comparações pareadas, classificações ou críticas.
- É crucial para o **Alinhamento** de LLMs, garantindo que as saídas do modelo sejam **úteis, honestas e inofensivas** ("helpful, honest, and harmless").

### Limitações do Instruction Finetuning (Ajuste Fino por Instrução)

O ajuste fino por instrução tem limitações que o Preference Finetuning visa resolver:

- É **caro** coletar dados de "verdade absoluta" (ground-truth) para muitas tarefas.
- Tarefas criativas e abertas (open-ended) **não têm uma resposta "certa" única**.
- O modelo de linguagem penaliza todos os erros de token igualmente, mas **alguns erros são piores do que outros**.

## Principais Métodos de Ajuste por Preferência

### 1. Reinforcement Learning from Human Feedback (RLHF)

O RLHF (Aprendizado por Reforço a partir de Feedback Humano) é a técnica tradicionalmente usada para alinhamento, notavelmente em modelos como o InstructGPT e o ChatGPT.

**Fluxo do RLHF:**

1. **Coleta de Feedback Humano:** Humanos comparam pares de respostas geradas pelo LLM e escolhem a que é "melhor". Comparações pareadas são consideradas mais confiáveis do que pontuações diretas.
2. **Treinamento do Reward Model (RM):** Um modelo de recompensa $R(s;p)$ é treinado com base nesses dados de comparação para produzir uma **recompensa escalar** (um único número) para qualquer saída. O objetivo é aumentar a diferença na recompensa prevista entre a amostra "vencedora" ($s^+$) e a "perdedora" ($s^-$).
3. **Otimização por Aprendizado por Reforço (RL):** O modelo de linguagem (a "política" $\pi$) é ajustado para maximizar a recompensa esperada fornecida pelo RM.
4. **Regularização (KL-Penalty):** Uma penalidade é adicionada à recompensa para evitar que o modelo otimize demais o RM gerando texto de baixa qualidade (conhecido como "reward hacking"). Essa penalidade mede a divergência Kullback-Leibler (KL) entre o modelo RL e o modelo pré-treinado inicial.
   $$\hat{R}(s;p):=R(s;p)-\beta\log\left(\frac{p^{RL}(s)}{p^{PT}(s)}\right)$$
   O algoritmo mais popular para esta etapa é o **Proximal Policy Optimization (PPO)**.

**Desvantagens do RLHF:**

- É complexo e pode ser instável numericamente.
- É intensivo em computação e memória.
- O treinamento online pode ser lento e a performance é sensível aos hiperparâmetros.
- Fácil de fazer _overfit_ no modelo de recompensa.

### 2. Direct Preference Optimization (DPO)

O DPO (Otimização por Preferência Direta) é uma alternativa mais recente e mais simples ao RLHF.

**Como Funciona o DPO:**

- **Elimina o Reward Model:** O DPO elimina a necessidade de treinar um modelo de recompensa separado e de usar algoritmos de Aprendizado por Reforço.
- **Otimização Direta:** O DPO transforma o problema de alinhamento em um problema de **classificação binária** e otimiza os parâmetros do LLM diretamente nos dados de preferência (pares de resposta preferida/rejeitada).
- **Simplicidade e Eficiência:** O DPO é considerado **offline, mais simples, estável** e **computacionalmente mais leve** do que o RLHF.

**Vantagens do DPO:**

- **Mais Simples:** Não requer ajuste fino de um RM ou amostragem (sampling) durante o ajuste fino, tornando-o mais fácil de implementar.
- **Performance:** Demonstrou alcançar performance comparável ou melhor que a do PPO em algumas tarefas, como utilidade de sumarização e diálogo.
- **Adoção:** Está permitindo que modelos de código aberto melhorem seu alinhamento.

## Comparação Chave

| Característica           | RLHF                                             | DPO                                               |
| :----------------------- | :----------------------------------------------- | :------------------------------------------------ |
| **Requer Reward Model?** | Sim, treina um RM explícito                      | Não, otimiza o LLM diretamente                    |
| **Tipo de Otimização**   | Aprendizado por Reforço (PPO)                    | Otimização por preferência direta (classificação) |
| **Complexidade**         | Mais complexo, instável, intensivo em computação | Mais simples, estável, leve em computação         |
| **Uso de Dados**         | **Online** (gera amostras em tempo real)         | **Offline** (apenas nos dados de preferência)     |

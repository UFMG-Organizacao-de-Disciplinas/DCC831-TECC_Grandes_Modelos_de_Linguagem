# Anotações gerais sobre termos recorrentes

## Vertical Domains

- **Chain-of-Thought (CoT):** Técnica de prompting que incentiva o modelo a gerar raciocínios passo a passo, melhorando o desempenho em tarefas complexas.
- **Shots:** Exemplos fornecidos no prompt para guiar o modelo (zero-shot, one-shot, few-shot).
- **RAG (Retrieval-Augmented Generation):** Técnica que combina LLMs com sistemas de recuperação de informação para melhorar a precisão das respostas.
- **Fine-Tuning:** Processo de ajustar os pesos de um modelo pré-treinado com dados específicos de um domínio.
- **Transfer Learning:** Técnica que reutiliza modelos pré-treinados como base para treinar em novos domínios.

## Alternative Architectures

- **Negative Log-Likelihood (NLL):** métrica padrão para avaliar o nível de incerteza do modelo em prever a próxima palavra.
- **Recurrent Neural Network (RNN):** arquitetura que processa sequências de dados mantendo um estado interno.
  - $h_t = f(h_{t-1}, x_t)$
- **MAMBA:** é visto como uma evolução dos modelos RNN, utilizando State Space Models (SSM) para capturar dependências de longo prazo de forma mais eficiente.
- **State Space Models (SSM):** modelos matemáticos que descrevem sistemas dinâmicos em termos de estados internos e observações. O cálculo é linear.
  - $h_t = Ah_{t-1} + Bx_t$
  - $y_t = Ch_t$
- **Selective SSM:** variação do SSM que permite ao modelo decidir quais informações manter ou esquecer, melhorando a eficiência em tarefas específicas.
- **Mean Squared Error (MSE):** métrica que mede a média dos quadrados dos erros entre valores previstos e reais.
- **Mean Absolute Error (MAE):** métrica que mede a média dos valores absolutos dos erros entre valores previstos e reais.
- **Floating Point Operations (FLOPs):** medida da quantidade de operações de ponto flutuante que um modelo realiza, usada para avaliar a eficiência computacional.

## Causality

- **Structural Causal Model (SCM):** modelo matemático que representa relações causais entre variáveis usando grafos direcionados. A representação em grafos auxilia nos cenários contrafactuais.
- **Causal Representation Learning (CRL):** A representação em questão é a codificaçã ode um conceito em seus pesos. Seu objetivo é que esses conceitos sejam desemaranháveis.
- Distributed Interchange Intervention:
- Interchange Intervenction Accuracy (AII):
- Boundless Distributed Alignment Search (Boundless DAS):
- **Sparse Autoencoders (SAEs):** decompõem alta dimensionalidade em outras dimensões.
- Instrumental Variable Learning (IVL)
- Out Of Distribution (OOD)

## Model Uncertainty

- **Uncertainty Quantification (UQ):** técnicas para medir a incerteza nas previsões de modelos de linguagem.
  - **Ex:** O modelo diz que tá 90% certo. Idealmente esse valor deve estar condizente com o real.
- **Conformal Prediction (CP):** técnica estatística que fornece garantias de cobertura para predições, útil para medir a incerteza.
  - **Ex:** O modelo afirma qual é o intervalo de valores onde há uma chance estatística considerável de que a resposta correta esteja presente.
- **Mechanistic Interpretability (MI):** análise dos componentes internos do modelo (neurônios, camadas) para entender como ele processa informações e toma decisões.
- **Conformal Risk Control (CRC):** técnica que ajusta predições para garantir que o risco de erro esteja dentro de limites aceitáveis. Ao invés de garantir cobertura, garante que o risco seja controlado.
- **Conformal Language Modeling (CLM):** aplicação de técnicas de predição conformal especificamente em modelos de linguagem para melhorar a confiabilidade das respostas geradas.

## Modern Transformers

- **Mixture-of-Experts (MoE):** arquitetura que utiliza múltiplos "especialistas" (sub-modelos) para melhorar a capacidade do modelo, ativando apenas alguns especialistas para cada entrada.
- **Multi-Head Latent Attention (MLA):** variação da atenção multi-cabeça que incorpora representações latentes para capturar dependências mais complexas (projeta K e V em um espaço latente menor).
- **Group Query Attention (GQA):** técnica de atenção que agrupa as matrizes de K e V para várias queries, reduzindo o armazenamento.
- **Rotary Positional Embeddings (RoPE):** técnica de incorporação posicional que utiliza rotações para representar posições, melhorando a capacidade do modelo de capturar relações sequenciais.
- **No Positional Embeddings (NoPE):** abordagem que elimina embeddings posicionais, explorando outras formas de capturar informações de posição na sequência.
- **Gating:** mecanismo que permite ao modelo controlar o fluxo de informações, ativando ou desativando certas partes da rede neural com base na entrada. Pode ser também o roteamento usado no MoE.
- **Local Attention:** técnica de atenção que limita o escopo da atenção a janelas locais na sequência, reduzindo a complexidade computacional.
- No geral, visam eficiência sem abrir mão do desempenho. Focam na escalabilidade.

## Slides

### 15 Augmentation

- Long-Tail Knowledge: técnicas para lidar com dados raros ou incomuns, garantindo que o modelo possa generalizar bem mesmo com informações limitadas.

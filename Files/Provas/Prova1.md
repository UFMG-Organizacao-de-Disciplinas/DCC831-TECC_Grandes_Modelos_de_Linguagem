# Anotações sobre como foi a prova

15 questões objetivas valendo 1 ponto. 1 questão aberta valendo 5.

- Diferença de um transformer para uma rede recorrente?
- O que mais impactaria no peso da inferência?
- O que menos impactaria no peso da inferência?
- Quantos vetores QKV sem KV caching?
  1. 10, 10, 10
  2. 100, 100, 100
  3. 1024, 1024, 1024
  4. 10240, 10240, 10240
  5. 55, 55, 55
- Quantos vetores QKV com KV caching?
  1. 10, 10, 10
  2. 100, 100, 100
  3. 1024, 1024, 1024
  4. 10240, 10240, 10240
  5. 55, 55, 55
- Calcule $z = Q \cdot K^T \cdot V$ para $z_3$, considerando que $X=Q=K=V$ e que
  \[
  X = \begin{bmatrix}
  0 & 1 \\
  1 & 1 \\
  1 & 0 \\
  2 & 0 \\
  0 & 2 \\
  \end{bmatrix}
  \]
  1. ?; 2. ?; 3. \[1; 6]; 4. ?; 5. ?

1. (Aberta) Você foi contratado por uma empresa de seguro de carro que deseja prever a chance de ocorrer um acidente
   1. Descreva como você modelaria o problema;
   2. Quais os prós e contras de uma arquitetura:
      1. Fechada
      2. Pesos disponíveis
      3. Completamente aberta (Pesos, função de ativação, função de perda, dados de treinamento, etc.)

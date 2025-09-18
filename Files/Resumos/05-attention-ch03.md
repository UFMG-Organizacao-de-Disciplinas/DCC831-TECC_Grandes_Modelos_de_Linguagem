# Resumo Aula 05, Notebook 03 - Attention

Nesse capítulo, exploramos o mecanismo de atenção, uma técnica fundamental em redes neurais que permite que o modelo foque em partes específicas da entrada ao gerar a saída. A atenção é especialmente útil em tarefas de processamento de linguagem natural, onde diferentes palavras ou frases podem ter diferentes níveis de importância dependendo do contexto.

## 3.1 The problem with modeling long sequences

Um exemplo dado é sobre o caso de traduação onde, devido à questões intrínsecas à linguagem, palavras podem ter ordenações diferentes em línguas diferentes. Anteriormente as RNNs (Redes Neurais Recorrentes) lidavam com isso ao armazenar uma célula de memória (hidden state) que armazenava as informações referentes a essa ordem e então a passava adiante para o processo de decodificação.

## 3.2 Capturing data dependencies with attention mechanisms

Com o mecanismo de atenção, o modelo pode aprender a focar em partes específicas da entrada para gerar a saída, o que implica em certas partes terem maior nível de relevância que outras.

O mecansimo de atenção nos transformers é uma técnica voltada a permitir que cada posição da sequência de entrada consiga definir quâo relevante são cada uma das outras posições na sequência.

## 3.3 Attending to different parts of the input with self-attention

### 3.3.1 1A simple self-attention mechanism without trainable weights

Consideremos o seguinte:

- **Sequência de entrada:** $X = (x^{(1)}, \dots, x^{(T)})$, onde:

  - cada $x^{(i)} \in \mathbb{R}^d$ é um vetor de dimensão $d$;
  - $T$ é o comprimento da sequência.
  - Ex.: Para a frase "Your journey starts with one step", $x^{(1)}$ é o embedding d-dimensional que representa a palavra "Your".

- **Objetivo:**
  - Desejamos gerar `vetores de contexto` $z^{(i)}$ para cada posição $i$ na sequência de entrada.
  - Obs.: Tanto $z^{(i)}$ quanto $x^{(i)}$ têm a mesma dimensão $d$.
  - Cada vetor de contexto $z^{(i)}$ é uma soma ponderada dos vetores de entrada $x^{(j)}$.
    - Ex.: $z^{(i)} = \sum_{j=1}^{T} \omega_{ij} x^{(j)}$, onde $\omega_{ij}$ são os pesos de atenção.
  - Podemos então visualizar $z^{(i)}$ como uma versão modificada de $x^{(i)}$, que incorpora informações de outras partes da sequência.

**Attention Scores ($\omega$):** São os pesos de atenção não normalizados.

**Attention Weights:** São os pesos de atenção normalizados, ou seja, quando somados resultam em 1.

Para calcular os **Attention Scores ($\omega$)** usamos o produto interno (dot product) entre a posição $i$ e todas as outras posições $j$. Porém, apelidaremos o vetor $x^{(i)}$ como `query` $q^{(i)}$. Assim, teremos que: $\omega_{ij} = x^{(j)} \cdot (q^{(i)})^\top$.

O subscrito ${}_{ij}$ indica que o score é calculado entre a query na posição $i$ e o token na posição $j$.

JV: Suponho que o $\omega$ afinal é um escalar que representa o grau de similaridade entre os vetores $x^{(j)}$ e $q^{(i)}$.

> **DÚVIDA:** o produto interno é calculado entre vetores: $\{a, b, c\} \cdot \{d, e, f\} = ad + be + cf$ Porém, na equação que ele ilustra, ele considera que está fazendo uma multiplicação entre matrizes $(1 \times 3) \times (3 \times 1) = (1 \times 1)$.
>
> $$
> \begin{bmatrix}
>   a & b & c
> \end{bmatrix}
> \cdot
> \begin{bmatrix}
>   d & e & f
> \end{bmatrix}^T =
> \begin{bmatrix}
>   ad + be + cf
> \end{bmatrix}
> $$
>
> Isso tá matematicamente correto?

Exemplificando o cálculo dos Attention Scores ($\omega$) temos:

$$
\begin{array}{ccc|l}
  0.43 & 0.15 & 0.89 & x^{1}: \text{Your} \\
  0.55 & 0.87 & 0.66 & x^{2}: \text{journey} \\
  0.57 & 0.85 & 0.64 & x^{3}: \text{starts} \\
  0.22 & 0.58 & 0.33 & x^{4}: \text{with} \\
  0.77 & 0.25 & 0.10 & x^{5}: \text{one} \\
  0.05 & 0.80 & 0.55 & x^{6}: \text{step} \\
\end{array}
$$

Onde cada linha é um embedding d-dimensional que representa uma das palavras da frase. E cada coluna representa uma feature específica do embedding.

Ele exemplifica o cálculo dos Attention Scores ($\omega$) para a posição $i=2$ (a palavra "journey"), que seria basicamente:

$$
\omega_{2j} =
\begin{cases}
  \omega_{21} = x^{1} \cdot q^{2} = 0.43*0.55 + 0.15*0.87 + 0.89*0.66 = 0.9544 \\
  \omega_{22} = x^{2} \cdot q^{2} = 0.55*0.55 + 0.87*0.87 + 0.66*0.66 = 1.4950 \\
  \omega_{23} = x^{3} \cdot q^{2} = 0.57*0.55 + 0.85*0.87 + 0.64*0.66 = 1.4754 \\
  \omega_{24} = x^{4} \cdot q^{2} = 0.22*0.55 + 0.58*0.87 + 0.33*0.66 = 0.8434 \\
  \omega_{25} = x^{5} \cdot q^{2} = 0.77*0.55 + 0.25*0.87 + 0.10*0.66 = 0.7070 \\
  \omega_{26} = x^{6} \cdot q^{2} = 0.05*0.55 + 0.80*0.87 + 0.55*0.66 = 1.0865 \\
\end{cases}
$$

O que seria computacionalmente equivalente a:

```python
# Início dos tensores de input
inputs = torch.tensor(
  [[0.43, 0.15, 0.89], # (x^1) Your
   [0.55, 0.87, 0.66], # (x^2) journey
   [0.57, 0.85, 0.64], # (x^3) starts
   [0.22, 0.58, 0.33], # (x^4) with
   [0.77, 0.25, 0.10], # (x^5) one
   [0.05, 0.80, 0.55]] # (x^6) step
)

query = inputs[1] # Como exemplo, define o segundo item dos inputs como a query (x^2) journey

attn_scores_2 = torch.empty(inputs.shape[0]) # Cria um tensor vazio para armazenar os attention scores com o mesmo tamanho da dimensão de cada vetor de input

for i, x_i in enumerate(inputs): # para cada índice e item dos inputs...
    attn_scores_2[i] = torch.dot(x_i, query) # Computa o produto interno entre o item e a query e armazena no tensor de attention scores

print(attn_scores_2)
```

Agora normalizaremos os **Attention Scores** ($\omega$) para obter os **Attention Weights** ($\alpha$). Uma visualização interessante é que, assim que esses pesos passam a ser normalizados, eles podem ser interpretados como uma distribuição de probabilidade sobre as posições da sequência de entrada.

```python
attn_weights_2_tmp = attn_scores_2 / attn_scores_2.sum()

print("Attention weights:", attn_weights_2_tmp)
print("Sum:", attn_weights_2_tmp.sum())
```

> Attention weights: tensor(\[0.1455, 0.2278, 0.2249, 0.1285, 0.1077, 0.1656\])

Apesar desse método de normalização ser válido, o mais comum é usar a função softmax pois este método "é melhor em lidar com valores extremos e tem propriedades de gradientes mais desejáveis".

> DÚVIDA: Por que a softmax é melhor em lidar com valores extremos? E qual sua diferença para o método de normalização pela soma?
> RESPOSTA JV: No meu entendimento, enquanto a normalização pela soma simplesmente escala os valores para que somem 1, a softmax força com que os valores maiores se tornem ainda maiores e os menores se tornem ainda menores, o que pode ajudar a destacar as diferenças entre os valores.

```python
def softmax_naive(x):
    return torch.exp(x) / torch.exp(x).sum(dim=0)

attn_weights_2_naive = softmax_naive(attn_scores_2)
```

> Attention weights: tensor(\[0.1385, 0.2379, 0.2333, 0.1240, 0.1082, 0.1581\])

Porém, é recomendado usar a implementação otimizada da biblioteca PyTorch:

```python
attn_weights_2 = torch.softmax(attn_scores_2, dim=0)
```

> Attention weights: tensor(\[0.1385, 0.2379, 0.2333, 0.1240, 0.1082, 0.1581\])

Agora que já calculamos os **Attention Weights** ($\alpha$), podemos calcular o vetor de contexto $z^{(i)}$ para a posição $i=2$ como uma soma ponderada dos vetores de entrada $x^{(j)}$:

$$
input \times Attention\ Weights = Context\ Vector
$$

$$
\begin{bmatrix}
  0.43 & 0.15 & 0.89 \\
  0.55 & 0.87 & 0.66 \\
  0.57 & 0.85 & 0.64 \\
  0.22 & 0.58 & 0.33 \\
  0.77 & 0.25 & 0.10 \\
  0.05 & 0.80 & 0.55 \\
\end{bmatrix} \times
\
\omega_{2j} =
\begin{cases}
  \omega_{21} = x^{1} \cdot q^{2} = 0.43*0.55 + 0.15*0.87 + 0.89*0.66 = 0.9544 \\
  \omega_{22} = x^{2} \cdot q^{2} = 0.55*0.55 + 0.87*0.87 + 0.66*0.66 = 1.4950 \\
  \omega_{23} = x^{3} \cdot q^{2} = 0.57*0.55 + 0.85*0.87 + 0.64*0.66 = 1.4754 \\
  \omega_{24} = x^{4} \cdot q^{2} = 0.22*0.55 + 0.58*0.87 + 0.33*0.66 = 0.8434 \\
  \omega_{25} = x^{5} \cdot q^{2} = 0.77*0.55 + 0.25*0.87 + 0.10*0.66 = 0.7070 \\
  \omega_{26} = x^{6} \cdot q^{2} = 0.05*0.55 + 0.80*0.87 + 0.55*0.66 = 1.0865 \\
\end{cases}
$$

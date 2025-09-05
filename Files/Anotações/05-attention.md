# Aula 05: Attention - Professor: Rodrygo/Anísio

## Attention - Ch03

### Anotações

O Rodrygo visualiza cada um dos elementos do embedding como sendo informações latentes. Como por exemplo: "quão bem essa palavra representa 'azul'?" ou "Essa palavra é um verbo?".

Cada palavra com seus embeddings gera um impacto no embedding do contexto. E os pesos das influências são aprendidos durante o treinamento.

Dúvida: "o que significa o produto interno entre dois vetores?" Será que é a intensidade de similaridade entre ambos? Por que não calcular a distância euclidiana?

O que é mesmo o "knn" (k-nearest neighbors)? Ah, obrigado, Copilot.

---

Depois de pegar o produto do contexto pelo embedding de cada palavra, ele divide cada um pela soma desses valores para que a distribuição some a 1.

Usa-se também o softmax que eleva eles a um determinado expoente e depois divide pela soma desses expoentes.

Poderíamos também aplicar outros métodos de comparação de distância além do produto interno. Talvez o ganho de qualidade seja boa mas não compense computacionalmente.

Aparentemente os scores também são normalizados a 1...

Dúvida: Por que eles são normalizados para que a soma deles seja 1 e não o resultado dos produtos deles?

o $\frac{1}{\sqrt{d_k}}$ é um fator de normalização para evitar que os produtos internos fiquem muito grandes. Isso porque o produto interno tende a crescer com a dimensionalidade dos vetores.

Dúvida: Não entendi bem o porquê disso.

A decisão da tokenização influencia na forma como o modelo dará peso às palavras.

Dúvida: Os embeddings dos tokens são intrínsecos a eles ou aos contextos em que eles aparecem e foram treinados?

1. Computar scores de atenção
2. Computar pesos de atenção
3. Computar o vetor de contexto

UM @ EM PYTHON É UMA MULTIPLICAÇÃO DE MATRIZES?!

### 3.4? Self-Attention

#### 3.4.1

No artigo

$$
Attention(Q, K, V) = softmax \left(\frac{Q(x)K(x)^T}{\sqrt{d_k}}\right)V(x)
$$

No quadro

$$
Z(x) = softmax \left(\frac{Q(x)K(x)^T}{\sqrt{d_k}}\right)V(x)
$$

- $X: 6 \times 3$
- $Z(X): 6 \times 2$
- $Q: 6 \times 3; W_{Q}: 3 \times 2; Q W_{Q}: 6 \times 2$ Query
- $K: 6 \times 3; W_{K}: 3 \times 2; K W_{K}: 6 \times 2$ Key
- $V: 6 \times 3; W_{V}: 3 \times 2; V W_{V}: 6 \times 2$ Value

Cada palavra passará por esses 3 papéis.

#### 3.4.2

Ler depois sobre a forma eficiente de aplicar a máscara

### 3.5

Máscara de drop-out: apagar parte dos valores durante o treinamento para impedir overfitting.

## Paper Attention is all you need

### 3.2.2 Multi-Head Attention

A multi head attention surge do uso de matrizes que tiveram máscaras de drop-out diferentes.

Por que uma matriz Wq de tamanho 4 ao invés de duas matrizes de tamanho 2? E de que forma eles analisam isso de forma diferente?

E dessas 4, se são subdivididos em 2, como são definidas quais são os dois pares? Como calcular qual a melhor? E qual critério usar?

## Resumo

O github do autor tem várias outras implementações. E ele tem um benchmark desses métodos.

## Referências

## Próxima Aula

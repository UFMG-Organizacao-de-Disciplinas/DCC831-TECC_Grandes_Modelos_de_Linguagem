# Aula 08 - Professor: Anísio

## CH06 - Finetuning for Text Classification

### 6.1 Different Categories of finetuning

O melhor que podemos fazer para encontrar spam é uma LLM?

Não. Podemos usar outras técnicas mais simples.

### 6.2 Preparing the dataset

- Ele faz undersampling para balancear o dataset.
- Depois converte ham/spam para 0/1.

Agora separamos em treino, validação e teste. Por que na aula anterior não separamos em Treino, Validação e Teste?

### 6.3 Creating data loaders (17h25 - 17min)

Não prestei atenção, mas imagino que seja fazer padding por batch.

(Viajei mais um pouco aqui) (17h37 - 29min)

O gradiente descendente calcula o erro médio por batch.

O drop_last é true para treinar e evitar aceitar valores com grande desvio. Já para validação é false pra aumentar a quantidade de valores de validação.

### 6.4 Initializing a model with pretrained weights (17h41 - 33min)

Na próxima aula será mostrado como treinar ele para responder a tarefas.

### 6.5 Adding a classification head (17h49 - 41min)

Por que não retreinar tudo? Por custo.
E seria benéfico? Não necessariamente. Eles já foram treinados de forma demorada e cara.

Aqui queremos só dar um empurrãozinho, não um empurrãozão.

Descartaremos a última camada, colocaremos uma nova e a treinaremos.

Também treinaremos o último bloco transformer.

Os gradientes mais atuais não consideram apenas o erro atual, mas também os erros passados.

É importante atualizarmos o learning rate para não caducar o modelo.

Assim que ele recria a última camada, ela descongela.

Ele então define quais camadas serão descongeladas.

Por causa da atenção causal, o último token é o único que viu todo o resto.

Pegar o último token não-padding resulta em melhores resultados do que pegar o último que seja padding.

### 6.6 Calculating the classification loss and accuracy > (18h20 - 1h12min)

### 6.7 Finetuning the model on supervised data > (18h27 - 1h19min)

### 6.8 Using the LLM as a spam classifier (18h34 - 1h26min)

## Extra (18h37 - 1h29min)

## Resumo

## Referências

## Próxima Aula

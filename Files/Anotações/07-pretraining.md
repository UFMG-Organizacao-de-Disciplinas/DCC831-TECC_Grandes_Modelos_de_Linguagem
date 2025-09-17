# Aula 07 - Professor: Rodrygo

## CH05 - 17h08 - 6min

- O que torna o pré-treinamento caro?
  - Dois componentes: o custo de cada operação e quantas vezes essa operação será demandada.
  - Uma métrica: throughput (quantidade de tokens processados por segundo).
    - Podendo ser de 15.000 a 150.000 tokens por segundo, dependendo das otimizações.
  - Para os bilhões de tokens, duraria uns 7,7 meses de treino.
  - Atualmente daria para reduzir isso para um treino de 1 hora.

> Como o GPT-2 gera texto?

- Não podemos nos dar ao luxo de deixar o modelo quebrar no meio do treinamento. Por isso é importante salvarmos checkpoints do treinamento.

### 5.1 Evaluating (17h19 - 17min)

1. Converte palavras em ids;
2. Manda pro GPT-Model converter ids em embeddings;
3. Converte o último embedding (?) (o mais relevante) para definir qual será a próxima palavra;

#### (17h24 - 22min) 5.1.2 Calculating the text generation loss: cross-entropy and perplexity

JV: Eu ainda tô incomodado com a ideia de que os últimos tokens de inputs repetem com os primeiros tokens dos targets. Não seria melhor se fossem uma lista só?

```python
inputs = torch.tensor([[16833, 3626, 6100],   # ["every effort moves",
                       [40,    1107, 588]])   #  "I really like"]

targets = torch.tensor([[3626, 6100, 345  ],  # [" effort moves you",
                        [1107,  588, 11311]]) #  " really like chocolate"]
```

Ao invés de checar se o modelo acertou/errou a previsão, ele vai calcular a probabilidade de cada palavra ser a próxima palavra correta.

Cross-entropy é algo para analisar a entropia entre as duas distribuições de probabilidade.

JV: até então eu estava entendendo que o modelo colocava as palavras em espaço de embedding, ou seja, colocava um ponto no espaço. Mas agora ele parece ser uma distribuição de probabilidade. Ainda não entendi como que isso foi feito, embora entenda de que forma eu poderia matematicamente fazer isso.

(17h38 - 36min)

- Dúvida JV: Em que momento deixamos de ter um ponto no espaço dos embeddings para passarmos a ter uma distribuição de probabilidade?
- Resposta Rodrygo: Quando tínhamos os embeddings espaciais, estávamos buscando algum tipo de similaridade semântica. Isso era no treinamento. Agora, no output, queremos apenas pegar o mais provável.

Um modelo não treinado começa com valores aleatórios/balanceados.

Uma forma interessante de usar as probabilidades é passar a usar o log das probabilidades. Isso porque o log de uma multiplicação vira uma soma de logs, assim aumentando a estabilidade numérica.

Ao invés de maximizar o log, passa-se a minimizar o negativo do log.

(17h45 - 43min)

- Ele primeiro pega as maiores probabilidades que tendem a ser próximos de 1/|Tamanho do vocabulário|.

1. Logits
2. Probabilidades
3. Probabilidades alvo
4. Log das probabilidades
5. Média dos logs
6. Negativo da média dos logs

(17h53 - 51min)

Em Machine Learning, não queremos dar a qualidade de uma predição para um item, mas sim para vários.

O valor encontrado, se eu fizer seu exponencial, encontro o perplexity. Que é "ele está em dúvida quanto a quantas palavras disponíveis que ele deve escolher?".

Idealmente queremos que a loss seja 0 e o perplexity seja 1, ele não teve dúvida sobre qual palavra escolher.

(18h02 - 1h) Como diferenciar se o modelo tá perdendo ponto por distribuir a ficha entre vários ou apostar em uma palavra errada?

- Resposta Rodrygo: Talvez isso matematicamente quebre a lógica dessa interpretação.

#### (18h05 - 1h03) 5.1.3 Calculating the training and validation loss

> De um médico!

(18h14 - 1h12) é esperado que mantenhamos subexemplos entre os treinos?

### (18h19 - 1h17) 5.2 Training a LLM

Para que a caminhada do gradiente seja mais amena, computa-se a média dos valores dos gradientes.

Otimizador: o mais básico é o gradiente descendente estocástico (SGD), mas geralmetne usa-se o AdamW.

(18h27 - 1h25) Explicando o loop de treino.

(18h31 - 1h29) Explicando overfitting.

### (18h34 - 1h32) 5.3 Decoding strategies to control randomness

Temperature scaling: normalizar a distribuição por uma temperatura. Primeiro achata todos.

Variação por temperatura:

1 = sem alteração
< 1 = mais confiante (aumenta os picos)

> 1 = mais incerto (achatamento dos picos)

Podemos fazer isso mas após fazer os top-k sampling e então aplicar a função de normalização.

### (18h40 - 1h38) 5.4 Loading and saving model weights in PyTorch

Determinado otimizador guarda também a progressão do treino.

Treinamos com poucos dados. Podemos carregar com os dados providos pela OpenAI.

## Resumo

## Referências

## Próxima Aula

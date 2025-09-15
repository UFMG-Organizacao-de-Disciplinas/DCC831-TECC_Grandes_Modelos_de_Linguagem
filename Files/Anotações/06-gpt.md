# Aula 06 - Professor: Rodrygo

## Conteúdo

### Artigo: ?

### Código

#### GPT_CONFIG_124M

- drop_rate/Drop-out: percentual de elementos apagados
- Projeções QKV: projeção linear com ou sem viés. Porém os mais recentes não usam viés.
  - Precisamos ter as mesmas configurações do GPT-2 para usarmos os mesmos pesos.
  - Esse viés seria um componente aditivo, que, linearmente seria como se deslocássemos o ponto inicial (0, 0).

#### DummyGPTModel (17h18 - 8min30)

- Revisitar essa parte dos embeddings do init. Não entendi isso que tá antes do dropout.

Dúvida: Os tais pesos são um conjunto dos pesos de cada uma das camadas?
Dúvida: As camadas podem/são convertidas em uma só?
Dúvida: por que uma camada não pode se retroalimentar?

##### Forward

- Logits: são os scores das palavras mais prováveis a serem emitidos.

Dúvida: qual a diferença entre positional embeddings e token embeddings?

A camada de atenção é o único momento em que os tokens "conversam" entre si.

(17h26)

#### Normalização (17h28)

Ele falou algo sobre regra da cadeia.

Números maiores que um pode gerar explosão de gradiente. (exploding gradient/overflow)

Números muito pequenos podem gerar o desaparecimento do gradiente. (vanishing gradient/underflow)

idealmente queremos média 0, variância 1.

RMSNorm: raiz da média dos quadrados

A normalização é feita recorrentemente para evitar que os números saiam de controle frequentemente.

(17h36)

- `x.mean(dim=-1)`:média ao longo da última dimensão, as colunas.
- `self.eps`: valor pequeno para evitar divisão por zero.

Ideia: e se, na normalização, apenas evitássemos que ela gerasse overflow ou underflow?

Dúvida aluno: (17h41)

A forma da normalização funciona como se fosse pra englobar todos os embeddings dentro de uma "bola" (hiperesfera) de raio 1, por exemplo.

Dúvida: mas isso ainda poderia causar o underflow, não?

(17h43)

Isso acaba sendo uma engenharia que é muito sensível.

Pra que se torna a rede profunda? Pra dar mais capacidade, mas isso pode acabar perdendo a linha.

#### Feed Forward (17h47)

Pós processamento da atenção

Sendo combinações lineares, podemos utilizar modificações não lineares para tentar enriquecer a rede.

Dúvida: qual é o impacto de haver ou não variação na quantidade de neurônios nas camadas?

Uma proporção linear seria quando há uma relação direta entre a entrada e a saída.

**ReLU:** função que zera os valores negativos.
**GELU:** função que suaviza a ReLU, fazendo uma transição mais gradual.

Um problema do ReLU é que a derivada pros valores negativos é zero.

Dúvida de aluno (17h55): Por que zerar os negativos?

- Onde é aplicada a camada de ativação?
  - Ele expande um vetor em um vetor 4 vezes maior, depois converte de volta pro tamanho original.

(18h01) Mixed Experts

Minha dúvida: (18h05) Remover os negativos não acabaria tornando inacessíveis alguns tokens que tenham embeddings com valores negativos?

Não seria melhor, ao invés de matar os negativos, apenas redistribuir os valores de 0 a 1 com valores distribuídos uniformemente e diferentes de 0?

##### 4.4 (18h14)

##### 4.5 (18h15)

(18h23)

No transformer é esperado que a entrada e a saída tenham a mesma dimensão.

Dúvida: e se não tiverem? E se forem intercalados? (aumenta, diminui, aumenta, etc.)

Dúvida: como que ocorre o aumento e diminuição do GELU? é como se fossem traçados 3 novos pontos entre cada par de pontos?

Queremos que a saída no fim, preveja a próxima palavra. Então, espera-se que após o aprendizado, a primeira palavra deve gerar algo próximo de sua segunda palavra.

##### 4.6 Codign the GPT Model (18h27)

Antes tínhamos ids por tokens, agora temos embeddings.

Token embeddings mapeia do tamanho do vocabulário pro tamanho da representação. E na saída, temos o linear output layer que faz o oposto. Ambas têm o mesmo formato, então o GPT reusa os pesos.

Dúvida: então os tais números de parâmetros são os pesos usados em cada uma dessas camadas?

##### 4.7 Generating Text (18h31)

- aparentemente as primeiras palavras viram tokens para alimentar o contexto pra última.

Softmax: transforma os logits em probabilidades. (eleva os valores para a escala exponencial, depois normaliza, então a soma dá 1)

Das probabilidades, encontra o mais provável, e usa o id do token correspondente para buscar a palavra.

Dúvida aluno: (18h36)

Por enquanto ele não gera nada relevante porque ainda não foi treinado.

### Artigo: Layer Normalization

### Artigo: ? (18h10)

O X entra, passa por um processo, depois ele é somado a esse valor modificado.

Meu entendimento é que antes buscava-se encontrar o "1,???" que multiplicará o X. Ao invés disso, procura-se o "1" +- "0,????". Essas camadas residuais auxiliam a evitar que valores muito pequenos sejam resultados.

## Resumo

Revisar o notebook. É tudo muito mecânico.

## Referências

## Próxima Aula

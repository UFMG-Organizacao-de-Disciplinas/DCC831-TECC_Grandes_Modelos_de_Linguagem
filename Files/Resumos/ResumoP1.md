# Resumão

## Resumo do Capítulo 2 - Preparação dos Dados

No capítulo 2 vimos sobre a criação do dataloader que através do tokenizador utilizando o BPE converte palavras em tokens, que, ao pertencer ao vocabulário terão seu Token ID e então divide esses token IDs em batches contendo pares de vetores de input-target que serão utilizados com a técnica de sliding window para treinar o modelos que serão desenvolvidos nos capítulos seguintes.

Porém, antes de treinar, é necessário converter esses token IDs em embeddings, que são representações densas e contínuas dos tokens em um espaço vetorial de alta dimensão. Esses embeddings são inicialmente aleatórios e são ajustados durante o treinamento do modelo para capturar relações semânticas entre palavras. Também é somado ao embedding do token o embedding posicional, que é uma representação absoluta e/ou relativa da posição do token na sequência, permitindo que o modelo capture a ordem dos tokens.

## Resumo do Capítulo 3 - Attention

No capítulo 3 vimos sobre o mecanismo de atenção. Eles são importantes por palavras terem significados diferentes em contextos diferentes. O mecanismo de atenção funciona atribuindo valores de relevância entre os tokens de uma sequência com relação aos outros tokens já vistos, os ainda não vistos, devido à atenção causal, não são considerados.

Partindo dos tokens-embeddings, são criadas matrizes treináveis Wq, Wk e Wv. Cada Token-embedding é multiplicado por Wq, Wk e Wv para gerar as matrizes (projeções) Q, K e V. E então, cada Token-embedding Qi será multiplicado por todos K1..n transposto que gera os scores de atenção para Qi, ou seja, quanto que cada Key influencia no token Qi. Esse score, após ser dividido por sqrt(dk) (para evitar o vanishing gradient) e normalizado com softmax (dando maior ênfase aos maiores valores positivos) se tornam os Pesos de Atenção.

Para evitar overfitting, também é feito o dropout que omite alguns dos pesos de atenção antes de serem multiplicados por V. O vetor de contexto é gerado multiplicando os pesos de atenção por V.

Todo esse processo é executado pelo módulo de Atenção Multi-Head, que executa o processo h vezes com diferentes matrizes Wq, Wk e Wv. A intenção é que cada cabeça aprenda diferentes aspectos de atenção. Os vetores de contexto de cada cabeça são então concatenados e projetados novamente para o espaço original do embedding.

## Resumo o Capítulo 4 - Implementing a GPT model from Scratch To Generate Text

Um modelo de linguagem funciona gerando o próximo token com base nos tokens anteriores. E é um processo autoregressivo, ou seja, o resultado gerado é inserido novamente como entrada para gerar o próximo token. Isso amostrando o token com base na distribuição de probabilidade gerada pelo modelo. Porém, antes de haver o treinamento, o modelo gera tokens aleatórios.

Um modelo de linguagem grande (LLM) como o GPT é construído através de um _transformer_ **Decoder-Only** composto inicialmente pelo pipeline de entrada que converte texto em tokens e posteriormente em embeddings com sua informação posicional, seguido por vários _transformer blocks_ empilhados. O transformer block é composto pelo mecanismo de atenção multi-head mascarado, seguido por uma rede feed forward (FFN) com a função de ativação GELU, ambos acompanhados por conexões residuais (shortcut connections) e normalização de camada (Layer Normalization). Por fim, temos a camada de saída (output head) que projeta a representação interna do modelo de volta para a dimensão do vocabulário, produzindo os _logits_ (scores brutos) que serão usados para prever o próximo _token_.

A Feed Forward Network (FFN) é uma rede linear de duas camadas que processa as informações enriquecidas pelo mecanismo de atenção. Ela expande a dimensão do embedding em um fator de 4 antes de revertê-la ao tamanho original, permitindo que a rede extraia _features_ mais complexas. O GPT utiliza a função de ativação GELU (Gaussian Error Linear Unit), que pesa as entradas por seu valor, proporcionando uma transição mais suave em comparação com funções como ReLU.

A Shortcut Connections (Conexões Residuais) ajudam a mitigar o problema do _vanishing gradient_ somando os valores de entrada do sub-bloco pré-MHA e somando-as à saída do FFN (posteriormente normalizada) com o objetivo de facilitar o fluxo do gradiente durante o _backpropagation_.

A Layer Normalization mantém a média e o desvio padrão dos embeddings consistentes, estabilizando o treinamento. Ela tem dois parâmetros treináveis por _feature_ ($\gamma$ e $\beta$) para ajustar a escala e o deslocamento após a normalização.

## Resumo do Capítulo 5 - Pretraining a GPT Model

Os modelos de geração de texto, como o GPT, são **autoregressivos**, ou seja, o token gerado em cada passo é adicionado ao contexto para prever o próximo token. Partindo do contexto inicial o modelo gera scores para os vocábulos e os converte em probabilidades. O próximo token é escolhido com base nessas probabilidades, geralmente selecionando o token com a maior probabilidade (**decodificação gulosa**), porém, isso pode o tornar muito repetitivo. Para adicionar diversidade, pode-se usar amostragem probabilística e **escalonamento de temperatura** onde baixa temperatura torna a distribuição mais nítida (mais coerente) e alta temperatura torna a distribuição mais plana (mais diversa); também podemos usar o **top-k sampling** que restringe o vocabulário para os k tokens mais prováveis.

Após a saída dos logits (scores para cada uma das palavras do vocabulário) do modelo e os converter em probabilidades através do **softmax**, podemos calcular a função de perda (**loss function**) que mede o quão boa foi a previsão do modelo. A entropia cruzada (**Cross-Entropy Loss**), se baseia na log-verossimilhança negativa do token alvo.Ao aplicar o log numa baixa probabilidade do token alvo, o resultado (a perda) é um valor alto, indicando que a previsão está muito errada, logo, esse valor alto (a perda) deve ser minimizado para se aproximar de 0. É a minimização dessa perda que, por sua vez, força o modelo a maximizar a probabilidade do token alvo correto.

A perplexity (PPL) é uma métrica que em média representa qual a quantidade de tokens dos quais o modelo deve escolher a próxima palavra e é dada pela fórmula $PPL = e^{\text{Cross-Entropy Loss}}$. Quanto menor a perplexidade, melhor é a capacidade do modelo de prever o texto.

Além de avaliar as perdas do treino, é necessário avaliar as perdas no conjunto de validação, isso porque, se houver uma diferença muito grande entre as perdas, o modelo pode estar sofrendo de overfitting, ou seja, decorou o conjunto de treino e não está generalizando. Apesar de que o cálculo da loss ocorre para cada uma das estimativas de geração de token, a loss resultante é na verdade a média das losses de cada batch. O valor de loss resultante é então usado na retropropagação (backpropagation) para atualizar os pesos do modelo para que suas previsões gerem menos perda na próxima vez.

Através da backpropagation, os gradientes necessários para aproximar a função de perda são calculados, e o otimizador AdamW é usado para atualizar os pesos do modelo. A diferença do Adam pro AdamW é que o primeiro aplica o decaimento de pesos diretamente na taxa adaptativa, enquanto o AdamW aplica o decaimento de pesos separadamente. Desse modo ele o força a reduzir a intensidade dos pesos, o que ajuda a evitar overfitting.

Após realizar o treino, para que não se perca o esforço computacional, é uma boa prática salvar os pesos do modelo em um arquivo. Esses pesos se encontram, usualmente, no dicionário de estado do modelo (`model.state_dict()`). Outros pesos relevantes são os do otimizador, que podem ser salvos de maneira similar (`optimizer.state_dict()`).

Apesar disso, o pré-treinamento de uma LLM em um grande _corpus_ é custoso em termos de tempo e recursos, é comum a prática de carregar **pesos pré-treinados abertamente disponíveis** (como os da OpenAI). Isso fornece um ponto de partida sólido para as fases subsequentes de _fine-tuning_.

## Resumo do Capítulo 6 -

## Somando os Parâmetros Treináveis

- vários (Q, K, V)
- 2x peso de feed forward
- |Transformer Block| \* LayerNorm (gamma e beta)
- Pesos da saída

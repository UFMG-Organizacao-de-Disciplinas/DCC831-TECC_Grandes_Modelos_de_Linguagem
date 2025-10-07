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

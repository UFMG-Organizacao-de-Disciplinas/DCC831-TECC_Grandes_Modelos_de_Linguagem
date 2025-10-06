# Chapter 2: Working with Text Data

Para construirmos uma LLM começamos com a preparação dos dados e o _sampling_ (amostragem) dos dados.

Todo o processo se resume em levar o **texto bruto** até os **vetores numéricos** que o modelo Transformer (Decoder-Only) pode processar.

## 2.1. Understanding word embeddings

Um `embedding` é uma representação numérica de um token (palavra, pedaço de palavra ou caractere) em um espaço vetorial d-dimensional ond `d` é a dimensão do embedding, ou seja, o número de _features_ que cada token terá. O embedding pode ser utilizado para texto, áudio, vídeo, etc.

Como geralmente esses _embeddings_ são de alta dimensionalidade, é difícil de visualizar, visto que os humanos usualmente conseguem visualizar até 3 dimensões. Então, para facilitar a visualização, consideremos um embedding de 3 dimensões (d=3) ($\mathbb{R}^3$).

Ao representarmos os tokens em um espaço vetorial, cada uma de suas coordenadas representa uma _feature_ (característica) do token. Por exemplo, em um embedding de 3 dimensões, poderíamos ter as seguintes features: 1. "Positividade" (quão positivo é o token), 2. "Formalidade" (quão formal é o token) e 3. "Complexidade" (quão complexo é o token).

Desse modo, poderíamos representar os tokens com valores similares a estes:

- "Ótimo": (0.9, 0.4, 0.3)
- "Excelentíssimo": (0.95, 0.8, 0.7)
- "Garotada": (0.5, 0.2, 0.1)
- "Pá": (0.3, 0.1, 0.05)

Perceba que os tokens "ótimo" e "Excelentíssimo" estão próximos no espaço vetorial, principalmente na dimensão de "Positividade", indicando que ambos são tokens positivos. Já os tokens "garotada" e "pá" se aproximam mais na dimensão de "Formalidade", sugerindo que ambos são informais.

Entretanto, o espaço vetorial dos _embeddings_ não é tão claro assim. Afinal, as _features_ não são evidentes, assim ofuscando o real significado de cada dimensão. Além disso, o número de dimensões é muito maior do que 3, o que dificulta a visualização.

Um ponto que o GPT me recordou aqui é que as _features_ aprendidas são **latentes**, ou seja, não são diretamente interpretáveis, elas capturam padrões complexos de coocorrência e significado distribuído através dos dados de treino, sendo então dependentes do contexto em que estão inseridas.

Apesar disso, ainda assim é possível calcular a similaridade entre os tokens utilizando métricas como a distância Euclidiana ou o cosseno do ângulo entre os vetores. Quanto ao segundo, pelo que entendi, quanto mais próximo de 1, mais similares são os tokens. Porém dois vetores que estão na mesma direção, mas com magnitudes diferentes, o que tiver maior magnitude será mais parecido, ou ao menos, mais relevante, mesmo que o outro esteja mais próximo euclidianamente do outro vetor comparado.

Outra situação sugerida pelo GPT é sobre a aritmética vetorial. Por exemplo, o vetor resultante de "rei" - "homem" + "mulher" deve ser próximo ao vetor de "rainha". Isso indica que o modelo capturou relações semânticas entre os tokens. E também que a magnitude nem sempre é tão relevante, visto que os embeddings são normalizados, o que geralmente significa que a soma de seus componentes é igual a 1.

## 2.2. Tokeninzing Text

O processo de tokenização envolve dividir o texto em unidades menores chamadas tokens. Esses tokens podem ser palavras, subpalavras ou caracteres, dependendo do método de tokenização utilizado.

Trançando a sequência de etapas para preparar o texto para uma LLM, temos:

1. **Texto bruto:** o texto de entrada completo. Ex.: "O rato roeu a roupa do rei de Roma."; Tamanho: 9 palavras
2. **Tokenização:** o texto é dividido em tokens. Ex.: \["O", "rato", "roeu", "a", "roupa", "do", "rei", "de", "Roma", "."]; Tamanho: 10 tokens
3. **Token IDs:** cada token é mapeado para um ID numérico. Ex.: \[101, 2023, 2003, 1037, 2173, 1997, 2986, 1998, 2035, 1012]; Tamanho: 10 IDs
4. **Token _embeddings_:** os IDs dos tokens são convertidos em vetores numéricos. Ex.: \[\[0.1, 0.2, \dots], \[0.3, 0.4, ...], ...]; Tamanho: 10 vetores de dimensão $d$ ($10 \times d$)
5. **Positional _embeddings_:** vetores adicionais são adicionados para indicar a posição de cada token na sequência.
6. **Entrada para a LLM:** a sequência de vetores (token _embeddings_ + positional _embeddings_) é alimentada na LLM para processamento.
7. **Saída da LLM:** a LLM gera uma sequência de vetores de saída, que podem ser convertidos de volta em tokens e texto.
   - **Dúvida:** inicialmente qual é o retorno? O próximo token mais provável? Uma lista de probabilidades do próximo token? :thinking:
8. **Decodificação:** os vetores de saída são mapeados de volta para tokens e, em seguida, para texto legível.
9. **Pós-processamento:** o texto gerado pode passar por etapas adicionais, como remoção de tokens especiais ou formatação.
10. **Texto final:** o texto gerado final após todas as etapas de processamento.

Para tokenizarmos o texto, podemos utilizar bibliotecas ou "fazer na mão".

```python
def tokenize(text="Hello, world. This, is a test.") -> list: # texto de entrada
    import re # biblioteca de expressões regulares
    text = "Hello, world. This, is a test."
    tokens = re.split(r'([,.:;?_!"()\']|--|\s)', text) # divide o texto em tokens, incluindo vírgulas, pontos e espaços
    cleaned_tokens = [item.strip() for item in tokens if item.strip()] # remove tokens vazios ou apenas espaços
    return cleaned_tokens
```

Obs.: Esse método inicial é uma tokenização simples por palavra e pontuação. Existem métodos mais sofisticados, como o Byte Pair Encoding (BPE), que será abordado mais adiante.

Que ótimo! Vamos preencher essas lacunas e solidificar as etapas da _pipeline_ de dados do LLM.

Como vimos, todo o processo se resume em levar o **texto bruto** até os **vetores numéricos** que o modelo Transformer (Decoder-Only) pode processar.

---

OBSERVAÇÃO: A PARTIR DAQUI O TEXTO FOI [GERADO PELO GEMINI](https://g.co/gemini/share/b68ffc1fcd86), COM ALGUNS AJUSTES MEUS.

## 2.3. Converting tokens into token IDs

Após a **Tokenização** (dividir o texto em unidades menores), a próxima conversão é de _strings_ (tokens de texto) para números inteiros, chamados **Token IDs**.

1. **Construção do Vocabulário:** É preciso primeiro construir um **Vocabulário**. Este vocabulário é um mapeamento que associa cada token **único** (palavra, pontuação, caractere especial) presente no conjunto de dados de treino a um **ID inteiro exclusivo**.
2. **Mapeamento:** O texto tokenizado é percorrido, e cada token é substituído pelo seu respectivo Token ID.
3. **Decodificação:** Para converter a saída numérica da LLM de volta para texto legível, é criado um **vocabulário inverso** que mapeia os IDs de volta para os _strings_ de token.

## 2.4. Adding special context tokens

O uso de tokens especiais visa melhorar o entendimento do modelo e tratar casos que o vocabulário não cobre.

- **`<|unk|>` (Unknown - Desconhecido):** Usado para representar palavras que não existiam no vocabulário durante o treinamento. Sua necessidade é minimizada com o BPE (Byte Pair Encoding) (próxima seção), mas é crucial em tokenizadores simples.
- **`<|endoftext|>`:** Atua como um marcador para **delimitar documentos** ou separar fontes de texto não relacionadas que foram concatenadas para o treinamento. Em modelos GPT, este token também é usado para **preenchimento** (_padding_) de sequências mais curtas em um lote (_batch_).
- **Outros:** Modelos mais antigos ou de propósito geral podem usar `[BOS]` (Beginning of Sequence), `[EOS]` (End of Sequence, análogo ao `<|endoftext|>`), e `[PAD]` (Padding).

## 2.5. Byte Pair Encoding (BPE)

BPE é um esquema de tokenização mais sofisticado usado em LLMs populares como GPT-2 e GPT-3.

- **Subword Units:** A principal inovação do BPE é que ele decompõe as palavras em **subunidades de palavras** (_subwords_) ou caracteres individuais, em vez de apenas palavras inteiras.
- **Lidando com o Desconhecido:** Esta técnica permite que o LLM processe **qualquer palavra** (mesmo as que ele nunca viu). Se uma palavra não estiver no vocabulário, ela é quebrada em seus _subwords_ ou caracteres conhecidos, eliminando a necessidade explícita do token `<|unk|>`.
- **Construção do Vocabulário:** O BPE é construído iterativamente. Ele começa com todos os caracteres individuais e, em seguida, **funde repetidamente os pares de _bytes_ ou _subwords_ mais frequentes** em novas unidades, criando o vocabulário a partir destas unidades combinadas.

---

Meu entendimento: ele parte de todos os caracteres únicos e vai juntando os pares mais frequentes, formando subpalavras. Assim, se a junção posterior de `ca` com `sa` for frequente, ele cria a subpalavra `casa`. Agora, quando partimos de palavras desconhecidas, ele tenta casar a maior _substring_ possível com o vocabulário, e o que sobrar, ele segue quebrando em _substrings_ menores, até chegar em caracteres únicos, se necessário.

Nota-se que é um processo estatístico de frequência, então não captura necessariamente a semântica, porém, como na prática os tokens mais frequentes tendem a ser semanticamente relevantes, o BPE acaba capturando padrões úteis para o modelo.

## 2.6. Data sampling with a sliding window

Esta é a etapa em que se prepara o _input_ e o _target_ (entrada e alvo) para o treinamento do modelo de **Next-Word Prediction** (Previsão da Próxima Palavra).

O mecanismo central é criar pares onde o _target_ é a sequência de _input_ **deslocada em uma posição**.

| Variável         | Conteúdo                                                           | Exemplo            |
| :--------------- | :----------------------------------------------------------------- | :----------------- |
| **Input (`x`)**  | Sequência de Tokens IDs usada como entrada para a LLM.             | `[T1, T2, T3, T4]` |
| **Target (`y`)** | Sequência de Tokens IDs que a LLM deve prever (a próxima palavra). | `[T2, T3, T4, T5]` |

- **Sliding Window (Janela Deslizante):** Uma janela de tamanho fixo (`max_length` ou _context size_) percorre o texto tokenizado para criar esses pares de `x` e `y`.
- **Stride (Passo):** O parâmetro `stride` (passo) controla o quanto a janela desliza para criar o próximo par. Um `stride` de **1** maximiza a sobreposição e os exemplos de treino, enquanto um `stride` igual ao `max_length` garante que não haja sobreposição entre os _batches_.
- **Tensors:** No final, `x` e `y` são formatados como _tensors_ (matrizes multi-dimensionais) do PyTorch, prontos para serem carregados para o treinamento.

---

Dúvida: Uma coisa que até agora não me desceu é por que não podemos juntar o `x` e o `y` em um único tensor `[T1, T2, T3, T4, T5]`, e durante o treinamento, a LLM prever `T2` dado `T1`, depois `T3` dado `T1, T2`, então `[T0:TK]` prever `TK+1`. Assim, não precisaríamos do `y` explicitamente, e o modelo aprenderia a prever o próximo token em cada passo. Isso reduziria a quantidade de dados a serem passados.

Com isso criamos um Batch de dados, que é um conjunto de pares `x` e `y` que serão usados para treinar o modelo em uma única iteração.

Exemplo de Batch:

```python
batch = {
    "input_ids": torch.tensor([
        [101, 523, 203, 137, 173],  # Exemplo 1
        [111, 557, 200, 137, 273],  # Exemplo 2
        # ... mais exemplos
        [128, 731, 100, 335, 102]   # Exemplo N
    ]),
    "labels": torch.tensor([
        [523, 203, 137, 173, 202],  # Target para Exemplo 1
        [557, 200, 137, 273, 300],  # Target para Exemplo 2
        # ... mais targets
        [731, 100, 335, 102, 150]   # Target para Exemplo N
    ])
}
```

Dessa forma, $N$ é o número de exemplos no _batch_, e cada exemplo tem uma sequência de `max_length` tokens.

E então o retorno será uma estrutura 3D, onde a primeira dimensão é o número de exemplos no _batch_, a segunda é a sequência de tokens, e a terceira é o vetor de _embedding_ para cada token.

## 2.7. Creation token embeddings

O objetivo é converter os **Token IDs** (números inteiros) em **vetores de _embedding_** (vetores numéricos contínuos).

- **Necessidade:** Redes neurais profundas, como as LLMs, não podem processar diretamente números inteiros discretos, mas sim tensores de valores contínuos.
- **Mecanismo:** A `Embedding Layer` (Camada de _Embedding_) funciona como uma **operação de consulta (_lookup_)**. Ela pega o Token ID e o usa como um índice para extrair uma linha da **matriz de pesos (_weight matrix_)** da camada.
- **Treinamento:** Essa matriz de pesos é inicialmente preenchida com valores aleatórios e é **otimizada** (ajustada) junto com todos os outros parâmetros da LLM durante o treinamento.

## 2.8. Encoding word positions

Esta etapa final garante que o modelo entenda a **ordem** das palavras, o que é crucial, já que o mecanismo de _self-attention_ é, por natureza, **agnóstico à posição**.

1. **O Problema:** Um _token embedding_ será sempre o mesmo vetor, não importa onde ele esteja na frase.
2. **A Solução:** Adicionar **Embeddings Posicionais** aos _token embeddings_.
3. **Tipos:**
   - **Embeddings Posicionais Absolutos:** Um vetor exclusivo é adicionado para cada **posição exata** na sequência (ex: um vetor para a posição 1, outro para a posição 2, etc.). Os modelos **GPT** usam essa abordagem.
   - **Embeddings Posicionais Relativos:** O foco está na **distância** ou relação entre os tokens, não na posição absoluta.
4. **Processo Final:** Os _embeddings_ posicionais têm a mesma dimensão dos _token embeddings_ e são **somados** a eles para criar os **Input Embeddings** (Embeddings de Entrada) finais. Em modelos GPT, esses _embeddings_ posicionais também são **aprendidos/otimizados** durante o treinamento.

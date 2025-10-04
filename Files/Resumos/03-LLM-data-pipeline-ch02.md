# Chapter 2: Working with Text Data

Para construirmos uma LLM começamos com a preparação dos dados e o _sampling_ (amostragem) dos dados.

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

## 2.3. Converting tokens into token IDs

## 2.4. Adding special context tokens

## 2.5. BytePair encoding

## 2.6. Data sampling with a sliding window

## 2.7. Creation token embeddings

## 2.8. Encoding word positions

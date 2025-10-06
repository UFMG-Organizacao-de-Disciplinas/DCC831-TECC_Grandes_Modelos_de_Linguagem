# Chapter 4: Implementing a GPT model from Scratch To Generate Text

Nessa etapa, vamos utilizar os conhecimentos adquiridos até agora para implementar a arquitetura do GPT. Para isso passamos primeiro pela preparação dos dados, seguido da implementação do componente de atenção.

## 4.1 Coding an LLM architecture

Os modelos como GPT e Llama geram palavras sequencialmente, isso se baseando na parte _decoder_ da arquitetura _transformer_.

DÚVIDA: Mas e o que seria um _transformer_?
RESPOSTA COPILOT: Um _transformer_ é uma arquitetura de rede neural que utiliza mecanismos de atenção para processar dados sequenciais, como texto ou áudio. Ele é composto por camadas de atenção e camadas feedforward, permitindo que o modelo capture relações de longo alcance em dados sequenciais de maneira eficiente.

DÚVIDA: Então o _transformer_ é basicamente o esqueleto de uma LLM antes de ter sido treinada com muitos dados?
RESPOSTA COPILOT: Sim, o _transformer_ é a arquitetura básica que serve como esqueleto para modelos de linguagem como GPT.

Então, meu entendimento é que primeiro iremos programar um transformer, que usa o mecanismo de atenção. Depois, com os dados preparados iremos treinar esse transformer para que ele se torne um modelo de linguagem como o GPT.

## 4.2 Normalizing activations with layer normalization

## 4.3 Implementing a feed forward network with GELU activations

## 4.4 Adding shortcut connections

## 4.5 Connecting attention and linear layers in a transformer block

## 4.6 Coding the GPT model

## 4.7 Generating text

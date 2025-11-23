# Model Uncertainty

## LLMs são muito boas

- Igualam e até superam humanos (ENEM, POSCOMP)
- Alucinações

## Mas...

- Algumas alucinações são óbvias.
- Às vezes são confiantes no erro. E confiantes em excesso.
- Maiores problemas: predição de modelos pra medicina.

---

- Uncertainty Quantification (UQ): técnicas para medir a incerteza nas previsões de modelos de linguagem.

## Por que estudar isso?

- Ter clareza no que não se sabe ajuda a decidir o que e como responder.
- Saber a incerteza ajuda a se abster quando o risco é alto.
  - {Parece ser meio que aprender a quando não assumir o BO}
- Predição Conformal (CP): técnica estatística que fornece garantias de cobertura para predições, útil para medir a incerteza.

## O que é

- Quantificação de Incerteza (UQ)
- Predição Conformal (CP)

Incertezas:

1. Epistêmica: O modelo não sabe (falta de dados)
2. Aleatória: Os dados são inerentemente ruidosos e mesmo com dados infinitos continuará existindo.

## Dificuldades

Não temos acesso à distribuição real dos dados no mundo, apenas amostras.

O modelo, como caixa preta, não nos fornece os dados numéricos reais. E mesmo as métricas simples, caso disponíveis, sofrem de vieses (comprimento, mudança de domínio, acúmulo de erros).

## Panorama

- **Token-level:** entropia, medidas sensíveis ao significado
- **Auto-verbalizada:** confiança declarada pelo próprio modelo
- **Consenso semântico:** verifica se várias respostas apresentam o mesmo significado.
- **Interpretabilidade mecanística:** análise dos neurônios ativados
- **Predição Conformal:** garante formalmente que a resposta correta está dentro de um intervalo

### UQ Token-level

Dado um prompt, medir os logits para analisar sua entropia

Problema: Se o primeiro token tá errado, ele pode ter muita certeza mesmo que tudo esteja errado.

Geralmente funciona muito bem quando tem alternativas fixas.

### Confiança Auto-verbalizada

Pedir pro modelo declarar a confiança em palavras ou números. O benefício é a legibilidade.

Problema: São superconfiantes.

### Consenso semântico

Gerar várias respostas e analisar em média se ela mantém a mesma resposta

## Interpretabilidade mecanística

Analisar quais neurônios são ativados pra quais inputs e outputs. Isso para entender quão certo ele tá.

Se você já mapeou o que ativa o quê, você pode redirecionar o modelo para disambiguação.

State of Art: Sparse Autoencoders.

## Predição Conformal

(Não entendi nada do que tá escrito no slide)

## Conformal Language Modeling

É importante entender se a saída faz sentido

## Métricas de avaliação

## Caminhos promissores

Faltam benchmarks padronizados.

Interpretabilidade mecanística e Uncertainty Quantification podem...

## Conclusões

## Dúvidas

Dúvida: por que os modelos menores são mais autoconfiantes?

Resposta: Chutam que seja por saberem menos e precisarem dar uma resposta.

---

Rodrygo: Em que casos ele gera coisas que tem pouco lastro vs está gerando coisas factuais?

Antes buscavam coisas concretas, agora geram coisas.

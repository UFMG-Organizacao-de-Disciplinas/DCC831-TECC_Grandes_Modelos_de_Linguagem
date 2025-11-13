# Model Uncertainty

## LLMs são boas

- Igualam e até superam humanos (ENEM, POSCOMP)
- Alucinações

## Mas...

- Algumas alucinações são óbvias.
- Maiores problemas: predição de modelos pra medicina.

## (Me desatentei)

## O que é

- Quantificação de Incerteza (UQ)
- Predição Conformal (CP)

Incertezas: Epistêmica e Aleatória (várias respostas certas)

## Dificuldades

Não temos acesso à distribuição real dos dados no mundo, apenas amostras.

## Panorama

- Token-level
- Auto-verbalizada
- Consenso semântico
- Interpretabilidade mecanística
- Predição Conformal

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

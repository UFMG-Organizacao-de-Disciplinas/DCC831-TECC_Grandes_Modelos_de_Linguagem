# Modern Transformers - 04/11/2025

## ?

## Group QKV

## Local Attention: Janelas Deslizantes

---

Redução no KV cache, computação local otimizada, mas perde em generalização do modelo.

## Mixture-of-Experts (MoE)

- Ao invés de ter uma grande camada densa, substitui por várias pequenas camadas chamadas "experts".
- É difícil treinar porque pode focar muito em alguns experts

## QK-Norm, RMS-Norm e NoPE

- RMS-Nome: uma simplificação
- O Layer Norm é aplicado no Attention is All You Need
- Layer Norm ajuda na convergência
  - Normaliza antes de somar com o embegging posicional oculto.
- Removendo embeddings poisicionais melhora em algumas coias, porém piora em outras.

## RoPE: Rotary Positional Embeddings

- POde ser usado com o KV-cache

## Arquiteturas

- DeepSeek V3/R1 (671B)
  - Proposto pra eficiência
- OLMo ..
- GPT-OSS 20B
- Llama4 e Grok-2.4 VS FGrok 2.5 (270 B)
- SmolLM3 vs OLMo 2
- Gemma-3 vs Mitral 3.1 Small 24
  - Mistral: mais rápido

## O Futuro dos LLMs

- MQD
- MOE

## Pergunta

- Como escolheram os pares: agruparam por dicotomias

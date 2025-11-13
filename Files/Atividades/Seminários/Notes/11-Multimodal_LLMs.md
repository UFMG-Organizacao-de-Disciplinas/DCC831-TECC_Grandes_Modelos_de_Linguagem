# 28/10/2025 - Multimodal LLMs

## Atrasado

## Motivation

## Main Challenges

- Janela de contexto limitada

## Classical Problems

- Visual Question Answering (VQA)
- Image Captioning and Description
- Document Understanding
- Multimodal Retrieval and Search
- Video Understanding
- Robotics and Embodied AI
- Medical Imaging Analysis
- Education and Accessibility

## MLLM Architecture: Overview

- Encoders
  - Primeiro extrai as features de cada modalidade
  - Caros de treinar
- Interface
- Generator

### Encoder

Usam encoders especializados (CLIP, ViT, HuBERT), ou unified encoders (ImageBind)

Os patches de imagens são convertidos em embeddings e entram no transformer.

### Interface

### Alignment/Fusion

Alinhamento antes das features entrarem na LLM

### Generator

- Stable Diffusion
- Brownian Motion

## Conclusão

- CLIP: aligned embedding
- BLIP-2: bridge via Q-Former
- Flamingo
- GPT-4V/Gemini

Não querem mais fazer fusão de feature, querem fazer algo mais conectado.

## Future Directions

## Applications

- Generative Fills
- Sora2, VEO3
- Descript
- Med-PaLM M
- Project Astra

## Dúvidas

DeepSeek OCR: ao invés da LLM processar o texto, deixar ela processar a imagem como pixel.

Conseguiram rodar Doom em IA

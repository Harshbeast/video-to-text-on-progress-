# video-to-text-on-progress-

This repository contains a PyTorch implementation of a video captioning model that combines spatial and temporal features using a dual-backbone encoder (ResNet152 + C3D) and a GPT-style Transformer decoder. To improve the alignment between visual features and text embeddings, the model utilizes a hybrid loss function combining Cross-Entropy and Contrastive Loss.
#Model Architecture
The model uses a dual-encoder strategy to capture both what is in the frame and how it moves:

Spatial Encoder (ResNet152): Extracts high-level visual features from 16 sampled frames.

Temporal Encoder (C3D): Captures motion dynamics and action sequences using 3D convolutions.

Fusion Layer: A projection and bottleneck layer that merges spatial and temporal cues into a 512-dimensional "Context Vector."

GPT-Style Decoder: A Transformer-based decoder that uses the Context Vector to autoregressively generate captions using tiktoken (GPT-2) subword embeddings.

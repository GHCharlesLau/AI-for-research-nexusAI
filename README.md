# AI-for-research-nexusAI
A replication project conducted by the group, NexusAI, for the course DOTE 6635 (AI for Business Research) which is instructed by Prof. Renyu (Philip) Zhang. Please refer to the link for detailed course information: https://github.com/rphilipzhang/AI-PhD-S26

The project replicates and extends the study published in https://pubsonline.informs.org/doi/abs/10.1287/mksc.2024.0990

Ye, Z., Yoganarasimhan, H., & Zheng, Y. (2025). Lola: Llm-assisted online learning algorithm for content experiments. _Marketing Science_, _44_(5), 995-1016.

Our replication is structured into two main parts, focusing on different methodological components of large language model (LLM) applications in business research.



## Part Ⅰ — Prompt- & Embedding-Based Methods

> The presentation recording for Part I (Prompt & Embedding) is available here: https://drive.google.com/file/d/1TGYBFzn8r9K5zAJ5JVqe4C5xfApNSNrR/view?usp=drive_link

This section focuses on reproducing and extending the prompt-based and embedding-based approaches described in the original paper.

### Prompt-Based Replication
- Reimplemented prompt design strategies described in the original study  
- Evaluated a frontier model (GPT-5.2) for performance comparison  

All prompt scripts and experiments are located in Prompt/For Prompt Engineering Method.ipynb

### Embedding-Based Replication
- Implemented the embedding pipeline described in the original study
- Replicated downstream analysis based on embedding features
- Compared embedding-based results with prompt-based approaches

The corresponding scripts and data processing pipelines can be found in Emedding/For CTR prediction using OpenAI Embedding.ipynb and Emedding/For CTR prediction using Word2Vec256 Embedding.ipynb.

> Due to file size limitations, some embedding outputs and intermediate data files are stored externally in https://drive.google.com/drive/folders/1a6JvDSDEV4S61U3jmvkfbh54J4TCytip?usp=drive_link



## Part Ⅱ — LLM-Assisted Online Learning Alogrithm

This section aims to replicate the LOLA framework which contains the finetuning technique as well because the finetuned model often reveals the optimal performance among pure LLM methods. As such, LOLA leverages the optimal LLM-based model to predict CTR priors. And then it further integrates 2-Upper Confidence Bounds (2-UCBs) policy to design a online learning algorithm based on the bandit framework. 

### Finetuning the LLMs

Finetuning is another LLM-based approach to predicting CTR in business, which often shows the optimal performance compared to the prompt-based method and the embedding-based method. Considering the limited computing resources, we opted for Low-Rank Adaptation (LoRA) as a parameter-efficient finetuning (PEFT) technique. 

Specifically, we chose Qwen3-8b as the foundation model due to the unavailability of Llama's model. We finetuned Qwen3-8b using the Aliyun's Platform for Artificial Intelligence ([PAI](https://www.aliyun.com/product/pai)). The corresponding applicable data and codes are in the folder `Finetuning`.

### LOLA: LLM-2UCBs

LOLA, as the proposed novel approach, integrates advantages of LLMs and the Bandit framework. Following the original article, we adopted a two-phase training process, including the LLM training stage and the online learning stage. The applicable data and codes are stored in the folder `LOLA`.

# 心理健康领域微调对话模型
# 项目概述

本项目旨在通过微调 LLaMA 模型来生成心理学领域的问答系统，具体包括焦虑症、抑郁症、PTSD、认知行为疗法等话题。项目通过对比微调前后模型的表现，以验证微调对模型回答质量的提升。

## 使用的工具

- **Transformers**：用于加载和微调 LLaMA 模型。
- **Torch**：深度学习框架，支持 GPU 加速。
- **PEFT (Parameter Efficient Fine-Tuning)**：用于优化微调过程，应用 LoRA技术，降低计算资源需求。
- **Huggingface Datasets**：用于加载和处理数据集。## 微调过程

## 微调数据集

-数据来源为灵心大模型所提供的开源数据，地址：https://www.modelscope.cn/datasets/YIRONGCHEN/SoulChatCorpus

### 1. 数据预处理

数据预处理部分主要是将原始数据（JSON 格式）转换为适用于 LLaMA 模型训练的格式，使用了 ChatML 格式。在 `Convert_Json.ipynb` 中实现了从 JSON 到 ChatML 的转换。

转换过程：
- 读取原始 JSON 数据。
- 遍历每个对话主题，将每个对话中的消息和角色（如 system、user、assistant）提取出来，并按 LLaMA 训练格式进行组织。
- 保存为 JSONL 格式，每行代表一个对话实例。

### 2. 微调模型准备

微调过程中，首先加载了 LLaMA 3.2 3B Instruct 模型，并对其进行了 LoRA 配置。LoRA 通过对模型的特定层进行低秩适配，有效减少了需要更新的参数数量，降低了显存占用。

**主要步骤**：
- **模型加载**：使用 `AutoTokenizer` 和 `AutoModelForCausalLM` 加载预训练的 LLaMA 3.2 3B Instruct 模型。
- **LoRA 配置**：通过 `LoraConfig` 设置 LoRA 参数，如 `r=8`，`lora_alpha=32`，以及选择性地对 `q_proj` 和 `v_proj` 层进行低秩适配。
- **数据处理**：数据被处理成适合 GPT 训练的数据格式，并分为训练集和验证集。

### 3. 训练过程

训练使用了 `Trainer` 类，它简化了训练过程，并提供了高效的多设备训练支持。训练时使用了半精度（FP16），并启用了梯度累积和检查点。

**训练参数**：
- **批量大小**：每设备训练批量为 2，评估批量为 1。
- **学习率**：设置为 `2e-5`，并使用余弦退火调度。
- **梯度检查点**：启用以减少内存消耗。
- **优化器**：使用 `AdamW` 优化器。
- **训练轮数**：训练了 3 个 epoch。

### 4. 微调后的模型保存

训练完成后，将微调后的模型和 tokenizer 保存到本地，以便后续使用和评估。

### 5. 模型评估

评估使用了 BLEU、余弦相似度（Cosine Similarity）、ROUGE 分数和简单的质量评分（基于回答长度）作为主要指标。

- **BLEU**：评估模型回答的精确度和流畅度，越高表示生成文本与标准答案的相似度越高。
- **余弦相似度**：通过计算句子嵌入的余弦相似度，评估文本的语义相似度。
- **ROUGE**：评估生成文本的召回率，特别是 ROUGE-1 作为关键指标。
- **质量评分**：简单地基于回答的字数判断回答的质量，字数较多的回答通常被认为更为详尽。

### 6. 微调前后比较

评估的结果显示了微调前后模型在各项指标上的变化，具体数据如下：

- **微调后**：
  - BLEU: 0.1831
  - 余弦相似度: 0.8920
  - ROUGE-1: 0.1176
  - 质量评分: 0.75
  
- **微调前**：
  - BLEU: 0.0203
  - 余弦相似度: 0.7482
  - ROUGE-1: 0.0422
  - 质量评分: 1.00

### 结果分析

- **BLEU 分数**：微调后的 BLEU 分数明显提高，表明微调后生成的回答更接近标准答案，文本质量得到了提升。
- **余弦相似度**：微调后的模型在语义上的相似度也有显著提高，表示模型在理解和生成更符合实际语境的回答。
- **ROUGE-1 分数**：微调后的 ROUGE-1 分数略有提升，表明生成的回答在与标准答案的覆盖度上有所增加。
- **质量评分**：微调后的质量评分有所下降，可能是因为微调后的回答更为精确和简洁，不再仅仅依赖字数来评估质量。

总体而言，微调显著提高了模型在多个评估指标上的表现，尤其是在生成文本的精确度和语义相似度方面。

## 文件说明

### `Convert_Json.ipynb`
负责将原始 JSON 数据转换为 ChatML 格式，以便用于训练。

### `FineTuning.py`
实现了模型的加载、LoRA 配置和微调过程，使用了训练数据集并保存微调后的模型。

### `Model_Evaluation_UnfineTuned.ipynb`
评估原始未微调模型的回答，并与标准答案进行对比，生成 JSON 格式的评估结果。

### `FineTuned_Evaluation.py`
评估微调后的模型回答，并与标准答案对比，输出评估结果。

### `Model_Comparison_Eval.py`
对比微调前后的模型表现，计算 BLEU、余弦相似度、ROUGE-1 和质量评分等评估指标，并输出平均结果。


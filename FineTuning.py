import os
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling
)
from datasets import load_dataset
from peft import LoraConfig, get_peft_model
import torch
from multiprocessing import freeze_support

# 自定义 prepare_model_for_training 函数
def prepare_model_for_training(model):
    """
    为训练准备模型：
    1. 禁用缓存（use_cache=False），以便与梯度检查点兼容
    2. 确保所有参数均可训练
    """
    if hasattr(model, "config"):
        model.config.use_cache = False
    model.train()  # 确保模型处于训练模式
    for param in model.parameters():
        param.requires_grad = True
    return model

# ==================== 路径配置 ====================
CACHE_DIR = "F:/transformers_cache"
DATA_PATH = os.path.abspath("../SoulChatCorpus/SoulChatCorpus-ChatML.json")
OUTPUT_DIR = os.path.abspath("../finetuned_model")

def main():
    # ==================== 硬件初始化 ====================
    torch.cuda.empty_cache()
    torch.backends.cuda.matmul.allow_tf32 = True

    # ==================== 模型加载 ====================
    model_path = "meta-llama/Llama-3.2-3B-Instruct"

    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        cache_dir=CACHE_DIR,
        padding_side="left"
    )
    tokenizer.pad_token = tokenizer.eos_token

    # 加载模型（不使用量化，直接加载全精度/半精度模型）
    base_model = AutoModelForCausalLM.from_pretrained(
        model_path,
        cache_dir=CACHE_DIR,
        device_map="auto",
        torch_dtype=torch.float16
    )

    # ==================== LoRA配置 ====================
    peft_config = LoraConfig(
        r=8,
        lora_alpha=32,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.05,
        task_type="CAUSAL_LM",
        inference_mode=False
    )
    base_model = prepare_model_for_training(base_model)
    model = get_peft_model(base_model, peft_config)
    model.config.use_cache = False  # 兼容梯度检查点

    # ==================== 数据处理管道 ====================
    def format_chatml(example):
        formatted_text = tokenizer.apply_chat_template(
            example["messages"],
            tokenize=False,
            add_generation_prompt=False
        )
        return {"text": formatted_text}

    dataset = load_dataset(
        "json",
        data_files=DATA_PATH,
        split="train"
    ).map(
        format_chatml,
        remove_columns=["messages"],
        num_proc=2 if torch.cuda.is_available() else None
    ).train_test_split(
        test_size=0.2,
        shuffle=True,
        seed=42
    )

    def tokenize_function(examples):
        tokenized = tokenizer(
            examples["text"],
            truncation=True,
            max_length=512,
            padding=False,
            add_special_tokens=False
        )
        # 显式设置 labels
        tokenized["labels"] = tokenized["input_ids"].copy()
        return tokenized

    dataset = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=["text"],
        num_proc=2 if torch.cuda.is_available() else None
    )

    # ==================== 训练参数配置 ====================
    training_args = TrainingArguments(
        output_dir="./results",
        per_device_train_batch_size=2,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=4,
        num_train_epochs=3,
        learning_rate=2e-5,
        weight_decay=0.01,
        fp16=True,
        tf32=True,
        gradient_checkpointing=True,
        optim="adamw_torch",
        evaluation_strategy="steps",
        eval_steps=200,
        logging_steps=50,
        save_strategy="steps",
        save_steps=400,
        lr_scheduler_type="cosine",
        warmup_ratio=0.1,
        group_by_length=True,
        dataloader_num_workers=2,
        label_names=["labels"],
        report_to=["tensorboard"] if os.path.exists("./logs") else [],
        load_best_model_at_end=True,
        torch_compile=True 
    )

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
        pad_to_multiple_of=16 
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"],
        data_collator=data_collator,
    )

    trainer.train()

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print(f"模型已保存至：{OUTPUT_DIR}")

if __name__ == '__main__':
    freeze_support()
    main()

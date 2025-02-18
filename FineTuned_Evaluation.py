import json
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# 配置微调后的模型路径
MODEL_PATH_FINETUNED = "../finetuned_model"  # 微调后模型

# 加载 Tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH_FINETUNED,
    cache_dir="F:/transformers_cache")

# 加载微调后的模型
def load_model(model_path):
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        cache_dir="F:/transformers_cache",
        device_map="auto"
    )
    return model

# 让模型回答问题
def generate_answer(model, question):
    inputs = tokenizer(question, return_tensors="pt").to("cuda")
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=200)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# 问题列表
questions = [
    "什么是焦虑症？它的常见症状是什么？",
    "抑郁症患者可能会有哪些负性认知？",
    "社交恐惧症和一般的害羞有什么区别？",
    "什么是 PTSD？它的核心症状是什么？",
    "双相情感障碍的‘躁狂发作’和‘抑郁发作’有什么区别？",
    "作为心理咨询师，如何在咨询过程中建立良好的信任关系？",
    "为什么心理咨询中要强调‘共情’？请举例说明。",
    "认知行为疗法（CBT）如何帮助患者改变负性思维模式？",
    "心理咨询师在面对情绪激动的来访者时应该如何处理？",
    "心理咨询师在什么情况下可以违反保密协议？",
    "抑郁症是什么？有哪些常见的治疗方法？",
    "如何有效地管理压力和焦虑？",
    "如何识别强迫症（OCD）的症状？",
    "如何识别 PTSD 的症状？",
    "情绪失调和情绪障碍有什么区别？",
    "情绪障碍的治疗方法有哪些？",
    "双相情感障碍的诊断标准是什么？",
    "心理咨询师如何帮助青少年应对心理压力？",
    "认知行为疗法的基本原则是什么？",
    "如何识别社交恐惧症？",
    "失眠如何影响心理健康？",
    "自残的心理原因是什么？",
    "边缘性人格障碍是什么？",
    "‘黑暗三合一’心理学指的是什么？",
    "什么是心理韧性？",
    "焦虑症的治疗方法有哪些？",
    "依恋理论的核心思想是什么？",
    "焦虑症和恐惧症有什么区别？",
    "什么是人格障碍？",
    "焦虑症的治疗方法有哪些？",
    "如何识别心理问题的早期迹象？",
    "社交焦虑症的应对策略有哪些？",
    "如何管理与创伤相关的情绪反应？",
    "心理咨询师如何帮助焦虑症患者？",
    "‘压力反应模型’在心理学中的含义是什么？",
    "自我意识在心理健康中的重要性是什么？",
    "情绪失调的迹象有哪些？",
    "强迫症（OCD）是什么？",
    "心理治疗中常用的治疗方法有哪些？",
    "依恋理论在亲密关系中的作用是什么？",
    "PTSD的长期影响是什么？",
    "认知失调在心理学中的含义是什么？",
    "抑郁症的认知模型是什么？",
    "如何管理情绪暴力？",
    "抑郁症患者如何有效寻求帮助？",
    "防御机制在心理学中的含义是什么？",
    "个体心理学是什么？",
    "如何评估一个人是否患有心理健康问题？",
    "如何治愈情感创伤？",
    "心理治疗如何帮助戒毒？",
    "慢性疼痛对心理健康有何影响？",
    "文化差异如何影响心理健康？"
]

# 加载微调后的模型
finetuned_model = load_model(MODEL_PATH_FINETUNED)

# 生成微调后的模型回答
finetuned_responses = {q: generate_answer(finetuned_model, q) for q in questions}

# 保存微调后模型的回答
with open("finetuned_responses.json", "w", encoding="utf-8") as f:
    json.dump(finetuned_responses, f, ensure_ascii=False, indent=2)

# 输出结果
print("微调后模型的回答已保存！")

# 也可以输出回答以查看
for q in questions:
    print(f"问题：{q}")
    print(f"微调后模型的回答：{finetuned_responses[q]}\n")

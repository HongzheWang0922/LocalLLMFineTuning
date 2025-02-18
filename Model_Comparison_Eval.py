import os
import json
import jieba
import numpy as np
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from sentence_transformers import SentenceTransformer, util
from rouge_score.rouge_scorer import RougeScorer

# 加载 JSON 文件（假设格式为 { "question_id": "回答文本", ... }）
def load_responses(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)

# 简单的回答质量评估（这里仅用回答的字数作为启发式指标）
def assess_quality(answer):
    # 这里简单认为字数多（说明回答较详细）的质量可能更高
    length = len(answer.strip())
    if length == 0:
        return 0.0
    # 可自行调整阈值
    if length < 20:
        return 0.5
    elif length < 50:
        return 0.7
    else:
        return 1.0

# 使用 jieba 分词处理中文文本
def tokenize(text):
    return list(jieba.cut(text))

def evaluate_responses(ft_responses, original_responses, standard_answers):
    # 初始化评估指标存储
    results = {}
    
    # 初始化平滑函数（用于 BLEU 计算）
    smoothie = SmoothingFunction().method4
    
    # 初始化 RougeScorer（使用 rouge_score 计算 ROUGE 分数）
    rouge_scorer_instance = RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    
    # 初始化 SentenceTransformer 模型（多语言模型可处理中文）
    embedder = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
    
    # 存储所有问题的各项指标，便于后续计算平均值
    ft_bleu_scores = []
    orig_bleu_scores = []
    ft_cosine_scores = []
    orig_cosine_scores = []
    ft_rouge1_scores = []
    orig_rouge1_scores = []
    ft_quality_scores = []
    orig_quality_scores = []
    
    # 对每个问题进行评估（以标准答案中的问题为准）
    for qid, standard_answer in standard_answers.items():
        # 获取微调后的回答和原始回答（如果缺失则为空字符串）
        ft_answer = ft_responses.get(qid, "")
        orig_answer = original_responses.get(qid, "")
        
        # 分词（参考回答和待评估回答均进行分词）
        ref_tokens = tokenize(standard_answer)
        ft_tokens = tokenize(ft_answer)
        orig_tokens = tokenize(orig_answer)
        
        # 计算 BLEU 分数（参考需要是 list of reference 分词列表）
        ft_bleu = sentence_bleu([ref_tokens], ft_tokens, smoothing_function=smoothie)
        orig_bleu = sentence_bleu([ref_tokens], orig_tokens, smoothing_function=smoothie)
        
        # 计算句子嵌入余弦相似度（原始文本直接输入）
        standard_emb = embedder.encode(standard_answer, convert_to_tensor=True)
        ft_emb = embedder.encode(ft_answer, convert_to_tensor=True)
        orig_emb = embedder.encode(orig_answer, convert_to_tensor=True)
        ft_cosine = util.cos_sim(ft_emb, standard_emb).item()
        orig_cosine = util.cos_sim(orig_emb, standard_emb).item()
        
        # 计算 ROUGE 分数，使用 RougeScorer
        # 注意：第一个参数为标准答案（参考答案），第二个为待评估回答
        ft_rouge_scores = rouge_scorer_instance.score(standard_answer, ft_answer)
        orig_rouge_scores = rouge_scorer_instance.score(standard_answer, orig_answer)
        # 这里我们取 rouge1 的 F1 分数作为指标
        ft_rouge1 = ft_rouge_scores["rouge1"].fmeasure
        orig_rouge1 = orig_rouge_scores["rouge1"].fmeasure
        
        # 评估回答质量（简单使用字数启发式）
        ft_quality = assess_quality(ft_answer)
        orig_quality = assess_quality(orig_answer)
        
        # 保存各项指标
        results[qid] = {
            "standard_answer": standard_answer,
            "ft_answer": ft_answer,
            "original_answer": orig_answer,
            "ft_bleu": ft_bleu,
            "original_bleu": orig_bleu,
            "ft_cosine": ft_cosine,
            "original_cosine": orig_cosine,
            "ft_rouge1": ft_rouge1,
            "original_rouge1": orig_rouge1,
            "ft_quality": ft_quality,
            "original_quality": orig_quality
        }
        
        ft_bleu_scores.append(ft_bleu)
        orig_bleu_scores.append(orig_bleu)
        ft_cosine_scores.append(ft_cosine)
        orig_cosine_scores.append(orig_cosine)
        ft_rouge1_scores.append(ft_rouge1)
        orig_rouge1_scores.append(orig_rouge1)
        ft_quality_scores.append(ft_quality)
        orig_quality_scores.append(orig_quality)
    
    # 计算所有问题的平均指标
    avg_scores = {
        "ft_bleu_avg": np.mean(ft_bleu_scores) if ft_bleu_scores else 0,
        "original_bleu_avg": np.mean(orig_bleu_scores) if orig_bleu_scores else 0,
        "ft_cosine_avg": np.mean(ft_cosine_scores) if ft_cosine_scores else 0,
        "original_cosine_avg": np.mean(orig_cosine_scores) if orig_cosine_scores else 0,
        "ft_rouge1_avg": np.mean(ft_rouge1_scores) if ft_rouge1_scores else 0,
        "original_rouge1_avg": np.mean(orig_rouge1_scores) if orig_rouge1_scores else 0,
        "ft_quality_avg": np.mean(ft_quality_scores) if ft_quality_scores else 0,
        "original_quality_avg": np.mean(orig_quality_scores) if orig_quality_scores else 0,
    }
    
    return results, avg_scores

def main():
    # 文件路径
    base_path = "./responses"
    ft_file = os.path.join(base_path, "FT_responses.json")
    original_file = os.path.join(base_path, "original_responses.json")
    standard_file = os.path.join(base_path, "standard_answers.json")
    
    # 加载数据
    ft_responses = load_responses(ft_file)
    original_responses = load_responses(original_file)
    standard_answers = load_responses(standard_file)
    
    # 评估
    results, avg_scores = evaluate_responses(ft_responses, original_responses, standard_answers)
    
    # 输出每个问题的评估结果（这里简单打印部分信息）
    for qid, metrics in results.items():
        print(f"问题 ID: {qid}")
        print(f"  标准答案: {metrics['standard_answer']}")
        print(f"  微调后回答: {metrics['ft_answer']}")
        print(f"    BLEU: {metrics['ft_bleu']:.4f}, 余弦相似度: {metrics['ft_cosine']:.4f}, ROUGE-1(F1): {metrics['ft_rouge1']:.4f}, 质量指标: {metrics['ft_quality']:.2f}")
        print(f"  微调前回答: {metrics['original_answer']}")
        print(f"    BLEU: {metrics['original_bleu']:.4f}, 余弦相似度: {metrics['original_cosine']:.4f}, ROUGE-1(F1): {metrics['original_rouge1']:.4f}, 质量指标: {metrics['original_quality']:.2f}")
        print("-" * 80)
    
    # 输出平均指标
    print("平均评估指标：")
    print(f"  微调后回答 - BLEU: {avg_scores['ft_bleu_avg']:.4f}, 余弦相似度: {avg_scores['ft_cosine_avg']:.4f}, ROUGE-1: {avg_scores['ft_rouge1_avg']:.4f}, 质量: {avg_scores['ft_quality_avg']:.2f}")
    print(f"  微调前回答 - BLEU: {avg_scores['original_bleu_avg']:.4f}, 余弦相似度: {avg_scores['original_cosine_avg']:.4f}, ROUGE-1: {avg_scores['original_rouge1_avg']:.4f}, 质量: {avg_scores['original_quality_avg']:.2f}")

if __name__ == "__main__":
    main()

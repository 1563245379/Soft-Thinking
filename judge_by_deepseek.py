import json
import argparse
import os
import re
import time
from openai import OpenAI


def extract_after_think(text: str) -> str:
    """提取</think>之后的内容，如果没有则返回最后2000字符"""
    pattern = r"</think>(.*)"
    match = re.search(pattern, text, re.DOTALL)
    return match.group(1).strip() if match else text[-2000:]


def build_judge_prompt(completion: str, ground_truth: str) -> str:
    """构建发送给DeepSeek的判断prompt"""
    answer_part = extract_after_think(completion)
    return f"""Please determine whether the final answer provided in the model-generated response is equivalent to the reference answer. The final answer may either be enclosed in \\boxed{{}} or appear after "Answer:" or "Final Answer". If they are equivalent, return "YES"; if they are not, return "NO". Only return "YES" or "NO", and do not generate any other content.

Model-generated answer: {answer_part}
Reference answer: {ground_truth}""".strip()


def judge_single(client: OpenAI, model: str, completion: str, ground_truth: str, max_retries: int = 3) -> bool:
    """调用DeepSeek API判断单个回答是否正确"""
    prompt = build_judge_prompt(completion, ground_truth)
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
            )
            msg = response.choices[0].message
            # deepseek-reasoner 把结果放在 reasoning_content，content 可能为空
            result = (msg.content or "").strip()
            if not result and hasattr(msg, "reasoning_content") and msg.reasoning_content:
                result = msg.reasoning_content.strip()
            print(f"  DeepSeek判断结果: {result}")
            return "YES" in result
        except Exception as e:
            print(f"  API调用失败 (尝试 {attempt + 1}/{max_retries}): {e}")
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)
    return False


def main():
    parser = argparse.ArgumentParser(description="使用DeepSeek API判断回答正确性")
    parser.add_argument("--input", type=str, default="results/results/QwQ-32B_aime2024_True_1_0.6_0.95_30_0.001_1.0_1.0_10_32768_0.01_256.json", help="输入JSON文件路径")
    parser.add_argument("--api_key", type=str, default=None, help="DeepSeek API Key (也可通过DEEPSEEK_API_KEY环境变量设置)")
    parser.add_argument("--api_base", type=str, default="https://api.deepseek.com", help="API base URL")
    parser.add_argument("--model", type=str, default="deepseek-reasoner", help="判断模型名称")
    parser.add_argument("--output", type=str, default=None, help="输出结果JSON文件路径")
    args = parser.parse_args()

    api_key = args.api_key or os.environ.get("DEEPSEEK_API_KEY")
    if not api_key:
        raise ValueError("请通过 --api_key 参数或 DEEPSEEK_API_KEY 环境变量提供API Key")

    client = OpenAI(api_key=api_key, base_url=args.api_base)

    with open(args.input, "r", encoding="utf-8") as f:
        data = json.load(f)

    total = len(data)
    correct = 0
    results = []

    print(f"共 {total} 道题目，开始评判...\n")

    for i, item in enumerate(data):
        ground_truth = item.get("ground_truth", "")
        completions = item.get("completion", [])
        prompt = item.get("prompt", "")[:80]

        # completion是一个列表，取第一个
        completion_text = completions[0] if completions else ""

        is_correct = judge_single(client, args.model, completion_text, ground_truth)

        if is_correct:
            correct += 1

        status = "✓" if is_correct else "✗"
        print(f"[{i + 1}/{total}] {status}  Ground Truth: {ground_truth}  | 题目: {prompt}...")

        results.append({
            "idx": item.get("idx", i),
            "ground_truth": ground_truth,
            "llm_judge_correct": is_correct,
        })

    accuracy = correct / total if total > 0 else 0
    print(f"\n{'=' * 50}")
    print(f"总题数: {total}")
    print(f"正确数: {correct}")
    print(f"正确率: {accuracy:.2%}")
    print(f"{'=' * 50}")

    # 保存结果
    output_path = args.output or args.input.replace(".json", "_llm_judge.json")
    summary = {
        "input_file": args.input,
        "judge_model": args.model,
        "total": total,
        "correct": correct,
        "accuracy": accuracy,
        "details": results,
    }
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"\n详细结果已保存到: {output_path}")


if __name__ == "__main__":
    main()

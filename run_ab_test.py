"""
A/B Test Runner: Qwen vs GPT on real Medical Monitoring questions.

Derives representative test cases from the actual EDC dataset:
1. AE术语归一化 (normalize_ae) -- 中文自由文本 -> MedDRA PT
2. CM-AE匹配 (match_cm_ae) -- 合并用药是否治疗某AE
3. 因果关系评估 (causality_assess) -- AE与药物因果推断
4. 数据清洗 (clean_data) -- typo/格式识别

Each question goes to both Qwen (qwen-max via DashScope compatible-mode)
and GPT (gpt-5.5 via rightcode proxy), results are compared and displayed.
"""

import sys
import os
import time
import json
from pathlib import Path

os.environ["PYTHONIOENCODING"] = "utf-8"
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.stderr.reconfigure(encoding="utf-8", errors="replace")

sys.path.insert(0, str(Path(__file__).parent))

from medical_monitoring.ai.engine import AIConfig, LLMEngine, LLMResponse


def print_header(title: str):
    print(f"\n{'='*70}")
    print(f"  {title}")
    print(f"{'='*70}")


def print_result(provider: str, resp: LLMResponse):
    latency = f"{resp.latency_ms:.0f}ms"
    tokens = resp.usage.get("output_tokens", "?")
    print(f"\n  [{provider.upper()}] (latency: {latency}, tokens: {tokens})")
    print(f"  {'─'*60}")
    content = resp.content.strip()
    for line in content.split("\n")[:20]:
        print(f"  │ {line}")
    if len(content.split("\n")) > 20:
        print(f"  │ ... ({len(content.split(chr(10)))} lines total)")


def compare_json_results(qwen_resp: LLMResponse, gpt_resp: LLMResponse) -> str:
    """Simple comparison of structured results."""
    qwen_json = qwen_resp.parse_json()
    gpt_json = gpt_resp.parse_json()

    if qwen_json is None and gpt_json is None:
        return "⚠ 双方均未返回有效JSON"
    if qwen_json is None:
        return "⚠ Qwen未返回有效JSON，GPT有"
    if gpt_json is None:
        return "⚠ GPT未返回有效JSON，Qwen有"

    if isinstance(qwen_json, dict) and isinstance(gpt_json, dict):
        common_keys = set(qwen_json.keys()) & set(gpt_json.keys())
        agree = sum(1 for k in common_keys
                    if str(qwen_json.get(k, "")).lower() == str(gpt_json.get(k, "")).lower())
        total = len(common_keys) if common_keys else 1
        score = agree / total
        return f"结构一致性: {score:.0%} ({agree}/{total} 字段匹配)"

    return "结构类型不匹配"


# =============================================================================
# TEST CASES derived from Y-6-LC-04 actual data
# =============================================================================

TEST_CASES = [
    # --- Task 1: AE术语归一化 ---
    {
        "id": "T1-normalize",
        "title": "AE术语归一化: 将自由文本映射到MedDRA PT",
        "task_type": "normalize_ae",
        "system": """你是MedDRA编码专家。将以下不良事件术语归一化为MedDRA首选术语(PT)。
输出严格JSON:
[
  {"original": "原始术语", "pt_cn": "中文PT", "pt_en": "English PT", "soc": "SOC", "confidence": 0.0-1.0}
]""",
        "user": """请归一化以下AE术语:
1. 脑渗血
2. 轻度贫血
3. 电解质紊乱
4. 卒中早期进展
5. 肝功能异常""",
    },

    # --- Task 2: CM-AE 治疗关系匹配 ---
    {
        "id": "T2-cm-ae",
        "title": "CM-AE匹配: 判断合并用药是否治疗某AE",
        "task_type": "match_cm_ae",
        "system": """你是临床药学专家。判断以下合并用药(CM)是否用于治疗对应的不良事件(AE)。
输出严格JSON:
[
  {"cm_drug": "药物名", "ae_name": "AE名", "treats_ae": true/false, "reasoning": "简要推理", "confidence": 0.0-1.0}
]""",
        "user": """请判断以下CM-AE对:
1. CM: 布洛芬缓释胶囊 → AE: 发热
2. CM: 阿托伐他汀钙片 → AE: 脑梗死复发
3. CM: 奥美拉唑肠溶胶囊 → AE: 消化道出血
4. CM: 氯化钾缓释片 → AE: 低钾血症
5. CM: 替罗非班注射液 → AE: 颅内出血""",
    },

    # --- Task 3: 因果关系评估 ---
    {
        "id": "T3-causality",
        "title": "因果关系评估: AE与抗血小板药物的因果推断",
        "task_type": "causality_assess",
        "system": """你是药物警戒专家。基于WHO-UMC因果关系评估标准，分析不良事件与研究药物的关联性。

WHO-UMC 因果关系类别：
- Certain: 时间合理 + 不能用其他原因解释 + 撤药后减轻 + 再激发阳性
- Probable/Likely: 时间合理 + 不太可能为其他原因 + 撤药后减轻
- Possible: 时间合理 + 可能由其他原因解释
- Unlikely: 时间关系不明确 + 其他原因更可能
- Conditional/Unclassified: 需要更多数据
- Unassessable/Unclassifiable: 信息不足

输出严格JSON:
{
  "temporal_relationship": "时间关系描述",
  "biological_plausibility": "生物学合理性",
  "confounding_factors": ["混杂因素列表"],
  "alternative_explanations": ["其他可能解释"],
  "suggested_category": "certain|probable|possible|unlikely|conditional|unassessable",
  "reasoning": "综合推理过程",
  "confidence": 0.0-1.0
}""",
        "user": """请评估以下AE与研究药物的因果关系：

研究药物: Y-6舌下片 (D-冰片6mg + 西洛他唑25mg), 抗血小板+神经保护
适应症: 急性缺血性卒中 (发病48h内)
IB已知风险: 出血(抗血小板机制)、头痛(血管扩张)、肝毒性(冰片代谢)

不良事件: 牙龈出血
AE严重度: Grade 1 (轻度)
AE开始日期: 给药后第3天 | AE结束日期: 第7天
研究者因果判断: 可能有关

合并症/病史: 高血压病史10年, 2型糖尿病
合并用药: 阿司匹林100mg qd, 依诺肝素 4000IU q12h, 氯吡格雷 75mg qd""",
    },

    # --- Task 4: 数据清洗/typo识别 ---
    {
        "id": "T4-clean",
        "title": "数据清洗: 识别EDC数据中的typo和格式问题",
        "task_type": "clean_data",
        "system": """你是临床数据管理专家。识别以下EDC数据字段中的问题（typo、格式错误、医学术语不规范等）。
输出严格JSON:
[
  {"original": "原始值", "issue_type": "typo|format|terminology|negation", "corrected": "修正值", "explanation": "说明", "confidence": 0.0-1.0}
]""",
        "user": """请检查以下EDC数据字段的质量问题:
1. AE名称: "脑渗血" (应为何种标准术语?)
2. 开始日期: "2026-13-05" (日期格式)
3. AE名称: "轻度头痛" (严重度是否应在Grade字段?)
4. 合并用药: "阿司匹林肠溶片片" (是否重复?)
5. 病史: "未见明显异常" (这是否定性描述, 如何在结构化字段处理?)""",
    },

    # --- Task 5: 实验室-AE关联 ---
    {
        "id": "T5-lab-ae",
        "title": "实验室异常-AE关联: 判断Lab异常是否应报告为AE",
        "task_type": "match_lb_ae",
        "system": """你是临床研究医学专家。判断以下实验室检查异常是否应该有对应的AE报告。
根据ICH E6(R2)，具有临床意义的实验室异常应报告为AE。

输出严格JSON:
[
  {"lab_test": "检查项目", "value": "异常值", "ref_range": "参考范围", "should_have_ae": true/false, "expected_ae_term": "期望AE术语", "reasoning": "推理", "confidence": 0.0-1.0}
]""",
        "user": """请判断以下实验室异常是否应有AE报告:
1. ALT: 185 U/L (参考范围: 0-40), 基线正常
2. 血红蛋白: 95 g/L (参考范围: 120-160), 基线110
3. 血小板: 85×10^9/L (参考范围: 100-300), 基线正常, 正在使用抗血小板药
4. 血钾: 3.3 mmol/L (参考范围: 3.5-5.5), 正在使用利尿剂
5. 白细胞: 3.2×10^9/L (参考范围: 4.0-10.0), 无合并用药""",
    },
]


def run_single_test(engine: LLMEngine, test: dict) -> dict:
    """Run a single test case against both providers."""
    print_header(f"[{test['id']}] {test['title']}")
    print(f"  问题概要: {test['user'][:80]}...")

    results = {"id": test["id"], "title": test["title"]}

    # Qwen
    try:
        t0 = time.time()
        qwen_resp = engine.complete(
            system=test["system"],
            user=test["user"],
            task_type=test["task_type"],
            provider="qwen",
        )
        print_result("Qwen", qwen_resp)
        results["qwen"] = {
            "content": qwen_resp.content,
            "latency_ms": qwen_resp.latency_ms,
            "tokens": qwen_resp.usage,
            "success": True,
        }
    except Exception as e:
        print(f"\n  [QWEN] ✗ 失败: {e}")
        results["qwen"] = {"success": False, "error": str(e)}
        qwen_resp = None

    # GPT
    try:
        gpt_resp = engine.complete(
            system=test["system"],
            user=test["user"],
            task_type=test["task_type"],
            provider="gpt",
        )
        print_result("GPT", gpt_resp)
        results["gpt"] = {
            "content": gpt_resp.content,
            "latency_ms": gpt_resp.latency_ms,
            "tokens": gpt_resp.usage,
            "success": True,
        }
    except Exception as e:
        print(f"\n  [GPT] ✗ 失败: {e}")
        results["gpt"] = {"success": False, "error": str(e)}
        gpt_resp = None

    # Comparison
    if qwen_resp and gpt_resp and qwen_resp.content and gpt_resp.content:
        comparison = compare_json_results(qwen_resp, gpt_resp)
        print(f"\n  [COMPARE] {comparison}")
        print(f"  [LATENCY] Qwen {qwen_resp.latency_ms:.0f}ms vs GPT {gpt_resp.latency_ms:.0f}ms")
        results["comparison"] = comparison
        results["latency_winner"] = "qwen" if qwen_resp.latency_ms < gpt_resp.latency_ms else "gpt"
    else:
        print(f"\n  [WARN] cannot compare (one or both failed)")

    return results


def main():
    print("""
╔══════════════════════════════════════════════════════════════════════╗
║          Medical Monitoring A/B Test: Qwen vs GPT                  ║
║                                                                      ║
║  Qwen: qwen-max (DashScope compatible-mode)                        ║
║  GPT:  gpt-4o (rightcode proxy: right.codes/codex/v1)              ║
╚══════════════════════════════════════════════════════════════════════╝
""")

    config_path = Path(__file__).parent / "medical_monitoring" / "config" / "ai_config.yaml"
    config = AIConfig.from_yaml(config_path)

    print(f"配置加载完成:")
    print(f"  - Qwen: model={config.providers['qwen'].model}, "
          f"base_url={config.providers['qwen'].base_url}")
    print(f"  - GPT:  model={config.providers['gpt'].model}, "
          f"base_url={config.providers['gpt'].base_url}")
    print(f"  - A/B Test: {config.ab_test_enabled}")
    print(f"  - Cache: {config.cache_enabled}")

    engine = LLMEngine(config)

    all_results = []
    total_start = time.time()

    for test in TEST_CASES:
        try:
            result = run_single_test(engine, test)
            all_results.append(result)
        except KeyboardInterrupt:
            print("\n\n[!] User interrupted")
            break
        except Exception as e:
            print(f"\n  [X] Test exception: {e}")
            all_results.append({"id": test["id"], "error": str(e)})

    total_time = time.time() - total_start

    # Summary
    print_header("A/B 测试汇总")
    qwen_ok = sum(1 for r in all_results if r.get("qwen", {}).get("success"))
    gpt_ok = sum(1 for r in all_results if r.get("gpt", {}).get("success"))
    total = len(all_results)

    print(f"  总测试数: {total}")
    print(f"  Qwen 成功: {qwen_ok}/{total}")
    print(f"  GPT  成功: {gpt_ok}/{total}")
    print(f"  总耗时: {total_time:.1f}s")

    if any(r.get("qwen", {}).get("success") for r in all_results):
        qwen_lats = [r["qwen"]["latency_ms"] for r in all_results if r.get("qwen", {}).get("success")]
        print(f"  Qwen 平均延迟: {sum(qwen_lats)/len(qwen_lats):.0f}ms")
    if any(r.get("gpt", {}).get("success") for r in all_results):
        gpt_lats = [r["gpt"]["latency_ms"] for r in all_results if r.get("gpt", {}).get("success")]
        print(f"  GPT  平均延迟: {sum(gpt_lats)/len(gpt_lats):.0f}ms")

    latency_wins = {"qwen": 0, "gpt": 0}
    for r in all_results:
        w = r.get("latency_winner")
        if w:
            latency_wins[w] += 1
    print(f"  延迟更快: Qwen {latency_wins['qwen']} / GPT {latency_wins['gpt']}")

    # Save results
    output_path = Path(__file__).parent / "ab_test_results.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2, default=str)
    print(f"\n  详细结果已保存: {output_path}")


if __name__ == "__main__":
    main()

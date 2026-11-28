"""Eyes-Agent 手机眼前节概率与临床文本融合模块。"""

from numbers import Real

from fusion import GlaucomaFusion


SMARTPHONE_RULES = """
你正在运行 Eyes-Agent 的手机眼前节研究演示。
请根据每只眼的手机眼前节模型预测概率，以及本例提供的临床表现，
分别给出该眼的青光眼预测概率。

【眼别对应】
1. 左眼、OS、O.S. 对应左眼。右眼、OD、O.D. 对应右眼。
2. 双眼、OU、双侧对应双眼。全身病史或明确的共同症状可供双眼参考。
3. 明确属于一侧眼睛的表现只用于该眼。
4. 仅评估提供了手机眼前节模型概率的眼睛。

【手机眼前节模型】
沿用本研究的模型分数分级：
- 概率 > 0.7：强阳性模型信号。
- 概率 0.6–0.7：中等阳性模型信号。
- 概率 0.5–0.6：弱阳性模型信号。
- 概率 < 0.5：阴性模型信号。

【临床表现】
从实际提供的文本中提取与该眼对应的信息，可以包括：
- 前房深度，以及已记录的房角、虹膜、角膜或瞳孔表现。
- 眼压和视野检查结果。
- 视力变化、眼部疼痛、头痛、恶心或虹视等已记录的症状。
模型提供的是一个概率值。请勿仅凭这个数值推断照片中看到了某个具体体征。

【综合判断】
1. 同时提供模型概率和临床文本时，沿用本研究提示词的相对权重：
   手机眼前节模型占 50%，临床表现占 50%。结合证据强弱给出最终概率。
2. 若模型概率低于 0.4，但该眼有明确的视野缺损或持续升高的眼压等
   强支持证据，可以上调最终概率。
3. 若模型概率低于 0.4，临床支持证据较弱，以模型判断为主。
4. 若模型概率高于 0.7，且有明确的相反证据，可以下调最终概率。
   临床文本未提及某项检查，与该项检查已有明确阴性结果应分别处理。
5. 如果没有提供临床文本，仅依据该眼已有的模型结果给出估计。
6. 双眼均有输入时，分别给出概率。每只眼以本眼证据为依据。

【输出】
仅输出所要求眼别的概率，每只眼一行，保留三位小数。
"""


def build_smartphone_prompt(eyes, patient_info=""):
    """
    构建手机专用融合提示词。

    单眼输入示例：
        {"L": {"anterior_prob": 0.68}}

    双眼输入示例：
        {
            "L": {"anterior_prob": 0.68},
            "R": {"anterior_prob": 0.31},
        }
    """
    if not isinstance(eyes, dict) or not eyes or set(eyes) - {"L", "R"}:
        raise ValueError("eyes 必须以 L 或 R 为键，可以传单眼或双眼。")

    if not isinstance(patient_info, str):
        raise TypeError("patient_info 应为临床文本；无文本时可使用默认空字符串。")

    evidence_lines = []
    output_lines = []

    for side, eye_name in (("L", "左眼"), ("R", "右眼")):
        if side not in eyes:
            continue

        values = eyes[side]
        if not isinstance(values, dict) or set(values) != {"anterior_prob"}:
            raise ValueError(
                f"{eye_name}的手机融合输入应只包含 anterior_prob。"
            )

        probability = values["anterior_prob"]
        if (
            isinstance(probability, bool)
            or not isinstance(probability, Real)
            or not 0 <= probability <= 1
        ):
            raise ValueError(
                f"{eye_name}的 anterior_prob 必须是 0 到 1 之间的有效数值。"
            )

        evidence_lines.append(
            f"{eye_name}手机眼前节模型青光眼概率：{probability:.3f}"
        )
        output_lines.append(f"{eye_name}：0.XXX")

    clinical_text = patient_info.strip() or "未提供"
    evidence = "\n".join(evidence_lines)
    output_format = "\n".join(output_lines)

    return (
        f"{SMARTPHONE_RULES.strip()}\n\n"
        f"=== 本例手机眼前节模型结果 ===\n{evidence}\n\n"
        f"=== 本例临床表现 ===\n{clinical_text}\n\n"
        f"=== 本例输出格式 ===\n{output_format}"
    )


class SmartphoneGlaucomaFusion(GlaucomaFusion):
    """使用手机专用提示词，复用 GlaucomaFusion 的 GPT-5 调用和结果解析。"""

    build_prompt = staticmethod(build_smartphone_prompt)
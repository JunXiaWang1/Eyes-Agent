"""Eyes-Agent 眼级概率融合模块。"""

import re
from numbers import Real

from openai import OpenAI


def build_prompt(eyes, patient_info=""):
    """
    根据单眼或双眼的模型输出构建提示词。

    eyes 以 "L" 和 "R" 表示左眼和右眼。
    每只眼可以提供以下字段：
        fundus_prob   眼底模型青光眼概率
        anterior_prob 眼前节模型青光眼概率，包括手机眼前节照片
        cdr           垂直杯盘比

    patient_info 为可选的脱敏临床文本。
    """
    if not eyes or set(eyes) - {"L", "R"}:
        raise ValueError("eyes 必须包含 L 或 R，双眼可同时提供。")

    labels = {
        "fundus_prob": "眼底模型青光眼概率",
        "anterior_prob": "眼前节模型青光眼概率",
        "cdr": "垂直杯盘比（CDR）",
    }

    has_left = "L" in eyes
    has_right = "R" in eyes

    prompt = (
        "请根据以下患者的检查数据和临床表现评估各眼的青光眼概率。\n"
        "=== 检查指标 ===\n"
    )

    for side, name in (("L", "左眼"), ("R", "右眼")):
        if side not in eyes:
            continue

        values = eyes[side]
        if not isinstance(values, dict) or set(values) - labels.keys():
            raise ValueError(f"{side} 眼输入字段有误。")

        lines = []
        for key, label in labels.items():
            value = values.get(key)
            if value is None:
                continue

            if (
                isinstance(value, bool)
                or not isinstance(value, Real)
                or not 0 <= value <= 1
            ):
                raise ValueError(f"{side}.{key} 应为 0 至 1 之间的数值。")

            lines.append(f"  - {label}: {value:.3f}")

        if not lines:
            raise ValueError(f"{side} 眼缺少有效输入。")

        prompt += name + ":\n" + "\n".join(lines) + "\n"

    patient_info = patient_info.strip() or "未提供"

    prompt += f"""
=== 临床表现 ===
{patient_info}

=== 重要说明：临床表现文本的左右眼对应方式 ===
临床表现来自同一个完整文本，可能同时包含左眼、右眼、双眼或未明确眼别的信息。
请你在内部先完成文本信息对应，不要直接把整段文本无差别套用到每只眼。

请按以下规则理解临床表现：
1. “左眼”、“左”、“OS”、“O.S.” 对应左眼。
2. “右眼”、“右”、“OD”、“O.D.” 对应右眼。
3. “双眼”、“OU”、“双侧”对应双眼共同信息。
4. 未明确眼别的信息，如果是全身病史、基础疾病、共同症状或无法区分眼别的描述，可以作为双眼共同参考。
5. 评估某只眼时，不要把对侧眼明确描述的异常直接套用到当前眼。
6. 如果当前只有一只眼有模型数据，请重点评估该眼，但仍需避免把明确属于对侧眼的信息错误套用过来。

=== 数据情况说明 ===
"""

    if has_left and not has_right:
        prompt += "当前只有左眼检查数据，请重点关注临床表现中与左眼相关的内容。\n"
    elif has_right and not has_left:
        prompt += "当前只有右眼检查数据，请重点关注临床表现中与右眼相关的内容。\n"
    else:
        prompt += "当前双眼都有检查数据，请分别评估每只眼睛的情况。\n"

    prompt += """
=== 诊断推理指南 ===
仅使用本例实际提供的证据；缺失指标不参与比较。若有垂直杯盘比，请结合该眼的眼底模型与临床表现综合解读。同一项测量在临床文本和工具中重复出现时，只计一次。
请综合考虑以下两个核心指标：

【核心指标1 - AI模型预测结果】：
1. 眼底模型预测：
   - 概率 > 0.7: 强阳性指标，高度提示青光眼
   - 概率 0.6-0.7: 中等阳性指标
   - 概率 0.5-0.6: 弱阳性指标
   - 概率 < 0.5: 阴性指标

2. 眼前节模型预测：
   - 概率 > 0.7: 强阳性指标，高度提示青光眼
   - 概率 0.6-0.7: 中等阳性指标
   - 概率 0.5-0.6: 弱阳性指标
   - 概率 < 0.5: 阴性指标

【模型差异处理】：
当两个AI模型预测结果存在显著差异时，请按以下规则处理：

1. 模型差异判断标准：
   - 差值 > 0.3: 显著差异
   - 差值 0.2-0.3: 中等差异
   - 差值 < 0.2: 轻微差异

2. 差异处理原则：
   - 当两个模型差值 > 0.3时，优先考虑眼底模型的结果
   - 当两个模型差值在0.2-0.3之间时，眼底模型权重稍高
   - 当两个模型差值 < 0.2时，两个模型权重相当

3. 优先眼底模型的理由：
   - 眼底模型直接观察视神经和视网膜，对青光眼诊断更直接
   - 眼底图像包含视盘、RNFL等青光眼关键特征
   - 眼前节模型主要观察前房角等结构，对开角型青光眼敏感性较低

4. 特殊情况处理：
   - 如果眼底模型概率 > 0.7且差值 > 0.3，强烈支持青光眼诊断
   - 如果眼前节模型概率 > 0.7但眼底模型概率 < 0.4且差值 > 0.3，结合临床表现判断

【核心指标2 - 临床表现】：
请仔细分析临床表现中的以下关键信息：

1. 眼压相关：
   - 眼压升高
   - 正常眼压范围：10-21mmHg

2. 视野检查：
   - 视野缺损
   - 视野检查异常是青光眼重要指标

3. 视神经相关：
   - 视盘改变
   - 杯盘比 C/D 增大
   - RNFL层变薄

4. 前房和房角：
   - 浅前房
   - 房角异常
   - 虹膜改变

5. 其他重要发现：
   - 晶体混浊程度
   - 玻璃体混浊
   - 黄斑区异常

【冲突处理规则 - 临床表现与模型判断不一致】：
当临床表现强烈提示青光眼但AI模型判断为阴性时，按以下规则处理：

1. 临床表现强烈提示青光眼的证据，满足以下任一条件：
   - 明确的视野缺损描述
   - 杯盘比 C/D > 0.7 且伴有视盘改变
   - 眼压持续升高 > 25mmHg
   - 明确的 RNFL 变薄描述

2. 冲突处理原则：
   - 如果临床表现有强烈青光眼证据，但两个AI模型概率都 < 0.4，可以提高最终判断概率
   - 如果临床表现只有中等或偏弱青光眼证据，但AI模型概率均 < 0.4，则以模型判断为主

3. 反向冲突处理：
   - 如果临床表现明确排除青光眼，但模型概率 > 0.7，可以适当降低最终判断概率
   - 如果临床表现无明显证据，但模型概率 > 0.7，则以模型判断为主

=== 单眼数据分析特别说明 ===
"""

    if has_left and not has_right:
        prompt += """- 当前只有左眼模型数据，请专注于左眼的判断。
- 在分析临床表现时，请优先使用与左眼相关的描述，例如“左眼”、“OS”等。
- 如果临床表现中明确提到右眼异常，不要直接套用到左眼。
- 如果临床表现描述的是双眼症状，可以合理参考到左眼。
- 如果临床表现未明确左右眼，但属于全身病史或共同表现，可以作为左眼辅助信息。
- 最终只输出左眼的青光眼概率。

"""
    elif has_right and not has_left:
        prompt += """- 当前只有右眼模型数据，请专注于右眼的判断。
- 在分析临床表现时，请优先使用与右眼相关的描述，例如“右眼”、“OD”等。
- 如果临床表现中明确提到左眼异常，不要直接套用到右眼。
- 如果临床表现描述的是双眼症状，可以合理参考到右眼。
- 如果临床表现未明确左右眼，但属于全身病史或共同表现，可以作为右眼辅助信息。
- 最终只输出右眼的青光眼概率。

"""
    else:
        prompt += """- 当前双眼都有模型数据，请分别独立评估每只眼睛。
- 左眼概率应主要基于左眼模型数据和左眼相关临床表现。
- 右眼概率应主要基于右眼模型数据和右眼相关临床表现。
- 临床表现中如果明确区分左右眼症状，请分别对应。
- 临床表现中如果写“双眼”或未明确眼别，可以作为双眼共同参考。
- 不要把一只眼明确异常的描述直接套用到另一只眼。

"""

    prompt += """=== 综合判断规则 ===
【优先级从高到低】：
1. 如果两个AI模型差值 > 0.3，优先考虑眼底模型结果
2. 如果任一AI模型显示强阳性，概率 > 0.7，考虑诊断为青光眼
3. 如果临床表现有强烈青光眼证据，考虑诊断为青光眼
4. 如果临床表现有明显青光眼证据，且任一AI模型显示弱阳性以上，考虑诊断为青光眼
5. 如果任一AI模型显示中等阳性，概率 > 0.6，且临床表现有任一支持性证据，考虑诊断为青光眼
6. 如果两个模型都显示弱阳性，概率 > 0.5，且临床表现有任一支持性证据，考虑诊断为青光眼
7. 如果任一模型显示弱阳性，概率 > 0.5，且临床表现有明显支持性证据，考虑诊断为青光眼
8. 如果AI模型和临床表现都无明显阳性证据，概率 < 0.5，考虑诊断为非青光眼
9. 如果临床表现与模型判断冲突，参照冲突处理规则

=== 权重说明 ===
- 眼底模型预测：重要，在模型差异大时权重更高
- 眼前节模型预测：重要
- 临床表现：重要，包含眼压、视野、视神经、前房等多维度信息
- 当两个模型差值 > 0.3 时：眼底模型权重 > 眼前节模型权重
- 总体重要性：AI模型 ≈ 临床表现

请给出一个0.0到1.0之间的概率值，表示该眼患有青光眼的可能性：
- 0.0-0.2: 极不可能，诊断为非青光眼
- 0.2-0.4: 可能性低，诊断为非青光眼
- 0.4-0.5: 可能性较低，诊断为非青光眼
- 0.5-0.6: 可能性较高，诊断为青光眼
- 0.6-0.8: 可能性高，诊断为青光眼
- 0.8-1.0: 极有可能，诊断为青光眼

=== 输出要求 ===
请严格按照以下格式输出，不要包含推理过程，只输出最终概率：
"""

    output_lines = []
    if has_left:
        output_lines.append("左眼：0.XXX")
    if has_right:
        output_lines.append("右眼：0.XXX")

    prompt += "\n".join(output_lines)
    prompt += "\n\n其中XXX是0到1之间的三位小数概率值\n\n你的判断结果: "
    return prompt


def parse_result(text, eyes):
    """解析 GPT 返回的逐眼概率，并核对眼别和数值范围。"""
    names = {"左": "L", "右": "R"}
    result = {}

    for line in text.strip().splitlines():
        line = line.strip()
        if not line:
            continue

        match = re.fullmatch(
            r"([左右])眼\s*[:：]\s*(0(?:\.\d+)?|1(?:\.0+)?)",
            line,
        )
        if not match:
            raise ValueError("GPT 输出格式异常，无法确定眼别或概率。")

        side = names[match[1]]
        if side in result:
            raise ValueError("GPT 返回了重复的眼别。")

        result[side] = float(match[2])

    if set(result) != set(eyes):
        raise ValueError("GPT 返回的眼别与输入不一致。")

    return result


class GlaucomaFusion:
    """使用 GPT-5 融合各眼的模型输出和可选临床信息。"""

    build_prompt = staticmethod(build_prompt)

    def __init__(self):
        # SDK 从运行环境中的 OPENAI_API_KEY 读取密钥。
        self.client = OpenAI(base_url="https://api.openai.com/v1")

    def predict(self, eyes, patient_info=""):
        """返回逐眼融合概率，例如 {"L": 0.623}。"""
        prompt = self.build_prompt(eyes, patient_info)

        response = self.client.chat.completions.create(
            model="gpt-5",
            reasoning_effort="medium",
            max_completion_tokens=32000,
            store=False,
            messages=[{"role": "user", "content": prompt}],
        )

        choice = response.choices[0]
        if (
            choice.finish_reason != "stop"
            or not choice.message.content
            or choice.message.refusal
        ):
            raise RuntimeError("GPT 未返回完整的融合概率。")

        return parse_result(choice.message.content, eyes)
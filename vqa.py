"""Eyes-Agent 单眼视觉问答模块。"""

import base64
import json
from pathlib import Path

from openai import OpenAI


IMAGE_TYPES = {
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".png": "image/png",
    ".webp": "image/webp",
}

BINARY_ANSWERS = ("Yes", "No")
TREATMENT_ANSWERS = ("surgery", "medication")

VQA_INSTRUCTIONS = """
You are answering a research visual question answering task for one eye.
Use the supplied images and the clinical text for the specified eye.

Eye side:
- Left eye: OS, LE, 左眼. Right eye: OD, RE, 右眼.
- Use findings attributed to the target eye and findings explicitly
  attributed to both eyes (OU, 双眼).
- Do not transfer a finding from the opposite eye to the target eye.

Question types:
- anterior_segment: prioritize the anterior-segment photographs.
  Use the clinical text for support when the photographs are unclear.
- fundus: prioritize the fundus photographs. Use the relevant clinical
  text for support when the photographs are unclear.
- treatment: answer the study's fixed-choice question using the documented
  treatment history or plan together with the images. Under the study's
  coding rules, surgery, laser, postoperative status, or planned surgery
  maps to "surgery". Medication-only treatment maps to "medication".
  If treatment is undocumented, follow the original benchmark convention:
  obvious advanced or urgent visual findings map to "surgery", otherwise
  use "medication". This is a benchmark label, not a treatment recommendation.

Output rules:
- For a binary question, return exactly "Yes" or "No".
- For a categorical treatment question, return "surgery" or "medication".
- The original benchmark uses "No" for an indeterminate binary finding.
- Answer each question_id exactly once. Return only the requested answers.
""".strip()


def load_questions(questions_path):
    """读取题目，仅保留推理需要的字段。"""
    data = json.loads(Path(questions_path).read_text(encoding="utf-8-sig"))
    rows = data.get("questions", [])

    if not isinstance(rows, list) or not rows:
        raise ValueError("问题.json 中必须包含非空的 questions 列表。")

    questions = []
    allowed_answers = {}

    for row in rows:
        question_id = row["question_id"]
        question_type = row["question_type"]
        answer_type = row["answer_type"]
        question_text = row["question"]

        if not isinstance(question_id, str) or not question_id.strip():
            raise ValueError("每道题都需要一个非空字符串 question_id。")
        if question_id in allowed_answers:
            raise ValueError(f"题号重复：{question_id}")
        if not isinstance(question_text, str) or not question_text.strip():
            raise ValueError(f"{question_id} 缺少题目正文。")
        if question_type not in {"anterior_segment", "fundus", "treatment"}:
            raise ValueError(f"{question_id} 的 question_type 无法识别。")

        if answer_type == "binary":
            choices = BINARY_ANSWERS
        elif question_type == "treatment" and answer_type == "categorical":
            choices = TREATMENT_ANSWERS
        else:
            raise ValueError(f"{question_id} 的题型与答案类型不匹配。")

        # 重新构造发给 GPT 的内容，原始行里的 expected_answer 不会被带入。
        questions.append({
            "question_id": question_id,
            "question_type": question_type,
            "answer_type": answer_type,
            "question": question_text,
            "options": list(choices),
        })
        allowed_answers[question_id] = choices

    return questions, allowed_answers


def image_input(image_path):
    """将本地图片转换为 Responses API 的图片输入。"""
    path = Path(image_path)
    mime = IMAGE_TYPES.get(path.suffix.lower())
    if mime is None:
        raise ValueError(f"不支持的图片格式：{path.name}")

    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return {
        "type": "input_image",
        "image_url": f"data:{mime};base64,{encoded}",
    }


class EyeVQA:
    """使用 GPT 回答一只眼的多道视觉问题。"""

    def __init__(self, model="gpt-5.2"):
        self.client = OpenAI()
        self.model = model

    def predict(
        self,
        *,
        side,
        fundus_images,
        anterior_images,
        clinical_text,
        questions_path,
    ):
        """
        side: "L" 表示左眼，"R" 表示右眼。
        fundus_images / anterior_images: 同一只眼的图片路径列表。
        clinical_text: 已脱敏的临床文本。
        questions_path: 问题 JSON 文件路径。
        """
        side_names = {"L": "left", "R": "right"}
        if side not in side_names:
            raise ValueError('side 必须为 "L" 或 "R"。')
        if not fundus_images and not anterior_images:
            raise ValueError("VQA 至少需要一张眼部图片。")

        questions, allowed_answers = load_questions(questions_path)

        case_data = {
            "target_eye": side_names[side],
            "clinical_text": clinical_text,
            "questions": questions,
        }
        content = [{
            "type": "input_text",
            "text": json.dumps(case_data, ensure_ascii=False),
        }]

        # 显式标记每组图片的模态，支持同一只眼有多张照片。
        for label, paths in (
            ("Anterior-segment photographs", anterior_images),
            ("Fundus photographs", fundus_images),
        ):
            if paths:
                content.append({"type": "input_text", "text": label})
                content.extend(image_input(path) for path in paths)

        # 每个题号单独指定合法答案。所有题号均为必填项。
        schema = {
            "type": "object",
            "properties": {
                qid: {"type": "string", "enum": list(choices)}
                for qid, choices in allowed_answers.items()
            },
            "required": list(allowed_answers),
            "additionalProperties": False,
        }

        response = self.client.responses.create(
            model=self.model,
            instructions=VQA_INSTRUCTIONS,
            input=[{"role": "user", "content": content}],
            text={
                "format": {
                    "type": "json_schema",
                    "name": "eye_vqa_answers",
                    "strict": True,
                    "schema": schema,
                }
            },
            store=False,
        )

        if response.status != "completed" or not response.output_text:
            raise RuntimeError("GPT 未返回完整的 VQA 答案。")

        predictions = json.loads(response.output_text)

        if set(predictions) != set(allowed_answers):
            raise ValueError("GPT 返回的题号与输入题号不一致。")

        answers = []
        for question in questions:
            qid = question["question_id"]
            answer = predictions[qid]
            if answer not in allowed_answers[qid]:
                raise ValueError(f"题目 {qid} 返回了无效答案：{answer}")
            answers.append({"question_id": qid, "answer": answer})

        return {"side": side, "answers": answers}
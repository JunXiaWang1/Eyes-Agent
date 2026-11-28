"""Eyes-Agent 统一演示入口。"""

import argparse
import json
from pathlib import Path
from statistics import mean


ROOT = Path(__file__).resolve().parent
EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}


def image_paths(eye_dir, name):
    """
    获取某只眼、某种模态的全部图片。

    支持以下放法：
    1. fundus.jpg
    2. fundus_01.jpg、fundus_02.jpg
    3. fundus/ 子目录中的图片

    眼前节对应的名称是 anterior_segment。
    """
    paths = [
        path
        for path in eye_dir.iterdir()
        if path.is_file()
        and path.suffix.lower() in EXTENSIONS
        and (
            path.stem == name
            or path.stem.startswith(name + "_")
        )
    ]

    folder_names = {
        "fundus": ("fundus", "眼底"),
        "anterior_segment": ("anterior_segment", "眼前节"),
    }

    for folder_name in folder_names[name]:
        folder = eye_dir / folder_name
        if folder.is_dir():
            paths.extend(
                path
                for path in folder.iterdir()
                if path.is_file() and path.suffix.lower() in EXTENSIONS
            )

    return sorted(paths)


def get_eye_dirs(case_dir, side):
    """
    确定病例中的眼别。

    图片直接放在病例目录时，用 --side L 或 --side R 指定眼别。
    如果病例目录中有 L/、R/，就按子目录识别眼别。
    """
    if not case_dir.is_dir():
        raise ValueError(f"病例目录不存在：{case_dir}")

    eye_dirs = {
        eye_side: case_dir / eye_side
        for eye_side in ("L", "R")
        if (case_dir / eye_side).is_dir()
    }

    if eye_dirs:
        if side and side not in eye_dirs:
            raise ValueError(f"病例中没有 {side} 子目录。")

        if side:
            return {side: eye_dirs[side]}

        return eye_dirs

    if not side:
        raise ValueError(
            "图片直接放在病例目录时，请使用 --side L 或 --side R 指定眼别。"
        )

    return {side: case_dir}


def read_clinical(case_dir, mode="standard"):
    """
    读取病例目录中的临床文本。

    手机模式只读取 clinical_smartphone.txt。其他模式读取完整临床文本。
    没有临床文本时返回空字符串。
    """
    if mode == "smartphone":
        names = ("clinical_smartphone.txt",)
    else:
        names = ("clinical.txt", "病人表现信息.txt")

    for name in names:
        path = case_dir / name

        if path.is_file():
            data = path.read_bytes()
            try:
                return data.decode("utf-8-sig").strip()
            except UnicodeDecodeError:
                return data.decode("gb18030").strip()

    return ""


def run_fusion(mode, eye_dirs, weights_dir, clinical_text, device):
    """运行常规融合或手机融合。"""
    fundus_model = None
    anterior_model = None
    evidence = {}

    if mode == "smartphone":
        from anterior_segment.smartphone import (
            SmartphoneAnteriorSegmentClassifier,
        )
        from smartphone_fusion import SmartphoneGlaucomaFusion

        anterior_class = SmartphoneAnteriorSegmentClassifier
        fusion_class = SmartphoneGlaucomaFusion
    else:
        from anterior_segment.anterior_segment import (
            AnteriorSegmentClassifier,
        )
        from fusion import GlaucomaFusion

        anterior_class = AnteriorSegmentClassifier
        fusion_class = GlaucomaFusion

    for side, eye_dir in eye_dirs.items():
        values = {}

        # 常规模式：有眼底图时，计算该眼的平均眼底模型概率。
        if mode == "standard":
            fundus_images = image_paths(eye_dir, "fundus")

            if fundus_images:
                if fundus_model is None:
                    from fundus.fundus import FundusClassifier

                    fundus_model = FundusClassifier(
                        weights_dir / "fundus.pth",
                        device=device,
                    )

                values["fundus_prob"] = mean(
                    fundus_model.predict(path)
                    for path in fundus_images
                )

                # 当前支持读取已经计算好的单眼 CDR。
                cdr_path = eye_dir / "cdr.txt"
                if cdr_path.is_file():
                    cdr_text = cdr_path.read_text(
                        encoding="utf-8-sig"
                    ).strip()
                    values["cdr"] = float(cdr_text)

        # 普通眼前节和手机眼前节使用各自的图片预处理。
        anterior_images = image_paths(eye_dir, "anterior_segment")

        if anterior_images:
            if anterior_model is None:
                anterior_model = anterior_class(
                    weights_dir / "anterior_segment.pth",
                    device=device,
                )

            values["anterior_prob"] = mean(
                anterior_model.predict(path)
                for path in anterior_images
            )

        if not values:
            raise ValueError(
                f"{side} 眼没有可用于 {mode} 模式的图像：{eye_dir}"
            )

        evidence[side] = values

    # 双眼病例用同一次 GPT 调用，分别返回左眼和右眼概率。
    fusion = fusion_class()
    predictions = fusion.predict(
        eyes=evidence,
        patient_info=clinical_text,
    )

    return {
        "mode": mode,
        "visual_results": evidence,
        "fusion_probabilities": predictions,
    }


def run_vqa(eye_dirs, clinical_text, model):
    """逐眼执行 VQA。每只眼使用自己的图片和问题文件。"""
    from vqa import EyeVQA

    vqa = EyeVQA(model=model)
    results = {}

    for side, eye_dir in eye_dirs.items():
        questions_path = eye_dir / "问题.json"

        if not questions_path.is_file():
            raise ValueError(
                f"缺少 {side} 眼的问题文件：{questions_path}"
            )

        results[side] = vqa.predict(
            side=side,
            fundus_images=image_paths(eye_dir, "fundus"),
            anterior_images=image_paths(eye_dir, "anterior_segment"),
            clinical_text=clinical_text,
            questions_path=questions_path,
        )

    return {
        "mode": "vqa",
        "results": results,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Eyes-Agent 常规融合、手机融合和视觉问答演示"
    )

    parser.add_argument(
        "--mode",
        required=True,
        choices=["standard", "smartphone", "vqa"],
        help="选择演示模式",
    )
    parser.add_argument(
        "--case",
        required=True,
        type=Path,
        help="病例目录，相对路径以 demo.py 所在目录为准",
    )
    parser.add_argument(
        "--side",
        choices=["L", "R"],
        help="图片直接放在病例目录时，指定实际眼别",
    )
    parser.add_argument(
        "--device",
        choices=["cuda", "cpu"],
        help="视觉模型的运行设备，默认自动选择",
    )
    parser.add_argument(
        "--vqa-model",
        default="gpt-5.2",
        help="VQA 使用的模型，默认沿用原 VQA 脚本的 gpt-5.2",
    )

    args = parser.parse_args()
    case_dir = ROOT / args.case

    try:
        eye_dirs = get_eye_dirs(case_dir, args.side)
        clinical_text = read_clinical(case_dir, args.mode)

        if args.mode == "vqa":
            result = run_vqa(
                eye_dirs=eye_dirs,
                clinical_text=clinical_text,
                model=args.vqa_model,
            )
        else:
            result = run_fusion(
                mode=args.mode,
                eye_dirs=eye_dirs,
                weights_dir=ROOT / "weights",
                clinical_text=clinical_text,
                device=args.device,
            )

    except (ValueError, FileNotFoundError) as exc:
        parser.exit(1, f"错误：{exc}\n")

    print(
        json.dumps(
            result,
            ensure_ascii=False,
            indent=2,
            allow_nan=False,
        )
    )


if __name__ == "__main__":
    main()
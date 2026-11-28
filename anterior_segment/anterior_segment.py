"""Eyes-Agent 眼前节图像分类模块。"""

import math

import torch
from PIL import Image
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from torchvision import transforms

from .models_vit import RETFound_mae


class AnteriorSegmentClassifier:
    """加载眼前节学生模型，预测单张眼前节图的青光眼概率。"""

    def __init__(self, checkpoint_path, device=None):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)

        # 参数与 evaluate_testset.py 中的 build_model() 一致。
        self.model = RETFound_mae(
            img_size=512,
            num_classes=2,
            drop_path_rate=0.1,
            global_pool=True,
        )

        # 原蒸馏 checkpoint 包含训练参数，仅加载来源可信的权重。
        checkpoint = torch.load(
            checkpoint_path,
            map_location="cpu",
            weights_only=False,
        )
        state_dict = checkpoint.get("model", checkpoint)

        # 兼容分布式训练保存的 module. 前缀。
        state_dict = {
            key[len("module."):] if key.startswith("module.") else key: value
            for key, value in state_dict.items()
        }
        self.model.load_state_dict(state_dict, strict=True)
        self.model.to(self.device)
        self.model.eval()

        # 与原 EyeDataset 的 test 预处理一致。
        self.transform = transforms.Compose([
            transforms.Resize(
                512,
                interpolation=transforms.InterpolationMode.BICUBIC,
            ),
            transforms.CenterCrop(512),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=IMAGENET_DEFAULT_MEAN,
                std=IMAGENET_DEFAULT_STD,
            ),
        ])

    @torch.inference_mode()
    def predict(self, image_path):
        """返回单张眼前节图的青光眼概率，范围为 0 到 1。"""
        with Image.open(image_path) as image:
            image_tensor = self.transform(image.convert("RGB"))

        inputs = image_tensor.unsqueeze(0).to(self.device)

        # 对应原测试中 B=1、N=1 的情况：每张图单独提特征并分类。
        features = self.model.forward_features(inputs)
        features = features.reshape(1, -1)
        logits = self.model.head(features)

        if logits.shape != (1, 2):
            raise RuntimeError(
                f"眼前节模型预期输出形状为 (1, 2)，实际为 {tuple(logits.shape)}。"
            )

        # 原测试脚本明确使用第 1 类作为青光眼概率。
        glaucoma_probability = torch.softmax(logits, dim=-1)[0, 1].item()

        if not math.isfinite(glaucoma_probability):
            raise ValueError("眼前节模型输出了无效概率，请检查图像和模型权重。")

        return glaucoma_probability
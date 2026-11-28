"""Eyes-Agent 眼底图像分类模块。"""

import math
from contextlib import nullcontext

import torch
from PIL import Image
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from torchvision import transforms

from .models_vit import RETFound_mae


class FundusClassifier:
    """加载眼底分类模型，并预测单张眼底图的青光眼概率。"""

    def __init__(self, checkpoint_path, device=None):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)

        # 参数与原眼底模型的测试命令一致。
        self.model = RETFound_mae(
            img_size=512,
            num_classes=2,
            drop_path_rate=0.2,
            global_pool=True,
        )

        # 原实验的 checkpoint 含有训练参数，仅加载来源可信的权重。
        checkpoint = torch.load(
            checkpoint_path,
            map_location="cpu",
            weights_only=False,
        )
        state_dict = checkpoint.get("model", checkpoint)
        self.model.load_state_dict(state_dict, strict=True)
        self.model.to(self.device)
        self.model.eval()

        # 与原 datasets.py 的测试预处理一致：短边缩放，再中心裁剪。
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
        """返回单张眼底图的青光眼概率，范围为 0 到 1。"""
        with Image.open(image_path) as image:
            image_tensor = self.transform(image.convert("RGB"))

        inputs = image_tensor.unsqueeze(0).to(self.device)

        # 与原 evaluate() 一致，GPU 前向推理使用混合精度。
        precision = (
            torch.autocast(device_type="cuda", dtype=torch.float16)
            if self.device.type == "cuda"
            else nullcontext()
        )
        with precision:
            logits = self.model(inputs)

        if logits.shape != (1, 2):
            raise RuntimeError(
                f"眼底模型预期输出形状为 (1, 2)，实际为 {tuple(logits.shape)}。"
                "请核对 models_vit.py 和 timm 版本是否与原测试环境一致。"
            )

        # 类别 0 = 非青光眼，类别 1 = 青光眼。
        glaucoma_probability = torch.softmax(logits, dim=1)[0, 1].item()

        if not math.isfinite(glaucoma_probability):
            raise ValueError("眼底模型输出了无效概率，请检查输入图像和模型权重。")

        return glaucoma_probability
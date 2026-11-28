"""Eyes-Agent 手机眼前节照片分类模块。"""

from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from torchvision import transforms

from .anterior_segment import AnteriorSegmentClassifier


class SmartphoneAnteriorSegmentClassifier(AnteriorSegmentClassifier):
    """使用眼前节模型预测单张手机眼前节照片的青光眼概率。"""

    def __init__(self, checkpoint_path, device=None):
        super().__init__(checkpoint_path, device=device)

        # 与手机逐图预测脚本 eval_save_results_shouji.py 保持一致。
        self.transform = transforms.Compose([
            transforms.Resize((512, 512)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=IMAGENET_DEFAULT_MEAN,
                std=IMAGENET_DEFAULT_STD,
            ),
        ])
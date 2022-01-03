import lpips
import torch
from mutil.pytorch_utils import ImageNetNormalizeTransformInverse
from pytorch_lightning.metrics.functional import psnr, ssim
from torchmetrics import FID as fid


class AbstractMetric(torch.nn.Module):
    scale = 255

    def __init__(self):
        super().__init__()
        self.unnormalize = ImageNetNormalizeTransformInverse(scale=self.scale)


class PSNR(AbstractMetric):
    scale = 255

    def forward(self, preds, target):
        preds = self.unnormalize(preds)
        target = self.unnormalize(target)
        return psnr(preds, target, data_range=self.scale)


class SSIM(AbstractMetric):
    scale = 255

    def forward(self, preds, target):
        preds = self.unnormalize(preds)
        target = self.unnormalize(target)
        return ssim(preds, target, data_range=self.scale)


class LPIPS(AbstractMetric):
    scale = 1

    def __init__(self):
        super().__init__()
        self.metric = lpips.LPIPS(net='alex', verbose=False)

    def forward(self, preds, target):
        preds = self.unnormalize(preds)
        target = self.unnormalize(target)
        # Might need a .mean()
        return self.metric(preds, target, normalize=True).mean()  # With normalize, lpips expects the range [0, 1]

class FID(AbstractMetric):
    scale = 255
    features = 2048
    def __init__(self):
        super().__init__()
        self.fid = fid(feature=self.features)

    def forward(self, preds, target):
        preds = self.unnormalize(preds)
        target = self.unnormalize(target)

        self.fid.update((preds.detach() * self.scale).type(torch.uint8), real = False)
        self.fid.update((target.detach() * self.scale).type(torch.uint8), real = True)

        return self.fid.compute()
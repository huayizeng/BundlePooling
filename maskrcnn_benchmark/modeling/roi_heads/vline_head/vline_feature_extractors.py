from torch import nn
from torch.nn import functional as F

from maskrcnn_benchmark.modeling.poolers import Pooler
from maskrcnn_benchmark.layers import Conv2d

class VLineFPNFeatureExtractor(nn.Module):
    """
    Feature extractor for VLine prediction
    """

    def __init__(self, cfg):
        """
        Arguments:
            cfg: YACS config node containing configuration settings
        """
        super(VLineFPNFeatureExtractor, self).__init__()

        resolution = cfg.MODEL.VLINE_HEAD.POOLER_RESOLUTION
        scales = cfg.MODEL.VLINE_HEAD.POOLER_SCALES
        sampling_ratio = cfg.MODEL.VLINE_HEAD.POOLER_SAMPLING_RATIO
        input_size = cfg.MODEL.RESNETS.BACKBONE_OUT_CHANNELS
        layers = cfg.MODEL.VLINE_HEAD.CONV_LAYERS
        self.pooler = Pooler(output_size=(resolution, resolution), scales=scales,
            sampling_ratio=sampling_ratio)

        next_feature = input_size
        self.blocks = []

        for layer_idx, layer_features in enumerate(layers, 1):
            layer_name = "vp_mask_fcn{}".format(layer_idx)
            module = Conv2d(next_feature, layer_features, 3, stride=1, padding=1)
            nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
            nn.init.constant_(module.bias, 0)
            self.add_module(layer_name, module)
            next_feature = layer_features
            self.blocks.append(layer_name)

    def forward(self, x, proposals):
        x = self.pooler(x, proposals)
        return x


_ROI_VP_MASK_FEATURE_EXTRACTORS = {
    "VLineFPNFeatureExtractor": VLineFPNFeatureExtractor,
}

class VLineSimpleFeatureExtractor(nn.Module):

    def __init__(self, cfg):
        super(VLineFeatureExtractor, self).__init__()

    def forward(self, x, proposals):
        x = F.relu(x[0])
        return x

def make_vline_feature_extractor(cfg):
    return VLineFPNFeatureExtractor(cfg)

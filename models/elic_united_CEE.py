from modules.transform import *

from .elic_united import ELIC_united


class ELIC_united_CEE(ELIC_united):
    def __init__(self, config, **kwargs):
        super().__init__(config, **kwargs)
        self.g_a = AnalysisTransformEXcro(config.N, config.M, act=nn.ReLU)
        self.g_s = SynthesisTransformEXcro(config.N, config.M, act=nn.ReLU)

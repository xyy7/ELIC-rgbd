from .Cheng2020withCKBD import Cheng2020AnchorwithCheckerboard
from .elic import ELIC
from .elic_master import ELIC_master
from .elic_united import ELIC_united
from .elic_united_CEE import ELIC_united_CEE
from .elic_united_CPT import ELIC_united_CPT
from .elic_united_R2D import ELIC_united_R2D
from .stf import SymmetricalTransFormer
from .mlicpp import MLICPlusPlus

# 先找复杂的
modelZoo = {
    "ckbd": Cheng2020AnchorwithCheckerboard,
    "ELIC_united_CPT": ELIC_united_CPT,
    "ELIC_united_CEE": ELIC_united_CEE,
    "ELIC_united_R2D": ELIC_united_R2D,
    "ELIC_united": ELIC_united,
    "ELIC_master": ELIC_master,
    "ELIC": ELIC,
    "STF": SymmetricalTransFormer,
    "MLIC":MLICPlusPlus
}

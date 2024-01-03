from .Cheng2020withCKBD import Cheng2020AnchorwithCheckerboard
from .elic import ELIC
from .elic_master import ELIC_master
from .elic_united import ELIC_united
from .elic_united_CEE import ELIC_united_CEE
from .elic_united_CPT import ELIC_united_CPT
from .elic_united_R2D import ELIC_united_R2D
from .mlicpp import MLICPlusPlus
from .stf import SymmetricalTransFormer
from .stf_united import SymmetricalTransFormerUnited
from .stf_united_CPT import SymmetricalTransFormerUnited_CPT
from .stf_united_EEM import SymmetricalTransFormerUnited_EEM

# 先找复杂的
modelZoo = {
    "ckbd": Cheng2020AnchorwithCheckerboard,
    "ELIC_united_CPT": ELIC_united_CPT,
    "ELIC_united_CEE": ELIC_united_CEE,
    "ELIC_united_R2D": ELIC_united_R2D,
    "ELIC_united": ELIC_united,
    "ELIC_master": ELIC_master,
    "ELIC": ELIC,
    "STF_united_CPT": SymmetricalTransFormerUnited_CPT,
    "STF_united_EEM": SymmetricalTransFormerUnited_EEM,
    "STF_united": SymmetricalTransFormerUnited,
    "STF": SymmetricalTransFormer,
    "MLIC": MLICPlusPlus,
}

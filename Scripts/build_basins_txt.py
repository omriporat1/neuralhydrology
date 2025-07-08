from nhWrap.neuralhydrology.neuralhydrology.nh_run import start_training
from nhWrap.neuralhydrology.neuralhydrology.utils.config import Config
from pathlib import Path
from utils.configs import *

selector = {
    'camels': [],
    'camelsaus': [],
    'camelsbr': [],
    'camelscl': [],
    'camelsgb': [],
    'hysets': [],
    'lamah': [],
    'il': "all"
}

generate_basins_txt(selector, 'configs/all_basins.yaml', 'RT_flood/basins/all_il_basins.txt')
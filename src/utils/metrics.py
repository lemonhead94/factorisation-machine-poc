import math

import numpy as np


def getHitRatio(recommend_list, gt_item) -> float:
    if gt_item in recommend_list:
        return 1
    else:
        return 0


def getNDCG(recommend_list, gt_item) -> float:
    idx = np.where(recommend_list == gt_item)[0]
    if len(idx) > 0:
        return math.log(2) / math.log(idx + 2)
    else:
        return 0

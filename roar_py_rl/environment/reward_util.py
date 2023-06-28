import numpy as np
from typing import Optional

def near_quadratic_bound(value, target, left_margin, right_margin, out_of_margin_activation : Optional[str] = "linear", power = 2.0, value_at_margin = 0.0):
    delta = value-target
    fract = delta/right_margin if delta > 0 else delta/left_margin
    
    if out_of_margin_activation is None or out_of_margin_activation != "near_quadratic":
        clipped_fract = np.clip(fract, -1.0, 1.0)
        rew = 1 - (1-value_at_margin) * (np.abs(clipped_fract) ** power)
        oodfract = fract - clipped_fract
        if out_of_margin_activation == "linear":
            rew -= (1-value_at_margin) * np.abs(oodfract)
        elif out_of_margin_activation == "quadratic":
            rew -= (1-value_at_margin) * (oodfract ** 2)
        elif out_of_margin_activation == "gaussian":
            rew += value_at_margin * np.exp(-oodfract**2/0.25)
    elif out_of_margin_activation == "near_quadratic":
        rew = 1 - (1-value_at_margin) * (np.abs(fract) ** power)
    return rew
from typing import Dict, Optional


def update_dict_ema(cur: Optional[Dict[str, float]], new: Dict[str, float], exp=0.9):
    if not cur:
        return new

    return {
        k: cur[k] * exp + (1 - exp) * v for k, v in new.items()
    }

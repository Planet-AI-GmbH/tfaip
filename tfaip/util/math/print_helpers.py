from typing import Any, Dict
import numpy as np
import logging

logger = logging.getLogger(__name__)


def printable_of_batched_dict(v: Dict[str, Any]) -> Dict[str, Any]:
    """Basically compute the mean of each element

    If there is no mean, the element is skipped with 'nan'
    """

    out = {}
    for k, v in v.items():
        try:
            out[k] = np.mean(v)
        except TypeError:
            out[k] = "nan"
        except Exception as e:
            logging.exception(e)
            logging.warning(f"An unknown exception occurred when converting {k}={v} to a string")
            out[k] = "Unknown exception"

    return out

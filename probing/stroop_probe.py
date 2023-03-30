from abc import ABC
from typing import List

import numpy as np
import torch

from probing.probe import Probe


class StroopProbe(Probe, ABC):
    def __init__(self, model_name: str, model_pretrained: str, device: torch.device = torch.device('cuda')):
        super().__init__(model_name, model_pretrained, device)

    def score(self, base_sentence: str, sentence_list: List[str]) -> np.ndarray:
        v = self._embed([base_sentence] + sentence_list)

        base = v[0]
        modified = v[1:]
        scores = modified @ base

        return scores

    def _tokenize(self, prompts: List[str]) -> np.ndarray:
        raise NotImplementedError

    def _embed(self, prompts: List[str]) -> np.ndarray:
        raise NotImplementedError

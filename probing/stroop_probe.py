from abc import ABC
from typing import List, Union, Dict

import numpy as np
import torch

from probing.probe import Probe


class StroopProbe(Probe, ABC):
    def __init__(self, model_pretrained: str, device: torch.device = torch.device('cuda')):
        super().__init__(model_pretrained, device)

    def score(self, base_sentence: str, sentence_list: List[str]) -> np.ndarray:
        v = self._embed([base_sentence] + sentence_list)

        base = v[0]
        modified = v[1:]
        scores = modified @ base

        return scores

    def score_from_options(self, sentence: str, options: List[str], as_dict: bool = True) -> Union[Dict[str, float], List[float]]:
        if "MASK" not in sentence:
            raise ValueError(f"the work MASK must appear within the given sentence, received: {sentence}")

        base_sentence = sentence.replace("MASK", "")
        sentence_list = [sentence.replace("MASK", option) for option in options]
        scores = self.score(base_sentence, sentence_list)

        return dict(zip(options, scores)) if as_dict else scores

    def _tokenize(self, prompts: List[str]) -> np.ndarray:
        raise NotImplementedError

    def _embed(self, prompts: List[str]) -> np.ndarray:
        raise NotImplementedError

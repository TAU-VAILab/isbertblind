from typing import List

import numpy as np
import torch


class Probe:
    def __init__(self, model_pretrained: str, device: torch.device = torch.device('cuda'), **kwargs):
        self.model_pretrained = model_pretrained
        self.device = device
        self.model = None
        self.tokenizer = None
        self._build_model()

    def _build_model(self):
        raise NotImplementedError

    def score(self, base_sentence: str, sentence_list: List[str]) -> np.ndarray:
        raise NotImplementedError

from typing import List

import numpy as np
import torch


class Probe:
    def __init__(self, model_name: str, model_pretrained: str, device: torch.device = torch.device('cuda')):
        self.model_name = model_name
        self.model_pretrained = model_pretrained
        self.device = device
        self.model = None
        self.tokenizer = None
        self._build_model()

    def _build_model(self):
        raise NotImplementedError

    def score(self, base_sentence: str, sentence_list: List[str]) -> np.ndarray:
        raise NotImplementedError

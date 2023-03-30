from abc import ABC
from typing import List

import numpy as np
import torch
from transformers import pipeline

from probing.probe import Probe


class MLMProbe(Probe, ABC):
    def __init__(self, model_name: str, model_pretrained: str, device: torch.device = torch.device('cuda:0')):
        super().__init__(model_name, model_pretrained, device)
        self.mask_token = self.model.tokenizer.mask_token

    def _build_model(self):
        self.model = pipeline('fill-mask', model=self.model_pretrained, device=self.device)

    def score(self, base_sentence: str, options: List[str]) -> np.ndarray:
        res_mapping = {option: 0 for option in options}
        results = self.model(base_sentence.replace('MASK', self.mask_token), targets=options)
        for result in results:
            if result['token_str'] in options:
                res_mapping[result['token_str']] = result['score']

        return np.fromiter(res_mapping.values(), dtype=float)


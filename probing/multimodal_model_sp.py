from abc import ABC
from typing import List

import numpy as np
import torch
from transformers import CLIPProcessor, CLIPModel, FlavaModel, FlavaProcessor

from probing.stroop_probe import StroopProbe


class CLIPStroopProbe(StroopProbe, ABC):
    def __init__(self, model_name: str, model_pretrained: str, device: torch.device = torch.device('cuda')):
        super().__init__(model_name, model_pretrained, device)

    def _build_model(self):
        self.model = CLIPModel.from_pretrained(self.model_pretrained).to(self.device)
        self.tokenizer = CLIPProcessor.from_pretrained(self.model_pretrained)

    def _tokenize(self, prompts: List[str]) -> torch.tensor:
        return self.tokenizer(text=prompts, return_tensors='pt', padding=True).to(self.device)

    def _embed(self, prompts: List[str]) -> np.ndarray:
        inp = self._tokenize(prompts)
        with torch.no_grad():
            v = self.model.text_model(**inp).pooler_output
            v = self.model.text_projection(v)
            v /= v.norm(p=2, dim=-1, keepdim=True)
            v = v.cpu().numpy()

            return v


class FLAVAStroopProbe(StroopProbe, ABC):
    def __init__(self, model_name: str, model_pretrained: str, device: torch.device = torch.device('cuda')):
        super().__init__(model_name, model_pretrained, device)

    def _build_model(self):
        self.model = FlavaModel.from_pretrained(self.model_pretrained).to(self.device)
        self.tokenizer = FlavaProcessor.from_pretrained(self.model_pretrained)

    def _tokenize(self, prompts: List[str]) -> torch.tensor:
        return self.tokenizer(text=prompts, return_tensors='pt', padding=True).to(self.device)

    def _embed(self, prompts: List[str]) -> np.ndarray:
        inp = self._tokenize(prompts)
        with torch.no_grad():
            v = self.model.get_text_features(**inp)
            v = v[:, 0, :]
            v /= v.norm(p=2, dim=-1, keepdim=True)
            v = v.cpu().numpy()

            return v

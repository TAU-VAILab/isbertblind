from abc import ABC
from typing import List

import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel

from probing.stroop_probe import StroopProbe


class TextStroopProbe(StroopProbe, ABC):
    def __init__(self, model_name: str, model_pretrained: str, device: torch.device = torch.device('cuda')):
        super().__init__(model_name, model_pretrained, device)

    def _build_model(self):
        self.model = AutoModel.from_pretrained(self.model_pretrained).to(self.device)
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_pretrained)
        except:
            print(f"Warning: unable to get tokenizer, falling back to bert tokenizer")
            self.tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

    def _tokenize(self, prompts: List[str]) -> torch.tensor:
        return self.tokenizer(prompts, return_tensors='pt', padding=True).to(self.device).input_ids

    def _embed(self, prompts: List[str]) -> np.ndarray:
        inp = self._tokenize(prompts)
        with torch.no_grad():
            v = self.model(inp).pooler_output
            v /= v.norm(p=2, dim=-1, keepdim=True)
            v = v.cpu().numpy()

            return v

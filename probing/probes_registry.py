from probing.mlm_probes import MLMProbe
from probing.text_model_sp import TextStroopProbe
from probing.multimodal_model_sp import CLIPStroopProbe, FLAVAStroopProbe

probes_registry = {
    "TEXT_MLM": MLMProbe,
    "TEXT_SP": TextStroopProbe,
    "CLIP_SP": CLIPStroopProbe,
    "FLAVA_SP": FLAVAStroopProbe
}

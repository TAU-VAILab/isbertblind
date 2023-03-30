import ast
import json
import os
from pathlib import Path
from typing import Literal, Dict, List

import pandas as pd
from pandas import DataFrame, Series
from torch import nn

TASK = Literal["choice", "cloze", "regression"]


def get_nof_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def parse_config(config_path: Path) -> Dict[str, any]:
    if not config_path.exists():
        raise FileNotFoundError(f"No file found for config in {config_path}")

    with open(config_path) as config_handler:
        config_dict = json.load(config_handler)

    return config_dict


def create_output_dir(config_dict: Dict[str, any]) -> Path:
    setup_params = config_dict['setup_params']
    output_path = Path(os.path.join(setup_params['output_dir'], setup_params['experiment_name']))
    output_path.mkdir(parents=True, exist_ok=True)

    return output_path


def get_output_dir(config_dict: Dict[str, any]) -> Path:
    setup_params = config_dict['setup_params']
    output_path = Path(os.path.join(setup_params['output_dir'], setup_params['experiment_name']))

    return output_path


def apply_sentence(row: Series, prompt: str) -> str:
    return prompt.replace("WORD", row['word'])


def apply_sentence_sp_choice(row: Series, prompt: str, pad_option: str = '') -> str:
    return prompt.replace("WORD", row['word']).replace("MASK", pad_option)


def apply_sentence_sp_cloze(row: Series, prompt: str, pad_option: str = '') -> str:
    return row['sentence'].replace("MASK", pad_option)


def apply_get_options(row: Series) -> List[str]:
    return [option for option in ast.literal_eval(row['options'])]


def apply_get_sp_sentences_choice(row: Series, prompt: str) -> List[str]:
    return [prompt.replace("WORD", row['word']).replace("MASK", option) for option in ast.literal_eval(row['options'])]


def apply_get_sp_sentences_cloze(row: Series) -> List[str]:
    return [row['sentence'].replace("MASK", option) for option in ast.literal_eval(row['options'])]


def parse_specific_df(df_orig: DataFrame, prompt: str, model_type: Literal["MLM", "SP"], task) -> DataFrame:
    df = None

    if task == "choice":
        df = parse_df_for_choice_task(df, df_orig, model_type, prompt)
    else:
        df = parse_df_for_cloze_task(df, df_orig, model_type, prompt)

    return df


def parse_df_for_choice_task(df, df_orig, model_type, prompt):
    if model_type == "MLM":
        df = pd.DataFrame(columns=["sentence", "options", "gt"])
        df['sentence'] = df_orig.apply(apply_sentence, axis=1, args=(prompt,))
        df['options'] = df_orig.apply(apply_get_options, axis=1)
        df['gt'] = df_orig['gt']

    elif model_type == "SP":
        df = pd.DataFrame(columns=["base_sentence", "options", "sentences", "gt"])
        df['base_sentence'] = df_orig.apply(apply_sentence_sp_choice, axis=1, args=(prompt,))
        df['options'] = df_orig.apply(apply_get_options, axis=1)
        df['sentences'] = df_orig.apply(apply_get_sp_sentences_choice, axis=1, args=(prompt,))
        df['gt'] = df_orig['gt']
    return df


def parse_df_for_cloze_task(df, df_orig, model_type, prompt):
    if model_type == "MLM":
        df = pd.DataFrame(columns=["sentence", "options", "gt"])
        df['sentence'] = df_orig['sentence']
        df['options'] = df_orig.apply(apply_get_options, axis=1)
        df['gt'] = df_orig['gt']

    elif model_type == "SP":
        df = pd.DataFrame(columns=["base_sentence", "options", "sentences", "gt"])
        df['base_sentence'] = df_orig.apply(apply_sentence_sp_cloze, axis=1, args=(prompt,))
        df['options'] = df_orig.apply(apply_get_options, axis=1)
        df['sentences'] = df_orig.apply(apply_get_sp_sentences_cloze, axis=1)
        df['gt'] = df_orig['gt']
    return df


def parse_and_validate_df(df_path: Path, task: TASK) -> DataFrame:
    df = pd.read_csv(df_path)

    if task == "choice":
        if "word" not in df.columns or "options" not in df.columns or "gt" not in df.columns:
            raise Exception(f"csv {df_path} does not have the required columns,"
                            f" must have 'word', 'options' and 'gt' and has {df.columns}")

    if task == "cloze":
        if "sentence" not in df.columns or "options" not in df.columns or "gt" not in df.columns:
            raise Exception(f"csv {df_path} does not have the required columns,"
                            f" must have 'sentence', 'options' and 'gt' and has {df.columns}")

    return df

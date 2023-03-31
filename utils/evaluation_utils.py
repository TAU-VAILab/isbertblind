import dataclasses
import itertools
from dataclasses import dataclass
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd
import transformers
from pandas import DataFrame

from probing.probes_registry import probes_registry
from utils.setup_utils import parse_specific_df, TASK
from utils.statistics_utils import get_statistics_results, ProbingStatistics

transformers.logging.set_verbosity_error()


@dataclass
class DFExperimentInfo:
    model_name: str
    model_type: str
    model_pretrained: str
    prompt: str
    accuracy: float
    macro_accuracy: float
    recall_at_5: float


def initialize_evaluation(df: DataFrame, model_info: Dict[str, any], output_path: Path, prompt: str, task: TASK):

    # Get basic info
    probe_name = probes_registry[model_info['model_type']]
    probe = probe_name(**model_info['model_params'])
    probe_type = model_info['model_type'].split("_")[-1]

    # Build df for task
    df_for_task = parse_specific_df(df, prompt, probe_type, task)
    df_for_task['prediction'] = None
    df_for_task['prediction_scores'] = None

    # Create log output path
    model_output_path = Path(output_path / f"{model_info['model_type']}_{model_info['model_params']['model_name']}")
    model_output_path.mkdir(parents=False, exist_ok=True)
    return df_for_task, model_output_path, probe, probe_type


def evaluate_model(model_info: Dict[str, any], df: DataFrame, output_path: Path, df_out: DataFrame,
                   task: TASK, prompt: str = None) -> DataFrame:

    model_type = model_info['model_type']
    model_name = model_info['model_params']['model_name']
    print(f"evaluating {model_type} {model_name} for task {task} {f'with prompt {prompt}' if prompt else ''}")

    # Initialize required data for evaluation
    df_for_task, model_output_path, probe, probe_type = initialize_evaluation(df, model_info, output_path, prompt, task)

    # Get results placeholders
    num_options = len(df_for_task.iloc[0]['options'])
    gt_array = np.zeros(len(df_for_task))
    predictions = np.zeros(len(df_for_task))
    predictions_scores = np.zeros((len(df_for_task), num_options))

    # Iterate over dataset and get results
    for index, row in df_for_task.iterrows():
        options = row['options']
        gt_array[index] = options.index(row['gt'])
        if probe_type == "MLM":
            scores = probe.score(row['sentence'], options)
        else:
            scores = probe.score(row['base_sentence'], row['sentences'])
        predictions_scores[index] = scores
        predictions[index] = np.argmax(scores)

        row['prediction'] = options[np.argmax(scores)]
        row['prediction_scores'] = scores

    # Write the results for this experiment in logs
    df_for_task.to_csv(model_output_path / f"{prompt.replace(' ', '_')}.csv")

    # Get statistics and log them to experiment df
    statistics_results = get_statistics_results(gt_array, predictions, predictions_scores)
    print(f"Results for {model_type} {model_name} with prompt {prompt}: {statistics_results}")
    experiment_info = DFExperimentInfo(model_name, model_type, model_info['model_params']['model_pretrained'], prompt,
                                       accuracy=statistics_results.accuracy, macro_accuracy=statistics_results.macro_accuracy,
                                       recall_at_5=statistics_results.recall_at_5)
    df_out = pd.concat([df_out, DataFrame(dataclasses.asdict(experiment_info), index=[0])], ignore_index=True)

    del probe

    return df_out


def extract_stats_for_experiment(df: DataFrame, output_path: Path):
    model_names = df.model_name.unique()
    model_types = df.model_type.unique()

    df_experiment_summary = DataFrame(columns=["model_name", "model_type",
                                               "max_accuracy", "max_macro_accuracy", "max_recall_at_5",
                                               "mean_accuracy", "mean_macro_accuracy", "mean_recall_at_5",
                                               "std_accuracy", "std_macro_accuracy", "std_recall_at_5",
                                               ])

    for model_name, model_type in itertools.product(model_names, model_types):
        df_part = df[(df['model_name'] == model_name) & (df['model_type'] == model_type)]
        if len(df_part) == 0:
            continue

        max_results = ProbingStatistics(accuracy=df_part.accuracy.max(),
                                        macro_accuracy=df_part.macro_accuracy.max(),
                                        recall_at_5=df_part.recall_at_5.max())

        mean_results = ProbingStatistics(accuracy=df_part.accuracy.mean(),
                                         macro_accuracy=df_part.macro_accuracy.mean(),
                                         recall_at_5=df_part.recall_at_5.mean())

        std_results = ProbingStatistics(accuracy=df_part.accuracy.std(),
                                        macro_accuracy=df_part.macro_accuracy.std(),
                                        recall_at_5=df_part.recall_at_5.std())

        model_result_dict = {
            "model_name": model_name,
            "model_type": model_type,
            "max_accuracy": max_results.accuracy,
            "max_macro_accuracy": max_results.macro_accuracy,
            "max_recall_at_5": max_results.recall_at_5,
            "mean_accuracy": mean_results.accuracy,
            "mean_macro_accuracy": mean_results.macro_accuracy,
            "mean_recall_at_5": mean_results.recall_at_5,
            "std_accuracy": std_results.accuracy,
            "std_macro_accuracy": std_results.macro_accuracy,
            "std_recall_at_5": std_results.recall_at_5
        }
        df_experiment_summary = pd.concat([df_experiment_summary, DataFrame(model_result_dict, index=[0])], ignore_index=True)

    df_experiment_summary.to_csv(output_path / "experiment_summary.csv")



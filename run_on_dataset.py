import argparse
from dataclasses import fields
from pathlib import Path

import pandas as pd

from utils.evaluation_utils import evaluate_model, DFExperimentInfo, extract_stats_for_experiment
from utils.setup_utils import parse_config, create_output_dir, parse_and_validate_df


def main(config_path: str):
    config_dict = parse_config(Path(config_path))
    output_path = create_output_dir(config_dict)
    task = config_dict['setup_params']['task']
    df = parse_and_validate_df(config_dict['setup_params']['dataset_path'], task)

    experiment_df = pd.DataFrame(columns=[fields(DFExperimentInfo)])

    for model_info in config_dict['models']:
        for prompt in config_dict['prompts']:
            experiment_df = evaluate_model(model_info, df, output_path, experiment_df, task, prompt)

    experiment_df.to_csv(f"{output_path}/experiment_results.csv")

    extract_stats_for_experiment(experiment_df, output_path)

    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run evaluation of dataset using text encoders with different methods")
    parser.add_argument('config', help='json config file path')
    args = parser.parse_args()
    main(args.config)

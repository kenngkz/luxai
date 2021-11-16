'''
Train and evaluate a stage (pool of models in the same phase of training).
'''

# Imports
from sys import path
path.append("C:\\Users\\kenng\\Desktop\\Coding\\CustomModules")  # allows custom_modules to be imported

from train_loop import train_loop
from eval_model import eval_model
from utility import get_existing_models
from model_score import ModelScore
from data_structures.quickselect import select_k


import random
import os
import pandas as pd
import matplotlib.pyplot as plt

def train_stage(params:dict, stage_path, replay=False, resume=False):
    '''
    Run a training stage.
    Resume=True can only be used if run_id is passed in params.
    '''
    TRAIN_PARAMS_FILE_NAME = "train_params.txt"

    # Auto-adjust model_path to end with '/'
    if stage_path[-1] != '/' and stage_path[-1] != '\\':
        stage_path += '/'

    def add_opt_args(input_args, output_args, arg_names):
        for name in arg_names:
            if name in input_args:
                output_args[name] = input_args[name]
        return output_args

    if resume:  # check that previous training params was saved. If not saved, do not resume
        if os.path.exists(stage_path + TRAIN_PARAMS_FILE_NAME):
            with open(stage_path + TRAIN_PARAMS_FILE_NAME, 'r') as f:
                old_train_params = eval(f.read())
            try:
                if sum([old_train_params[i]['run_id'] != params[i]['run_id'] for i in range(len(params))]) != 0:
                    resume = False
            except KeyError:
                resume = False
        else:
            resume = False

    n_models = len(params)
    model_ids = []
    for train_index, model_train_params in enumerate(params):
        if resume:  # look for info.json -> info.json is saved after training.
            if os.path.exists(stage_path + model_train_params['run_id'] + '/info.json'):
                model_ids += [model_train_params['run_id']]
                print(f"Training for model {model_train_params['run_id']} skipped. Already completed.")
                continue
        # Check that 'model_path' or 'model_policy' is provided
        print(f"Commencing training for model {model_train_params['run_id']}. Pending: {n_models - train_index - 1} / {n_models}")
        assert 'model_path' in model_train_params or 'model_policy' in model_train_params, "Keys 'model_path' and 'model_policy' not found in params argument. Please enter either one."
        # Set up train_loop args
        train_loop_args = {
            'save_folder':stage_path,
            'modellist_file':stage_path + 'modellist.txt',
            'tensorboard_log':stage_path + 'tensorboard_log/'
        }
        optional_args = ['run_id', 'step_count', 'learning_rate', 'gamma', 'gae_lambda', 'opp_path', 'replay_freq', 'replay_num']
        train_loop_args = add_opt_args(model_train_params, train_loop_args, optional_args)
        # Defining replay behaviour
            # if replay_freq provided, pass the value on into train_loop()
            # elif replay=True, set replay_freq=step_count
                # if step_count not provided, use default value of 100000
            # else, set replay_freq=step_count+1 (no replay created)
                # if step_count not provided, use default value of 100000+1
        if 'step_count' in model_train_params:
            if replay:
                train_loop_args['replay_freq'] = model_train_params['step_count']
            elif 'replay_freq' not in model_train_params:
                train_loop_args['replay_freq'] = model_train_params['step_count'] + 1
        elif not replay and 'replay_freq' not in model_train_params:
            train_loop_args['replay_freq'] = 100001
        # Set the model to be trained
        if 'model_path' in model_train_params:
            train_loop_args['ply_path'] = model_train_params['model_path']
        if 'model_policy' in model_train_params:
            train_loop_args['ply_policy'] = model_train_params['model_policy']
        # Number of copies of the model to generate
        n = model_train_params['n_copies'] if 'n_copies' in model_train_params else 1
        # Run training loop to create n copies of a model
        model_id = [train_loop(**train_loop_args) for _ in range(n)]
        model_ids += model_id

    print(f"Training stage {stage_path[-2]} complete. Models {model_ids} saved in {stage_path}")

    return model_ids

def eval_stage(stage_path, n_select, benchmark_models, model_ids=None, n_games=5, max_steps=1000, resume=False):
    '''
    Second version of eval_stage: 2.0
    Evaluates all models in a given stage folder, then selects the models with the top n_select scores.
    '''
    # Auto-adjust paths to end with '/'
    if stage_path[-1] != '/' and stage_path[-1] != '\\':
        stage_path += '/'

    # File names and paths
    BEST_MODELS_FILE_NAME = "best_models.txt"
    BENCHMARK_MODELS_FILE_NAME = "benchmark_models.txt"
    EVAL_RESULTS_FILE_NAME = "eval_results.txt"
    best_models_file_path = stage_path + BEST_MODELS_FILE_NAME
    benchmark_models_file_path = stage_path + BENCHMARK_MODELS_FILE_NAME
    eval_results_file_path = stage_path + EVAL_RESULTS_FILE_NAME

    # Get list of models to evaluate
    if not model_ids:
        model_ids = get_existing_models(stage_path + 'modellist.txt')
    n_models = len(model_ids)
    n_benchmarks = len(benchmark_models)

    # Load eval_results from file if the file exists
    if os.path.exists(eval_results_file_path):
        with open(eval_results_file_path, 'r') as f:
            eval_results = eval(f.read())
    else:
        eval_results = {}

    # Run evaluation games against benchmark models
    for model_index, model_id in enumerate(model_ids):
        if resume:
            if model_id in eval_results:
                print(f"Evaluation for model {model_id} skipped.")
                continue
        print(f"Evaluation for model {model_id}. Pending: {n_models - model_index - 1} / {n_models}")
        results = eval_model(stage_path+model_id, benchmark_models, n_games=n_games, max_steps=max_steps, overwrite=True)
        eval_results[model_id] = results
        with open(eval_results_file_path, 'w') as f:
            f.write(str(eval_results))
    
    # Calculate average winrate
    model_scores = [ModelScore(model_id, sum([eval_results[model_id][opp_id]['winrate'] for opp_id in eval_results[model_id].keys()])/n_benchmarks) for model_id in model_ids]

    # Select the top 'n_select' models
    _, i, sorted_scores = select_k(model_scores, n_select)
    best_models = sorted_scores[:i+1]
    best_models = {model_score.id:model_score.score for model_score in best_models}

    # Save the best models into a file
    with open(best_models_file_path, 'w') as f:
        f.write(str(best_models))

    # Get new benchmark models
    _, i, sorted_scores = select_k(model_scores, n_benchmarks)
    benchmark_models = sorted_scores[:i+1]
    benchmark_models = [stage_path + model_score.id for model_score in benchmark_models]

    # Save the new benchmark models paths
    with open(benchmark_models_file_path, 'w') as f:
        f.write(str(benchmark_models))

    return best_models
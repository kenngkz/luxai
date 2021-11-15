'''
Train and evaluate a stage (pool of models in the same phase of training).
'''

# Imports
from sys import path
path.append("C:\\Users\\lenovo\\Desktop\\Coding\\CustomModules")  # allows custom_modules to be imported

from train_loop import train_loop
from eval_model import eval_model
from utility import get_existing_models
# from model_score import ModelScore
from data_structures.quickselect import select_k
from score_graph import ScoreGraph, ModelScore

import random
import os
import pandas as pd
import matplotlib.pyplot as plt

def train_stage(params:dict, stage_path, replay=False):
    '''
    Run a training stage
    '''
    # Auto-adjust model_path to end with '/'
    if stage_path[-1] != '/' or stage_path[-1] != '\\':
        stage_path += '/'

    def add_opt_args(input_args, output_args, arg_names):
        for name in arg_names:
            if name in input_args:
                output_args[name] = input_args[name]
        return output_args

    model_ids = []
    for model_train_params in params:
        # Check that 'model_path' or 'model_policy' is provided
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
        id = [train_loop(**train_loop_args) for _ in range(n)]
        model_ids += id

    print(f"Training stage complete. Models {model_ids} saved in {stage_path}")

    return model_ids

def eval_stage(stage_path, n_select, benchmark_models, model_ids=None, n_games=5, max_steps=1000, resume=False):
    '''
    First version of eval_stage
    Evaluates all models in a given stage folder, then selects the models with the top n_select scores.
    '''
    # Auto-adjust paths to end with '/'
    if stage_path[-1] != '/' or stage_path[-1] != '\\':
        stage_path += '/'
    for path in benchmark_models:
        if path[-1] != '/' or path[-1] != '\\':
            path += '/'

    # File names and paths
    BEST_MODELS_FILE_NAME = "best_models.txt"
    best_models_file_path = stage_path + BEST_MODELS_FILE_NAME



def get_scores_data(stage_path):
    '''
    Retrieve score for all models in a given stage and returns the scores as a pd.DataFrame
    '''
    # Auto-adjust model_path to end with '/'
    if stage_path[-1] != '/' or stage_path[-1] != '\\':
        stage_path += '/'

    # File names and paths
    GRAPH_FILE_NAME = "score_graph.txt"
    graph_file_path = stage_path + GRAPH_FILE_NAME

    score_graph = ScoreGraph.load(graph_file_path)
    model_ids = score_graph.info()["nodes"]
    scores_data = pd.DataFrame(list(score_graph.get_scores(model_ids).items()), columns=['id', 'score'])
    for model_id in model_ids:
        scores_data.loc[scores_data['id']==model_id, 'score'] = score_graph.get_score(model_id)
    return scores_data

def plot_scores(stage_path):
    scores = get_scores_data(stage_path)
    plt.hist(scores["score"], bins=40)
    # plt.axvline(x=scores["score"].mean(), color='r', label='mean')
    # plt.legend()
    plt.show()
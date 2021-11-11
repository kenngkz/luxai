'''
Train and evaluate a stage (pool of models in the same phase of training).
'''

# Imports
from sys import path
path.append("C:\\Users\\lenovo\\Desktop\\Coding\\CustomModules")  # allows custom_modules to be imported

from train_loop import train_loop
from eval_model import eval_model
from utility import get_existing_models
from model_score import ModelScore
from data_structures.quickselect import select_k

import random


def train_stage(params:dict, stage_path, replay=False):
    '''
    Run a training stage
    '''

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

def eval_stage(stage_path, n_select, model_ids=None, n_games=3, max_steps=1000, resume=False):
    '''
    Evaluates all models in a given stage folder, then selects the models with the top n_select scores.
    '''

    def update_eval_file(path, eval_completed_status, score=None):
        with open(path, 'r') as f:
            eval_info = eval(f.read())
        if score:
            eval_info['score'] = score
        eval_info['eval_completed'] = eval_completed_status
        with open(path, 'w') as f:
            f.write(str(eval_info))

    if not model_ids:
        model_ids = get_existing_models(stage_path + 'modellist.txt')
    n_models = len(model_ids)

    # Initiliaze a ModelScore for every model to be evaluated
        # the skip dict indicates if a model eval should be skipped
    model_scores = {}
    skip = {}
    for id in model_ids:
        model_scores[id] = ModelScore(id)
        skip[id] = False
    # Allow every ModelScore to access all other ModelScores
    for model_score in model_scores.values():
        model_score.add_pool(model_scores)

    # If selecting a small portion of n_models and there are at least 8 models, 
        # eliminate half of the pool to reduce computation time
    if n_select/n_models < 0.45 and n_models > 7:
        for model_index, id in enumerate(model_ids):
            model_score = model_scores[id]
            if resume:  # check if model has been evaluated before (score value recorded in eval.json)
                try:
                    with open(stage_path + id + '/eval.json', 'r') as f:
                        eval_info = eval(f.read())
                    if 'eval_completed' in eval_info:
                        if eval_info['eval_completed']:
                            skip[id] = True
                            continue
                    if 'score' in eval_info:  # if the model already has a score, skip only the preliminary evaluation
                        print(f"Preliminary evaluation for model {id} skipped. Preliminary evaluation already completed.")
                        played_opps = list(eval_info['history'].keys())
                        model_score.skip_opps(played_opps)
                        continue
                except FileNotFoundError:
                    pass
            print(f"Preliminary Evaluation for model {id}. Pending: {n_models - model_index - 1} / {n_models}")
            opp_paths = random.sample([stage_path+opp_id for opp_id in model_score.remaining_opp], int(n_models/2))
            results = eval_model(stage_path+id, opp_paths, n_games=n_games, max_steps=max_steps)
            model_score.update(results)
            update_eval_file(stage_path+id+'/eval.json', False, model_score.score)
            
        _, i, sorted_scores = select_k([model_score for model_score in model_scores.values()], int(n_models/2))
        final_model_ids  = sorted_scores[:i+1]
        dropped_models = sorted_scores[i+1:]
    else:
        final_model_ids = model_ids
        dropped_models = []

    # Set all dropped_models eval_completed status to True
    for id in dropped_models:
        update_eval_file(stage_path+id+'/eval.json', True)

    # Select the final n best models 
    for model_index, id in enumerate(final_model_ids):
        if resume and skip[id]:
            print(f"Model {id} skipped: Evaluation already complete")
            continue
        print(f"Final Evaluation for model {id}. Pending: {n_models - model_index - 1} / {n_models}")
        results = eval_model(stage_path+id, [stage_path+opp_id for opp_id in model_score.remaining_opp], n_games=n_games, max_steps=max_steps)
        model_scores[id].update(results)
        update_eval_file(stage_path+id+'/eval.json', True, model_score.score)

    _, i, selected_scores = select_k([model_score for model_score in model_scores.values()], n_select)

    # Update all eval.json files for every model to include their score
    for id, model_score in model_scores.items():
        with open(stage_path + id + '/eval.json', 'r') as f:
            eval_info = eval(f.read())
        eval_info['score'] = model_score.score
        with open(stage_path + id + '/eval.json', 'w') as f:
            f.write(str(eval_info))

    return [model_score.id for model_score in selected_scores[:i+1]]
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

def eval_stage(stage_path, n_select, model_ids=None, n_games=3, max_steps=1000, resume=False):
    '''
    Evaluates all models in a given stage folder, then selects the models with the top n_select scores.
    '''
    # Auto-adjust stage_path to end with '/'
    if stage_path[-1] != '/' or stage_path[-1] != '\\':
        stage_path += '/'

    # File names and paths
    GRAPH_FILE_NAME = "score_graph.txt"
    FILTERED_MODELS_FILE_NAME = "filtered_models.txt"
    BEST_MODELS_FILE_NAME = "best_models.txt"
    graph_file_path = stage_path + GRAPH_FILE_NAME
    filtered_models_file_path = stage_path + FILTERED_MODELS_FILE_NAME
    best_models_file_path = stage_path + BEST_MODELS_FILE_NAME

    # function not needed in latest version
    def update_eval_file(path, eval_completed_status, score=None):
        '''
        Edits eval.json to update the given score and eval_completed_status.
        '''
        with open(path, 'r') as f:
            eval_info = eval(f.read())
        if score:
            eval_info["score"] = score
        eval_info["eval_completed"] = eval_completed_status
        with open(path, 'w') as f:
            f.write(str(eval_info))

    if not model_ids:
        model_ids = get_existing_models(stage_path + 'modellist.txt')
    n_models = len(model_ids)
    # function to determine how many opponents in the preliminary stage
    m = - 1 / 3 / 100
    c = 0.5 - 10 * m
    n_samples = int(n_models * max(0.26, n_models * m + c))

    # Initialize score graph
    if os.path.exists(graph_file_path):  # if a graph save file already exists, load graph from save file
        print("ScoreGraph loaded from existing save file")
        graph = ScoreGraph.load(graph_file_path)
    else:
        graph = ScoreGraph(model_ids)

    # Preliminary Evaluation
    if n_select/n_models < 0.45 and n_models > 7:  # if eligible for prelim evaluation
        for model_index, model_id in enumerate(model_ids):
            # Check if model achieved preliminary number of matches 'n_samples'
            if graph.n_links(model_id) >= n_samples:
                print(f"Preliminary evaluation for model {model_id} already completed. Score: {graph.get_score(model_id)}")
                continue
            # Run prelim eval
            print(f"Preliminary Evaluation for model {model_id}. Pending: {n_models - model_index - 1} / {n_models}")
            opp_paths = random.sample([stage_path+opp_id for opp_id in graph.missing_links(model_id)], n_samples-graph.n_links(model_id))
            results = eval_model(stage_path+model_id, opp_paths, n_games=n_games, max_steps=max_steps, overwrite=not resume)
            graph.update(model_id, results)
            graph.save(graph_file_path)
            print(f"Score for model {model_id}: {graph.get_score(model_id)}")
        # Convert all ids and scores into ModelScore objects and select the top half using select_k
        model_scores = [ModelScore(score_node.id,score_node.get_score()) for score_node in graph.nodes.values()]
        model_scores = random.sample(model_scores, len(model_scores))  # randomize the order of model_scores
        _, i, sorted_scores = select_k([model_score for model_score in model_scores], int(n_models/2))
        final_model_ids  = [model_score.id for model_score in sorted_scores[:i+1]]
        n_dropped = len(sorted_scores) - i

        # Save ids of the models with score in the top half
        with open(filtered_models_file_path, 'w') as f:
            f.write(str(final_model_ids))
    else:
        final_model_ids = model_ids
        n_dropped = 0

    model_ids.reverse()
    new_n_samples = 30
    #  Inverted Preliminary Evaluation
    if n_select/n_models < 0.45 and n_models > 7:  # if eligible for prelim evaluation
        for model_index, model_id in enumerate(model_ids):
            # Check if model achieved preliminary number of matches 'n_samples'
            if graph.n_links(model_id) >= new_n_samples:
                print(f"Preliminary evaluation for model {model_id} already completed. Score: {graph.get_score(model_id)}")
                continue
            # Run prelim eval
            print(f"Preliminary Evaluation for model {model_id}. Pending: {n_models - model_index - 1} / {n_models}")
            opp_paths = random.sample([stage_path+opp_id for opp_id in graph.missing_links(model_id)], new_n_samples-graph.n_links(model_id))
            results = eval_model(stage_path+model_id, opp_paths, n_games=n_games, max_steps=max_steps, overwrite=not resume)
            graph.update(model_id, results)
            graph.save(graph_file_path)
            print(f"Score for model {model_id}: {graph.get_score(model_id)}")
        # Convert all ids and scores into ModelScore objects and select the top half using select_k
        model_scores = [ModelScore(score_node.id,score_node.get_score()) for score_node in graph.nodes.values()]
        model_scores = random.sample(model_scores, len(model_scores))  # randomize the order of model_scores
        _, i, sorted_scores = select_k([model_score for model_score in model_scores], int(n_models/2))
        final_model_ids  = [model_score.id for model_score in sorted_scores[:i+1]]
        n_dropped = len(sorted_scores) - i

        # Save ids of the models with score in the top half
        with open(filtered_models_file_path, 'w') as f:
            f.write(str(final_model_ids))
    else:
        final_model_ids = model_ids
        n_dropped = 0

    # Convert all ids and scores into ModelScore objects and select the top 'n_select' using select_k
    model_scores = [ModelScore(score_node.id,score_node.get_score()) for score_node in graph.nodes.values()]
    model_scores = random.sample(model_scores, len(model_scores))  # randomize the order of model_scores
    _, i, sorted_scores = select_k([model_score for model_score in model_scores], n_select)
    best_model_ids = {model_score.id:model_score.score for model_score in sorted_scores[:i+1]}  # best models id:score

    # Save the best models
    with open(best_models_file_path, 'w') as f:
        f.write(str(best_model_ids))
    
    return best_model_ids

    ####### TODO: delete the above section and continue below when done with getting the benchmark

    # Final Evaluation
    print(f"Final Evaluation Stage for models: {final_model_ids}")
    for model_index, model_id in enumerate(final_model_ids):
        if graph.n_links(model_id) >= len(model_ids) - n_dropped - 1:
            print(f"Final evaluation for model {model_id} already completed. Score: {graph.get_score(model_id)}")
            continue
        print(f"Final Evaluation for model {model_id}. Pending: {n_models - model_index - 1} / {n_models}")
        opp_paths = [stage_path+opp_id for opp_id in graph.missing_links(model_id)]
        results = eval_model(stage_path+model_id, opp_paths, n_games=n_games, max_steps=max_steps, overwrite=not resume)
        graph.update(results)
        graph.save(graph_file_path)
        print(f"Score for model {model_id}: {graph.get_score(model_id)}")
    # Convert all ids and scores into ModelScore objects and select the top 'n_select' using select_k
    model_scores = [ModelScore(score_node.id,score_node.get_score()) for score_node in graph.nodes.values()]
    model_scores = random.sample(model_scores, len(model_scores))  # randomize the order of model_scores
    _, i, sorted_scores = select_k([model_score for model_score in model_scores.values()], n_select)
    best_model_ids = {model_score.id:model_score.score for model_score in sorted_scores[:i+1]}  # best models id:score

    # Save the best models
    with open(best_models_file_path, 'w') as f:
        f.write(str(best_model_ids))
    
    return best_model_ids
    #################################

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
        # check for existing history and load it into current model_score if exists.
        if os.path.exists(stage_path+id+'/eval.json') and resume:
            with open(stage_path+id+'/eval.json', 'r') as f:
                eval_info = eval(f.read())
                if "history" in eval_info:
                    model_score.update(eval_info["history"])

    # if models with above average preliminary score has already been found, skip to the final evaluation.
    if os.path.exists(stage_path + 'top_half.txt'):
        with open(stage_path + 'top_half.txt', 'r') as f:
            final_model_ids = eval(f.read())
    # If selecting a small portion of n_models and there are at least 8 models, 
        # eliminate half of the pool to reduce computation time
    elif n_select/n_models < 0.45 and n_models > 7:
        for model_index, id in enumerate(model_ids):
            model_score = model_scores[id]
            if resume:  # check if model has been evaluated before (score value recorded in eval.json)
                try:
                    with open(stage_path + id + '/eval.json', 'r') as f:
                        eval_info = eval(f.read())
                    if 'eval_completed' in eval_info:
                        if eval_info["eval_completed"]:
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
            opp_paths = random.sample([stage_path+opp_id for opp_id in model_score.remaining_opp], int(n_samples))
            results = eval_model(stage_path+id, opp_paths, n_games=n_games, max_steps=max_steps, overwrite=not resume)
            model_score.update(results)
            update_eval_file(stage_path+id+'/eval.json', False, model_score.score)
            print(f"Score for model {id}: {model_score.score}")

        _, i, sorted_scores = select_k([model_score for model_score in model_scores.values()], int(n_models/2))
        # print([ms.score for ms in sorted_scores])
        final_model_ids  = [model_score.id for model_score in sorted_scores[:i+1]]
        dropped_model_ids = [model_score.id for model_score in sorted_scores[i+1:]]

        # Save ids of the models with score in the top half
        with open(stage_path + 'top_half.txt', 'w') as f:
            f.write(str(final_model_ids))
        # Set all dropped_models eval_completed status to True
        for ids in dropped_model_ids:
            update_eval_file(stage_path+id+'/eval.json', True)
    # if preliminary trimming not required
    else:
        final_model_ids = model_ids

    print(f"Final Evaluation Stage for models: {final_model_ids}")

    # Select the final n best models 
    for model_index, id in enumerate(final_model_ids):
        if resume and skip[id]:
            print(f"Model {id} skipped: Evaluation already complete")
            continue
        print(f"Final Evaluation for model {id}. Current Score: {model_scores[id].score}. Pending: {n_models - model_index - 1} / {n_models}")
        results = eval_model(stage_path+id, [stage_path+opp_id for opp_id in model_score.remaining_opp], n_games=n_games, max_steps=max_steps, overwrite=not resume)
        model_scores[id].update(results)
        update_eval_file(stage_path+id+'/eval.json', True, model_score.score)

    _, i, selected_scores = select_k([model_score for model_score in model_scores.values()], n_select)

    # Update all eval.json files for every model to include their score
    for id, model_score in model_scores.items():
        with open(stage_path + id + '/eval.json', 'r') as f:
            eval_info = eval(f.read())
        eval_info["score"] = model_score.score
        with open(stage_path + id + '/eval.json', 'w') as f:
            f.write(str(eval_info))

    return [model_score.id for model_score in selected_scores[:i+1]]

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
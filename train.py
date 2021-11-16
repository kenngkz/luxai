# Allow custom modules to be imported
from sys import path
path.append("C:\\Users\\kenng\\Desktop\\Coding\\CustomModules")
from stage import eval_stage, train_stage, get_scores_data
from utility import get_existing_models

import os
import sys

def clear_eval_files():
    import shutil
    stage_path = 'pool/stage_0/'

    model_ids = get_existing_models(stage_path + 'modellist.txt')
    for model_id in model_ids:
        shutil.rmtree(stage_path + model_id + '/eval_replay')
        if os.path.exists(stage_path + model_id + '/eval.json'):
            os.remove(stage_path + model_id + '/eval.json')

def main():
    '''
    Main function to execute.
    '''
    # Variables
    stage_size = 30  # number of models in 1 stage
    select = 10  # select the top 'select' models to pass into the next stage
    spawn_new = 5  # each selected model should spawn 'spawn_new' new models
    ini_steps = 1000000  # number of steps to train in the first stage (stage 0)
    stage_steps = 100000  # number of steps to train between each stage
    n_stages = 10  # number of stages before quitting the algorithm
    policies = ['agent1', 'agent2', 'agent3']
    stage_name_prefix = 'stage'
    directory = None
    default_model_params = {
        'learning_rate': 0.001,
        'gamma':0.995,
        'gae_lambda':0.95
    }

    '''
    train_params = {
        'run_id': optional = None,
        'model_path': ,  either model_path or model_policy must be provided
        'model_policy: ,
        'opponent': optional = 'agent_blank',
        'step_count': optional = 100000,
        'learning_rate': optional = 0.001,
        'gamma': optional = 0.995,
        'gae_lambda': optional = 0.95,
        'n_copies': optional = 1,
        'replay_freq': optional = step_count,
    }
    '''

    # Create relevant folders
    # if directory is specified, set the current working directory
    if directory:
        os.chdir(directory)

    # if pool directory does not exist, create it
    if not os.path.exists('pool'):
        os.mkdir('pool')
    
    # Define paths to all stages
    stage_paths = ['pool/' + stage_name_prefix + f'_{i}/' for i in range(n_stages)]    

    # Define parameters
    ini_train_params = [
        {
            'model_policy':'agent2',
            'n_copies':3,
            'step_count': 1000000,
            'learning_rate':0.003,
        },
        {
            'model_policy':'agent2',
            'n_copies':3,
            'step_count': 1000000,
            'learning_rate':0.0005,
        },
        {
            'model_policy':'agent2',
            'n_copies':3,
            'step_count': 1000000,
            'learning_rate':0.0001,
            'gamma': 0.995,
        },
        {
            'model_policy':'agent2',
            'n_copies':3,
            'step_count': 1000000,
            'learning_rate':0.001,
            'gamma': 0.990,
            'gae_lambda': 0.95
        },
        {
            'model_policy':'agent2',
            'n_copies':3,
            'step_count': 1000000,
            'learning_rate':0.001,
            'gamma': 0.995,
            'gae_lambda': 0.99
        },
        {
            'model_policy':'agent2',
            'n_copies':3,
            'step_count': 1000000,
            'learning_rate':0.001,
            'gamma': 0.995,
            'gae_lambda': 0.90
        }
    ]

    stage_path = stage_paths[0]

    if not os.path.exists(stage_path):
        os.mkdir(stage_path)
    # models = train_stage(ini_train_params, stage_path)
    with open(stage_path + 'best_models.txt', 'r') as f:
        benchmark_models = eval(f.read())
        # benchmark_models = [stage_path + opp_id for opp_id in benchmark_models]
        # with open(stage_path + 'benchmark_models.txt', 'w') as f:
        #     f.write(str(benchmark_models))
    best_models = eval_stage(stage_path, select, benchmark_models, resume=True)
    print(f"Best models in stage_0: {best_models}")
    
    return None

    '''
    train_params_template = [
        {
            'model_policy':'agent1',
            'n_copies': 1,
            'step_count': 100000,
            'learning_rate':0.001,
            'gamma':0.995,
            'gae_lambda':0.95,
        },
        {
            'model_policy':'agent1',
            'n_copies': 1,
            'step_count': 100000,
            'learning_rate':0.005,
            'gamma':0.995,
            'gae_lambda':0.95,
        },
        {
            'model_policy':'agent1',
            'n_copies': 1,
            'step_count': 100000,
            'learning_rate':0.001,
            'gamma':0.999,
            'gae_lambda':0.95,
        },
        {
            'model_policy':'agent1',
            'n_copies': 1,
            'step_count': 100000,
            'learning_rate':0.001,
            'gamma':0.995,
            'gae_lambda':0.99,
        },
        {
            'model_policy':'agent1',
            'n_copies': 1,
            'step_count': 100000,
            'learning_rate':0.001,
            'gamma':0.995,
            'gae_lambda':0.90,
        },
    ]
    full_params = dict([(i, {'train_params':train_params_template, 'stage_path':stage_paths[i]}) for i in range(1, n_stages)])
    full_params[0] = {'train_params':ini_train_params, 'stage_path':stage_paths[0]}

    for stage_num, train_stage_args in full_params.items():
        stage_path, args = train_stage_args['stage_path'], train_stage_args['train_params']
        if not os.path.exists(train_stage_args['stage_path']):
            os.mkdir(train_stage_args['stage_path'])
        if stage_num > 0:
            previous_stage_path = full_params[stage_num-1]['stage_path']
            template = args
            for model_id in best_models:
                
        ### TODO

    # if stage_0
    stage_path = 'pool/' + stage_name_prefix + '_0/'
    if not os.path.exists(stage_path):
        os.mkdir(stage_path)

        stage0params = {'models':{'agent1':stage_size}, 'steps':ini_steps}
        for name, value in train_params.items():
            stage0params[name] = value
        
        models = train_stage(stage0params, stage_path, new_model=True)
        best_models = eval_stage(stage_path, select)
        print(f"Stage_0 best models: {best_models}")

        with open(stage_path + 'best_models.txt', 'w') as f:
            f.write(str(best_models))

    # stages 1 to n_stages
    for stage_num in range(1, n_stages):
        previous_stage_path = stage_path
        stage_path[-2] = stage_num
        if not os.path.exists(stage_path):
            os.mkdir(stage_path)

            stage_params = {'models':dict([(previous_stage_path+id, spawn_new) for id in best_models]), 'steps':stage_steps}
            for name, value in train_params.items():
                stage_params[name] = value
            
            models = train_stage(stage_params, stage_path)
            best_models = eval_stage(stage_path, select)
            print(f"Stage_{stage_num} best models: {best_models}")

            with open(stage_path + 'best_models.txt', 'w') as f:
                f.write(str(best_models))

    # select the final best model
    best_model = eval_stage(stage_path, 1, model_ids=best_models)
    with open(stage_path + best_model + '/eval.json', 'r') as f:
        bm_score = eval(f.read())['score']
    print(f"Best model: {best_model}, Score: {bm_score}")
    return best_model
    '''


if __name__ == "__main__":
    if sys.version_info < (3,7) or sys.version_info >= (3,8):
        os.system("")
        class style():
            YELLOW = '\033[93m'
        version = str(sys.version_info.major) + "." + str(sys.version_info.minor)
        message = f'/!\ Warning, python{version} detected, you will need to use python3.7 to submit to kaggle.'
        message = style.YELLOW + message
        print(message)

    # scores = get_scores_data('pool/stage_0/')
    # for i in range(len(scores)):
    #     print(scores.iloc[i, :])

    main()

    # Note: run this file from LuxAI/rl/scripts





##############################
# Plan:
# Invert the list and do again to get at least 30 opponents for each model.

# Create a benchmark set of opp_models (20 models). 
# Run all the models to be evaluated against these benchmarks and sum winrates (n_games=5)
##############################
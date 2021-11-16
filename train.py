# Allow custom modules to be imported
from sys import path
path.append("C:\\Users\\kenng\\Desktop\\Coding\\CustomModules")
from stage import eval_stage, train_stage
from utility import get_existing_models, get_benchmarks, get_best_stats, gen_run_ids

import os
import sys



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
    PATH_PREFIX = "../modelcp/pool"  # if running from scripts or modelcp, this is the path prefix to use

    # Create relevant folders
    # if directory is specified, set the current working directory
    if directory:
        os.chdir(directory)
        if not os.path.exists('pool'):
            os.mkdir('pool')
        stage_paths = ['pool/' + stage_name_prefix + f'_{i}/' for i in range(n_stages)]
    else:
        # if pool directory does not exist, create it
        if not os.path.exists(PATH_PREFIX):
            os.mkdir(PATH_PREFIX)
    
        # Define paths to all stages
        stage_paths = ['../modelcp/pool/' + stage_name_prefix + f'_{i}/' for i in range(n_stages)]    

    # Define parameters
    tree_branch_template = [
        {
            'model_path':None,
            'model_policy':'agent1',
            'step_count':100000,
            'n_copies':1,
            'learning_rate':0.001,
            'gamma':0.995,
            'gae_lambda':0.95,
        },
        {
            'model_path':None,
            'model_policy':'agent1',
            'step_count':100000,
            'n_copies':1,
            'learning_rate':0.0004,
            'gamma':0.995,
            'gae_lambda':0.95,
        },
        {
            'model_path':None,
            'model_policy':'agent1',
            'step_count':100000,
            'n_copies':1,
            'learning_rate':0.0001,
            'gamma':0.995,
            'gae_lambda':0.95,
        },
        {
            'model_path':None,
            'model_policy':'agent1',
            'step_count':100000,
            'n_copies':1,
            'learning_rate':0.001,
            'gamma':0.995,
            'gae_lambda':0.99,
        },
        {
            'model_path':None,
            'model_policy':'agent2',
            'step_count':100000,
            'n_copies':1,
            'learning_rate':0.001,
            'gamma':0.995,
            'gae_lambda':0.95,
        },
        {
            'model_path':None,
            'model_policy':'agent2',
            'step_count':100000,
            'n_copies':1,
            'learning_rate':0.0004,
            'gamma':0.995,
            'gae_lambda':0.95,
        },
        {
            'model_path':None,
            'model_policy':'agent2',
            'step_count':100000,
            'n_copies':1,
            'learning_rate':0.0001,
            'gamma':0.995,
            'gae_lambda':0.95,
        },
        {
            'model_path':None,
            'model_policy':'agent3',
            'step_count':100000,
            'n_copies':1,
            'learning_rate':0.001,
            'gamma':0.995,
            'gae_lambda':0.95,
        },
        {
            'model_path':None,
            'model_policy':'agent3',
            'step_count':100000,
            'n_copies':1,
            'learning_rate':0.0004,
            'gamma':0.995,
            'gae_lambda':0.95,
        },
        {
            'model_path':None,
            'model_policy':'agent3',
            'step_count':100000,
            'n_copies':1,
            'learning_rate':0.0001,
            'gamma':0.995,
            'gae_lambda':0.95,
        }
    ]

    stage_path = stage_paths[0]

    BEST_MODELS_FILE_NAME = "best_models.txt"
    for i in range(1, n_stages+1):
        # Check if stage is already completed
        if os.path.exists(stage_paths[i] + BEST_MODELS_FILE_NAME):
            print(f"Stage {i} completed. {n_stages - i} stages remaining.")
            continue
        else:
            print(f"Stage {i} begins. {n_stages - i} stages remaining.")
        if not os.path.exists(stage_paths[i]):
            os.mkdir(stage_paths[i])
        prev_best_models = get_best_stats(stage_paths[i-1])['id']
        prev_best_models_paths = [stage_paths[i-1] + model_id for model_id in prev_best_models]
        benchmark_models = get_benchmarks(stage_paths[i-1])
        params = tree_branch_template * len(prev_best_models)
        parent_path_vector = prev_best_models_paths * len(tree_branch_template)
        for n in range(len(params)):
            params[n]['model_path'] = parent_path_vector[n]
        params = gen_run_ids(params, stage_path=stage_paths[i], resume=True)
        train_stage(params, stage_paths[i], replay=False, resume=True)
        eval_stage(stage_paths[i], n_select=select, benchmark_models=benchmark_models, resume=True)
        print('#' * 50 + '\n' + 'Best Model Statistics')
        print(get_best_stats(stage_paths[i]))
        print('#' * 50)
    
    return None


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
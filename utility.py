'''
Utility functions that are useful for other functions
'''

# Imports
import random
import os
import pandas as pd

from luxai2021.game.constants import LuxMatchConfigs_Default
from luxai2021.env.lux_env import LuxEnvironment
from luxai2021.env.agent import Agent

def get_existing_models(modellist_path=None):
    '''
    Reads the modellist file given and returns the list of models in the file
    If file does not exist, returns an empty list.
    '''
    if not modellist_path:
        modellist_path = "C:/Users/lenovo/Desktop/Coding/LuxAI/rl/modelcp/modellist.txt"
    try:
        with open(modellist_path, 'r') as f:
            modellist =  f.read().split('\n')
        modellist.remove('')
    except FileNotFoundError:
        print(f"File not found. File created at path: {modellist_path}")
        with open(modellist_path, 'w') as f:
            f.write('')
        modellist = []
    return modellist

def is_same(model1, model2, num_envs=3) -> bool:
    '''
    Function to test if 2 models are the same. Generates 1 environment and use both models to predict on the same env.
    Then, compare the predictions made.
    '''
    from models.agent_policy import AgentPolicy

    for _ in range(num_envs):
        configs = LuxMatchConfigs_Default
        configs['seed'] = random.randint(0, 10000)
        env = LuxEnvironment(configs, AgentPolicy(), Agent())
        obs = env.reset()
        for i in range(600):
            action_code1, _status1 = model1.predict(obs, deterministic=True)
            action_code2, _status2 = model2.predict(obs, deterministic=True)
            if action_code1 != action_code2:
                print(obs, action_code1, action_code2)
                return False
            obs, reward, done, info = env.step(action_code1)
            if done:
                break
    return True

def clear_eval_files():
    import shutil
    stage_path = 'pool/stage_0/'

    model_ids = get_existing_models(stage_path + 'modellist.txt')
    for model_id in model_ids:
        shutil.rmtree(stage_path + model_id + '/eval_replay')
        if os.path.exists(stage_path + model_id + '/eval.json'):
            os.remove(stage_path + model_id + '/eval.json')

def get_best_stats(stage_path):
    '''
    Use with stage 0 only.
    '''
    BEST_MODELS_FILE_NAME = "best_models.txt"

    # Auto-adjust model_path to end with '/'
    if stage_path[-1] != '/' and stage_path[-1] != '\\':
        stage_path += '/'
    
    with open(stage_path + BEST_MODELS_FILE_NAME, 'r') as f:
        best_models = eval(f.read())
    
    data = []
    for model_id, score in best_models.items():
        with open(stage_path + model_id + '/info.json', 'r') as f:
            model_info = eval(f.read())
        n_parents = len(model_info['parents'])
        if n_parents == 0:
            parent = None
        else:
            parent = model_info['parents'][-1]
        if stage_path[-2] == '0':
            last_policy = model_info['policy']
        else:
            last_policy = model_info['train_params']['policy']
        last_train_params = model_info['train_history'][-1]
        last_lr = last_train_params['train_params']['learning_rate']
        last_gamma = last_train_params['train_params']['gamma']
        last_gl = last_train_params['train_params']['gae_lambda']
        last_opponent = 'agent_blank' if last_train_params['opponent']['model'] == None else last_train_params['opponent']['model']
        last_train_date = model_info['last_train']
        train_history = model_info['train_history']
        row = [model_id, score, parent, n_parents, last_policy, last_lr, last_gamma, last_gl, last_opponent, last_train_date, train_history]
        data.append(row)

    data = pd.DataFrame(data, columns=['id', 'score', 'parent', 'n_parents', 'last_policy', 'last_lr', 'last_gamma', 'last_gl', 'last_opponent', 'last_train_date', 'train_history'])
    data = data.sort_values('score', ascending=False)
    return data

def get_benchmarks(stage_path):
    # Auto-adjust model_path to end with '/'
    if stage_path[-1] != '/' and stage_path[-1] != '\\':
        stage_path += '/'

    BENCHMARK_MODELS_FILE_NAME = 'benchmark_models.txt'
    if os.path.exists(stage_path  + BENCHMARK_MODELS_FILE_NAME):
        with open(stage_path  + BENCHMARK_MODELS_FILE_NAME, 'r') as f:
            benchmark_models_paths = eval(f.read())
    else:
        benchmark_models_paths = []
    return benchmark_models_paths

def gen_run_ids(params, stage_path=None, resume=False):
    RAN_RANGE = 99999
    if resume:
        if stage_path:
            # Auto-adjust model_path to end with '/'
            if stage_path[-1] != '/' and stage_path[-1] != '\\':
                stage_path += '/'
            if os.path.exists(stage_path + 'train_params.txt'):
                with open(stage_path + 'train_params.txt', 'r') as f:
                    p = eval(f.read())
                return p
    p = params.copy()
    for param in p:
        param['run_id'] = str(random.randint(0, RAN_RANGE))
    if stage_path:
        # Auto-adjust model_path to end with '/'
        if stage_path[-1] != '/' and stage_path[-1] != '\\':
            stage_path += '/'
        with open(stage_path + 'train_params.txt', 'w') as f:
            f.write(str(p))
    return p
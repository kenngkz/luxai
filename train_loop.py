'''
Trains a single model through many timesteps and games and saves the trained model in a zip file.
'''
# Imports
from utility import get_existing_models

import random
from importlib import import_module
from stable_baselines3 import PPO
from datetime import datetime
import os

from luxai2021.game.constants import LuxMatchConfigs_Default
from luxai2021.env.lux_env import LuxEnvironment, SaveReplayAndModelCallback

def train_loop(
    ply_path=None, 
    ply_policy=None,
    opp_path=None,
    run_id=None,
    replay_freq=100000,
    replay_num=3,
    learning_rate=0.001,
    gamma=0.995,
    gae_lambda=0.95,
    step_count=100000,
    save_folder='../modelcp/pool/',
    modellist_file='../modelcp/models.txt',
    tensorboard_log='../modelcp/lux_tensorboard/'
    ):
    '''
    Implements training loop between a given player and opponent model. Only the player can learn in one training loop.
    '''
    # Set env configs
    configs = LuxMatchConfigs_Default

    # Set run id
    run_id_range = 100000
    existing_models = get_existing_models(modellist_file)
    if run_id==None:
        for _ in range(run_id_range):
            run_id = random.randint(1, run_id_range)
            if str(run_id) not in existing_models:
                break
    run_id = str(run_id)

    # Set new model save folder
    new_model_path = save_folder + f'{run_id}/'

    if ply_path:
        # auto-adjust ply_path to add '/' if not already included
        if ply_path[-1] != '/' and ply_path[-1] != '\\':
            ply_path += '/'
        # get info and set player policy and model
        with open(ply_path + 'info.json', 'r') as f:
            ply_info = eval(f.read())
        if not ply_policy:
            ply_policy = ply_info['policy']
        ply_policy_obj = import_module('models.' + ply_policy).AgentPolicy
        ply_model = PPO.load(ply_path + 'model.zip')
    else:
        ply_info = None
        if not ply_policy:
            ply_policy = 'agent_policy'
        ply_policy_obj = import_module('models.' + ply_policy).AgentPolicy
        ply_model = None
    # initialize player
    player = ply_policy_obj(mode='train')

    # Set opponent
    if opp_path:
        # auto-adjust opp_path to add '/' if not already included
        if opp_path[-1] != '/' and opp_path[-1] != '\\':
            opp_path += '/'
        # get info and set opponent policy and model
        with open(opp_path + 'info.json', 'r') as f:
            opp_info = eval(f.read())
        opp_policy = opp_info['policy']
        opp_policy_obj = import_module('models.' + opp_policy).AgentPolicy
        opp_model = PPO.load(opp_path + 'model.zip')
    else:
        opp_info = None
        opp_policy = 'agent_blank'
        opp_policy_obj = import_module('models.' + opp_policy).AgentPolicy
        opp_model = None
    # initialize opponent
    opponent = opp_policy_obj(mode='inference', model=opp_model)

    # Set up env and PPO model
    env = LuxEnvironment(configs, player, opponent)
    if ply_model:
        ply_model.set_env(env)
    else:
        ply_model = PPO("MlpPolicy",
        env,
        verbose=1,
        tensorboard_log=tensorboard_log,
        learning_rate = learning_rate,
        gamma=gamma,
        gae_lambda = gae_lambda
    ) 
    
    # Save a checkpoint and 'replay_num' match replay files every 'replay_freq' steps
    callbacks = []
    player_replay = ply_policy_obj(mode="inference", model=ply_model)
    if opp_model:
        opponent_replay = opp_policy_obj(mode="inference", model=opp_model)
    else:
        opponent_replay = opp_policy_obj()
    callbacks.append(
        SaveReplayAndModelCallback(
                                save_freq=replay_freq,
                                save_path=new_model_path + 'replays/',
                                name_prefix=f'model{run_id}',
                                replay_env=LuxEnvironment(
                                                configs=configs,
                                                learning_agent=player_replay,
                                                opponent_agent=opponent_replay
                                ),
                                replay_num_episodes=replay_num
                            )
    )

    # Training model
    print("Training model...")
    ply_model.learn(total_timesteps=step_count, callback=callbacks, tb_log_name=f'{run_id}')
    print("Done training model.")

    # Save model name to list of models
    with open(modellist_file, 'a') as f:
        f.write(str(run_id) + '\n')

    # Save model info
    if ply_info:
        parents = ply_info['parents']
        parents.append(ply_info['run_id'])
        train_history = ply_info['train_history']
    else:
        parents = []
        train_history = []
    
    last_train = datetime.now().strftime("%d/%m/%Y-%H:%M:%S")
    train_params = {'policy':ply_policy, 'learning_rate':learning_rate, 'gamma':gamma, 'gae_lambda':gae_lambda}
    opp_info = {'policy':opp_policy, 'model':opp_path}
    train_history.append({'step_count':step_count, 'train_params':train_params, 'opponent':opp_info, 'date_time':last_train})

    info = {'run_id':run_id, 'last_train':last_train, 'parents':parents, 'train_params':train_params, 'train_history':train_history}
    with open(new_model_path + 'info.json', 'w') as f:
        f.write(str(info))

    # Save model zip file
    if not os.path.exists(new_model_path + 'model.zip'):
        ply_model.save(path=new_model_path + 'model.zip')
    else:
        print(f"Existing model.zip found in file location {new_model_path}. Creating a new file with name model2.zip in the same directory ({new_model_path})")
        ply_model.save(path=new_model_path + 'model2.zip')

    print(f"Save complete. Folder name: {run_id}")
    return run_id
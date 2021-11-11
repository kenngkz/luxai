# Allow custom modules to be imported
from sys import path
path.append("C:\\Users\\lenovo\\Desktop\\Coding\\CustomModules")
from stage import eval_stage, train_stage

import argparse
import glob
import os
import sys
import random

from stable_baselines3 import PPO  # pip install stable-baselines3
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.utils import set_random_seed, get_schedule_fn
from stable_baselines3.common.vec_env import SubprocVecEnv

from models.agent0 import AgentPolicy
from luxai2021.env.agent import Agent
from luxai2021.env.lux_env import LuxEnvironment, SaveReplayAndModelCallback
from luxai2021.game.constants import LuxMatchConfigs_Default

"""
ERROR TRACEBACK
Preliminary Evaluation for model 2313. pending_models / total_models: 64 / 67
Traceback (most recent call last):
  File "../scripts/train.py", line 907, in <module>
  File "../scripts/train.py", line 779, in main
    if stage_num > 0:
  File "../scripts/train.py", line 619, in eval_stage
    '''
  File "C:\Users\lenovo\anaconda3\envs\luxai\lib\random.py", line 321, in sample
    raise ValueError("Sample larger than population or is negative")
ValueError: Sample larger than population or is negative
"""



# https://stable-baselines3.readthedocs.io/en/master/guide/examples.html?highlight=SubprocVecEnv#multiprocessing-unleashing-the-power-of-vectorized-environments
def make_env(local_env, rank, seed=0):
    """
    Utility function for multi-processed env.

    :param local_env: (LuxEnvironment) the environment 
    :param seed: (int) the initial seed for RNG
    :param rank: (int) index of the subprocess
    """

    def _init():
        local_env.seed(seed + rank)
        return local_env

    set_random_seed(seed)
    return _init


def get_command_line_arguments():
    """
    Get the command line arguments
    :return:(ArgumentParser) The command line arguments as an ArgumentParser
    """
    parser = argparse.ArgumentParser(description='Training script for Lux RL agent.')
    parser.add_argument('--id', help='Identifier of this run', type=str, default=str(random.randint(0, 100000)))
    parser.add_argument('--learning_rate', help='Learning rate', type=float, default=0.001)
    parser.add_argument('--gamma', help='Gamma', type=float, default=0.995)
    parser.add_argument('--gae_lambda', help='GAE Lambda', type=float, default=0.95)
    parser.add_argument('--batch_size', help='batch_size', type=int, default=2048)  # 64
    parser.add_argument('--step_count', help='Total number of steps to train', type=int, default=10000000)
    parser.add_argument('--n_steps', help='Number of experiences to gather before each learning period', type=int, default=2048)
    parser.add_argument('--path', help='Path to a checkpoint to load to resume training', type=str, default=None)
    parser.add_argument('--n_envs', help='Number of parallel environments to use in training', type=int, default=1)
    parser.add_argument('--opp_path', help='Path to a checkpoint to load as the opponent.', type=str, default=None)
    args = parser.parse_args()

    return args



def train(args):
    """
    The main training loop
    :param args: (ArgumentParser) The command line arguments
    """
    print(args)

    # Run a training job
    configs = LuxMatchConfigs_Default

    # Create an opponent agent
    opponent = Agent()

    # Create a RL agent in training mode
    player = AgentPolicy(mode="train")

    # Train the model
    env_eval = None
    if args.n_envs == 1:
        env = LuxEnvironment(configs=configs,
                             learning_agent=player,
                             opponent_agent=opponent)
    else:
        env = SubprocVecEnv([make_env(LuxEnvironment(configs=configs,
                                                     learning_agent=AgentPolicy(mode="train"),
                                                     opponent_agent=opponent), i) for i in range(args.n_envs)])
    
    run_id = args.id
    print("Run id %s" % run_id)

    if args.path:
        # by default previous model params are used (lr, batch size, gamma...)
        model = PPO.load(args.path)
        model.set_env(env=env)

        # Update the learning rate
        model.lr_schedule = get_schedule_fn(args.learning_rate)

        # TODO: Update other training parameters
    else:
        model = PPO("MlpPolicy",
                    env,
                    verbose=1,
                    tensorboard_log="./lux_tensorboard/",
                    learning_rate=args.learning_rate,
                    gamma=args.gamma,
                    gae_lambda=args.gae_lambda,
                    batch_size=args.batch_size,
                    n_steps=args.n_steps
                    )

    
    
    callbacks = []

    # Save a checkpoint and 5 match replay files every 100K steps
    player_replay = AgentPolicy(mode="inference", model=model)
    callbacks.append(
        SaveReplayAndModelCallback(
                                save_freq=100000,
                                save_path='./replays/',
                                name_prefix=f'model{run_id}',
                                replay_env=LuxEnvironment(
                                                configs=configs,
                                                learning_agent=player_replay,
                                                opponent_agent=Agent()
                                ),
                                replay_num_episodes=3
                            )
    )
    
    # Since reward metrics don't work for multi-environment setups, we add an evaluation logger
    # for metrics.
    if args.n_envs > 1:
        # An evaluation environment is needed to measure multi-env setups. Use a fixed 4 envs.
        env_eval = SubprocVecEnv([make_env(LuxEnvironment(configs=configs,
                                                     learning_agent=AgentPolicy(mode="train"),
                                                     opponent_agent=opponent), i) for i in range(4)])

        callbacks.append(
            EvalCallback(env_eval, best_model_save_path=f'./logs_{run_id}/',
                             log_path=f'./logs_{run_id}/',
                             eval_freq=args.n_steps*2, # Run it every 2 training iterations
                             n_eval_episodes=30, # Run 30 games
                             deterministic=False, render=False)
        )

    print("Training model...")
    model.learn(total_timesteps=args.step_count, callback=callbacks)
    if not os.path.exists(f'models/rl_model_{run_id}_{args.step_count}_steps.zip'):
        model.save(path=f'models/rl_model_{run_id}_{args.step_count}_steps.zip')
    print("Done training model.")

    # Inference the model
    print("Inference model policy with rendering...")
    saves = glob.glob(f'models/rl_model_{run_id}_*_steps.zip')
    latest_save = sorted(saves, key=lambda x: int(x.split('_')[-2]), reverse=True)[0]
    model.load(path=latest_save)
    obs = env.reset()
    for i in range(600):
        action_code, _states = model.predict(obs, deterministic=True)
        obs, rewards, done, info = env.step(action_code)
        if i % 5 == 0:
            print("Turn %i" % i)
            env.render()

        if done:
            print("Episode done, resetting.")
            obs = env.reset()
    print("Done")

    '''
    # Learn with self-play against the learned model as an opponent now
    print("Training model with self-play against last version of model...")
    player = AgentPolicy(mode="train")
    opponent = AgentPolicy(mode="inference", model=model)
    env = LuxEnvironment(configs, player, opponent)
    model = PPO("MlpPolicy",
        env,
        verbose=1,
        tensorboard_log="./lux_tensorboard/",
        learning_rate = 0.0003,
        gamma=0.999,
        gae_lambda = 0.95
    )

    model.learn(total_timesteps=2000)
    env.close()
    print("Done")
    '''




def main(args):
    '''
    Main function to execute.
    '''
    # Variables
    stage_size = 30  # number of models in 1 stage
    select = 12  # select the top 'select' models to pass into the next stage
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
    best_models = eval_stage(stage_path, select, resume=True)
    scores = {}
    for model_id in best_models:
        with open(stage_path + model_id + '/eval.json', 'r') as f:
            scores[model_id] = eval(f.read())['score']
    print(f"Best models in stage_0:")
    for id, score in scores.items():
        print(f"  {id} : {score}")

    with open(stage_path + 'best_models.txt', 'w') as f:
            f.write(str(best_models))
    
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
    
    # Get the command line arguments
    local_args = get_command_line_arguments()

    # Train the model
    main(local_args)

    # modellist = get_existing_models()

    # for i, id in enumerate(modellist):
    #     opps = modellist[:i] + modellist[i+1:]
    #     eval_model(f'../modelcp/pool/{id}', [f'../modelcp/pool/{opp}' for opp in opps], n_games=5, save=False)

    # Note: run this file from LuxAI/rl/scripts
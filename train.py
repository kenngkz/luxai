from sys import path
path.append("C:\\Users\\kenng\\Desktop\\Coding\\CustomModules")

import argparse
import glob
import os
import sys
import random
from importlib import import_module
from datetime import datetime
from data_structures.quickselect import select_k
from math import exp
from numpy import sign

from stable_baselines3 import PPO  # pip install stable-baselines3
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.utils import set_random_seed, get_schedule_fn
from stable_baselines3.common.vec_env import SubprocVecEnv

from models.agent0 import AgentPolicy
from luxai2021.env.agent import Agent
from luxai2021.env.lux_env import LuxEnvironment, SaveReplayAndModelCallback
from luxai2021.game.constants import LuxMatchConfigs_Default

class ModelScore:
    '''
    DataClass to store model_ids and model scores. To be used with the quickselect algorithm
    '''
    def __init__(self, id, results:dict={}, score=None):
        self.id = id
        self.results = results
        self.score = score
        self.opp_pool = None
        self.remaining_opp = None
    
    def cal_winrate(self):
        # calculate the average winrate
        winrates = [summary['winrate'] for summary in self.results.values()]
        return sum(winrates)/len(winrates)

        # calculates the average winrates weighted by number of games played
        total_games = 0
        weighted_sum = []
        for summary in self.results.values():
            total_games += summary['n_games']
            weighted_sum.append((summary['n_games'], summary['winrate']))
        return sum([n_games*winrate for n_games, winrate in weighted_sum])/total_games

    def add_pool(self, opp_pool):
        self.opp_pool = opp_pool
        self.remaining_opp = list(opp_pool.keys())
        self.remaining_opp.remove(self.id)
        self.update(self.results)

    def cal_score(self, results):
        '''
        Function to calculate the score of a model. Higher winrate against opponents with higher relative score is rewarded more.
        If opponent does not have a existing score, the score of the opponent is estimated from their weighted winrate.
        '''
        assert self.opp_pool != None, "opp_pool must be specified for cal_score() to run. Add opp_pool using ModelScore.add_pool(opp_pool)"
        if self.score == None:
            self.score = 100
        for opp_id, summary in results.items():
            winrate = summary['winrate']
            if self.opp_pool[opp_id].score == None:
                opp_winrate = self.opp_pool[opp_id].cal_winrate()  # calculate the weighted winrate of the opponent.
                opp_score = 50 + 100 * opp_winrate  
                    # if 50% winrate, score will be 100 (default score)
                    # range of estimated opp_score: 50 - 150
            else:
                opp_score = self.opp_pool[opp_id].score
            # Score Calculation Formula
            relative_score_diff = (opp_score - self.score)/100  # weighted relative difference in score
            # print(self.score, opp_score, relative_score_diff)
            score_weight = exp(sign(winrate) * relative_score_diff) * 10  # score weight defines how relative score difference affects the actual change in score linearly to winrate
            self.score += score_weight * (winrate - relative_score_diff)

    def update(self, results, cal_score=True):
        self.skip_opps(results.keys())
        for id, summary in results.items():
            self.results[id] = summary  # replace previous summary if this model has played against that opponent before
            if cal_score:
                self.cal_score(results)

    def skip_opps(self, opps):
        for id in opps:
            try:
                self.remaining_opp.remove(id)
            except ValueError:
                print(f"Opponent ID {id} not in the remaining opponent list of ModelScore with id {self.id}. Model {self.id} has likely played against {id} before.\n{self.results}")

    def __ge__(self, other):
        return self.score >= other.score

    def __le__(self, other):
        return self.score <= other.score


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

def get_existing_models(modellist_path=None):
    if not modellist_path:
        modellist_path = "C:/Users/lenovo/Desktop/Coding/LuxAI/rl/modelcp/models.txt"
    try:
        with open(modellist_path, 'r') as f:
            modellist =  f.read().split('\n')
        modellist.remove('')
    except FileNotFoundError:
        with open(modellist_path, 'w') as f:
            f.write('')
        modellist = []
    return modellist

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
        if ply_path[-1] != '/' or ply_path[-1] != '\\':
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
        if opp_path[-1] != '/' or opp_path[-1] != '\\':
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
    train_params = {'learning_rate':learning_rate, 'gamma':gamma, 'gae_lambda':gae_lambda}
    opp_info = {'policy':opp_policy, 'model':opp_path}
    train_history.append({'step_count':step_count, 'train_params':train_params, 'opponent':opp_info, 'date_time':last_train})

    info = {'run_id':run_id, 'last_train':last_train, 'policy':ply_policy, 'parents':parents, 'train_params':train_params, 'train_history':train_history}
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

def eval_model(
    model_path,
    opp_paths:list,
    n_games=10,
    max_steps=10000,
    seed=random.randint(0, 10000),
    replay_dir='eval_replay/',
    replay_prefix='replay',
    save=True,  # True to save replays
    overwrite=False
    ):
    '''
    Evaluate how good a model is. Plays the given model with an opponent and records the win percentage.
    '''
    # Auto-adjust model_path to end with '/'
    if model_path[-1] != '/' or model_path[-1] != '\\':
        model_path += '/'

    # Set env configs
    configs = LuxMatchConfigs_Default
    # Set player
    with open(model_path + 'info.json', 'r') as f:
        ply_info = eval(f.read())
    ply_policy = ply_info['policy']
    ply_policy_obj = import_module('models.' + ply_policy).AgentPolicy
    player = ply_policy_obj(mode='train')
    model=PPO.load(model_path+'model.zip')

    # Check for existing eval history
    try:
        with open(model_path+'eval.json', 'r') as f:
            eval_history = eval(f.read())['history']
    except FileNotFoundError:
        eval_history = {}

    print(f"Evaluation for model {ply_info['run_id']} against {len(opp_paths)} opponents...")
    
    results = {}
    for opp_path in opp_paths:
        # auto-adjust opp_path to end with '/'
        if opp_path[-1] != '/' or opp_path[-1] != '\\':
            opp_path += '/'
        # get info and set opponent policy
        with open(opp_path + 'info.json', 'r') as f:
            opp_info = eval(f.read())
        opp_id = opp_info['run_id']
        opp_policy = opp_info['policy']
        opp_policy_obj = import_module('models.' + opp_policy).AgentPolicy
        # initialize opponent
        opponent = opp_policy_obj(mode='inference', model=PPO.load(opp_path+'model.zip'))

        # Get past history if not overwrite and add the new match results into the existing eval history
        if not overwrite and (opp_id in eval_history):
            summary = eval_history[opp_id]
        else:
            summary = {'n_games':0, 'n_wins':0, 'winrate':0}

        new_matches = []

        # Run games and save replay
        for i in range(n_games):
            configs['seed'] = seed
            if save:
                replay_name = replay_prefix + f'_{opp_id}_{i}'
                env = LuxEnvironment(configs, player, opponent, replay_folder=model_path+replay_dir, replay_prefix=replay_name)
            else:
                env = LuxEnvironment(configs, player, opponent)
            obs = env.reset()
            for i in range(max_steps):
                action_code, _states = model.predict(obs, deterministic=True)
                obs, rewards, done, info = env.step(action_code)
                if done:
                    print("Episode done, resetting.")
                    new_matches.append(int(env.game.get_winning_team() == player.team))
                    obs = env.reset()
                    break
    
        n_wins = sum(new_matches)
        print(f"All matches against opponent {opp_id} completed. Win - Loss : {n_wins} - {n_games - n_wins}")
        
        # add matches to evaluation match history
        summary['n_games'] += n_games
        summary['n_wins'] += n_wins
        summary['winrate'] = summary['n_wins']/summary['n_games']

        eval_history[opp_id] = summary
        results[opp_id] = summary

    with open(model_path+'eval.json', 'w') as f:
        f.write(str({'history':eval_history}))

    print(f"Evaluation for model {ply_info['run_id']} complete.")
    print(f"Results: {results}")

    return results

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

    models = params['models']
    model_ids = []
    n_models = 0
    # run training using the given params
    for model, n in models.items():
        n_models += n
        train_loop_args = {
                    'step_count': params['steps'],
                    'learning_rate': params['learning_rate'],
                    'gamma': params['gamma'],
                    'gae_lambda': params['gae_lambda'],
                    'save_folder':stage_path,
                    'modellist_file':stage_path + 'modellist.txt',
                    'tensorboard_log':stage_path + 'tensorboard_log/'
                }
        # determine if a new model is to be generated
        if not new_model:
            train_loop_args['ply_path'] = model
        else:
            train_loop_args['ply_policy'] = model
        # check for opp_path in params
        if 'opp_path' in params:
            train_loop_args['opp_path'] = params['opp_path']
        # check if training replays should be saved
        if replay:
            train_loop_args['replay_freq'] = params['steps']
        else:
            if 'replay_freq' in params:
                train_loop_args['replay_freq'] = params['replay_freq']
        if 'run_id' in params:
            train_loop_args['run_id'] = params['run_id']
        id = [train_loop(**train_loop_args) for _ in range(n)]
        model_ids += id

    return model_ids

def eval_stage(stage_path, select, model_ids=None, n_games=3, max_steps=1000):

    if not model_ids:
        model_ids = get_existing_models(stage_path + 'modellist.txt')
    n_models = len(model_ids)

    # Initiliaze a ModelScore for every model to be evaluated
    model_scores = {}
    for id in model_ids:
        model_scores[id] = ModelScore(id)
    # Allow every ModelScore to access all other ModelScores
    for model_score in model_scores.values():
        model_score.add_pool(model_scores)

    # If selecting a small portion of n_models and there are at least 8 models, 
        # eliminate half of the pool to reduce computation time
    if n_models/select < 0.45 and n_models > 7:
        for id in model_ids:
            results = eval_model(stage_path+id, [stage_path+opp_id for opp_id in model_score.remaining_opp], n_games=n_games, max_steps=max_steps)
            model_score.update(results)
        _, i, sorted_scores = select_k([model_score for model_score in model_scores.values()], int(n_models/2))
        final_model_ids  = sorted_scores[:i+1]
    else:
        final_model_ids = model_ids

    # Select the final n best models 
    for id in final_model_ids:
        results = eval_model(stage_path+id, [stage_path+opp_id for opp_id in model_score.remaining_opp], n_games=n_games, max_steps=max_steps)
        model_scores[id].update(results)

    _, i, selected_scores = select_k([model_score for model_score in model_scores.values()], select)

    # Update all eval.json files for every model to include their score
    for id, model_score in model_scores.items():
        with open(stage_path + id + '/eval.json', 'r') as f:
            eval_info = eval(f.read())
        eval_info['score'] = model_score.score
        with open(stage_path + id + '/eval.json', 'w') as f:
            f.write(str(eval_info))

    return [model_score.id for model_score in selected_scores[:i+1]]

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

def main(args):
    '''
    Main function to execute.
    '''
    # Variables
    stage_size = 30  # number of models in 1 stage
    select = 6  # select the top 'select' models to pass into the next stage
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
            'model_policy':'agent1',
            'n_copies':3,
            'step_count': 1000000,
            'learning_rate':0.005,
        },
        {
            'model_policy':'agent1',
            'n_copies':3,
            'step_count': 1000000,
            'learning_rate':0.001,
            'gamma':0.999,
        },
        {
            'model_policy':'agent1',
            'n_copies':3,
            'step_count': 1000000,
            'learning_rate':0.001,
            'gamma':0.995,
            'gae_lambda':0.99,
        },
        {
            'model_policy':'agent1',
            'n_copies':3,
            'step_count': 1000000,
            'learning_rate':0.001,
            'gamma':0.995,
            'gae_lambda':0.90,
        },
        {
            'model_policy':'agent2',
            'n_copies':3,
            'step_count': 1000000,
            'learning_rate':0.001,
        },
        {
            'model_policy':'agent2',
            'n_copies':3,
            'step_count': 1000000,
            'learning_rate':0.005,
        },
        {
            'model_policy':'agent2',
            'n_copies':3,
            'step_count': 1000000,
            'learning_rate':0.001,
            'gamma': 0.999,
        },
        {
            'model_policy':'agent2',
            'n_copies':3,
            'step_count': 1000000,
            'learning_rate':0.005,
            'gamma': 0.995,
            'gae_lambda': 0.99
        },
        {
            'model_policy':'agent2',
            'n_copies':3,
            'step_count': 1000000,
            'learning_rate':0.005,
            'gamma': 0.995,
            'gae_lambda': 0.90
        }
    ]
    ini_train_params = [
        {
            'model_policy':'agent3',
            'n_copies':3,
            'step_count':1000000
        }
    ]

    stage_path = stage_paths[0]
    models = train_stage(ini_train_params, stage_path)
    # best_models = eval_stage(stage_path, select)
    # print(f"Best models in stage_0: {best_models}")

    # with open(stage_path + 'best_models.txt', 'w') as f:
            # f.write(str(best_models))
    
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
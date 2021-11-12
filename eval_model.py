'''
Evaluates 1 model by playing it against a list of opponents (list of paths to other models.).
Records the results in eval.json.
'''

# Imports
import random
from importlib import import_module
from stable_baselines3 import PPO

from luxai2021.env.lux_env import LuxEnvironment
from luxai2021.game.constants import LuxMatchConfigs_Default

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
    n_opps = len(opp_paths)
    for opp_index, opp_path in enumerate(opp_paths):
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
        for n in range(n_games):
            configs['seed'] = seed
            if save:
                replay_name = replay_prefix + f'_{opp_id}_{n}'
                env = LuxEnvironment(configs, player, opponent, replay_folder=model_path+replay_dir, replay_prefix=replay_name)
            else:
                env = LuxEnvironment(configs, player, opponent)
            obs = env.reset()
            for i in range(max_steps):
                action_code, _states = model.predict(obs, deterministic=True)
                obs, rewards, done, info = env.step(action_code)
                if done:
                    new_matches.append(int(env.game.get_winning_team() == player.team))
                    obs = env.reset()
                    break
    
        n_wins = sum(new_matches)
        print(f"All matches against opponent {opp_id} completed. Win - Loss : {n_wins} - {n_games - n_wins}. Progress: {opp_index+1} / {n_opps}")
        
        # add matches to evaluation match history
        summary['n_games'] += n_games
        summary['n_wins'] += n_wins
        summary['winrate'] = summary['n_wins']/summary['n_games']

        eval_history[opp_id] = summary
        results[opp_id] = summary

    with open(model_path+'eval.json', 'w') as f:
        f.write(str({'history':eval_history}))

    print(f"Evaluation for model {ply_info['run_id']} complete.")
    total_games = n_games * n_opps
    total_wins = sum([summ['n_wins'] for summ in results.values()])
    print(f"Results -> Total Wins - Total Losses : {total_wins} - {total_games-total_wins}")

    return results
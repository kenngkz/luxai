from base_utils import path_join
from constants import WORKER_DATABASE_DIR

import os
import random
from importlib import import_module

from stable_baselines3 import PPO
from luxai2021.env.lux_env import LuxEnvironment
from luxai2021.game.constants import LuxMatchConfigs_Default

def eval_model(
    model_path,
    opp_paths:list,
    n_games=5,
    max_steps=1000,
    seeds=None,
    replay_dir="eval_replays/",
    replay_prefix="replay",
    save_replays=False,
    database=WORKER_DATABASE_DIR,
):
    '''
    Evaluate a model by running games against a set of opponents and returns the games results.
    '''
    # Set seeds
    if not seeds:
        seeds = [random.randint(0, 10000) for _ in range(n_games)]
    assert len(seeds) == n_games, "Length of 'seeds' arg does not match n_games."

    # Set env configs
    configs = LuxMatchConfigs_Default

    # Initialize player
    with open(path_join(database, model_path, 'info.json'), 'r') as f:
        model_info = eval(f.read())
    policy = model_info['train_params']['policy']
    policy_obj = import_module(f"worker.models.{policy}").AgentPolicy
    player = policy_obj(mode='train')
    model=PPO.load(path_join(database, model_path, "model.zip"))

    print(f"Evaluation for model {model_info['run_id']} against {len(opp_paths)} opponents...")

    # Loop through opponents and run multiple games against them
    n_opps = len(opp_paths)
    results = {}
    for index, opp_path in enumerate(opp_paths):
        # Initialize opponent
        assert os.path.exists(path_join(database, opp_path, 'info.json')), f"File not found: {path_join(opp_path, 'info.json')}"
        with open(path_join(database, opp_path, 'info.json'), 'r') as f:
            opp_info = eval(f.read())
        opp_id = opp_info['run_id']
        opp_policy = opp_info['train_params']['policy']
        opp_policy_obj = import_module('worker.models.' + opp_policy).AgentPolicy
        opponent = opp_policy_obj(mode='inference', model=PPO.load(path_join(database, opp_path,'model.zip')))

        match_wins = []
        opp_performance_summary = {}
        # Run games and save replay
        for n in range(n_games):
            configs['seed'] = seeds[n]
            if save_replays:
                replay_name = f'{replay_prefix}_{opp_id}_{n}'
                env = LuxEnvironment(configs, player, opponent, replay_folder=path_join(database, model_path, replay_dir), replay_prefix=replay_name)
            else:
                env = LuxEnvironment(configs, player, opponent)
            obs = env.reset()
            for i in range(max_steps):
                action_code, _states = model.predict(obs, deterministic=True)
                obs, rewards, done, info = env.step(action_code)
                if done:
                    match_wins.append(int(env.game.get_winning_team() == player.team))
                    obs = env.reset()
                    break

            n_wins = sum(match_wins)
        print(f"All matches against opponent {opp_id} completed. Win - Loss : {n_wins} - {n_games - n_wins}. Progress: {index+1} / {n_opps}")

        # add matches to evaluation match history
        opp_performance_summary["n_games"] = n_games
        opp_performance_summary["n_wins"] = n_wins
        opp_performance_summary["winrate"] = opp_performance_summary["n_wins"]/opp_performance_summary["n_games"]

        results[opp_id] = opp_performance_summary

    print(f"Evaluation for model {model_info['run_id']} complete.")
    total_games = n_games * n_opps
    total_wins = sum([summ["n_wins"] for summ in results.values()])
    print(f"Results -> Total Wins - Total Losses : {total_wins} - {total_games-total_wins}")

    return results

def random_result(
    model_path,
    opp_paths:list,
    n_games=5,
    max_steps=1000,
    seeds=None,
    replay_dir="eval_replays/",
    replay_prefix="replay",
    save_replays=False,
    database=WORKER_DATABASE_DIR,
):
    '''
    Generate random wins/losses for one eval_model.
    '''
    with open(path_join(database, model_path, 'info.json'), 'r') as f:
        model_info = eval(f.read())
    n_opps = len(opp_paths)
    results = {}
    for index, opp_path in enumerate(opp_paths):
        # Initialize opponent
        assert os.path.exists(path_join(database, opp_path, 'info.json')), f"File not found: {path_join(opp_path, 'info.json')}"
        with open(path_join(database, opp_path, 'info.json'), 'r') as f:
            opp_info = eval(f.read())
        opp_id = opp_info['run_id']

        match_wins = []
        opp_performance_summary = {}
        # Run games and save replay
        for n in range(n_games):
            match_wins.append(random.choice([0, 1]))
            n_wins = sum(match_wins)
        # add matches to evaluation match history
        opp_performance_summary["n_games"] = n_games
        opp_performance_summary["n_wins"] = n_wins
        opp_performance_summary["winrate"] = opp_performance_summary["n_wins"]/opp_performance_summary["n_games"]

        results[opp_id] = opp_performance_summary

    print(f"Evaluation (randomly generated) for model {model_info['run_id']} complete.")
    total_games = n_games * n_opps
    total_wins = sum([summ["n_wins"] for summ in results.values()])
    print(f"Results -> Total Wins - Total Losses : {total_wins} - {total_games-total_wins}")
    return results

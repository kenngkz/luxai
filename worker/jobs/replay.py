from luxai2021.env.lux_env import LuxEnvironment
from luxai2021.game.constants import LuxMatchConfigs_Default

from base_utils import path_join
from constants import WORKER_DATABASE_DIR

from importlib import import_module
from stable_baselines3 import PPO
import random
import os
import json

def replay(n_replays, model_path, opp_path=None, database=WORKER_DATABASE_DIR, seeds=None, max_steps=5000, player_names=[], autoname=False):
    # Set seeds
    if not seeds:
        seeds = [random.randint(0, 10000) for _ in range(n_replays)]
    assert len(seeds) == n_replays, "Length of 'seeds' arg does not match n_games."

    # Set env configs
    configs = LuxMatchConfigs_Default

    # Initialize player
    with open(path_join(database, model_path, 'info.json'), 'r') as f:
        model_info = eval(f.read())
    model_id = model_info["run_id"]
    model_policy = model_info["train_params"]["policy"]
    policy_obj = import_module(f"worker.models.{model_policy}").AgentPolicy
    player = policy_obj(mode="train")
    model = PPO.load(path_join(database, model_path, "model.zip"))

    # Initialize opponent
    if opp_path:
        with open(path_join(database, opp_path, "info.json"), 'r') as f:
            opp_info = eval(f.read())
        opp_policy = opp_info["policy"]
        opp_id = opp_info["run_id"]
        opp_model = PPO.load(path_join(database, opp_path, "model.zip"))
    else:
        opp_info = None
        opp_id = "blank"
        opp_policy = 'agent_blank'
        opp_model = None
    opp_policy_obj = import_module(f"worker.models.{opp_policy}").AgentPolicy
    opponent = opp_policy_obj(mode="inference", model=opp_model)

    print(f"Generating {n_replays} replays for model {model_id} vs {opp_id}...")
    replay_folder = f"replays_vs_{opp_id}"
    replay_folder_path = path_join(database, model_path, replay_folder)

    # Run replay games'
    if os.path.exists(replay_folder_path):
        existing_replay_files = os.listdir(replay_folder_path)
    else:
        existing_replay_files = []
    teams = {}
    for n in range(n_replays):
        game_complete = False
        configs['seed'] = seeds[n]
        replay_name = f'replay_{n}'
        env = LuxEnvironment(configs, player, opponent, replay_folder=replay_folder_path, replay_prefix=replay_name)
        obs = env.reset()
        for i in range(max_steps):
            action_code, _states = model.predict(obs, deterministic=True)
            obs, rewards, done, info = env.step(action_code)
            if done:
                game_complete = True
                break
        if game_complete:
            new_replay_file = [replay_file for replay_file in os.listdir(replay_folder_path) if replay_file not in existing_replay_files]
            assert len(new_replay_file) < 2, f"Error: new replay files > 1. new_replay_file: {new_replay_file}. existing_replay_files: {existing_replay_files}. current replay files: {os.listdir(replay_folder_path)}"
            existing_replay_files = os.listdir(replay_folder_path)
            teams[new_replay_file[0]] = [agent.team for agent in env.match_controller.agents]
        obs = env.reset()
        print(f"Game {n+1} / {n_replays} done. Game complete: {game_complete}")

    # Edit replay names to provided player and opponent names
    if autoname:
        player_names = ["player", "opponent"]
    if len(player_names) > 0:
        assert len(player_names) == 2, "player_names arg must be a list with 2 names. The first name is the player name and the 2nd is the opponent name"
        for filename, team in teams.items():
            with open(path_join(database, model_path, replay_folder, filename), "r") as f:
                replay_json = json.loads(f.read())
            replay_json["teamDetails"] = {"name": player_names[team[0]], "tournamentID": ""}, {"name": player_names[team[1]], "tournamentID": ""}
            with open(path_join(database, model_path, replay_folder, filename), "w") as f:
                f.write(json.dumps(replay_json))

    print(f"Replay generation complete. Model: {model_path}. Replay folder: {replay_folder}.")

    return f"replays_vs_{opp_id}"
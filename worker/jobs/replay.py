from luxai2021.env.lux_env import LuxEnvironment, SaveReplayAndModelCallback
from luxai2021.game.constants import LuxMatchConfigs_Default

from base_utils import path_join
from constants import WORKER_DATABASE_DIR

from importlib import import_module
import random
from stable_baselines3 import PPO

def replay(n_replays, model_path, opp_path=None, database=WORKER_DATABASE_DIR, seeds=None, max_steps=1000):
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

    # Initialize env
    env = LuxEnvironment(configs, player, opponent)

    print(f"Generating {n_replays} replays for model {model_id} vs {opp_id}...")

    for n in range(n_replays):
        configs['seed'] = seeds[n]
        replay_name = f'replay_{n}'
        env = LuxEnvironment(configs, player, opponent, replay_folder=path_join(database, model_path, f"replays_vs_{opp_id}"), replay_prefix=replay_name)
        obs = env.reset()
        for i in range(max_steps):
            action_code, _states = model.predict(obs, deterministic=True)
            obs, rewards, done, info = env.step(action_code)
            if done:
                obs = env.reset()
                print(f"Game {n+1}/{n_replays} done.")
                break

    print(f"Replay generation complete. Model: {model_path}. Replay folder: replays_vs_{opp_id}.")

    return f"replays_vs_{opp_id}"
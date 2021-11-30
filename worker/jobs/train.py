from base_utils import path_join
from constants import WORKER_DATABASE_DIR

import os
from datetime import datetime
from importlib import import_module
from stable_baselines3 import PPO

from luxai2021.game.constants import LuxMatchConfigs_Default
from luxai2021.env.lux_env import LuxEnvironment, SaveReplayAndModelCallback

def train(
    run_id=None,
    model_path = None,
    model_policy = None,
    opp_path = None,
    step_count = 100000,
    learning_rate=0.001,
    gamma=0.995,
    gae_lambda=0.95,
    replay_freq = None,
    replay_num = 3,
    stage_prefix = "stage",
    tensorboard_log="tensorboard_log",
    database=WORKER_DATABASE_DIR
):
    # Set env configs
    configs = LuxMatchConfigs_Default

    # Set run id
    run_id = str(run_id)

    # Set new model save folder
    old_stage = os.path.dirname(model_path)
    stage_name = f"{stage_prefix}_{int(old_stage.split('_')[-1])+1}"
    new_model_path = path_join(stage_name, run_id)
    if not os.path.exists(path_join(database, new_model_path)):
        os.makedirs(path_join(database, new_model_path))

    # Initialize player
    if model_path:  
        with open(path_join(database, model_path, 'info.json'), 'r') as f:
            model_info = eval(f.read())
        if not model_policy:
            model_policy = model_info["policy"]
    else:
        model_info = None
        if not model_policy:
            model_policy = "agent_policy"
    policy_obj = import_module(f"worker.models.{model_policy}").AgentPolicy
    player = policy_obj(mode="train")

    # Initialize opponent
    if opp_path:
        if opp_path == "self":
            opp_info = model_info
            opp_policy = model_policy
            opp_model = PPO.load(path_join(database, model_path, "model.zip"))
        else:
            with open(path_join(database, opp_path, "info.json"), 'r') as f:
                opp_info = eval(f.read())
            opp_policy = opp_info["policy"]
            opp_model = PPO.load(path_join(database, opp_path, "model.zip"))
    else:
        opp_info = None
        opp_policy = 'agent_blank'
        opp_model = None
    opp_policy_obj = import_module(f"worker.models.{opp_policy}").AgentPolicy
    opponent = opp_policy_obj(mode="inference", model=opp_model)

    # Set up env
    env = LuxEnvironment(configs, player, opponent)

    # Set up the model
    if model_path:
        model = PPO.load(
            path_join(database, model_path, "model.zip"), 
            env=env, 
            tensorboard_log=path_join(database, new_model_path, tensorboard_log), 
            learning_rate=learning_rate,
            gamma=gamma,
            gae_lambda=gae_lambda
        )
    else:
        model = PPO("MlpPolicy",
        env,
        verbose=1,
        tensorboard_log = path_join(database, new_model_path, tensorboard_log),
        learning_rate = learning_rate,
        gamma=gamma,
        gae_lambda = gae_lambda
    )


    # Save a checkpoint and 'replay_num' match replay files every 'replay_freq' steps
    callbacks = []
    player_replay = policy_obj(mode="inference", model=model)
    if opp_model:
        opponent_replay = opp_policy_obj(mode="inference", model=opp_model)
    else:
        opponent_replay = opp_policy_obj()
    if replay_freq:
        callbacks.append(
            SaveReplayAndModelCallback(
                                    save_freq=replay_freq,
                                    save_path=path_join(database, new_model_path, 'replays'),
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
    print(f"Training model {run_id} in {stage_name}...")
    model.learn(total_timesteps=step_count, callback=callbacks, tb_log_name=f'{run_id}')
    print("Done training model.")

    # Save model info
    if model_info:
        parents = model_info['parents']
        parents.append(model_info['run_id'])
        train_history = model_info['train_history']
    else:
        parents = []
        train_history = []
    
    last_train = datetime.now().strftime("%d/%m/%Y-%H:%M:%S")
    train_params = {'policy':model_policy, 'learning_rate':learning_rate, 'gamma':gamma, 'gae_lambda':gae_lambda}
    opp_info = {'policy':opp_policy, 'model':opp_path}
    train_history.append({'step_count':step_count, 'train_params':train_params, 'opponent':opp_info, 'date_time':last_train})

    info = {'run_id':run_id, 'last_train':last_train, 'parents':parents, 'train_params':train_params, 'train_history':train_history}
    with open(path_join(database, new_model_path, "info.json"), 'w') as f:
        f.write(str(info))

    # Save model zip file
    model.save(path_join(database, new_model_path, "model.zip"))

    print(f"Model files saved at {new_model_path}")

    return new_model_path
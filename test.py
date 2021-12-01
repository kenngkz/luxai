from sys import path
from constants import PROJECT_DIR
path.insert(1, PROJECT_DIR)

from base_utils import path_join
from utils import reset_jobs, reset_worker_data, edit_stage_benchmarks, clear_replays, gen_stage_tree_file, copy_database, reset_master_data
from worker.main import run_job
from master.manager.stage_manager import StageManager
from master.manager.job_manager import JobManager
from constants import POOL_DIR, MASTER_DATABASE_DIR, WORKER_DATABASE_DIR, MASTER_DATABASE_BACKUP_DIR, SECOND_DEF_TEMPLATE, THIRD_TEMPLATE
from worker.jobs.replay import replay

import os
import requests
import shutil
import random

master_host = "192.168.0.78"
master_port = 5001

short_param_template = [
    {
        'model_path':None,
        'model_policy':'agent1',
        'step_count':1000,
        'learning_rate':0.001,
        'gamma':0.995,
        'gae_lambda':0.95,
    },
    {
        'model_path':None,
        'model_policy':'agent1',
        'step_count':1000,
        'learning_rate':0.001,
        'gamma':0.995,
        'gae_lambda':0.95,
    }
]

# clear_replays(MASTER_DATABASE_BACKUP_DIR)

# copy_database(POOL_DIR, MASTER_DATABASE_BACKUP_DIR)

# stages_to_clear = ["stage_2", "stage_3", "stage_4", "stage_5"]
# for stage in stages_to_clear:
#     if os.path.exists(path_join(WORKER_DATABASE_DIR, stage)):
#         shutil.rmtree(path_join(WORKER_DATABASE_DIR, stage))
#     if os.path.exists(path_join(MASTER_DATABASE_DIR, stage)):
#         shutil.rmtree(path_join(MASTER_DATABASE_DIR, stage))
# reset_master_data()
# reset_jobs(path_join(MASTER_DATABASE_DIR, "stage_11"), step_count=100000, param_template=SECOND_DEF_TEMPLATE)
# reset_worker_data()
# run_job(master_host, master_port, 22, share=False, param_template=short_param_template, n_select=2, n_benchmarks=4)

# reset_master_data()
stage = "stage_14"
stage_path = path_join(MASTER_DATABASE_DIR, stage)
job_manager = JobManager()
job_manager.clear_queue()
stage_manager = StageManager(short_param_template)

with open(path_join(MASTER_DATABASE_DIR, "stage_13", "best_models.txt"), "r") as f:
    seed_models = eval(f.read())

model_ids = [dir_name for dir_name in os.listdir(stage_path) if os.path.isdir(path_join(stage_path, dir_name))]
model_ids = [model_id for model_id in model_ids if not "tensorboard" in model_id]

new_stage_params = []
available_model_ids = list(range(100000))
for model_id in seed_models:
    templates = [{arg_name:arg_val for arg_name, arg_val in template.items()} for template in THIRD_TEMPLATE]
    for template in templates:
        new_run_id = random.choice(available_model_ids)
        available_model_ids.remove(new_run_id)
        template["model_path"] = path_join(stage, model_id)
        template["run_id"] = str(new_run_id)
        new_stage_params.append(template)

stage_manager.add_stage(stage, "stage_13", new_stage_params)
stage_info = stage_manager.get_stage_info(stage)

with open(path_join(MASTER_DATABASE_DIR, "stage_13", "benchmark_models.txt"), "r") as f:
    benchmarks = eval(f.read())

old_stage_info = stage_manager.get_stage_info("stage_13")
old_stage_info["best_models"] = seed_models
old_stage_info["benchmark_models"] = benchmarks
stage_manager.update_stage("stage_13", old_stage_info)

# print(model_ids)
# print(stage_info["stage_params"])
# print(stage_info["eval_results"].keys())
# print(f"Missing model: {[model_id for model_id in model_ids if model_id not in stage_info['eval_results'].keys()]}")

# missing_model = '9038'
# job_args = {"model_path":path_join(stage, missing_model), "opp_paths":benchmarks, "save_replays":False}
# job_manager.add_queue("eval_model", job_args)

for model_id in model_ids:
    job_type = "eval_model"
    model_path = path_join(stage, model_id)
    job_args = {"model_path":model_path, "opp_paths":benchmarks, "save_replays":False}
    job_manager.add_queue(job_type, job_args)

# replacement_stage_params = stage_manager.gen_train_params("stage_11")
# stage_info["stage_params"] = replacement_stage_params
# print(stage_info["stage_params"])
# with open(path_join(MASTER_DATABASE_DIR, stage, "train_params.txt"), "w") as f:
#     f.write(str(replacement_stage_params))
stage_info["models"] = model_ids

stage_manager.update_stage(stage, stage_info)
stage_manager.save()

# model_path = "stage_0/2313"
# database = MASTER_DATABASE_DIR
# n_replays = 10

# replay_folder_path = path_join(database, model_path, "replays_vs_blank")
# if os.path.exists(replay_folder_path):
#     shutil.rmtree(replay_folder_path)

# replay(n_replays, model_path, None, database, player_names=["2313", "blank"])
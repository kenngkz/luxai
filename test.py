from sys import path
from constants import PROJECT_DIR, POOL_DIR, MASTER_DATABASE_DIR, WORKER_DATABASE_DIR
path.insert(1, PROJECT_DIR)

from base_utils import path_join
from utils import reset_jobs, reset_worker_data, edit_stage_benchmarks, clear_eval_replays, gen_stage_tree_file
from worker.main import run_job

import os
import requests
import shutil

master_host = "192.168.0.78"
master_port = 5001
url_prefix = f"http://{master_host}:{master_port}"

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

# clear_eval_replays()
stages_to_clear = ["stage_2", "stage_3", "stage_4", "stage_5"]
for stage in stages_to_clear:
    if os.path.exists(path_join(WORKER_DATABASE_DIR, stage)):
        shutil.rmtree(path_join(WORKER_DATABASE_DIR, stage))
    if os.path.exists(path_join(MASTER_DATABASE_DIR, stage)):
        shutil.rmtree(path_join(MASTER_DATABASE_DIR, stage))
gen_stage_tree_file()
reset_jobs(path_join(MASTER_DATABASE_DIR, "stage_1"), n_seed_models=2, step_count=1000, param_template=short_param_template)
reset_worker_data()
run_job(master_host, master_port, 22, short_param_template, 2, 4)
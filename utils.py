from master.info_storage import StageTree
from master.manager.job_manager import JobManager
from master.manager.stage_manager import StageManager
from worker.tracker import update_ongoing
from base_utils import path_join
from constants import (
    POOL_DIR, 
    MASTER_DATABASE_DIR, 
    MASTER_DATABASE_BACKUP_DIR,
    STAGETREE_FILE, 
    ID_RANGE, 
    DEFAULT_PARAM_TEMPLATE, 
    WORKER_DATABASE_DIR, 
    WORKER_CACHE_DIR, 
    JOBS_ONGOING_FILE
)

import os
import random
import shutil
import requests
import time

def gen_stage_tree_file(path=MASTER_DATABASE_DIR):
    stage_tree = StageTree.convert(path)
    stage_tree.save(path_join(MASTER_DATABASE_DIR, STAGETREE_FILE))

def gen_train_params(stage_path, n_seed_models=10, stage_prefix="stage", param_template=DEFAULT_PARAM_TEMPLATE, step_count=100000, random_seeds=True):
    '''
    Generate train jobs from the given stage. Can generate from random_seeds or from best_models
    '''
    job_manager = JobManager()
    stage_manager = StageManager(param_template)

    stage = os.path.basename(stage_path)
    stage_info = stage_manager.get_stage_info(stage)

    if random_seeds:
        model_ids = random.sample(stage_info["models"], n_seed_models)  # old stage model ids
    else:
        model_ids = stage_info["best_models"]
    new_stage_params = []
    available_model_ids = list(range(ID_RANGE))
    for model_id in model_ids:
        templates = [{arg_name:arg_val for arg_name, arg_val in template.items()} for template in param_template]
        for template in templates:
            new_run_id = random.choice(available_model_ids)
            available_model_ids.remove(new_run_id)
            template["model_path"] = stage + '/' + model_id
            template["run_id"] = str(new_run_id)
            template["step_count"] = step_count
            new_stage_params.append(template)
    # add new stage
    new_stage = f"{stage_prefix}_{int(stage[-1]) + 1}"
    stage_manager.add_stage(new_stage, stage, new_stage_params)
    # add new train jobs to queue
    for params in new_stage_params:
        job_manager.add_queue("train", params)
    stage_manager.save()

def clear_jobs():
    job_manager = JobManager()
    job_manager.clear_queue()

def reset_jobs(stage_path, n_seed_models=10, stage_prefix="stage", param_template=DEFAULT_PARAM_TEMPLATE, step_count=100000):
    clear_jobs()
    gen_train_params(stage_path, n_seed_models, stage_prefix, param_template, step_count)

def reset_worker_data():
    if os.path.exists(WORKER_DATABASE_DIR):
        shutil.rmtree(WORKER_DATABASE_DIR)
    update_ongoing("")

def edit_stage_benchmarks():
    stage = "stage_0"
    stage_manager = StageManager(DEFAULT_PARAM_TEMPLATE)
    old_benchmarks = stage_manager.get_benchmarks(f"stage_{int(stage[-1]) + 1}")
    ids = [os.path.basename(model_path) for model_path in old_benchmarks]
    new_benchmarks = [path_join(stage, model_id) for model_id in ids]
    stage_manager.update_stage(stage, {"benchmark_models":new_benchmarks})
    stage_manager.save()

def clear_replays(direc=MASTER_DATABASE_DIR):
    stages = [dir_name for dir_name in os.listdir(direc) if os.path.isdir(path_join(direc, dir_name))]
    stage_paths = [path_join(direc, stage_name) for stage_name in stages]
    for stage, stage_path in zip(stages, stage_paths):
        model_ids = [dir_name for dir_name in os.listdir(stage_path) if os.path.isdir(path_join(stage_path, dir_name))]
        model_ids = [dir_name for dir_name in model_ids if not "tensorboard" in dir_name]
        for model_id in model_ids:
            replay_dir_names = ["eval_replay", "eval_replays", "replays", "replay"]
            for dirname in replay_dir_names:
                replay_dir = path_join(direc, stage, model_id, dirname)
                if os.path.exists(replay_dir):
                    shutil.rmtree(replay_dir)

def connect_master(req_type, url, wait_time=3, **kwargs):
    for i in range(20):
        try:
            if req_type == "GET":
                response = requests.get(url, **kwargs)
            elif req_type == "POST":
                response = requests.post(url, **kwargs)
            return response
        except requests.exceptions.ConnectionError:
            print(f"Error connecting to master. Retrying in {wait_time}s...")
            time.sleep(wait_time)
    raise Exception(f"Unable to connect to master. req_type: {req_type}, url: {url}")

def reset_master_data():
    if os.path.exists(MASTER_DATABASE_BACKUP_DIR):
        if os.path.exists(MASTER_DATABASE_DIR):
            shutil.rmtree(MASTER_DATABASE_DIR)
        shutil.copytree(MASTER_DATABASE_BACKUP_DIR, MASTER_DATABASE_DIR)
    gen_stage_tree_file()
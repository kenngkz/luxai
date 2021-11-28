from constants import CUSTOM_MODULES_DIR
from sys import path
path.append(CUSTOM_MODULES_DIR)

from data_structures.quickselect import select_k
from worker.jobs.model_score import ModelScore

from utils import gen_train_params, clear_jobs
from base_utils import path_join
from constants import POOL_DIRECTORY, DEFAULT_PARAM_TEMPLATE
from master.api_func.job_complete import manage_completion
from master.api_func.job_assign import assign
from master.manager.stage_manager import StageManager
from worker.jobs.eval_stage import eval_stage

import os
import random

clear_jobs()
gen_train_params(path_join(POOL_DIRECTORY, "stage_0"))
for i in range(500):
    job = assign()
    print(i, job)
    if job["type"] == "eval_stage":
        eval_results = {}
        generated_results = [ModelScore(model_id, random.random()) for model_id in job["args"]["eval_results"]]
        _, i, sorted_scores = select_k(generated_results, 10)
        eval_results["best_models"] = {model_score.id:model_score.score for model_score in sorted_scores[:i+1]}
        eval_results["benchmark_models"] = {model_score.id:model_score.score for model_score in sorted_scores[:21]}
    else:
        eval_results = None
    manage_completion(job, eval_results)

# stage_manager = StageManager(DEFAULT_PARAM_TEMPLATE)

# stage_info = stage_manager.get_stage_info("stage_1")
# print(stage_info["models"])
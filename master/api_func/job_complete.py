from constants import DEFAULT_PARAM_TEMPLATE, N_SELECT, N_BENCHMARKS, ID_RANGE, N_REPLAYS
from master.manager.job_manager import JobManager
from master.manager.stage_manager import StageManager
from base_utils import path_join

import os
import random

def manage_completion(completed_job, results=None, param_template=DEFAULT_PARAM_TEMPLATE, n_select=N_SELECT, n_benchmarks=N_BENCHMARKS, n_replays=N_REPLAYS):
    '''
    Manages the job and stage info in response to job completion. 

    'results' arg can be either eval_results for 1 model or {'best_models':[], 'benchmark_models':[]} for type eval and eval_stage respectively.
    '''
    job_print = {"type": completed_job["type"], "args":{key:val for key, val in completed_job["args"].items() if key != "eval_results"}, "info":completed_job["info"]}
    print(f"Job Completion Report: {job_print}")
    # Initialize JobManager and StageManager
    job_manager = JobManager()
    stage_manager = StageManager(param_template)

    model_path = completed_job["args"]["model_path"]
    stage = os.path.dirname(model_path)
    stage_info = stage_manager.get_stage_info(stage)

    # If job_type is train, add a eval job to queue
    if completed_job['type'] == 'train':
        stage = os.path.dirname(results) # update to new stage, results passed is the new model path
        stage_info = stage_manager.get_stage_info(stage)
        stage_info["models"] += [completed_job["args"]["run_id"]]
        stage_manager.update_stage(stage, stage_info)
        new_model_path = f'{stage}/{completed_job["args"]["run_id"]}'
        benchmark_models = stage_manager.get_benchmarks(stage)
        new_job = job_manager.add_queue("eval_model", {"model_path":new_model_path, "opp_paths":benchmark_models, "save_replays":False})
        job_print = {"type": new_job["type"], "args":{key:val for key, val in new_job["args"].items() if key != "eval_results"}, "info":new_job["info"]}
        print(f"Job Queued: {job_print}")
    # If job_type is eval, check if stage is complete
    elif completed_job['type'] == 'eval_model':
        # update the stage eval_results
        model_id = os.path.basename(model_path)
        stage_info["eval_results"][model_id] = results
        stage_manager.update_stage(stage, stage_info)
        # if all models have been evaluated
        print(f"Stage evaluation progress: {len(stage_info['eval_results'])} / {len(stage_info['stage_params'])}")
        if len(stage_info["eval_results"]) >= len(stage_info["stage_params"]):
            # add a eval_stage job to queue
            new_job = job_manager.add_queue("eval_stage", {"model_path":model_path, "eval_results":stage_info["eval_results"], "n_select":n_select, "n_benchmarks":n_benchmarks}, {"stage_name":stage})
            job_print = {"type": new_job["type"], "args":{key:val for key, val in new_job["args"].items() if key != "eval_results"}, "info":new_job["info"]}
            print(f"Job Queued: {job_print}")
    # If job_type is eval_stage, assign new train jobs
    elif completed_job["type"] == "eval_stage":
        stage_manager.update_best_bench(stage, results["best_models"], results["benchmark_models"])
        # generate replays for the best models
        for model_id in results["best_models"]:
            model_path = path_join(stage, model_id)
            opp_pool = stage_manager.get_benchmarks(stage)
            for _ in range(n_replays):
                opp_path = random.choice(opp_pool)
                opp_pool.remove(opp_path)
                new_job = job_manager.add_queue("replay", {"n_replays":1, "model_path":model_path, "opp_path":opp_path, "player_names":[model_id, os.path.basename(opp_path)], "autoname":False})
                job_print = {"type": new_job["type"], "args":{key:val for key, val in new_job["args"].items() if key != "eval_results"}, "info":new_job["info"]}
                print(f"Job Queued: {job_print}")
            # model_path = path_join(stage, model_id)
            # new_job = job_manager.add_queue("replay", {"n_replays":n_replays, "model_path":model_path})  # can add opp_path in the future
            # job_print = {"type": new_job["type"], "args":{key:val for key, val in new_job["args"].items() if key != "eval_results"}, "info":new_job["info"]}
            # print(f"Job Queued: {job_print}")
        # generate new train params for the new stage
        model_ids = results["best_models"]  # old stage model ids
        print(f"Best models in {stage}: {model_ids}")
        new_stage = '_'.join(stage.split('_')[:-1]) + '_' + str(int(stage.split('_')[-1]) + 1)
        new_stage_params = []
        available_model_ids = list(range(ID_RANGE))
        print(f"New stage: {new_stage}. Generating stage params...", end=" ")
        for model_id in model_ids:
            templates = [{arg_name:arg_val for arg_name, arg_val in template.items()} for template in param_template]
            for template in templates:
                new_run_id = random.choice(available_model_ids)
                available_model_ids.remove(new_run_id)
                template["model_path"] = path_join(stage, model_id)
                template["run_id"] = str(new_run_id)
                new_stage_params.append(template)
        print("Done.")
        # add new stage
        stage_manager.add_stage(new_stage, stage, new_stage_params)
        # add new train jobs to queue
        new_jobs = []
        for params in new_stage_params:
            new_job = job_manager.add_queue("train", params)
            new_jobs.append(new_job)
            job_print = {"type": new_job["type"], "args":{key:val for key, val in new_job["args"].items() if key != "eval_results"}, "info":new_job["info"]}
            print(f"Job Queued: {job_print}")
    else: # if job_type is replay, pass
        pass
    stage_manager.save()
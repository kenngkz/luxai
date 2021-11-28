from constants import DEFAULT_PARAM_TEMPLATE, N_SELECT, N_BENCHMARKS, ID_RANGE
from master.manager.job_manager import JobManager
from master.manager.stage_manager import StageManager

import os
import random

def manage_completion(completed_job, results=None, param_template=DEFAULT_PARAM_TEMPLATE, n_select=N_SELECT, n_benchmarks=N_BENCHMARKS):
    '''
    Manages the job and stage info in response to job completion. 

    'results' arg can be either eval_results for 1 model or {'best_models':[], 'benchmark_models':[]} for type eval and eval_stage respectively.
    '''
    print(f"Job Completion Report: {completed_job}")
    # Initialize JobManager and StageManager
    job_manager = JobManager()
    stage_manager = StageManager(param_template)

    model_path = completed_job["args"]["model_path"]
    stage = os.path.dirname(model_path)
    stage_info = stage_manager.get_stage_info(stage)

    # If job_type is train, add a eval job to queue
    if completed_job['type'] == 'train':
        stage = stage[:-1] + str(int(stage[-1]) + 1)  # update the new stage
        stage_info = stage_manager.get_stage_info(stage)
        stage_info["models"] += [completed_job["args"]["run_id"]]
        stage_manager.update_stage(stage, stage_info)
        new_model_path = f'{stage}/{completed_job["args"]["run_id"]}'
        benchmark_models = stage_manager.get_benchmarks(stage)
        new_job = job_manager.add_queue("eval_model", {"model_path":new_model_path, "opp_paths":benchmark_models, "save_replays":False})
        print(f"Job Queued. Type: {new_job['type']}. Number of jobs: 1")
    # If job_type is eval, check if stage is complete
    elif completed_job['type'] == 'eval_model':
        # update the stage eval_results
        model_id = os.path.basename(model_path)
        stage_info["eval_results"][model_id] = results
        stage_manager.update_stage(stage, stage_info)
        # if all models have been trained
        print(f"Stage evaluation progress: {len(stage_info['models'])} / {len(stage_info['stage_params'])}")
        if len(stage_info["models"]) >= len(stage_info["stage_params"]):
            # add a eval_stage job to queue
            # print(f"stage_info['eval_results']: {stage_info['eval_results']}")
            new_job = job_manager.add_queue("eval_stage", {"model_path":model_path, "eval_results":stage_info["eval_results"], "n_select":n_select, "n_benchmarks":n_benchmarks}, {"stage_name":stage})
            print(f"Job Queued. Type: {new_job['type']}. Number of jobs: 1")
    # If job_type is eval_stage, assign new train jobs
    else:
        stage_info["best_models"] = results["best_models"]
        stage_info["benchmark_models"] = results["benchmark_models"]
        stage_manager.update_stage(stage, stage_info)
        # generate new train params for the new stage
        model_ids = stage_info["best_models"]  # old stage model ids
        new_stage = stage[:-1] + str(int(stage[-1]) + 1)
        new_stage_params = []
        available_model_ids = list(range(ID_RANGE))
        for model_id in model_ids:
            templates = [{arg_name:arg_val for arg_name, arg_val in template.items()} for template in param_template]
            for template in templates:
                new_run_id = random.choice(available_model_ids)
                available_model_ids.remove(new_run_id)
                template["model_path"] = stage + '/' + model_id
                template["run_id"] = str(new_run_id)
                new_stage_params.append(template)
        # add new stage
        stage_manager.add_stage(new_stage, stage, new_stage_params)
        # add new train jobs to queue
        new_jobs = []
        for params in new_stage_params:
            new_job = job_manager.add_queue("train", params)
            new_jobs.append(new_job)
        print(f"Jobs Queued: Type: {new_job['type']}. Number of jobs: {len(new_jobs)}")

    # save stage info into file
    stage_manager.save()
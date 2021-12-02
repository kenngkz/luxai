from constants import WORKER_DATABASE_DIR, DEFAULT_PARAM_TEMPLATE, MASTER_DATABASE_DIR
from worker.tracker import read_ongoing, update_ongoing
from worker.jobs.train import train
from worker.jobs.eval_model import eval_model, random_result
from worker.jobs.eval_stage import eval_stage
from worker.jobs.replay import replay
from utils import connect_master
from base_utils import path_join

import os
import requests
import time

master_host = "192.168.0.7"
master_port = 5001
url_prefix = f"http://{master_host}:{master_port}"

job_funcs = {"train":train, "eval_model":eval_model, "eval_stage":eval_stage, "replay":replay}

def run_job(master_host, master_port, max_jobs, share=False, param_template=DEFAULT_PARAM_TEMPLATE, n_select=8, n_benchmarks=20):
    if share:
        database = MASTER_DATABASE_DIR
    else:
        database = WORKER_DATABASE_DIR
    for i in range(max_jobs):
        # get the job to do
        job = read_ongoing()
        if job is None:  # get request from master
            job = get_job(url_prefix, database)
            if job == None:
                print(f"No job available. Retrying in 30s...")
                time.sleep(30)
                continue
        else:
            print("-" * 100)
            job_print = {"type": job["type"], "args":{key:val for key, val in job["args"].items() if key != "eval_results"}, "info":job["info"]}
            print(f"Job Started: {job_print}")
        # get function to run
        func = job_funcs[job["type"]]
        report = func(**job["args"], database=database)

        completion_url = f"{url_prefix}/job/report"
        if job["type"] == "train":
            # Upload the new model files
            new_model_path = report
            if not share:
                local_new_model_path = path_join(database, new_model_path)
                print(f"Uploading {new_model_path}...", end=' ')
                files = ["model.zip", "info.json"]
                for filename in files:
                    with open(path_join(local_new_model_path, filename), 'rb') as f:
                        file = f.read()
                    response = connect_master("POST", f"{url_prefix}/upload", files={"file":file}, params={"path":path_join(new_model_path, filename)})
                    if "Upload complete." not in response.text:
                        raise Exception(f"Upload failed. File: {path_join(local_new_model_path, filename)}")
                # upload tensorboard logs
                tensorboard_log_folder = "tensorboard_log"
                logs = os.listdir(path_join(local_new_model_path, tensorboard_log_folder))
                latest_log = sorted(logs, key=lambda x: int(x.split("_")[-1]), reverse=True)[0]
                log_files = os.listdir(path_join(local_new_model_path, tensorboard_log_folder, latest_log))
                for logfile in log_files:
                    with open(path_join(local_new_model_path, tensorboard_log_folder, latest_log, logfile), 'rb') as f:
                        file = f.read()
                    response = connect_master("POST", f"{url_prefix}/upload", files={"file":file}, params={"path":path_join(new_model_path, tensorboard_log_folder, latest_log, logfile)})
                    if "Upload complete." not in response.text:
                        raise Exception(f"Upload failed. File: {path_join(local_new_model_path, filename)}")
                print(f"Done.")
            # Report completion of job to master
            response = connect_master("POST", completion_url, params={"completed_job":str(job), "results":new_model_path})
        
        elif job["type"] == "eval_model":
            results = report
            response = connect_master("POST", completion_url, params={"completed_job":str(job), "results":str(results), "n_select":n_select, "n_benchmarks":n_benchmarks})
        
        elif job["type"] == "eval_stage":
            best_models, benchmark_models = report
            response = connect_master("POST", completion_url, params={"completed_job":str(job), "results":str({"best_models":best_models, "benchmark_models":benchmark_models}), "param_template":str(param_template)})
        
        else: # replay job completed
            replay_folder = report
            model_path = job["args"]["model_path"]
            replays = os.listdir(path_join(database, model_path, replay_folder))
            print(f"Uploading {len(replays)} replays for {model_path} in {replay_folder}...", end=' ')
            for filename in replays:
                with open(path_join(database, model_path, replay_folder, filename), 'rb') as f:
                    file = f.read()
                response = connect_master("POST", f"{url_prefix}/upload", files={"file":file}, params={"path":path_join(model_path, replay_folder, filename)})
                if "Upload complete." not in response.text:
                        raise Exception(f"Upload failed. File: {path_join(database, model_path, replay_folder, filename)}")
            print("Done.")
            response = connect_master("POST", completion_url, params={"completed_job":str(job)})

        update_ongoing("")
        job_print = {"type": job["type"], "args":{key:val for key, val in job["args"].items() if key != "eval_results"}, "info":job["info"]}
        print(f"Job Completed: {job_print}")

def download_url(url, save_path, chunk_size=128):
    '''
    Downloads file from url and saves it in save_path.
    '''
    r = connect_master("GET", url, stream=True)
    with open(save_path, 'wb') as fd:
        for chunk in r.iter_content(chunk_size=chunk_size):
            fd.write(chunk)

def check_model_file(model_path, url_prefix, database):
    '''
    Checks if the model file exists, if it doesnt, download from master.
    '''
    url_folder = f"{url_prefix}/get/{model_path}"
    if not os.path.exists(path_join(database, model_path)):
        print(f"Downloading {model_path}...", end=' ')
        os.makedirs(path_join(database, model_path))
        for filename in ["info.json", "model.zip"]:
            url_complete = f"{url_folder}/{filename}"
            download_url(url_complete, path_join(database, model_path, filename))
        print(f"Done.")

def get_job(url_prefix, database):
    '''
    Gets job assignment from master and downloads all required files in order to perform job.
    '''
    response = connect_master("GET", f"{url_prefix}/job/request")
    job = eval(response.text)
    if job == None:
        return None
    print("-"*100)
    job_print = {"type": job["type"], "args":{key:val for key, val in job["args"].items() if key != "eval_results"}, "info":job["info"]}
    print(f"Job Started: {job_print}")
    update_ongoing(job)
    if "model_path" in job["args"]:
        check_model_file(job["args"]["model_path"], url_prefix, database=database)
    if "opp_paths" in job["args"]:
        for model_path in job["args"]["opp_paths"]:
            check_model_file(model_path, url_prefix, database=database)
    if "opp_path" in job["args"]:
        if job["args"]["opp_path"] != "self":
            check_model_file(job["args"]["opp_path"], url_prefix, database=database)
    return job
from constants import WORKER_DATABASE_DIR, DEFAULT_PARAM_TEMPLATE
from worker.tracker import read_ongoing, update_ongoing
from worker.jobs.train import train
from worker.jobs.eval_model import eval_model, random_result
from worker.jobs.eval_stage import eval_stage
from utils import connect_master
from base_utils import path_join

import os
import requests
import time

master_host = "192.168.0.7"
master_port = 5001

job_funcs = {"train":train, "eval_model":random_result, "eval_stage":eval_stage}

def run_job(master_host, master_port, max_jobs, template=DEFAULT_PARAM_TEMPLATE, n_select=8, n_benchmarks=20):
    for i in range(max_jobs):
        url_prefix = f"http://{master_host}:{master_port}"
        # get the job to do
        try:
            job = read_ongoing()
            if job is None:  # get request from master
                job = get_job(url_prefix)
                if job == None:
                    print(f"No job available. Retrying in 3s...")
                    time.sleep(3)
                    continue
            else:
                print("-" * 100)
                print(f"Job Started: {job}")
        except requests.exceptions.ConnectionError:
            print(f"Error connecting to master. Retrying in 3s...")
            time.sleep(3)
            continue
        # get function to run
        func = job_funcs[job["type"]]
        report = func(**job["args"])

        completion_url = f"{url_prefix}/job/report"
        if job["type"] == "train":
            # Report completion of job to master
            new_model_path = report
            response = connect_master("POST", completion_url, params={"completed_job":str(job), "results":new_model_path})
            # response = requests.post(completion_url, params={"completed_job":str(job), "results":new_model_path})
            # Upload the new model files
            local_new_model_path = path_join(WORKER_DATABASE_DIR, new_model_path)
            print(f"Uploading {new_model_path}...", end=' ')
            files = ["model.zip", "info.json"]
            for filename in files:
                with open(path_join(local_new_model_path, filename), 'rb') as f:
                    file = f.read()
                response = connect_master("POST", f"{url_prefix}/upload2", files={"file":file}, params={"path":path_join(new_model_path, filename)})
                # response = requests.post(f"{url_prefix}/upload2", files={"file":file}, params={"path":path_join(new_model_path, filename)})
                if "Upload complete." not in response.text:
                    raise Exception(f"Upload failed. File: {path_join(local_new_model_path, filename)}")
            print(f"Done.")
        elif job["type"] == "eval_model":
            results = report
            response = connect_master("POST", completion_url, params={"completed_job":str(job), "results":str(results), "n_select":n_select, "n_benchmarks":n_benchmarks})
            # response = requests.post(completion_url, params={"completed_job":str(job), "results":str(results), "n_select":n_select})
        else:
            best_models, benchmark_models = report
            response = connect_master("POST", completion_url, params={"completed_job":str(job), "results":str({"best_models":best_models, "benchmark_models":benchmark_models}), "param_template":str(template)})
            # response = requests.post(completion_url, params={"completed_job":str(job), "results":str({"best_models":best_models, "benchmark_models":benchmark_models})})
        update_ongoing("")
        job_alert = f"Job Completed: {job['type']}. Stage: {os.path.dirname(job['args']['model_path'])}" if job["type"] == "eval_stage" else f"Job Complete: {job}"
        print(job_alert)

def upload_url(url, file_path, chunk_size=128):
    r = requests.post(url, files=file_path)

def download_url(url, save_path, chunk_size=128):
    '''
    Downloads file from url and saves it in save_path.
    '''
    r = connect_master("GET", url, stream=True)
    # r = requests.get(url, stream=True)
    with open(save_path, 'wb') as fd:
        for chunk in r.iter_content(chunk_size=chunk_size):
            fd.write(chunk)

def check_model_file(model_path, url_prefix):
    '''
    Checks if the model file exists, if it doesnt, download from master.
    '''
    url_folder = f"{url_prefix}/get/{model_path}"
    if not os.path.exists(path_join(WORKER_DATABASE_DIR, model_path)):
        print(f"Downloading {model_path}...", end=' ')
        os.makedirs(path_join(WORKER_DATABASE_DIR, model_path))
        for filename in ["info.json", "model.zip"]:
            url_complete = f"{url_folder}/{filename}"
            download_url(url_complete, path_join(WORKER_DATABASE_DIR, model_path, filename))
        print(f"Done.")

def get_job(url_prefix):
    '''
    Gets job assignment from master and downloads all required files in order to perform job.
    '''
    response = connect_master("GET", f"{url_prefix}/job/request")
    if response == None:
        return None
    # response = requests.get(f"{url_prefix}/job/request", verify=False)
    job = eval(response.text)
    print("-"*100)
    job_alert = f"Job Started: {job['type']}. Stage: {os.path.dirname(job['args']['model_path'])}" if job["type"] == "eval_stage" else f"Job Started: {job}"
    print(job_alert)
    update_ongoing(job)
    if "model_path" in job["args"]:
        check_model_file(job["args"]["model_path"], url_prefix)
    if "opp_paths" in job["args"]:
        for model_path in job["args"]["opp_paths"]:
            check_model_file(model_path, url_prefix)
    if "opp_path" in job["args"]:
        check_model_file(job["args"]["opp_path"], url_prefix)
    return job
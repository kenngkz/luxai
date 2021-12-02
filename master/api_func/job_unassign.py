from master.manager.job_manager import JobManager

def unassign(job):
    job_manager = JobManager()
    new_job = job_manager.add_queue(job["type"], job["args"], job["info"])
    job_print = {"type": new_job["type"], "args":{key:val for key, val in new_job["args"].items() if key != "eval_results"}, "info":new_job["info"]}
    print(f"Job Unassigned: {job_print}")
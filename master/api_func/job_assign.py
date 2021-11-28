from master.manager.job_manager import JobManager

def assign():
    job_manager = JobManager()
    return str(job_manager.assign_job())
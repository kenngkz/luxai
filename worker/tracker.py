from base_utils import path_join
from constants import WORKER_CACHE_DIR, JOBS_ONGOING_FILE

import os

ongoing_path = path_join(WORKER_CACHE_DIR, JOBS_ONGOING_FILE)

def read_ongoing(file=ongoing_path):
    if os.path.exists(file):
        with open(file, 'r') as f:
            ongoing = f.read()
            ongoing = eval(ongoing) if ongoing != "" else None
    else:
        ongoing = None
    return ongoing

def write_ongoing(ongoing, file=ongoing_path):
    with open(file, 'w') as f:
        f.write(str(ongoing))

def update_ongoing(job, file=ongoing_path):
    ongoing = job
    write_ongoing(ongoing, file=file)
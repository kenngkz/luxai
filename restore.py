'''
Restore database to database_backup.
'''

from sys import path
from constants import PROJECT_DIR
path.insert(1, PROJECT_DIR)

from base_utils import path_join
from utils import reset_master_data, reset_worker_data, reset_jobs
from constants import SECOND_DEF_TEMPLATE, MASTER_DATABASE_DIR

if __name__ == "__main__":
    stage = "stage_11"

    reset_master_data()
    reset_worker_data()
    reset_jobs(path_join(MASTER_DATABASE_DIR, stage), step_count=100000, param_template=SECOND_DEF_TEMPLATE)

    print("Restore complete.")
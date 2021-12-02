'''
Restore database to database_backup.
'''

from sys import path
from constants import PROJECT_DIR
path.insert(1, PROJECT_DIR)

from base_utils import path_join
from utils import reset_master_data, reset_worker_data, reset_jobs
from constants import SECOND_DEF_TEMPLATE, MASTER_DATABASE_DIR, THIRD_TEMPLATE

if __name__ == "__main__":
    stage = "stage_12"

    reset_master_data()
    reset_worker_data()
    reset_jobs(path_join(MASTER_DATABASE_DIR, stage), step_count=200000, param_template=THIRD_TEMPLATE, random_seeds=False)

    print("Restore complete.")
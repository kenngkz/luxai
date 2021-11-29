from sys import path
from constants import PROJECT_DIR
path.insert(1, PROJECT_DIR)

from worker.main import run_job
from utils import reset_worker_data, reset_master_data, reset_jobs
from constants import DEFAULT_PARAM_TEMPLATE, N_BENCHMARKS, N_SELECT, SECOND_DEF_TEMPLATE

import argparse

def get_terminal_args():
    """
    Get the command line arguments
    :return:(ArgumentParser) The command line arguments as an ArgumentParser
    """
    parser = argparse.ArgumentParser(description='Script to run worker.')
    parser.add_argument('--ip', help='IP address of machine that master is running on.', type=str, default='localhost')
    parser.add_argument('--port', help='Port number that master is running on.', type=int, default=5001)
    parser.add_argument('--max_jobs', help='Max number of jobs that the worker will run.', type=int, default=999999999)
    parser.add_argument('--share', help='If True, use model files from master database.', type=bool, default=False)
    parser.add_argument('--reset', help='If True, clear worker data and ongoing jobs.', type=bool, default=False)
    args = parser.parse_args()

    return args

def run_worker(args, template, n_select, n_benchmarks):
    if args.reset:
        reset_worker_data()
    run_job(args.ip, args.port, args.max_jobs, args.share, template, n_select, n_benchmarks)

if __name__ == "__main__":
    # Variables to edit
    template = SECOND_DEF_TEMPLATE
    n_select = 6
    n_benchmarks = N_BENCHMARKS

    # Reset master and worker data and job queue
    # short_param_template = [
    #     {
    #         'model_path':None,
    #         'model_policy':'agent1',
    #         'step_count':1000,
    #         'learning_rate':0.001,
    #         'gamma':0.995,
    #         'gae_lambda':0.95,
    #     },
    #     {
    #         'model_path':None,
    #         'model_policy':'agent1',
    #         'step_count':1000,
    #         'learning_rate':0.001,
    #         'gamma':0.995,
    #         'gae_lambda':0.95,
    #     }
    # ]
    # reset_master_data()
    # reset_jobs("stage_1", n_seed_models=4, param_template=short_param_template, step_count=1000)

    # run worker
    args = get_terminal_args()
    run_worker(args, template, n_select, n_benchmarks)
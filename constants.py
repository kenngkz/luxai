'''
Master Constants
'''

# Directories
PROJECT_DIR = "C:/Users/lenovo/Desktop/Coding/LuxAI/collab_project"
CUSTOM_MODULES_DIR = "C:/Users/lenovo/Desktop/Coding/CustomModules/"
MASTER_DATABASE_DIR = f"{PROJECT_DIR}/database/"
MASTER_CACHE_DIR = f"{PROJECT_DIR}/master/"
WORKER_DATABASE_DIR = f"{PROJECT_DIR}/worker/data/"
WORKER_CACHE_DIR = f"{PROJECT_DIR}/worker/"
POOL_DIR = "C:/Users/lenovo/Desktop/Coding/LuxAI/rl/modelcp/pool"

# File Names
STAGETREE_FILE = "stage_data.txt"
EVAL_RESULTS_FILE = "eval_results.txt"
BEST_MODELS_FILE = "best_models.txt"
BENCHMARK_MODELS_FILE = "benchmark_models.txt"
JOBS_QUEUE_FILE = "job_queue.txt"
JOBS_ONGOING_FILE = "job_ongoing.txt"

# API parameters
API_HOST = "0.0.0.0"
API_PORT = 5001

# Defualt Training Parameters
N_SELECT = 8  # number of best models to select
N_BENCHMARKS = 20  # number of benchmark models to select
ID_RANGE = 99999
DEFAULT_PARAM_TEMPLATE = [
    {
        'model_path':None,
        'model_policy':'agent1',
        'step_count':100000,
        'learning_rate':0.001,
        'gamma':0.995,
        'gae_lambda':0.95,
    },
    {
        'model_path':None,
        'model_policy':'agent1',
        'step_count':100000,
        'learning_rate':0.0004,
        'gamma':0.995,
        'gae_lambda':0.95,
    },
    {
        'model_path':None,
        'model_policy':'agent1',
        'step_count':100000,
        'learning_rate':0.0001,
        'gamma':0.995,
        'gae_lambda':0.95,
    },
    {
        'model_path':None,
        'model_policy':'agent1',
        'step_count':100000,
        'learning_rate':0.001,
        'gamma':0.995,
        'gae_lambda':0.99,
    },
    {
        'model_path':None,
        'model_policy':'agent2',
        'step_count':100000,
        'learning_rate':0.001,
        'gamma':0.995,
        'gae_lambda':0.95,
    },
    {
        'model_path':None,
        'model_policy':'agent2',
        'step_count':100000,
        'learning_rate':0.0004,
        'gamma':0.995,
        'gae_lambda':0.95,
    },
    {
        'model_path':None,
        'model_policy':'agent2',
        'step_count':100000,
        'learning_rate':0.0001,
        'gamma':0.995,
        'gae_lambda':0.95,
    },
    {
        'model_path':None,
        'model_policy':'agent3',
        'step_count':100000,
        'learning_rate':0.001,
        'gamma':0.995,
        'gae_lambda':0.95,
    },
    {
        'model_path':None,
        'model_policy':'agent3',
        'step_count':100000,
        'learning_rate':0.0004,
        'gamma':0.995,
        'gae_lambda':0.95,
    },
    {
        'model_path':None,
        'model_policy':'agent3',
        'step_count':100000,
        'learning_rate':0.0001,
        'gamma':0.995,
        'gae_lambda':0.95,
    }
]
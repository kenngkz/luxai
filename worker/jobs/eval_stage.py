from sys import path
path.append("C:\\Users\\lenovo\\Desktop\\Coding\\CustomModules")  # allows custom_modules to be imported

from data_structures.quickselect import select_k
from base_utils import path_join

import os

class ModelScore:
    '''
    Links model_id to score. Used in the quickselect algorithm (select_k()).
    '''
    def __init__(self, model_id, score):
        self.id = model_id
        self.score = score

    def __ge__(self, other):
        return self.score >= other.score

    def __le__(self, other):
        return self.score <= other.score

def eval_stage(model_path, eval_results, n_select, n_benchmarks, database=None):
    '''
    Return the top few models ids and a list of benchmark models ids from eval results.
    '''
    # Calculate average winrates for all models in eval_results
    winrates = {model_id:[eval_results[model_id][opp_id]["winrate"] for opp_id in eval_results[model_id]] for model_id in eval_results}
    avg_winrates = {model_id:sum(winrates[model_id]) for model_id in winrates}
    model_scores = [ModelScore(model_id, avg_winrate) for model_id, avg_winrate in avg_winrates.items()]

    # Select the top 'n_select' models
    _, i, sorted_scores = select_k(model_scores, n_select)
    best_models = sorted_scores[:i+1]
    best_models = [model_score.id for model_score in best_models]

    # Select the models to be used as benchmarks
    stage = os.path.dirname(model_path)
    _, i, sorted_scores = select_k(model_scores, n_benchmarks)
    benchmark_models = sorted_scores[:i+1]
    benchmark_models = [path_join(stage, model_score.id) for model_score in benchmark_models]

    print(f"Eval stage complete. Best models: {best_models}")

    return best_models, benchmark_models
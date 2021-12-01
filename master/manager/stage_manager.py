from constants import MASTER_DATABASE_DIR, STAGETREE_FILE
from master.info_storage import StageTree
from base_utils import path_join

import os

class StageManager:
    '''
    Responsible for managing information regarding stages.

    1. generating new stage params (train params for all new models)
    2. managing the StageTree (updating and saving)
    3. retrieve benchmark model information  (when queried by JobManager)
    4. check if stage is complete  (when queried by JobManager)
    '''
    

    def __init__(self, param_template, tree_path=path_join(MASTER_DATABASE_DIR, STAGETREE_FILE), convert_dir=MASTER_DATABASE_DIR):
        self.param_template = param_template
        self.tree_path = tree_path
        if os.path.exists(tree_path):
            self.tree = StageTree.load(tree_path)
        else:
            self.tree = StageTree.convert(convert_dir)
    
    def add_stage(self, stage_name, parent, stage_params):
        self.tree.add_new_stage(stage_name, parent, stage_params)
        if not os.path.exists(path_join(MASTER_DATABASE_DIR, stage_name)):
            os.mkdir(path_join(MASTER_DATABASE_DIR, stage_name))
        with open(path_join(MASTER_DATABASE_DIR, stage_name, "train_params.txt"), "w") as f:
            f.write(str(stage_params))

    def update_best_bench(self, stage, best_models, benchmark_models):
        stage_info = self.tree.get_stage_info(stage)
        stage_info["best_models"] = best_models
        stage_info["benchmark_models"] = benchmark_models
        self.update_stage(stage, stage_info)
        with open(path_join(MASTER_DATABASE_DIR, stage, "best_models.txt"), "w") as f:
            f.write(str(best_models))
        with open(path_join(MASTER_DATABASE_DIR, stage, "benchmark_models.txt"), "w") as f:
            f.write(str(benchmark_models))

    def update_stage(self, stage_name, new_stage_info:dict):
        stage_node = self.tree.get_stage(stage_name)
        for name, val in new_stage_info.items():
            setattr(stage_node, name, val)

    def get_stage_info(self, stage_name):
        if stage_name in self.tree.nodes:
            return self.tree.get_stage_info(stage_name)
        else:
            print(f"{stage_name} not found in stage_tree: Enter one of {self.tree.list_stages()}")
            return None
        
    def get_benchmarks(self, stage_name):
        '''
        Get benchmarks from the parent stage.
        '''
        prev_stage = self.tree.get_parent(stage_name).name
        if prev_stage == None:
            raise Exception(f"Parent not found for {stage_name}")
        prev_stage = self.tree.get_stage(prev_stage)
        # return [prev_stage.name + '/' + model_id for model_id in prev_stage.benchmark_models]
        return prev_stage.benchmark_models

    def gen_train_params(self, stage_name):
        '''
        Generate new train params based on the best models in the stage provided and the param template.
        '''
        seed_models = self.tree.get_stage(stage_name).best_models
        prev_stage = self.tree.get_stage(stage_name).name
        new_params = []
        for model_id in seed_models:
            for template in self.param_template:
                new_train_param = template.copy()
                new_train_param["model_path"] = prev_stage + '/' + model_id
                new_params.append(new_train_param)
        return new_params

    def save(self):
        self.tree.save(self.tree_path)
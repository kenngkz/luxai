from base_utils import path_join

import os

class StageNode:
    '''
    Represents information about a stage. Used in StageTree defined below.
    '''
    def __init__(self, name, parent, stage_params):
        self.name = name
        self.parent = parent
        self.stage_params = stage_params  # training params for this whole stage
        self.models = []
        self.eval_results = {}
        self.best_models = []
        self.benchmark_models = []

    def _add_result(self, model_id, result):
        self.eval_results[model_id] = result

    @classmethod
    def _load(cls, name, parent, stage_params, models, eval_results, best_models, benchmark_models):
        stage_node = cls(name, parent, stage_params)
        stage_node.models = models
        stage_node.eval_results = eval_results
        stage_node.best_models = best_models
        stage_node.benchmark_models = benchmark_models
        return stage_node

    def _get_info(self):
        info = {
            "name":self.name,
            "parent":self.parent,
            "models":self.models,
            "stage_params":self.stage_params,
            "eval_results":self.eval_results,
            "best_models":self.best_models,
            "benchmark_models":self.benchmark_models,
        }
        return info

    def __eq__(self, other):
        return self.name == other.name


class StageTree:
    '''
    Holds information about stages.

    Info:
        - name
        - parent stage
        - stage train params: training params for every model in the stage
        - eval results: n_wins, n_games and winrates for every evaluation round for every model
        - best_models: ids
        - benchmark_models: ids
        - completed: bool -> whether the stage has been filled with models completely
    '''
    def __init__(self):
        self.nodes = {}

    def list_stages(self):
        return list(self.nodes.keys())
    
    def add_stage(self, node_obj):
        '''
        Add StageNode obj.
        '''
        self.nodes[node_obj.name] = node_obj

    def add_new_stage(self, name, parent, stage_params):
        '''
        Creates and adds a new StageNode obj.
        '''
        self.nodes[name] = StageNode(name, parent, stage_params)

    def get_stage(self, name):
        return self.nodes[name]

    def get_stage_info(self, name):
        return self.get_stage(name)._get_info()

    def save(self, path):
        tree_info = []
        for node_name, node in self.nodes.items():
            node_info = self.get_stage_info(node_name)
            tree_info.append(node_info)
        with open(path, 'w') as f:
            f.write(str(tree_info))
    
    @classmethod
    def load(cls, path):
        stage_tree = cls()
        with open(path, 'r') as f:
            tree_info = eval(f.read())
        for node_info in tree_info:
            stage_tree.add_stage(StageNode._load(**node_info))
        return stage_tree

    @classmethod
    def convert(cls, pool_directory):
        '''
        Convert stage and model files into a StageTree.
        '''
        stage_tree = cls()
        stages = [dir_name for dir_name in os.listdir(pool_directory) if os.path.isdir(path_join(pool_directory, dir_name))]
        stage_paths = [path_join(pool_directory, dir_name) for dir_name in stages]
        for stage, stage_path in zip(stages, stage_paths):
            # Skip tensorboard logs folder
            if "tensorboard" in stage:
                continue
            # Stage initialization params
            prev_stage = stage[:-1] + str(int(stage[-1]) - 1)
            if not prev_stage in stages:
                prev_stage = None
            if "train_params.txt" in os.listdir(stage_path):
                with open(path_join(stage_path, "train_params.txt"), 'r') as f:
                    stage_params = eval(f.read())
            else:
                stage_params = None
            stage_node = StageNode(stage, prev_stage, stage_params)
            # Stage details: models, eval_results, best_models, benchmark_models
            models = [dir_name for dir_name in os.listdir(stage_path) if os.path.isdir(path_join(stage_path, dir_name))]
            models = [model_id for model_id in models if not "tensorboard" in model_id]
            stage_node.models = models
            files = {"eval_results":"eval_results.txt", "best_models":"best_models.txt", "benchmark_models":"benchmark_models.txt"}
            for attr_name, file_name in files.items():
                if os.path.exists(path_join(stage_path, file_name)):
                    with open(path_join(stage_path, file_name), 'r') as f:
                        file = eval(f.read())
                        if attr_name == "benchmark_models":
                            file = [path_join(stage, os.path.basename(model_path)) for model_path in file]
                        setattr(stage_node, attr_name, file)
            stage_tree.add_stage(stage_node)
        return stage_tree

    def get_children(self, node_name):
        '''
        Returns names of children
        '''
        children = []
        for child_name in self.nodes.keys():
            if self.get_parent(node_name).name == node_name:
                children.append(child_name)
        return children

    def get_parent(self, node_name):
        try:
            return self.nodes[self.nodes[node_name].parent]
        except Exception as e:
            print(e)
            return None
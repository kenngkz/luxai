'''
Data Structure for model scoring and evaluation results.
'''
from sys import path
path.append("C:\\Users\\lenovo\\Desktop\\Coding\\CustomModules")  # allows custom_modules to be imported

from data_structures.bidirected_graph import Graph, Node
from math import tanh

class ScoreNode(Node):

    def __init__(self, node_id, score=0):
        super().__init__(node_id)
        self.score = 0

    def get_score(self):
        return sum(self.links.values())

    def n_links(self):
        return len(self.links)

class ScoreGraph(Graph):
    '''
    Graph to store evaluation results between models and compute the score
    '''
    NODE_OBJ = ScoreNode

    def missing_links(self, node_id) -> set:
        '''
        Get the potential links that have not yet been made. 
        Returns the set of nodes that are not connected to this node.
        '''
        return self.nodes.keys() - self.nodes[node_id].links.keys()

    def update(self, node_id, results):
        '''
        Add links to the graph given a dict of evaluation results.
        '''
        for opp_id, result in results.items():
            winrate = 2 * result["winrate"] - 1  # change the range from (0.0 - 1.0) -> (-1.0 - 1.0)
            self.nodes[node_id].add_link(opp_id, winrate)
            self.nodes[opp_id].add_link(node_id, -winrate)

    def get_score(self, node_id):
        return self.nodes[node_id].get_score()

    def n_links(self, node_id):
        return self.nodes[node_id].n_links()

    def get_scores(self, node_ids):
        scores = {}
        for node_id in node_ids:
            scores[node_id] = self.get_score(node_id)
        return scores

class ModelScore:
    '''
    Holds id and score for a model. To be used with quickselect algorithm.
    '''
    def __init__(self, id, score):
        self.id = id
        self.score = score

    def __ge__(self, other):
        return self.score >= other.score
    
    def __le__(self, other):
        return self.score <= other.score
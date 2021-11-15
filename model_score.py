from numpy import sign
from math import exp, tanh


class ModelScore:
    '''
    DataClass to store model_ids and model scores. To be used with the quickselect algorithm
    '''
    def __init__(self, id, score):
        self.id = id
        self.score = score

    def __ge__(self, other):
        try:
            return self.score >= other.score
        except TypeError as e:
            raise Exception(f"{e}\nself.id: {self.id}, self.score: {self.score}, other.id: {other.id}, other.score: {other.score}")

    def __le__(self, other):
        try:
            return self.score <= other.score
        except TypeError as e:
            raise Exception(f"{e}\nself.id: {self.id}, self.score: {self.score}, other.id: {other.id}, other.score: {other.score}")
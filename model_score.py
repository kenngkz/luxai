from numpy import sign
from math import exp, tanh


class ModelScore:
    '''
    DataClass to store model_ids and model scores. To be used with the quickselect algorithm
    '''
    def __init__(self, id, results:dict={}, score=None):
        self.id = id
        self.results = results
        self.score = score
        self.opp_pool = None
        self.remaining_opp = None
    
    def cal_winrate(self):
        # calculate the average winrate
        winrates = [summary["winrate"] for summary in self.results.values()]
        return sum(winrates)/len(winrates)

    def add_pool(self, opp_pool):
        self.opp_pool = opp_pool
        self.remaining_opp = list(opp_pool.keys())
        self.remaining_opp.remove(self.id)
        self.update(self.results)

    def cal_score(self, results):
        '''
        Function to calculate the score of a model. Higher winrate against opponents with higher relative score is rewarded more.
        If opponent does not have a existing score, the score of the opponent is estimated from their weighted winrate.
        '''
        assert self.opp_pool != None, "opp_pool must be specified for cal_score() to run. Add opp_pool using ModelScore.add_pool(opp_pool)"
        if self.score == None:
            self.score = 0
        for opp_id, summary in results.items():
            winrate = 2 * summary["winrate"] - 1  # convert winrate to be between -1.0 and 1.0
            self.score += winrate
            '''
            if self.opp_pool[opp_id].score == None:
                opp_winrate = self.opp_pool[opp_id].cal_winrate()  # calculate the average winrate of the opponent.
                opp_score = -10 + 10 * opp_winrate  
                    # if 50% winrate, score will be 100 (default score)
                    # range of estimated opp_score: -20 - 20
            else:
                opp_score = self.opp_pool[opp_id].score
            # Score Calculation Formula
            relative_score_diff = tanh((opp_score - self.score)/100)  # weighted relative difference in score
            try:
                score_weight = exp(sign(winrate) * relative_score_diff)  # score weight defines how relative score difference affects the actual change in score linearly to winrate
            except OverflowError:
                raise OverflowError(f"Overflow Error occured -> winrate: {winrate}, self_score: {self.score}, opp_score: {opp_score}, relative_score_diff: {relative_score_diff}")
            self.score += score_weight * (winrate - relative_score_diff) * 0.1
            '''

    def update(self, results, cal_score=True):
        self.skip_opps(results.keys())
        for id, summary in results.items():
            self.results[id] = summary  # replace previous summary if this model has played against that opponent before
            if cal_score:
                self.cal_score(results)

    def skip_opps(self, opps):
        for id in opps:
            try:
                self.remaining_opp.remove(id)
            except ValueError:
                pass
                # print(f"Opponent ID {id} not in the remaining opponent list of ModelScore with id {self.id}. Model {self.id} has likely played against {id} before.\n{self.results}")

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
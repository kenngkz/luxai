'''
Experimental agent policy
Rewards: 
    - incentivised to create units, ratio of 10 carts to 1 worker is optimal
'''
from worker.models.agent_policy import AgentPolicy as BaseAgentPolicy
from luxai2021.game.constants import Constants

class AgentPolicy(BaseAgentPolicy):

    def game_start(self, game):
        """
        This function is called at the start of each game. Use this to
        reset and initialize per game. Note that self.team may have
        been changed since last game. The game map has been created
        and starting units placed.

        Args:
            game ([type]): Game.
        """
        self.workers_last = 0
        self.carts_last = 0

    def get_reward(self, game, is_game_finished, is_new_turn, is_game_error):

        rewards = {}

        units = game.state["teamStates"][self.team]["units"].values()
        workers = 0
        carts = 0
        for unit in units:
            if unit.is_worker():
                workers += 1
            else:
                carts += 1

        rewards['worker_diff'] = (workers - self.workers_last) * 0.01
        self.workers_last = workers
        rewards['carts_diff'] = (carts - self.carts_last) * 0.1
        self.carts_last = carts

        total_rew = 0
        for name, value in rewards.items():
            total_rew += value

        return total_rew
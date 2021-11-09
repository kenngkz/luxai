'''
Agent with custom reward policy
Comparative citytile and unit creation/death and comparative alive at game end instead of only looking at own citytile_count
Rewards:
    - comparative unit creation/death = 0.05
    - comparative citytile creation/death = 0.01
    - collecting fuel = 0.0001
    - comparative citytile alive at game end = 1
'''

from agent_policy import AgentPolicy as BaseAgentPolicy

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
        self.units_last = 0
        self.units_last_opp = 0
        self.citytile_last = 0
        self.citytile_last_opp = 0
        self.fuel_collected_last = 0

    def get_reward(self, game, is_game_finished, is_new_turn, is_game_error):
        """
        Returns the reward function for this step of the game. Reward should be a
        delta increment to the reward, not the total current reward.
        """
        if is_game_error:
            # Game environment step failed, assign a game lost reward to not incentivise this
            print("Game failed due to error")
            return -1.0

        if not is_new_turn and not is_game_finished:
            # Only apply rewards at the start of each turn or at game end
            return 0

        # Get stats
        teams = [agent.team for agent in game.agents]
        teams.pop(self.team)
        opp_team = teams[0]
        unit_count = len(game.state["teamStates"][self.team]["units"])
        unit_count_opp = len(game.state["teamStates"][opp_team]["units"])

        city_count = 0
        city_count_opp = 0
        citytile_count = 0
        citytile_count_opp = 0
        for city in game.cities.values():
            if city.team == self.team:
                city_count += 1
            else:
                city_count_opp += 1

            for cell in city.city_cells:
                if city.team == self.team:
                    citytile_count += 1
                else:
                    citytile_count_opp += 1

        fuel_collected = game.stats["teamStats"][self.team]["fuelGenerated"]
        
        # Assigning rewards
        rewards = {}
        
        rewards['units'] = ((unit_count - self.units_last) - (unit_count_opp - self.units_last_opp)) * 0.05
        self.units_last, self.units_last_opp = unit_count, unit_count_opp

        rewards['citytile'] = ((citytile_count - self.citytile_last) - (citytile_count_opp - self.citytile_last_opp)) * 0.1
        self.citytiles_last, self.citytile_last_opp = citytile_count, citytile_count_opp

        rewards['fuel'] = (fuel_collected - self.fuel_collected_last) * 0.0001
        self.fuel_collected_last = fuel_collected

        if is_game_finished:
            self.is_last_turn = True
            rewards['citytile_end'] = citytile_count - citytile_count_opp

        # Compile rewards
        total_rew = 0
        for name, value in rewards.items():
            total_rew += value
        return total_rew

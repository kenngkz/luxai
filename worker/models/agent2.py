'''
Agent with custom reward policy.
Gym example + research rewards
Rewards:
    - unit creation/death = (unit_count - unit_count_last) * 0.05
    - citytile creation/death = (citytile_count - citytile_count_last) * 0.1
    - collecting fuel = (fuel_collected - fuel_collected_last) / 20000
    - research = (research_points - research_points_last) * 0.01
    - citytile alive at game end = if game_end: citytile_count
'''

from worker.models.agent_policy import AgentPolicy as BaseAgentPolicy

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
        self.city_tiles_last = 0
        self.fuel_collected_last = 0
        self.research_points_last = 0

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
        
        # Get some basic stats
        unit_count = len(game.state["teamStates"][self.team]["units"])

        city_count = 0
        city_count_opponent = 0
        city_tile_count = 0
        city_tile_count_opponent = 0
        for city in game.cities.values():
            if city.team == self.team:
                city_count += 1
            else:
                city_count_opponent += 1

            for cell in city.city_cells:
                if city.team == self.team:
                    city_tile_count += 1
                else:
                    city_tile_count_opponent += 1
        
        research_points = game.state["teamStates"][self.team]["researchPoints"]

        rewards = {}

        # Give a reward for unit creation/death. 0.05 reward per unit.
        rewards["rew/r_units"] = (unit_count - self.units_last) * 0.05
        self.units_last = unit_count

        # Give a reward for city creation/death. 0.1 reward per city.
        rewards["rew/r_city_tiles"] = (city_tile_count - self.city_tiles_last) * 0.1
        self.city_tiles_last = city_tile_count

        # Reward collecting fuel
        fuel_collected = game.stats["teamStats"][self.team]["fuelGenerated"]
        rewards["rew/r_fuel_collected"] = ( (fuel_collected - self.fuel_collected_last) / 20000 )
        self.fuel_collected_last = fuel_collected

        # Reward research up until 200 research points, 0.01 reward per point
        rewards["rew/r_research_points"] = (research_points - self.research_points_last) * 0.01 if research_points <= 200 else 0
        self.research_points_last = research_points

        # Give a reward of 1.0 per city tile alive at the end of the game
        rewards["rew/r_city_tiles_end"] = 0
        if is_game_finished:
            self.is_last_turn = True
            rewards["rew/r_city_tiles_end"] = city_tile_count

        # Compile rewards
        reward = 0
        for name, value in rewards.items():
            reward += value
        return reward
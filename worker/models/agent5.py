'''
Agent with custom reward policy
Tries to maintain a certain ratio of units to citytiles
Rewards:
    - unit creation/death = (unit_count - self.unit_last) * 0.05
    - city creation/death = (city_tile_count - self.city_tiles_last) * 0.1
    - collecting fuel = (fuel_collected - self.fuel_collected_last) * 0.00003
    - collecting coal (on top of fuel collection) = * 0.00005 per point
    - collecting uranium (on top of fuel collection) = * 0.00005 per point
    - research up to 200 points = (research_points - self.research_points_last) * 0.001
    - citytile alive at game end = if game_end: city_tile_count
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
        self.research_points_last = 0
        self.fuel_generated_last = 0
        self.coal_collected_last = 0
        self.uranium_collected_last = 0

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
        fuel_generated = game.stats["teamStats"][self.team]["fuelGenerated"]
        coal_collected = game.stats["teamStats"][self.team]["resourcesCollected"]["coal"]
        uranium_collected = game.stats["teamStats"][self.team]["resourcesCollected"]["uranium"]

        # Rewards
        rewards = {}
        
        # unit creation/death. 0.05 reward per unit.
        rewards["rew/r_units"] = (unit_count - self.units_last) * 0.08
        self.units_last = unit_count

        # city creation/death. 0.1 reward per city.
        rewards["rew/r_city_tiles"] = (city_tile_count - self.city_tiles_last) * 0.1
        self.city_tiles_last = city_tile_count

        # research
        rewards["rew/r_research_points"] = (research_points - self.research_points_last) * 0.001 if research_points <= 200 else 0
        self.research_points_last = research_points

        # collecting fuel from coal and uranium if research points is high enough
        if game.state["teamStates"][self.team]["researched"]["coal"]:
            rewards["coal_collected"] = ((coal_collected- self.coal_collected_last) * 0.00005)
            self.coal_collected_last = coal_collected
            
        if game.state["teamStates"][self.team]["researched"]["uranium"]:
            # reward collecting uranium
            rewards["uranium_collected"] = ((uranium_collected - self.uranium_collected_last) * 0.00005)
            self.uranium_collected_last = uranium_collected

        # generating fuel
        rewards["rew/r_fuel_collected"] = ((fuel_generated - self.fuel_generated_last) * 0.00003)
        self.fuel_collected_last = fuel_generated
        
        # Give a reward of 1.0 per city tile alive at the end of the game
        rewards["rew/r_city_tiles_end"] = 0
        if is_game_finished:
            self.is_last_turn = True
            rewards["rew/r_city_tiles_end"] = city_tile_count

            '''
            # Example of a game win/loss reward instead
            if game.get_winning_team() == self.team:
                rewards["rew/r_game_win"] = 100.0 # Win
            else:
                rewards["rew/r_game_win"] = -100.0 # Loss
            '''
        
        reward = 0
        for name, value in rewards.items():
            reward += value

        return reward
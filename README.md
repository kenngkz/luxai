# Lux AI
Kaggle Lux AI Competition submission

Plan: Use RL technique self-play with 2 neural networks: target network and policy network. Only the policy network will be submitted.

Offload path scheduling and organization to the pathplanner script. To allow the network to focus more on strategy and wider actions (move from here to there and not worry about how to get there)

##Potential Strategies:

Policy towards opponents
1. Ignore the opponents movements and focus on building your own base
2. Actively sabotage your opponents bases/resources
3. Focus on protecting your base

Expansion policy
1. Keep to one big city (lowers the resource upkeep for the city but far from other resources)
2. Multiple small cities close to resources (being close to resources allows less turns to be wasted on movement)
3. Aggressive expansion (expand as quickly as possible for the first 20 turns then use 10 turns to collect resources)
4. Aggressive tech (research as quickly as possible to reach higher levels of tech)
  - can pair with aggressively taking opponent's uranium/coal to quickly deplete their supply

Attacking theory:
If we can completely surround a worker away from resources, it cannot move anywhere to collect resources and it will die. 

"""
A factory class for creating routing models for the given type of environmen (cvrp or tsp).

"""
from copy import deepcopy

from src.models.cvrp_model.models import CVRPModel
from src.models.tsp_model.models import TSPModel


class RoutingModel:
    def __init__(self, model_params, env_params):
        self.model_params = model_params
        self.env_params = env_params

    def create_model(self, env_type):
        model_params = deepcopy(self.model_params)

        if 'cvrp' in env_type:
            model_params['action_size'] = self.env_params['num_depots'] + self.env_params['num_nodes']
            return CVRPModel(**self.model_params)
        elif 'tsp' in env_type:
            model_params['action_size'] = self.env_params['num_nodes']
            return TSPModel(**self.model_params)
        else:
            raise NotImplementedError
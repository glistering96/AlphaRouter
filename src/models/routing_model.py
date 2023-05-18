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
        if env_type == 'cvrp':
            return CVRPModel(**self.model_params)
        elif env_type == 'tsp':
            return TSPModel(**self.model_params)
        else:
            raise NotImplementedError

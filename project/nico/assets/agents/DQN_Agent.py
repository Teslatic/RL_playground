#!/usr/bin/python3
from assets.agents.RL_Agent import RL_Agent
from assets.helperFunctions.discretize import discretize
from assets.estimators.NN_estimator import NN_estimator


class _DQN_Agent(RL_Agent):
    """
    An model based RL agent that uses DQN as a model of the environment.

    Members:
        model (object): The model of the MB agent
    """

    def __init__(self, env, hyperparameters, model):
        self._init_env_parameters(env)
        self._unzip_hyperparameters(hyperparameters)
        self.action_space = discretize(self.action_low, self.action_high, hyperparameters["N_ACTION"])
        architecture = self._create_architecture(model)
        self.estimator = self._build_model(architecture)

    def _create_architecture(self, model):
        architecture = {
                        'D_IN': self.D_state,
                        'D_OUT': self.D_action,
                        'ACTION_SPACE': self.action_space,
                        'ACTIVATION': model["ACTIVATION"],
                        'LOSS': model["LOSS"],
                        'OPTIMIZER': model["OPTIMIZER"],
                        'LEARNING_RATE': model["LEARNING_RATE"],
                        'WEIGHT_FILE': model["WEIGHT_FILE"],
                        'WEIGHT_FILE_OUT': model["WEIGHT_FILE_OUT"]
                        }
        self.weights_file_out = architecture["WEIGHT_FILE_OUT"]
        return architecture

    def _build_model(self, architecture):
        """
        Builds a DQN model. Architecture should be defined by input parameters
        """
        estimator = NN_estimator(architecture)
        weight_file = architecture["WEIGHT_FILE"]
        if weight_file is None:
            pass
        else:
            estimator.load_weights(weight_file)
        return estimator

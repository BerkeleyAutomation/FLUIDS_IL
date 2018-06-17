import gym_urbandriving as fluids
from gym_urbandriving.agents import *
from copy import deepcopy
import numpy as np

class InitialState:

    def __init__(self,fluids_config,il_config):

        self.fluids_config = fluids_config
        self.il_config = il_config



    def sample_state(self):

        self.num_cars = np.random.randint(self.il_config['min_cars'],high=self.il_config['max_cars'])

        self.fluids_config['agents']['controlled_cars'] = self.num_cars

        env = fluids.UrbanDrivingEnv(self.fluids_config)

        return env

    def create_supervisors(self,supervisor_type):

        supervisors = []

        supervisor_class = {'VelocitySupervisor':VelocitySupervisor, 
                            'SteeringSupervisor':SteeringSupervisor,
                            "VelocityCSPSupervisor":VelocityCSPSupervisor,
                            'VelocityNeuralSupervisor':VelocityNeuralSupervisor}[supervisor_type]


        for i in range(self.num_cars):
            supervisor = supervisor_class(agent_num = i)
            supervisors.append(supervisor)

        return supervisors


    def sample_test_enviroment(self,f):

        self.fluids_config['agents']['background_cars'] = np.random.randint(0, 3)
        self.fluids_config['agents']['use_traffic_lights'] = np.random.random() < 0.5
        self.fluids_config['agents']['use_pedestrians'] = True
        self.fluids_config['agents']['number_of_pedestrians'] = np.random.randint(0, 6)
        self.fluids_config['agents']['bg_state_space_config']['noise'] = np.random.random()
        self.fluids_config['agents']['bg_state_space_config']['omission_prob'] = np.random.random()

        f.write(str(self.fluids_config['agents']['background_cars']) + ",")
        f.write(str(int(self.fluids_config['agents']['use_traffic_lights'])) + ",")
        f.write(str(int(self.fluids_config['agents']['use_pedestrians'])) + ",")
        f.write(str(self.fluids_config['agents']['number_of_pedestrians']) + ",")
        f.write(str(self.fluids_config['agents']['bg_state_space_config']['noise']) + ",")
        f.write(str(self.fluids_config['agents']['bg_state_space_config']['omission_prob']) + ",")
        


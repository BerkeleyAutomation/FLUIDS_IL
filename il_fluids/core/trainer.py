import gym
import gym_urbandriving as uds
import cProfile
import time
import numpy as np
import numpy.linalg as LA
import IPython
import gym_urbandriving as fluids
from gym_urbandriving.agents import VelocitySupervisor
from il_fluids.utils import DataLogger
from il_fluids.core import Learner
from il_fluids.distributions import InitialState
from il_fluids.data_protocols import BehaviorCloning
from il_fluids.core  import Plotter



class Trainer:

    def __init__(self,fluids_config,il_config):

        self.fluids_config = fluids_config
        self.il_config = il_config

        self.file_path =  il_config['file_path'] + il_config['experiment_name']

        self.d_logger = DataLogger(self.file_path)

        self.il_learn = Learner(il_config)

        self.initial_state = InitialState(fluids_config,il_config)

        self.protocol = BehaviorCloning()


    def set_data_protocol(self,protocol):
        self.protocol = protocol

    def set_tracker(self,tracker):
        self.Tracker = tracker


    def train_model(self):
        """
        Loads the data and trains the model
        """

        self.il_learn.load_data()
        self.il_learn.train_model()
        self.protocol.update()

    def get_stats(self):
        """
        Collect statistics of each iteration

        Returns
        --------
        stats: A dictonary with each measurment corresponding to some stastistic
        """

        stats = {}
        stats['train_sup'] = self.il_learn.get_train_error()
        stats['loss_sup']  = self.il_learn.get_test_error()
        stats['reward_sup'] = self.il_learn.get_supervisor_reward()

        loss_robot,reward = self.evaluate_policy()
        stats['loss_robot'] = loss_robot
        stats['reward_robot'] = reward



        return stats


    def train_robot(self):

        #Plotter class
        plotter = Plotter(self.file_path)
        stats = []


        for i in range(self.il_config['num_iters']):
            #Collect demonstrations 
            self.collect_supervisor_rollouts()
            #update model 
            self.train_model()
            #Evaluate Policy 
            stats.append(self.get_stats())
            #Save plots
            plotter.save_plots(stats)
       


    def evaluate_policy(self):
        """
        Use to measure the learned policy's performance

        Returns
        --------
        loss_robot: float, corresponding to the surrogate loss on the robot's distributions
        success_rate: float, corresponding to the reward of the robot
        """

        evaluations = self.collect_policy_rollouts()


        loss_robot = self.il_learn.get_cs(evaluations)
        reward = self.il_learn.get_robot_reward(evaluations)

        return loss_robot,reward



    def rollout_supervisor(self):
        """
        Rollout the supervior, by generating a plan through OMPL and then executing it. 

        Returns
        --------
        Returns the rollout and the goal states
        """

        rollout = []

        env = self.initial_state.sample_state()
        self.supervisors = self.initial_state.create_supervisors()
        

        state = env.get_current_state()

        if self.Tracker: 

            tracker = self.Tracker(state,self.il_config)

            if tracker.load_state:

                env.current_state = tracker.load_initial_state()


        curr_observations = env.get_initial_observations()
        # Simulation loop
        for i in range(self.il_config['time_horizon']):

            actions = []
            sup_actions = []
            if self.protocol.use_robot_action:
                robot_actions = self.il_learn.eval_policy(curr_observations)
            for i in range(len(self.supervisors)):


                supervisor = self.supervisors[i]
                sup_action = supervisor.eval_policy(state)


                if self.protocol.use_robot_action:
                    action = self.protocol.get_action(robot_action = robot_actions[i],supervisor_action = sup_action)
                else:
                    action = self.protocol.get_action(robot_action = None, supervisor_action = sup_action)

                actions.append(action)
                sup_actions.append(sup_action)

                if self.Tracker:
                    tracker.catch_bug(state,i,sup_action)

            sar = {}

            next_observations, reward, done, info_dict = env._step(actions)

            sar['state'] = curr_observations
            sar['reward'] = reward
            sar['action'] = sup_actions
            sar['noise_action'] = actions

            curr_observations = next_observations

            state = env.get_current_state()

            rollout.append(sar)
        
        return rollout


    def rollout_policy(self):
        """
        Rolls out the policy by executing the learned behavior

        Returns
        --------
        Returns the rollout and the goal states
        """

        rollout = []
        agents = []

        env = self.initial_state.sample_state() 
        self.supervisors = self.initial_state.create_supervisors()

        state = env.get_current_state()   

        observations = env.get_initial_observations()
       
        # Simulation loop
        for i in range(self.il_config['time_horizon']):

            actions = self.il_learn.eval_policy(observations)

            sup_actions = []
            for supervisor in self.supervisors:
                action = supervisor.eval_policy(state)
                sup_actions.append(action)

            sar = {}

            observations, reward, done, info_dict = env._step(actions)

            state = env.get_current_state()

            sar['state'] = observations
            sar['reward'] = reward
            sar['action'] = actions
            sar['sup_action'] = sup_actions

            rollout.append(sar)


        return rollout


    def collect_policy_rollouts(self):
        """
        Collects a number of policy rollouts and measures success

        Returns
        ----------
        The evaulations and the reported success rate
        """
     
        evaluations = []
        for i in range(self.il_config['num_policy_rollouts']):

            rollout  = self.rollout_policy()
 
            evaluations.append(rollout)
            
        return evaluations

    def collect_supervisor_rollouts(self):
        """
        Collects a number of policy rollouts and measures success

        Returns
        ----------
        The recorded success rate of the planner
        """
    

        for i in range(self.il_config['num_sup_rollouts']):
            rollout = self.rollout_supervisor()
            self.d_logger.save_rollout(rollout)

        return


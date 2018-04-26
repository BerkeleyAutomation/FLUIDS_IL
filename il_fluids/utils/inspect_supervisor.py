import numpy as np
import IPython
import os
import glob
from sklearn.tree import DecisionTreeRegressor
from numpy.random import uniform
import numpy.linalg as LA
from gym_urbandriving.actions import VelocityAction
from il_fluids.utils.losses import loss
from il_fluids.utils.bar_chart import make_bar_graph
import _pickle as pickle

from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

###Class created to store relevant information for learning at scale

class InspectSupervisor():

	def __init__(self,il_config):

		self.data = []
		self.rollout_info = {}
		self.file_path = il_config['file_path'] + il_config['experiment_name']

		self.model = il_config['model']

		self.il_config = il_config
		self.iter_count = 0

		self.load_data()
		self.file_path = self.file_path+'/inspect_supervisor'
		if not os.path.exists(self.file_path):
			os.makedirs(self.file_path)



	def load_data(self):
		"""
		Loads the data from the specified path 

		Returns
		----------
		path: list
			Containing the rollouts
		"""
		i = 0

		paths = glob.glob(self.file_path+'/rollout_*')
		self.rollouts = []


		for path in paths:
			data_point = np.load(path,encoding='latin1')
			self.rollouts.append(data_point)

		return paths


	def create_data_structure(self,rollout):
		num_supervisor = len(rollout[0][0]['action'])
		supervisors = []
		robots = []


		for i in range(num_supervisor):
			supervisors.append([])

		for i in range(num_supervisor):
			robots.append([])
	
		for datum in rollout[0]:

			observations = datum['state']
			sup_actions = datum['action']
			robot_actions = datum["noise_action"]
			
			for i in range(len(sup_actions)):

				supervisors[i].append(sup_actions[i].get_value())
				robots[i].append(robot_actions[i].get_value())

		return supervisors,robots

	def compute_variances(self,agents):

		count = 0
		variances = []
		for agent in agents:
			
			std = np.std(agent)

			variances.append(std**2)

		return variances

			



	def plot_varaince_per_agent(self,agents,file_name):

		varainces = self.compute_variances(agents)

		labels = []
		for i in range(len(varainces)):
			labels.append(str(i))

		make_bar_graph(varainces,file_name,labels,'varaince')


	def plot_rollout_actions(self):

		count = 0
		for rollout in self.rollouts:
			
			supervisors,robots = self.create_data_structure(rollout)

			for supervisor in supervisors:
				plt.plot(supervisor)
			
			plt.savefig(self.file_path+'/sup_rollout'+str(count)+'.png')
			plt.clf()

			for robot in robots:
				plt.plot(robot)
			
			plt.savefig(self.file_path+'/robot_rollout'+str(count)+'.png')
			plt.clf()

			count += 1



	def plot_rollout_variances(self):

		count = 0
		for rollout in self.rollouts:
			
			supervisors,robots = self.create_data_structure(rollout)

			filename = self.file_path+'/sup_variance'+str(count)+'.png'
			
			self.plot_varaince_per_agent(supervisors,filename)

			filename = self.file_path+'/robot_variance'+str(count)+'.png'
			
			self.plot_varaince_per_agent(robots,filename)

			count += 1


	

				




import numpy as np
import IPython
import os
import glob
from sklearn.tree import DecisionTreeRegressor
from numpy.random import uniform
import numpy.linalg as LA
from gym_urbandriving.actions import VelocityAction
from il_fluids.utils.losses import loss
import _pickle as pickle

from sklearn.preprocessing import StandardScaler

###Class created to store relevant information for learning at scale

class Learner():

	def __init__(self,il_config):

		self.data = []
		self.rollout_info = {}
		self.file_path = il_config['file_path'] + il_config['experiment_name']

		self.model = il_config['model']

		self.il_config = il_config
		self.iter_count = 0


	def standardize(self):

		self.ss = StandardScaler()
		self.ss.fit(self.X_train)

		self.ss.transform(self.X_train)
		self.ss.transform(self.X_test)

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


	def train_model(self):

		"""
		Trains a model on the loaded data, for know its a sklearn model
		"""

		self.X_train = []
		self.Y_train = []

		self.X_test = []
		self.Y_test = []

		#We are currently using a decision tree, however this can be quite modular
		if self.model == "none":
			self.model = DecisionTreeRegressor()


		for rollout in self.rollouts:

			if uniform() > 0.2:
				train = True
			else:
				train = False

			for datum in rollout[0]:

				observations = datum['state']
				actions = datum['action']

				for i in range(len(actions)):

					a_ = actions[i].get_value()

					s_ = observations[i]

					if train:
						self.X_train.append(s_)
						self.Y_train.append(a_)
					else:
						self.X_test.append(s_)
						self.Y_test.append(a_)

		#self.standardize()
		self.model.fit(self.X_train,self.Y_train) 

		if self.il_config['save_model']:
			if not os.path.exists(self.file_path+'/model'):
				os.makedirs(self.file_path+'/model')
			model_file_path = self.file_path+'/model/model_iter_'+str(self.iter_count)
			np.save(model_file_path,self.model)
			#pickle.dump(self.model,open(model_file_path,'w'))
			self.iter_count += 1
		


	def get_train_error(self):
		"""
        Reports the training error of the model

        Returns
        ------------
        float specifying L2 error
        """

		avg_err = 0.0

		for i in range(len(self.X_train)):

			x = np.array([self.X_train[i]])
			y = self.Y_train[i]

			y_ = self.model.predict(x)

			err = loss(self.il_config['loss_type'],y,y_[0])

			avg_err += err

		return avg_err/float(len(self.X_train))


	def get_cs(self,evaluations):
		"""
        Report the on-policy surrogate loss to measure covariate shift

        Returns
        ------------
        float specifying L2 error
        """

		count = 0.0
		avg_err = 0.0

		for rollout in evaluations:
			for datum in rollout:

				robot_action = datum['action']
				sup_action = datum['sup_action']

				for i in range(len(robot_action)):
					_ar = robot_action[i].get_value()
					_sr = sup_action[i].get_value()

					err = loss(self.il_config['loss_type'],_ar,_sr)

					avg_err += err
					count += 1.0


		return avg_err/count

	def get_robot_reward(self,evaluations):
		"""
        Report the on-policy surrogate loss to measure covariate shift

        Returns
        ------------
        float specifying L2 error
        """

		count = 0.0
		avg_reward = 0.0

		for rollout in evaluations:
			for datum in rollout:
				avg_reward += datum['reward']
				count += 1.0


		return avg_reward/count

	def get_supervisor_reward(self):
		"""
        Report the on-policy surrogate loss to measure covariate shift

        Returns
        ------------
        float specifying L2 error
        """

		count = 0.0
		avg_reward = 0.0

		for rollout in self.rollouts:
			for datum in rollout[0]:
				avg_reward += datum['reward']
				count += 1.0


		return avg_reward/count

	def get_test_error(self):
		"""
        Reports the test error of the model

        Returns
        ------------
        float specifying L2 error
        """


		avg_err = 0.0

		for i in range(len(self.X_test)):

			x = np.array([self.X_test[i]])
			y = self.Y_test[i]
			
			y_ = self.model.predict(x)
	
			err = loss(self.il_config['loss_type'],y,y_[0])

			avg_err += err

		if len(self.X_test) == 0:
			return avg_err

		return avg_err/float(len(self.X_test))



	def eval_policy(self,observations):
		"""
        Evaluates model, which is used in execution 
        
		Parameters
        ----------
        state: state of the enviroment
        goal_state: list of [x,y,velocity, theta] states

       	Returns
        ------------
        list of each action for the agent
        """
		actions = []
		for s_ in observations:

			x = np.array([s_])
			action = self.model.predict(x)

			
			actions.append(VelocityAction(action[0]))

		return actions






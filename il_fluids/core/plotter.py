from gym_urbandriving.state.state import PositionState
from gym_urbandriving.assets import Terrain, Lane, Street, Sidewalk,\
    Pedestrian, Car, TrafficLight
import numpy as np
import IPython
import os
import glob
from sklearn.tree import DecisionTreeRegressor
from numpy.random import uniform
import numpy.linalg as LA
import matplotlib
matplotlib.use("TkAgg")
from matplotlib import pyplot as plt
from il_fluids.utils.confusion_matrix import create_matrix
from copy import deepcopy
plt.style.use('fivethirtyeight')

class Plotter():

	def __init__(self,file_path,il_config=None):

		'''
		Class to plot performance of methods
		'''

		self.file_path = file_path
		
		if il_config:
			self.il_config = il_config
		if not os.path.exists(self.file_path+'/plots'):
			os.makedirs(self.file_path+'/plots')

		self.keys = {"reward_sup":[],
					"reward_robot":[],
					"loss_sup":[],
					"loss_robot":[],
					"train_sup":[],
					"bias":[],
					"variance":[]}

	def update_file_path(self,file_path):

		self.file_path = file_path

		if not os.path.exists(self.file_path+'/plots'):
			os.makedirs(self.file_path+'/plots')

	def save_matrix(self,stats):
		
		stat = stats[-1]
		_y = stat['confusion_matrix_sup']['robot']
		y =  stat['confusion_matrix_sup']['supervisor']

		filename = self.file_path+'/plots/sup_cm.png'
		create_matrix(y,_y,filename)


		_y = stat['confusion_matrix_robot']['robot']
		y =  stat['confusion_matrix_robot']['supervisor']

		filename = self.file_path+'/plots/robot_cm.png'
		create_matrix(y,_y,filename)
		plt.clf()


	def get_statistics(self,stats):
		raw_data = deepcopy(self.keys)

		for i in range(len(stats)):
			for key in stats[i].keys():
				raw_data[key] = stats[i][key]

		return raw_data




	def save_plots(self,stats):
		'''
		Save the plots to measure different aspects of the experiment

		Paramters
		-----------
		stats: list of dict
			Contains the measured statistics 
		'''

		raw_data = self.get_statistics(stats)


		if self.il_config:
			if self.il_config['action'] == 'velocity':
				self.save_matrix(stats)
		else:
			self.save_matrix(stats)

		
		plt.plot(raw_data['reward_sup'],label = 'R.S.' )
		plt.plot(raw_data['reward_robot'],label = 'R.R.' )
		plt.legend()

		plt.savefig(self.file_path+'/plots/reward.png')
		plt.clf()

		plt.plot(raw_data['loss_sup'],label = 'L.S.' )
		plt.plot(raw_data['loss_robot'],label = 'L.R.' )
		plt.legend()

		plt.savefig(self.file_path+'/plots/covariate_shift.png')
		plt.clf()

		plt.plot(raw_data['loss_sup'],label = 'L.S.' )
		plt.plot(raw_data['train_sup'],label = 'T.S.' )
		plt.legend()

		plt.savefig(self.file_path+'/plots/generalization.png')
		plt.clf()

		if self.il_config["measure_est_stats"]:
			plt.plot(raw_data['bias'],label = 'L.S.' )
			plt.legend()
			plt.savefig(self.file_path+'/plots/bias.png')
			plt.clf()

			plt.plot(raw_data['variance'],label = 'L.S.' )
			plt.legend()
			plt.savefig(self.file_path+'/plots/variance.png')
			plt.clf()
			

		np.save(self.file_path+'/plots/raw_data',raw_data)

	def save_agg_plots(self,agg_stats):

		raw_data = deepcopy(self.keys)
		raw_data_err = deepcopy(self.keys)

		num_point = float(len(agg_stats))
		for i in range(self.il_config['num_iters']):
			values = deepcopy(self.keys)

			for stat in agg_stats:
				for key in stat[0].keys():
					values[key].append(stat[i][key])

			for key in agg_stats[0][0].keys():
				raw_data[key].append(np.mean(values[key]))
				raw_data_err[key].append(np.std(values[key])/np.sqrt(num_point))

		
		x = range(self.il_config['num_iters'])
		
		plt.errorbar(x,raw_data['reward_sup'],yerr=raw_data_err['reward_sup'], label= 'R.S.' )
		plt.errorbar(x,raw_data['reward_robot'],yerr=raw_data_err['reward_robot'],label = 'R.R.' )
		plt.legend()

		plt.savefig(self.file_path+'/plots/reward.png')
		plt.clf()

		plt.errorbar(x,raw_data['loss_sup'],yerr=raw_data_err['loss_sup'],label = 'L.S.' )
		plt.errorbar(x,raw_data['loss_robot'],yerr=raw_data_err['loss_robot'],label = 'L.R.' )
		plt.legend()

		plt.savefig(self.file_path+'/plots/covariate_shift.png')
		plt.clf()

		plt.errorbar(x,raw_data['loss_sup'],yerr=raw_data_err['loss_sup'],label = 'L.S.' )
		plt.errorbar(x,raw_data['train_sup'],yerr=raw_data_err['train_sup'],label = 'T.S.' )
		plt.legend()

		plt.savefig(self.file_path+'/plots/generalization.png')
		plt.clf()


		if self.il_config["measure_est_stats"]:
			plt.errorbar(x,raw_data['bias'],yerr=raw_data_err['bias'],label = 'L.S.' )
			plt.legend()
			plt.savefig(self.file_path+'/plots/bias.png')
			plt.clf()


			plt.errorbar(x,raw_data['variance'],yerr=raw_data_err['variance'],label = 'T.S.' )
			plt.legend()
			plt.savefig(self.file_path+'/plots/variance.png')
			plt.clf()
			

		np.save(self.file_path+'/plots/raw_data',raw_data)


		




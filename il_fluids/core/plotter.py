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
import matplotlib.pyplot as plt

plt.style.use('fivethirtyeight')

class Plotter():

	def __init__(self,file_path):

		'''
		Class to plot performance of methods
		'''

		self.file_path = file_path
		

		if not os.path.exists(self.file_path+'/plots'):
			os.makedirs(self.file_path+'/plots')


	def save_plots(self,stats):
		'''
		Save the plots to measure different aspects of the experiment

		Paramters
		-----------
		stats: list of dict
			Contains the measured statistics 
		'''

		reward_sup = []
		reward_robot = []

		loss_sup = []
		loss_robot = []

		loss_sup = []
		train_sup = []

		for i in range(len(stats)):

			# reward_sup.append(stats[i]['reward_sup'])
			# reward_robot.append(stats[i]['reward_robot'])

			loss_sup.append(stats[i]['loss_sup'])
			loss_robot.append(stats[i]['loss_robot'])

			
			train_sup.append(stats[i]['train_sup'])



		# plt.plot(reward_sup,label = 'R.S.' )
		# plt.plot(reward_robot,label = 'R.R.' )
		# plt.legend()

		# plt.savefig(self.file_path+'/plots/reward.png')
		# plt.clf()

		plt.plot(loss_sup,label = 'L.S.' )
		plt.plot(loss_robot,label = 'L.R.' )
		plt.legend()

		plt.savefig(self.file_path+'/plots/covariate_shift.png')
		plt.clf()

		plt.plot(loss_sup,label = 'L.S.' )
		plt.plot(train_sup,label = 'T.S.' )
		plt.legend()

		plt.savefig(self.file_path+'/plots/generalization.png')
		plt.clf()





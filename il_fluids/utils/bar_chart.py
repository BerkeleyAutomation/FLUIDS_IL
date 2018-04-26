import matplotlib.pyplot as plt
import numpy as np

def make_bar_graph(values,plot_name,x_labels,y_axis):
	objects = []
	performance = []

	for indx in range(len(values)):
		objects.append(x_labels[indx])
		performance.append(values[indx])

	y_pos = np.arange(len(objects))
	 
	plt.bar(y_pos, performance, align='center', alpha=0.5)
	plt.xticks(y_pos, objects)
	
	 
	plt.savefig(plot_name+'.png')
	plt.clf()
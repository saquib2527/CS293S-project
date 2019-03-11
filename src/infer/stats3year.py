from __future__ import print_function
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

def printVals(avgs):
	for i in avgs:
		for j in i:
			print('{:4f}'.format(j),end = ' ')
		print()

outputdf = pd.DataFrame()
df = pd.read_csv('../CS293S/hourly_report-2016-2017-2018.csv')
df['Date'] = pd.to_datetime(df.Date)
stations = list(set(df.StationNbr))
for i in stations:
	tempdf = df[(df['StationNbr'] == i)]
	yearly = []
	for y in [2016,2017,2018]:
		avgs=[]
		print("Stationt{}:".format(i))
		for j in range(1,13):
			houravgs = []
			for t in range(100, 2500, 100):
				hourdf = tempdf[(tempdf.Date.dt.year == y) & (tempdf.Date.dt.month == j) & (tempdf['Hour'] == t)]
				houravgs.append(np.average(hourdf.HlyEto))
			avgs.append(houravgs)
		avgs = np.array(avgs)
		yearly.append(avgs)
	fig = plt.figure()
	h = ['{}:00'.format(c) for c in range(24)]
	years = [(2016,'r', .7),(2017,'g', .6),(2018,'b', .5)]
	for a in range(0,3):
		avgs = yearly[a]
		lab, color, alph = years[a]
		for z in range(12):
			x = avgs[z,:]
			plt.plot(h, x, color, linewidth=3, alpha=alph)
	custom_lines = [Line2D([0], [0], color='r', lw=4),
                Line2D([0], [0], color='g', lw=4),
                Line2D([0], [0], color='b', lw=4)]
	#fig1, ax = plt.subplots()
	#lines = ax.plot(data)
	plt.legend(custom_lines, ['2016', '2017', '2018'], prop={'size': 20})
	#plt.legend(prop={'size': 20})
	axes = plt.gca()
	axes.set_ylim([0.0,1.0])
	plt.xticks(rotation=45)
	plt.xlabel('time (hour)', fontsize=20)
	plt.ylabel('ETo (mm)', fontsize=20)
	plt.tick_params(axis='both', which='major', labelsize=24)
	plt.tick_params(axis='both', which='minor', labelsize=24)
	fig.set_size_inches(18.5, 10.5)
	plt.tight_layout()
	fig.savefig('3yeargroup/{}multi.svg'.format(i,y), bbox_inches='tight',dpi=100, format='svg')
	plt.close()
	#exit()
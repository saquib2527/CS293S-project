from __future__ import print_function
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt 

def printVals(avgs):
	for i in avgs:
		for j in i:
			print('{:4f}'.format(j),end = ' ')
		print()

outputdf = pd.DataFrame()
df = pd.read_csv('../CS293S/hourly_report-2016-2017-2018.csv')
df['Date'] = pd.to_datetime(df.Date)
stations = list(set(df.StationNbr))
for y in [2016, 2017, 2018]:
	tdf = df[(df.Date.dt.year == y)]
	for i in stations:
		tempdf = tdf[(tdf['StationNbr'] == i)]
		avgs=[]
		print("Stationt{}:".format(i))
		for j in range(1,13):
			houravgs = []
			for t in range(100, 2500, 100):
				hourdf = tempdf[(tempdf.Date.dt.month == j) & (tempdf['Hour'] == t)]
				houravgs.append(np.average(hourdf.HlyEto))
			avgs.append(houravgs)
		avgs = np.array(avgs)
		months = ['jan', 'feb', 'mar', 'apr', 'may', 'jun', 'jul', 'aug', 'sep', 'oct', 'nov','dec']
		fig = plt.figure()
		h = ['{}:00'.format(c) for c in range(24)]
		for z in range(12):
			x = avgs[z,:]
			plt.plot(h, x, label = months[z], linewidth=3)
		plt.legend(prop={'size': 20})
		axes = plt.gca()
		axes.set_ylim([0.0,1.0])
		plt.xticks(rotation=45)
		plt.xlabel('time (hour)', fontsize=20)
		plt.ylabel('ETo (mm)', fontsize=20)
		plt.tick_params(axis='both', which='major', labelsize=24)
		plt.tick_params(axis='both', which='minor', labelsize=24)
		fig.set_size_inches(18.5, 10.5)
		plt.tight_layout()
		fig.savefig('2018monthly/{}-{}.svg'.format(i,y), bbox_inches='tight',dpi=100, format='svg')
		plt.close()
		#exit()
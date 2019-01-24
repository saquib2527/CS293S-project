import math
from operator import itemgetter

from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

import pandas as pd
import numpy as np

class Classifier:

    def __init__(self):
        self.data_file = '../data/hourly_report.csv'
        self.stations_file = '../data/stations.csv'
        self.df = pd.read_csv(self.data_file)
        self.target = 'HlyEto'
        self.features = ['HlyAirTmp', 'HlySolRad', 'HlySoilTmp', 'HlyRelHum']

    def populate_nearest_neighbors(self):
        stations = pd.read_csv(self.stations_file)
        distances = {}
        for i in range(len(stations)):
            distances[stations.iloc[i]['StationNbr']] = []
        for i in range(len(stations) - 1):
            i_s_n = stations.iloc[i]['StationNbr']
            for j in range(i + 1, len(stations)):
                j_s_n = stations.iloc[j]['StationNbr']
                distance = (stations.iloc[i]['HmsLatitude'] - stations.iloc[j]['HmsLatitude'])**2 + (stations.iloc[i]['HmsLongitude'] - stations.iloc[j]['HmsLongitude'])**2
                distances[i_s_n].append({'StationNbr': j_s_n, 'Distance': distance})
                distances[j_s_n].append({'StationNbr': i_s_n, 'Distance': distance})
        with open('../data/distances.csv', 'w') as fh:
            for k in distances:
                distances[k] = sorted(distances[k], key=itemgetter('Distance')) 
                row = '{0},{1}'.format(k, ','.join('{0}:{1}'.format(d['StationNbr'], d['Distance']) for d in distances[k]))
                fh.write('{0}\n'.format(row))
                #print(distances[k])

    def linear_regression(self):
        train, test = train_test_split(self.df, test_size=0.2)
        reg = linear_model.LinearRegression()
        reg.fit(train[self.features], train[self.target])
        predictedY = reg.predict(test[self.features])
        print(mean_squared_error(test[self.target], predictedY))
        for pY, tY in zip(predictedY, test[self.target]):
            print('{0} {1}'.format(pY, tY))
        #print(reg.coef_)
        pass

    def print_stats(self):
        print(self.df.shape)
        counts = self.df.HlyEto.value_counts()
        print(counts)
        print(self.df.columns)


if __name__ == '__main__':
    c = Classifier()
    c.populate_nearest_neighbors()
    #c.linear_regression()
    #c.print_stats()

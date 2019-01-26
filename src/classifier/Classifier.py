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
        self.distance_file = '../data/distances.csv'
        self.df = pd.read_csv(self.data_file)
        self.target = 'HlyEto'
        self.features = ['HlyAirTmp', 'HlySolRad', 'HlySoilTmp', 'HlyRelHum']
        self.stations = pd.read_csv(self.stations_file)
        self.distances = {}

    def populate_nearest_neighbors(self):
        stations = self.stations
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

    def geographically_nearest_classifier(self, n=3):
        with open(self.distance_file, 'r') as fh:
            rows = fh.readlines()
            for row in rows:
                items = row.split(',')
                i = int(items.pop(0))
                self.distances[i] = []
                for item in items:
                    j, distance = item.split(':')
                    self.distances[i].append({'StationNbr': int(j), 'Distance': float(distance)})
        correct = 0
        incorrect = 0
        asd = 0
        for i in range(len(self.df)):
            station_number = self.df.iloc[i]['StationNbr']
            date = self.df.iloc[i]['Date']
            hour = self.df.iloc[i]['Hour']
            found = 0
            idx = 0
            eto = []
            while found != n:
                s_n = self.distances[station_number][idx]['StationNbr']
                idx += 1
                if idx == 135:
                    break
                row = self.df[(self.df['StationNbr'] == s_n) & (self.df['Date'] == date) & (self.df['Hour'] == hour)]
                if len(row) == 1:
                    eto.append(row.iloc[0]['HlyEto'])
                    found += 1
            predicted = max(set(eto), key = eto.count)
            actual = self.df.iloc[i]['HlyEto']
            if actual == predicted:
                correct += 1
            else:
                print('{0} {1}'.format(actual, predicted))
                incorrect += 1
            asd += 1
            if asd == 1000:
                break
        print('correct: {0} incorrect: {1}'.format(correct, incorrect))

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

    def fahrenheit_to_celsius(self, temp):
        return (temp - 32) * (5.0 / 9.0)

    def feet_to_meter(self, elevation):
        return (elevation * 0.3048)

    def CIMIS_penman(self):
        #print(self.df[(self.df['StationNbr'] == 5) & (self.df['Date'] == '2014-01-01') & (self.df['Hour'] == 100)]['HlyAirTmp'])
        for x in range(self.df.shape[0]):
            row = self.df.iloc[x]
            station_number = row['StationNbr']
            T = self.fahrenheit_to_celsius(row['HlyAirTmp']) #mean air temperature in celsius
            T_k = T + 273.16 #temperature in kelvin
            es = 0.6108 * math.exp((T * 17.27)/(T + 237.3)) #saturation vapor pressure
            VPD = es - (row['HlyVapPres'] * 0.1) #vapor pressure deficit, notice conversion from mBar to KPa
            DEL = (4099 * es) / ((T + 237.3) ** 2) #slope of the saturation vapor pressure vs. air temperature curve at the average hourly air temperature
            Z = self.feet_to_meter(self.stations[self.stations['StationNbr'] == station_number]['Elevation'].iloc[0]) #elevation of the station above mean sea level
            P = 101.3 - (0.0115 * Z) + 5.44 * (Z ** 2) / (10 ** 7) #barometric pressure
            GAM = 0.000646 * (1 + 0.000946 * T) * P #psychrometric constant
            W = DEL / (DEL + GAM) #weighting function
            Rn = (row['HlyNetRad'] / (60 * 24)) * 697.3 #hourly net radiation in Wm^-2
            U = row['HlyWindSpd'] * 0.44704 #hourly wind speed at 2 meters (ms^-1)
            FU2 = (0.125 + 0.0439 * U) if (Rn <= 0) else (0.030 + 0.0576 * U) #wind function
            NR = Rn / (694.5 * (1-0.000946 * T)) 
            RET = W * NR + (1-W) * VPD * FU2
            print('{0} {1} {2}'.format(row['HlyEto'], row['HlyAsceEto'], round(RET, 2)))
            if x == 100:
                break

    def print_stats(self):
        print(self.df.shape)
        counts = self.df.HlyEto.value_counts()
        print(counts)
        print(self.df.columns)


if __name__ == '__main__':
    c = Classifier()
    c.CIMIS_penman()
    #c.geographically_nearest_classifier(1)
    #c.populate_nearest_neighbors()
    #c.linear_regression()
    #c.print_stats()

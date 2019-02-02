import math
from operator import itemgetter
import time

from geopy import distance

from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.preprocessing import PolynomialFeatures

from sklearn import linear_model
from sklearn.neural_network import MLPRegressor

from sklearn.metrics import mean_squared_error, r2_score, confusion_matrix

import pandas as pd
import numpy as np

class Classifier:

    def __init__(self):
        self.data_file = '../data/hourly_report.csv'
        self.stations_file = '../data/stations.csv'
        self.distance_file = '../data/distances.csv'
        self.df = pd.read_csv(self.data_file)
        self.target = 'HlyEto'
        self.features = ['HlyAirTmp', 'HlyVapPres', 'HlyNetRad', 'HlyWindSpd', 'HlyRelHum']
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
                distance_ = distance.distance((stations.iloc[i]['HmsLatitude'], stations.iloc[i]['HmsLongitude']), (stations.iloc[j]['HmsLatitude'], stations.iloc[j]['HmsLongitude'])).miles
                distances[i_s_n].append({'StationNbr': j_s_n, 'Distance': distance_})
                distances[j_s_n].append({'StationNbr': i_s_n, 'Distance': distance_})
        with open('../data/distances.csv', 'w') as fh:
            for k in distances:
                distances[k] = sorted(distances[k], key=itemgetter('Distance')) 
                row = '{0},{1}'.format(k, ','.join('{0}:{1}'.format(d['StationNbr'], d['Distance']) for d in distances[k]))
                fh.write('{0}\n'.format(row))

    def geographically_nearest_classifier(self, n=3, metric='average'):
        # load distances
        with open(self.distance_file, 'r') as fh:
            rows = fh.readlines()
            for row in rows:
                items = row.split(',')
                i = int(items.pop(0))
                self.distances[i] = []
                for item in items:
                    j, distance = item.split(':')
                    self.distances[i].append({'StationNbr': int(j), 'Distance': float(distance)})
        
        asd = 0
        MSE = 0
        for i in range(len(self.df)):
            station_number = self.df.iloc[i]['StationNbr']
            date = self.df.iloc[i]['Date']
            hour = self.df.iloc[i]['Hour']
            found = 0
            idx = 0
            eto = []
            dis = []
            while found != n:
                s_n = self.distances[station_number][idx]['StationNbr']
                idx += 1
                if idx == 135:
                    print('searched all, did not find {0}, only {1}'.format(n, found))
                    break
                row = self.df[(self.df['StationNbr'] == s_n) & (self.df['Date'] == date) & (self.df['Hour'] == hour)]
                if len(row) == 1:
                    eto.append(row.iloc[0]['HlyEto'])
                    dis.append(self.distances[station_number][idx]['Distance'])
                    found += 1

            if metric == 'average':
                predicted = sum(eto) / len(eto)
            elif metric == 'idw':
                predicted = 0
                for x in range(len(dis)):
                    dis[x] = 1 / dis[x]
                dis_t = sum(dis)
                for x in range(len(dis)):
                    dis[x] = dis[x] / dis_t
                    predicted += eto[x] * dis[x]

            actual = self.df.iloc[i]['HlyEto']
            MSE += (actual - predicted) * (actual - predicted)
            asd += 1
            if asd == 1000:
                break
        MSE /= asd
        print('n: {0} metric: {1} MSE: {2}'.format(n, metric, MSE))

    def regression(self, features, lm_type='linear',alpha=0.5):
        # split into train and test
        train, test = train_test_split(self.df, test_size=0.2)

        if lm_type == 'mlp':
            reg = MLPRegressor()
            scaler = preprocessing.StandardScaler()
            X_train_scaled = scaler.fit_transform(train[features])
            X_test_scaled = scaler.fit_transform(test[features])
            reg.fit(X_train_scaled, train[self.target])
            predictedY = reg.predict(X_test_scaled)
        else:
            if lm_type == 'linear':
                reg = linear_model.LinearRegression()
            elif lm_type == 'ridge':
                reg = linear_model.Ridge(alpha=alpha)
            elif lm_type == 'lasso':
                reg = linear_model.Lasso(alpha=alpha)
            elif lm_type == 'bayesian':
                reg = linear_model.BayesianRidge()

            reg.fit(train[features], train[self.target])
            predictedY = reg.predict(test[features])

        for x in range(len(predictedY)):
            if predictedY[x] < 0:
                predictedY[x] = 0
            else:
                predictedY[x] = round(predictedY[x], 2)

        print(mean_squared_error(test[self.target], predictedY))

    def fahrenheit_to_celsius(self, temp):
        return (temp - 32) * (5.0 / 9.0)

    def feet_to_meter(self, elevation):
        return (elevation * 0.3048)

    def penman(self):
        for x in range(self.df.shape[0]):
            row = self.df.loc[x]
            station_number = row['StationNbr']
            T_a = row['HlyAirTmp']
            e_s = 0.6108 * math.exp((17.27 * T_a) / (T_a + 237.3))
            delta = (4099 * e_s) / ((T_a + 237.3) ** 2)
            R_n = row['HlyNetRad'] * 3.6 #conversion to MJm^-2h^-1, equation has calculated value for HlyNetRad
            A = self.feet_to_meter(self.stations[self.stations['StationNbr'] == station_number]['Elevation'].iloc[0]) #elevation of the station above mean sea level
            P = 101.3 * (((293 - 0.0065 * A) / 293) ** 5.26) #barometric pressure in KPa
            Lambda = 2.501 - 0.002361 * T_a
            Y = 0.00163 * (P / Lambda)
            e_a = e_s * (row['HlyRelHum'] / 100.0) #notice this is HlyVapPres once expressed in KPa, varified it works!
            if R_n > 0:
                G = 0.1 * R_n
                C_d = 0.24
            else:
                G = 0.5 * R_n
                C_d = 0.96
            U_2 = row['HlyWindSpd']
            first_part = (delta * (R_n - G)) / (Lambda * (delta + Y * (1 + C_d * U_2)))
            second_part = (Y * (37 / (T_a + 273.16)) * U_2 * (e_s - e_a)) / (delta + Y * (1 + C_d * U_2))
            ET_o = first_part + second_part
            print('{0} {1}'.format(row['HlyEto'], round(ET_o,2)))
            if x == 100:
                break

    def CIMIS_penman(self):
        #print(self.df[(self.df['StationNbr'] == 5) & (self.df['Date'] == '2014-01-01') & (self.df['Hour'] == 100)]['HlyAirTmp'])
        for x in range(self.df.shape[0]):
            row = self.df.iloc[x]
            station_number = row['StationNbr']
            T = row['HlyAirTmp'] #mean air temperature in celsius
            T_k = T + 273.16 #temperature in kelvin
            es = 0.6108 * math.exp((T * 17.27)/(T + 237.3)) #saturation vapor pressure in KPa
            VPD = es - row['HlyVapPres'] #vapor pressure deficit in KPa
            DEL = (4099 * es) / ((T + 237.3) ** 2) #slope of the saturation vapor pressure vs. air temperature curve at the average hourly air temperature
            Z = self.feet_to_meter(self.stations[self.stations['StationNbr'] == station_number]['Elevation'].iloc[0]) #elevation of the station above mean sea level
            P = 101.3 - (0.0115 * Z) + 5.44 * (Z ** 2) / (10 ** 7) #barometric pressure
            GAM = 0.000646 * (1 + 0.000946 * T) * P #psychrometric constant
            W = DEL / (DEL + GAM) #weighting function
            Rn = row['HlyNetRad'] #hourly net radiation in Wm^-2
            U = row['HlyWindSpd'] #hourly wind speed at 2 meters (ms^-1)
            FU2 = (0.125 + 0.0439 * U) if (Rn <= 0) else (0.030 + 0.0576 * U) #wind function
            NR = Rn / (694.5 * (1-0.000946 * T)) 
            RET = W * NR + (1-W) * VPD * FU2
            if RET < 0:
                RET = 0
            print('{0} {1}'.format(row['HlyEto'], round(RET, 2)))
            if x == 100:
                break

    def print_stats(self):
        print(self.df.shape)
        counts = self.df.HlyEto.value_counts()
        print(counts)
        print(self.df.columns)


if __name__ == '__main__':
    start = time.time()

    c = Classifier()
    #c.populate_nearest_neighbors()
    #c.CIMIS_penman()
    #c.penman()


    #for metric_type in ['average', 'idw']:
    #    for x in range(1, 6):
    #        c.geographically_nearest_classifier(x, metric_type)

    c.regression(c.features, 'linear')
    for x in range(len(c.features)):
        features = c.features[:]
        print('leaving out {0}'.format(features[x]))
        features.pop(x)
        c.regression(features, 'linear')

    end = time.time()
    print(end - start)

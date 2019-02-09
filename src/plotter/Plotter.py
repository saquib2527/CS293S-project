import datetime
import itertools
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.ticker import MaxNLocator
import numpy as np
import operator
import os
import pandas as pd
from pandas.plotting import scatter_matrix
import time

from sklearn.model_selection import train_test_split
from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score

class Plotter:

    def __init__(self):
        self.stations_file = '../data/stations.csv'
        self.data_file = '../data/hourly_report.csv'
        self.distances_file = '../data/distances.csv'
        self.image_folder = 'images'
        self.image_format = 'eps'
        self.stations = pd.read_csv(self.stations_file)
        self.df = pd.read_csv(self.data_file)
        self.load_distances()
        self.find_soi()

    def load_distances(self):
        self.distances = {}
        with open(self.distances_file, 'r') as fh:
            lines = fh.readlines()
            for line in lines:
                vals = line.split(',')
                target = vals.pop(0)
                self.distances[target] = []
                for val in vals:
                    station, distance = val.split(':')
                    self.distances[target].append({'StationNbr': int(station), 'distance': float(distance)})

    def CoCoRaHS(self):
        cocorahs_stations = {}
        cocorahs_stations['CASD132'] = {'lat': 32.9083, 'long': -116.6227, 'elev': 3327}
        cocorahs_stations['CASD12'] = {'lat': 32.9956, 'long': -117.0044, 'elev': 1206}
        cocorahs_stations['CABT5'] = {'lat': 39.4932, 'long': -121.477, 'elev': 669}

        for c_station in cocorahs_stations:
            print('finding nearest CIMIS stations of {0}'.format(c_station))
            distances = {}
            for x in range(len(self.stations)):
                station = self.stations.iloc[x]
                distances[station['StationNbr']] = ((station['HmsLatitude'] - cocorahs_stations[c_station]['lat']) ** 2) + ((station['HmsLongitude'] - cocorahs_stations[c_station]['long']) ** 2)
            distances = sorted(distances.items(), key=operator.itemgetter(1))
            print(distances[0][1])
            print(distances[1][1])
            print(distances[2][1])

    def hour_to_string(self, hour):
        hour /= 100
        if hour < 10:
            hour = '0' + str(hour) + ':00'
        else:
            hour = str(hour) + ':00'
        return hour

    def plot_hourly_eto_by_station_date(self, station_nbr, date, check_missing_data=True):
        df = self.df[(self.df['StationNbr'] == station_nbr) & (self.df['Date'] == date)]
        df = df.sort_values(by=['Hour'])
        if df.shape[0] != 24:
            print('all hourly data not present, aborting')
            return
        etos = df['HlyEto']
        hours = df['Hour']
        timestamps = []
        ts_idx = 0
        for hour, eto in zip(hours, etos):
            if hour < 1000:
                hour = '0' + str(hour)
            elif hour == 2400:
                hour = '2359'
            else:
                hour = str(hour)
            timestamps.append(datetime.datetime.strptime('{0} {1}'.format(date, hour), '%Y-%m-%d %H%M'))
            ts_idx += 1
        plt.plot(timestamps, etos, marker='o')
        ax = plt.gca()
        ax.xaxis_date()
        plt.xticks(rotation=45)
        myFmt = mdates.DateFormatter('%H:%M')
        ax.xaxis.set_major_formatter(myFmt)
        plt.xlabel('time (hour)')
        plt.ylabel('ETo (mm)')
        plt.title('Hourly ETo values for station {0} on {1}'.format(station_nbr, date))
        plt.tight_layout()
        plt.show()

    def plot_hourly_summary(self, filename='mean-of-hourly-eto-values'):
        hour_group = self.df[['Hour', 'HlyEto']].groupby(['Hour'])
        hours = self.df.Hour.unique()
        x = []
        y = []
        for hour in hours:
            x.append(self.hour_to_string(hour))
            y.append(hour_group.get_group(hour).mean()['HlyEto'])
        plt.plot(x, y, marker='o')
        ax = plt.gca()
        plt.xticks(rotation=45)
        plt.xlabel('time (hour)')
        plt.ylabel('ETo (mm)')
        plt.title('Mean hourly ETo values')
        plt.tight_layout()
        plt.savefig(os.path.join(self.image_folder, filename), format=self.image_format)

    def plot_soi_latitude_hourly_summary(self, filename='soi-latitude-mean-of-hourly-eto-values'):
        X = []
        Y = []
        label = ['south', 'middle', 'north']
        for station in self.soi_latitude:
            df = self.df[self.df['StationNbr'] == station]
            hour_group = df[['Hour', 'HlyEto']].groupby(['Hour'])
            hours = df.Hour.unique()
            x = []
            y = []
            for hour in hours:
                x.append(self.hour_to_string(hour))
                y.append(hour_group.get_group(hour).mean()['HlyEto'])
            X.append(x)
            Y.append(y)
        for i in range(len(self.soi_latitude)):
            plt.plot(X[i], Y[i], marker='o', label=label[i])
        plt.legend()
        plt.xticks(rotation=45)
        plt.xlabel('time (hour)')
        plt.ylabel('ETo (mm)')
        plt.tight_layout()
        plt.savefig(os.path.join(self.image_folder, filename), format=self.image_format)

    def plot_scatterplot_matrix(self, features=['HlyAirTmp', 'HlyVapPres', 'HlyWindSpd', 'HlyNetRad'], target='HlyEto'):
        filename = 'scatterplot-matrix-' + '-'.join(features)
        scatter_matrix(self.df[features], rasterized=True)
        plt.savefig(os.path.join(self.image_folder, filename), format=self.image_format)
        #plt.show()

    def get_regression_result(self, features=['HlyAirTmp', 'HlyVapPres', 'HlyWindSpd', 'HlyNetRad'], target='HlyEto'):
        filename = 'regression-' + '-'.join(features)
        train, test = train_test_split(self.df, test_size=0.2)
        reg = linear_model.LinearRegression()
        reg.fit(train[features], train[target])
        predictedY = reg.predict(test[features])

        for x in range(len(predictedY)):
            if predictedY[x] < 0:
                predictedY[x] = 0
            else:
                predictedY[x] = round(predictedY[x], 2)

        return {'r2':r2_score(test[target], predictedY) , 'mse': mean_squared_error(test[target], predictedY)}

    def get_regression_result_all_combo(self, features=['HlyAirTmp', 'HlyVapPres', 'HlyWindSpd', 'HlyNetRad'], target='HlyEto'):
        filename = 'regression_results.tex'
        results = []
        for num in range(1, len(features) + 1):
            combos = list(itertools.combinations(features, num))
            for combo in combos:
                combo = sorted(list(combo))
                result = self.get_regression_result(combo)
                result['f_list'] = ','.join(combo)
                results.append(result)
        results = sorted(results, key=operator.itemgetter('mse'))

        table = "\\begin{tabular}{|l|l|l|}\n"
        table += "\hline\n"
        table += "\\textbf{Features} & \\textbf{Mean Squared Error} & \\textbf{$R^2$ Value}\\\\\n"
        table += "\hline\n"
        for r in results:
            table += "{0} & {1} & {2}\\\\\n".format(r['f_list'], r['mse'], r['r2'])
            table += "\hline\n"
        table += "\end{tabular}"
        with open(filename, 'w') as fh:
            fh.write('{0}'.format(table))

    def find_soi(self):
        '''
        finds Stations of Interest:
            stations that are at the very North, South, and center
            stations that have nearest neighbor at the closest and the farthest distance
        '''
        self.soi_latitude = []
        self.soi_nearest_distance = []

        stations = self.stations.sort_values(by='HmsLatitude')
        min_lat = stations.iloc[0]['HmsLatitude']
        max_lat = stations.iloc[-1]['HmsLatitude']
        mid_lat = (min_lat + max_lat) / 2
        error = 10000
        mid_station = 1000
        for index, row in self.stations.iterrows():
            err = (row['HmsLatitude'] - mid_lat) ** 2
            if err < error:
                error = err
                mid_station = row['StationNbr']

        self.soi_latitude.append(stations.iloc[0]['StationNbr'])
        self.soi_latitude.append(mid_station)
        self.soi_latitude.append(stations.iloc[-1]['StationNbr'])

        temp = []
        for station in self.distances:
            temp.append({'StationNbr': station, 'distance': self.distances[station][0]['distance']})
        temp = sorted(temp, key=operator.itemgetter('distance'))
        min_distance = temp[0]['distance']
        max_distance = temp[-1]['distance']
        mid_distance = (min_distance + max_distance) / 2
        error = 10000
        mid_station = 1000
        for row in temp:
            err = (row['distance'] - mid_distance) ** 2
            if err < error:
                error = err
                mid_station = row['StationNbr']
        self.soi_nearest_distance.append(temp[0]['StationNbr'])
        self.soi_nearest_distance.append(mid_station)
        self.soi_nearest_distance.append(temp[-1]['StationNbr'])

    def stat(self):
        print(self.df.shape)


if __name__ == '__main__':
    start = time.time()
    
    p = Plotter()
    #p.plot_hourly_summary()
    #p.plot_soi_latitude_hourly_summary()
    #p.plot_scatterplot_matrix()
    p.get_regression_result_all_combo()

    end = time.time()
    print('elapsed time: {0} seconds'.format(end - start))

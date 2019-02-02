import operator
import pandas as pd
import matplotlib.pyplot as plt

class Plotter:

    def __init__(self):
        self.stations_file = '../data/stations.csv'
        self.data_file = '../data/hourly_report.csv'
        self.stations = pd.read_csv(self.stations_file)
        self.df = pd.read_csv(self.data_file)

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


    def accuracy_et(self, sn1, sn2):
        df1 = self.df[self.df['StationNbr'] == sn1]
        df2 = self.df[self.df['StationNbr'] == sn2]
        dates = df1.Date.unique()
        correct = incorrect = 0
        count = 0
        for date in dates:
            for hour in range(100, 2500, 100):
                et1 = df1[(df1['Date'] == date) & (df1['Hour'] == hour)]['HlyEto']
                et2 = df2[(df2['Date'] == date) & (df2['Hour'] == hour)]['HlyEto']
                if et1.empty or et2.empty:
                    continue
                et1 = et1.iloc[0]
                et2 = et2.iloc[0]
                if et1 == et2:
                    correct += 1
                else:
                    incorrect += 1
                count += 1
                if count % 5000 == 0:
                    print('{0} {1} {2} {3}'.format(correct, incorrect, float(correct) / (correct + incorrect), count))
                    return
        print('{0} {1} {2}'.format(correct, incorrect, float(correct) / (correct + incorrect)))

    def stat(self):
        print(self.df.shape)


if __name__ == '__main__':
    p = Plotter()
    p.CoCoRaHS()

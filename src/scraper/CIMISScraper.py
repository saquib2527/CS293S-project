import json
import os
import requests

class CIMISScraper:

    def __init__(self):
        """Initialize appkey, urls, supporting data, and folders"""

        self.app_key = '6f4e09b0-ad7f-4820-94b2-a38c6ac7d6ad'
        self.station_url = 'http://et.water.ca.gov/api/station'
        self.report_url = 'http://et.water.ca.gov/api/data'
        self.station_headers = {'Accept': 'application/json'}
        self.data_folder = '../data'
        self.hourly_report_folder = os.path.join(self.data_folder, 'hourly')
        self.daily_report_folder = os.path.join(self.data_folder, 'daily')
        if not os.path.isdir(self.data_folder):
            os.mkdir(self.data_folder)
        if not os.path.isdir(self.hourly_report_folder):
            os.mkdir(self.hourly_report_folder)
        if not os.path.isdir(self.daily_report_folder):
            os.mkdir(self.daily_report_folder)
        self.station_filename = 'stations.csv'
        self.station_numbers = []
        self.features = [
                "day-air-tmp-avg",
                "day-air-tmp-avg",
                "day-air-tmp-min",
                "day-dew-pnt",
                "day-eto",
                "day-asce-eto",
                "day-asce-etr", 
                "day-precip",
                "day-rel-hum-avg",
                "day-rel-hum-max",
                "day-rel-hum-min",
                "day-soil-tmp-avg",
                "day-soil-tmp-max",
                "day-soil-tmp-min",
                "day-sol-rad-avg", 
                "day-sol-rad-net",
                "day-vap-pres-max",
                "day-vap-pres-avg",
                "day-wind-ene", 
                "day-wind-ese",
                "day-wind-nne",
                "day-wind-nnw",
                "day-wind-run",
                "day-wind-spd-avg", 
                "day-wind-ssw",
                "day-wind-wnw",
                "day-wind-wsw",
                "hly-air-tmp",          #27
                "hly-dew-pnt",
                "hly-eto",
                "hly-net-rad",
                "hly-asce-eto",
                "hly-asce-etr",
                "hly-precip",
                "hly-rel-hum",
                "hly-res-wind",
                "hly-soil-tmp",
                "hly-sol-rad",
                "hly-vap-pres",
                "hly-wind-dir",
                "hly-wind-spd"
                ]

    def get_stations(self):
        """Creates a CSV file (as initialized in init) containing active and ETo stations
            info includes station number, name, city, county, elevation, latitude, and longitude
            note that it is required to call this function to populate valid stations
        """
        response = requests.get(self.station_url, headers=self.station_headers)
        data = json.loads(response.content)
        keys = ['StationNbr', 
                'Name',
                'City', 
                'County',
                'Elevation',
                'HmsLatitude',
                'HmsLongitude'
                ]
        with open(os.path.join(self.data_folder, self.station_filename), 'w') as fh:
            fh.write(','.join('{}'.format(k) for k in keys))
            fh.write('\n')
            for station in data['Stations']:
                if((station['IsActive'] == 'True') and (station['IsEtoStation'] == 'True')):
                    self.station_numbers.append(int(station['StationNbr']))
                    station['HmsLatitude'] = station['HmsLatitude'].split(' ')[2]
                    station['HmsLongitude'] = station['HmsLongitude'].split(' ')[2]
                    fh.write(','.join('{}'.format(station[k].encode('utf-8')) for k in keys))
                    fh.write('\n')

    def get_hourly_report_from_station(self, station_number=2, start_date='2018-11-01', end_date='2018-12-31'):
        """Creates a CSV file with report from a single station as indicated by station_number
            details about API call can be found here: http://et.water.ca.gov/Rest/Index
        """
        data_items = ','.join(x for x in self.features[27:]) #only the hourly features
        url = '{0}?appKey={1}&targets={2}&startDate={3}&endDate={4}&dataItems={5}'.format(self.report_url, self.app_key, station_number, start_date, end_date, data_items)
        response = requests.get(url, headers=self.station_headers)
        data = json.loads(response.content)
        try:
            records = data['Data']['Providers'][0]['Records']
        except:
            return -1
        keys = []
        for x in self.features[27:]:
            keys.append(''.join(y.capitalize() for y in x.split('-')))
        with open(os.path.join(self.hourly_report_folder, '{0}_{1}_{2}.csv'.format(station_number, start_date, end_date)), 'w') as fh:
            fh.write('{0},{1},{2}'.format(','.join(keys), 'Hour', 'Date'))
            fh.write('\n')
            for record in records:
                row = ','.join('{}'.format(record[k]['Value']) for k in keys)
                row = '{0},{1},{2}'.format(row, record['Hour'], record['Date'])
                fh.write(row)
                fh.write('\n')
                
        return 0


    def debug(self):
        """Prints debug statements during development"""
        
        print(self.station_numbers)
        print(self.features[27:])

if __name__ == '__main__':
    print('please use Driver.py')

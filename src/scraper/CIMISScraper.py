import json
import requests

class CIMISScraper:

    def __init__(self):
        """Initialize appkey and urls"""

        self.app_key = 'df2440aa-0896-4da7-9feb-a8ecf8ed3046'
        self.station_url = 'http://et.water.ca.gov/api/station'
        self.station_headers = {'Accept': 'application/json'}
        self.station_filename = 'stations.csv'

    def get_stations(self):
        """Creates a CSV file (as initialized in init) containing active and ETo stations
            info includes station number, name, city, county, elevation, latitude, and longitude
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
        with open(self.station_filename, 'w') as fh:
            fh.write(','.join('{}'.format(k) for k in keys))
            fh.write('\n')
            for station in data['Stations']:
                if((station['IsActive'] == 'True') and (station['IsEtoStation'] == 'True')):
                    station['HmsLatitude'] = station['HmsLatitude'].split(' ')[2]
                    station['HmsLongitude'] = station['HmsLongitude'].split(' ')[2]
                    fh.write(','.join('{}'.format(station[k].encode('utf-8')) for k in keys))
                    fh.write('\n')

    def debug(self):
        """Prints debug statements during development"""

        pass

if __name__ == '__main__':
    cs = CIMISScraper()
    cs.get_stations()

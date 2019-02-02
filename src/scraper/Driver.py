import datetime
import time
from dateutil.relativedelta import *

from CIMISScraper import CIMISScraper

class Driver:

    def __init__(self):
        self.cs = CIMISScraper()
        self.cs.get_stations()
        self.num_of_requests = 0
        self.wait_time = 10

    def get_hourly_data_start_end(self, start_date, end_date):
        current_date = start_date
        while True:
            print current_date
            for station_number in self.cs.station_numbers:
                ret_val = -1
                while ret_val != 0:
                    ret_val = self.cs.get_hourly_report_from_station(station_number=station_number, start_date=str(current_date.date()), end_date=str(current_date.date()))
                    if ret_val == -1:
                        print 'exception occurred, waiting for {0} seconds'.format(self.wait_time)
                        time.sleep(self.wait_time)
                print('{0} {1}'.format(station_number, str(start_date.date())))
            current_date += relativedelta(days=+1)
            if current_date > end_date:
                break
        pass

    def get_N_years_hourly_data(self, N, start_date):
        ''' start_date is a datetime object
        '''
        years = N
        end_date = start_date + relativedelta(days=+60)
        for x in range(1, years * 12, 2):
            for station_number in self.cs.station_numbers:
                ret_val = -1
                while ret_val != 0:
                    ret_val = self.cs.get_hourly_report_from_station(station_number=station_number, start_date=str(start_date.date()), end_date=str(end_date.date()))
                    if ret_val == -1:
                        print 'exception occurred, waiting for {0} seconds'.format(self.wait_time)
                        time.sleep(self.wait_time)
                print('{0} {1} {2}'.format(station_number, str(start_date.date()), str(end_date.date())))
                self.num_of_requests += 1
            start_date = start_date + relativedelta(days=+61)
            end_date = end_date + relativedelta(days=+61)

    def aggregate_data(self):
        self.cs.aggregate_data()

    def preprocess_data(self, input_file, output_file):
        self.cs.preprocess_data(input_file, output_file)

    def debug(self):
        print(len(self.cs.station_numbers))

if __name__ == '__main__':
    d = Driver()
    # d.get_hourly_data_start_end(datetime.datetime(2018, 01, 01), datetime.datetime(2018, 12, 31))
    # d.get_N_years_hourly_data(1, datetime.datetime(2018, 01, 01))
    # d.aggregate_data()
    d.preprocess_data('../data/aggregated_hourly_report.csv', '../data/hourly_report.csv')

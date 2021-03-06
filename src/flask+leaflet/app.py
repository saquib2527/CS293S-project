import sys
import pandas as pd
import os
from array import array
from flask import Flask, render_template, jsonify
from flask import request
from Plotter import Plotter


app = Flask(__name__)


@app.route('/')
def index():
	
	return render_template('index.html')


@app.route('/getStationCoords')
def fullmap():
    file = open('stations.csv','r')
    lines = file.read().split('\n')
    stations = []
    # (latitude, longitude) = lines[1].split(',')[5:]
    for r in lines[1:]:
        if not r:
            continue
        data = r.split(',')
        # print(data)
        stations.append({
            'id': data[0],
            'name': data[1],
            'lat': float(data[5]),
            'lng': float(data[6]),
            'infobox': data[1]
            # 'infobox': '%s\nLat: %s\nLong: %s' % (data[1], data[5], data[6])
            })
    return jsonify({"data": stations})

@app.route('/data', methods = ['POST'])
def getData():
    station = request.form.getlist('station[]')
    print(station)
    
    df_list = []
    for file in station:
        df = pd.read_csv('./output/{0}.csv'.format(file))
        df_list.append(df)
    combined_df = pd.concat(df_list, ignore_index=True)
    return render_template('data.html', data=combined_df.to_html())

@app.route('/reg', methods = ['POST'])
def calcReg():
	params = request.form.getlist('params[]')
	p = Plotter()
	# print(params)
	# print(p.get_regression_result(params))
	if(len(params)>0):
		return jsonify(p.get_regression_result(params))
	else:
		return jsonify(p.get_regression_result())
	
@app.route('/getStationET', methods = ['POST'])
def plotStationET():
	station = request.form.getlist('station[]')
	print(station)
	date = request.form.get('date')
	print(date)
	p = Plotter()
	figdata_png = p.plot_hourly_eto_by_station_date(int(station[0]), date)
	return render_template('img.html', result=figdata_png)


if __name__ == '__main__':

    app.run(debug=True)

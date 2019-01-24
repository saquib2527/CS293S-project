# coding: utf-8

from flask import Flask, render_template, request
from flask_googlemaps import GoogleMaps
from flask_googlemaps import Map, icons

app = Flask(__name__, template_folder="templates")

#app.config['GOOGLEMAPS_KEY'] = "AIzaSyDP0GX-Wsui9TSDxtFNj2XuKrh7JBTPCnU"

GoogleMaps(
    app,
    #key="AIzaSyDP0GX-Wsui9TSDxtFNj2XuKrh7JBTPCnU"
)

@app.route('/')
def fullmap():
    file = open('stations.csv','r')
    lines = file.read().split('\n')
    stations = []
    (latitude, longitude) = lines[0].split(',')[5:]
    for r in lines:
        data = r.split(',')
        stations.append({
            'icon': icons.dots.green,
            'lat': float(data[5]),
            'lng': float(data[6]),
            'infobox': 'station# %s, Name: %s, Lat: %s, Long: %s' % (data[0], data[1], data[5], data[6])
            })

    fullmap = Map(
        identifier="fullmap",
        varname="fullmap",
        style=(
            "height:100%;"
            "width:100%;"
            "top:0;"
            "left:0;"
            "position:absolute;"
            "z-index:200;"
        ),
        lat=latitude,
        lng=longitude,
        markers=stations,
        # maptype = "TERRAIN",
        zoom="8"
    )
    return render_template(
        'index.html',
        fullmap=fullmap,
        GOOGLEMAPS_KEY=request.args.get('apikey')
    )

if __name__ == "__main__":
    app.run(debug=True, use_reloader=True)

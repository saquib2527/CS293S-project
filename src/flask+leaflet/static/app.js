BASECOORDS = [36.336222,-120.112910];

function makeMap() {
    var TILE_URL = "https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png";
    var MB_ATTR = 'Map data &copy; <a href="http://openstreetmap.org">OpenStreetMap</a> contributors';
    mymap = L.map('llmap').setView(BASECOORDS, 8);
    L.tileLayer(TILE_URL, {attribution: MB_ATTR}).addTo(mymap);
}

var layer = L.layerGroup();

function renderData() {
    $.getJSON("/getStationCoords", function(obj) {
        var markers = $.map(obj.data, function(val) {
            var marker = L.marker(L.latLng(parseFloat(val.lat), parseFloat(val.lng)));
            marker.stationId = val.id;
            marker.bindPopup(val.infobox, {
                autoClose: false
            });
            marker.on('mouseover',function(ev) {
                marker.openPopup();
            });
            marker.on('mouseout',function(ev) {
                marker.closePopup();
            });
            marker.on('click', function(ev) {
                if($(marker._icon).hasClass("chooseStn")) {
                    $(marker._icon).removeClass("chooseStn");
                    $("#"+marker.stationId).remove();
                }
                else {
                    $(marker._icon).addClass("chooseStn");
                    $("#chosenStns").append($("<span id='"+marker.stationId+"'>"+marker.stationId+"</span>"));
                }
            });

            return marker;
        });
        mymap.removeLayer(layer);
        layer = L.layerGroup(markers);
        mymap.addLayer(layer);
    });
}

function getChosenStns() {
    var stations = [];
    var $chosenStns = $("#chosenStns");
    $chosenStns.children().each(function(i, stn) {
        stations.push(parseInt($(stn).text()));
    });
    return stations
}

$(function() {
    makeMap();
    renderData();
    $('#getData').click(function() {
       var stations = getChosenStns();
       $.ajax({type: 'POST',
               url:'/data', 
               data: {"station": getChosenStns()},
                success: function(stationData) {
                    $('#dataContainer').html(stationData);
                }
        });
    });
})

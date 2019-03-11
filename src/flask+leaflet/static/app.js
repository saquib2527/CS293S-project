BASECOORDS = [36.336222,-120.112910];

function makeMap() {
    var TILE_URL = "https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png";
    var MB_ATTR = 'Map data &copy; <a href="http://openstreetmap.org">OpenStreetMap</a> contributors';
    mymap = L.map('llmap').setView(BASECOORDS, 8);
    L.tileLayer(TILE_URL, {attribution: MB_ATTR}).addTo(mymap);
}

var layer = L.layerGroup();
var markers;

function renderData() {
    $.getJSON("/getStationCoords", function(obj) {
        markers = $.map(obj.data, function(val) {
            var marker = L.marker(L.latLng(parseFloat(val.lat), parseFloat(val.lng)));
            marker.stationId = val.id;
            marker.stationName = val.name;
            marker.stationLat = val.lat;
            marker.stationLong = val.lng;
            marker.bindPopup("<div><p>"+val.infobox+"</p></div><img height='250' width='350' src='static/picssvg/"+val.id+"-2018.svg'/>", {
                autoClose: false,
                maxWidth:350,
                maxHeight:300
            });
            marker.on('mouseover',function(ev) {
                marker.openPopup();
            });
            marker.on('mouseout',function(ev) {
                marker.closePopup();
            });
            marker.on('click', function(ev) {
                $("#stationInfo").hide();
                $("#imgContainer").hide();
                if($(marker._icon).hasClass("chooseStn")) {
                    $(marker._icon).removeClass("chooseStn");
                    $("#"+marker.stationId).remove();
                }
                else {
                    // console.log(markers);
                    for(m in markers) {
                        if($(markers[m]._icon).hasClass("chooseStn")) {
                            $(markers[m]._icon).removeClass("chooseStn");
                        }
                    }
                    $("#chosenStns").empty();
                    $(marker._icon).addClass("chooseStn");
                    $("#chosenStns").append($("<span id='"+marker.stationId+"'>"+marker.stationId+"</span>"));
                    // update station info
                    $("#stnId").text(marker.stationId);
                    $("#stnName").text(marker.stationName);
                    $("#stnLat").text(marker.stationLat);
                    $("#stnLong").text(marker.stationLong);
                }
            });

            return marker;
        });
        mymap.removeLayer(layer);
        layer = L.layerGroup(markers);
        mymap.addLayer(layer);
    });
}

function clearSelection() {
    // remove marker bg
    renderData();
    // clear list
    var $chosenStns = $("#chosenStns");
    $chosenStns.empty();
    // remove data grid
    $("#dataGrid").empty();
    


}

function getChosenStns() {
    var stations = [];
    var $chosenStns = $("#chosenStns");
    $chosenStns.children().each(function(i, stn) {
        stations.push(parseInt($(stn).text()));
    });
    return stations
}


// function addHeatmap() {
//     var testData = {
//       max: 8,
//       data: [{lat: 36.336222, lng:-120.112910, count: 1},{lat: 35.532556, lng:-119.28179, count: 1}]
//     };

//     var cfg = {
//       "radius": 2,
//       "maxOpacity": .8, 
//       "scaleRadius": true, 
//       "useLocalExtrema": true,
//       latField: 'lat',
//       lngField: 'lng',
//       valueField: 'count'
//     };

//     var heatmapLayer = new HeatmapOverlay(cfg);
//     heatmapLayer.setData(testData);
//     mymap.addLayer(heatmapLayer);
//     setTimeout(function(){
//           mymap.removeLayer(heatmapLayer);      
//     }, 5000);


// }


$(function() {
    makeMap();
    renderData();
    $("#datepicker").datepicker({ dateFormat: 'yy-mm-dd', defaultDate: new Date(2019-01-01) });

    $('#getData').click(function() {
       var stations = getChosenStns();
       if (stations.length > 0) {
            $.ajax({type: 'POST',
               url:'/data', 
               data: {"station": getChosenStns()},
                success: function(stationData) {
                    $('#dataGrid').html(stationData);
                }
            });
       }
       
    });
    $('#clearData').click(function() {
        clearSelection();

    });

    $("#calcReg").click(function() {
        var params = [];
        $("#checkBoxList input").each(function() {
            if ($(this).is(':checked'))
                params.push($(this).val());
        });

        if(params.length == 0)
        {
            $("#paramAlert").show();
            return;
        }

        $(this).hide();
        $("#calcRegSpinner").show();

        
        $.ajax({type: 'POST',
           url:'/reg', 
           data: {"params": params},
            success: function(result) {
                $("#paramAlert").hide();
                console.log(result);
                $("#r2").val(result.r2);
                $("#mse").val(result.mse);
                $("#calcRegSpinner").hide();
                $("#calcReg").show();
            }
        });
    });

    $('#paramAlert').on('close.bs.alert', function (event) {
        event.preventDefault();
        $("#paramAlert").hide();
    })

    $("#calcStn").click(function() {
        var stations = getChosenStns();
        selectedDate = $("#datepicker").val();
        if(stations.length == 0 || selectedDate == "") {
            if (stations.length == 0)
                $("#stationAlert").show();
            else
                $("#dateAlert").show();
            return;
        }

        $(this).hide();
        $("#calcStnSpinner").show();
        
        
        if (stations.length > 0 && selectedDate != "") {
            $.ajax({type: 'POST',
               url:'/getStationET', 
               data: {"station": getChosenStns(), date : selectedDate},
                success: function(stationData) {
                    $("#stationInfo").show();
                    $("#stationAlert").hide();
                    $("#dateAlert").hide();
                    $("#imgContainer").html(stationData).show();
                    $("#calcStnSpinner").hide();
                    $("#calcStn").show();
                }
            });
        }
    });

    $('#stationAlert').on('close.bs.alert', function (event) {
        event.preventDefault();
        $("#stationAlert").hide();
    })

    $('#dateAlert').on('close.bs.alert', function (event) {
        event.preventDefault();
        $("#dateAlert").hide();
    })

    

})

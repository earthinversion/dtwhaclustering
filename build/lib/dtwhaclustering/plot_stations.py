"""
dtwhaclustering.plot_stations
----------------------------------
DTW HAC analysis support

:author: Utpal Kumar
:date: 2021/06
:copyright: Copyright 2021 Institute of Earth Sciences, Academia Sinica.
"""

import numpy as np
import pygmt
import pandas as pd
import os


def plot_station_map(station_data, minlon=None, maxlon=None, minlat=None, maxlat=None,
                     outfig='station_map.png', datacolor='blue', topo_data="@earth_relief_15s",
                     cmap='etopo1', projection='M4i',
                     datalabel='Stations', markerstyle="i10p"):
    '''
    Plot topographic station map using PyGMT

    :param station_data: Pandas dataframe containing columns `lon`, `lat`
    :param minlon: Minimum longitude of the map (optional)
    :param maxlon: Maximum longitude of the map (optional)
    :param minlat: Minimum latitude of the map (optional)
    :param maxlat: Maximum latitude of the map (optional)
    :param datacolor: color of the data point to plot
    :param topo_data: etopo data file
    :param cmap: colormap for the output topography
    :param projection: projection of the map. Mercator of width 4 inch by default
    :param datalabel: Label for the data
    :param markerstyle: Style of the marker. Inverted triangle of size 10p by default.
    :param outfig: Output figure path
    '''
    df = pd.read_csv(station_data)
    # print(df.head())

    if minlon is None and maxlon is None and minlat is None and maxlat is None:
        minlon = df['lon'].min()-1
        maxlon = df['lon'].max()+1
        minlat = df['lat'].min()-1
        maxlat = df['lat'].max()+1

    # Visualization
    fig = pygmt.Figure()
    # make color pallets
    pygmt.makecpt(
        cmap=cmap,
        series='-8000/5000/1000',
        continuous=True
    )

    # plot high res topography
    fig.grdimage(
        grid=topo_data,
        region=[minlon, maxlon, minlat, maxlat],
        projection=projection,
        shading=True,
        frame=True
    )

    # plot coastlines
    fig.coast(
        region=[minlon, maxlon, minlat, maxlat],
        projection=projection,
        shorelines=True,
        frame=True
    )
    leftjustify, rightoffset = "TL", "5p/-5p"
    rightjustify, leftoffset = "TR", "-8p/-1p"
    for stn, lon, lat in zip(df["stn"].values, df["lon"].values, df["lat"].values):
        # plot east coast stations in color
        fig.plot(
            x=lon,
            y=lat,
            style=markerstyle,
            color=datacolor,
            pen="black",
        )
        fig.text(
            x=lon,
            y=lat,
            text=stn,
            justify=leftjustify,
            angle=0,
            offset=rightoffset,
            fill="white",
            font=f"6p,Helvetica-Bold,black",
        )

    fig.plot(
        x=np.nan,
        y=np.nan,
        style=markerstyle,
        color=datacolor,
        pen="black",
        label=datalabel
    )

    fig.legend(position="JTR+jTR+o0.2c", box=True)

    fig.savefig(outfig, crop=True, dpi=300)
    print(f"Output figure saved at {outfig}")


if __name__ == '__main__':
    plot_station_map()

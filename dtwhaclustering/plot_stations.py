"""
Plot topographic station map

:author: Utpal Kumar, Institute of Earth Sciences, Academia Sinica
"""

import numpy as np
import pygmt
import pandas as pd
import os


def plot_station_map(station_data, minlon=None, maxlon=None, minlat=None, maxlat=None,
                     outfig='station_map.png', datacolor='blue', topo_data="@earth_relief_15s",
                     cmap='etopo1', projection='M4i',
                     datalabel='Stations', markerstyle="i10p",
                     random_station_label=False, stn_labels=None, justify='left',
                     labelfont="6p,Helvetica-Bold,black", offset="5p/-5p", stn_labels_color='red', rand_justify=False):
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
    :param random_station_label: int. label randomly selected `random_station_label` of stations 

    .. code-block:: python

        from dtwhaclustering import plot_stations

        outloc=os.path.join("Figures","Maps")
        figname = f'{outloc}/station_map.pdf'
        kwargs={
            "labelfont":"10p,Helvetica-Bold,black",
            "outfig":figname,
            "stn_labels":['PKGM','YMSM', 'VR02', 'HUWE', 'LNCH','CHEN','TUNH','ERPN','FALI','WANS'],
            "justify":'right',
            "offset":"-10p/-1p",
            'markerstyle': 'i12p',
            'stn_labels_color': 'red',
            'rand_justify': True
        }
        plot_stations.plot_station_map(station_data = 'helper_files/selected_stations_info.txt',**kwargs)

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

    def get_justify(justify):
        if justify == 'left':
            justifystr = "TL"
        else:
            justifystr = "TR"
        return justifystr

    justifystr = get_justify(justify)

    stnvalues = df["stn"].values
    lonvalues = df["lon"].values
    latvalues = df["lat"].values
    for istn in np.arange(0, len(stnvalues)):
        if stn_labels is not None and stnvalues[istn] in stn_labels:
            color = stn_labels_color
        else:
            color = datacolor
        # plot east coast stations in color
        fig.plot(
            x=lonvalues[istn],
            y=latvalues[istn],
            style=markerstyle,
            color=color,
            pen="black",
        )
    if random_station_label:
        staiter = np.random.randint(
            0, len(stnvalues), size=(random_station_label,))

    else:
        staiter = np.arange(0, len(stnvalues))

    for istn in staiter:
        # print(istn, stnvalues[istn])
        if stn_labels is not None and stnvalues[istn] not in stn_labels:
            continue

        if rand_justify:
            if np.random.rand() > 0.5:
                justifystr = get_justify(justify='left')
                offset = "5p/-5p"
            else:
                justifystr = get_justify(justify='right')
                offset = "-10p/-1p"
        fig.text(
            x=lonvalues[istn],
            y=latvalues[istn],
            text=stnvalues[istn],
            justify=justifystr,
            angle=0,
            offset=offset,
            fill="white",
            font=labelfont,
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

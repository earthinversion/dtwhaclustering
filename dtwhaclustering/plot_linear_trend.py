"""
dtwhaclustering.plot_linear_trend
----------------------------------
DTW HAC analysis support

:author: Utpal Kumar
:date: 2021/06
:copyright: Copyright 2021 Institute of Earth Sciences, Academia Sinica.
"""

import pandas as pd
import numpy as np
from scipy.interpolate import griddata
import xarray as xr
import pygmt


def compute_interpolation(df, method='nearest', lonrange=(120., 122.), latrange=(21.8, 25.6), step=0.01):
    '''
    Interpolate linear trend values using Scipy's griddata and returns xarray object

    :param df: pandas dataframe containing the columns `lon`, `lat` and `slope`
    :param lonrange: minimum and maximum values of the longitude to be interpolated
    :type lonrange: tuple
    :param latrange: minimum and maximum values of the latitude to be interpolated
    :type latrange: tuple
    '''
    coordinates0 = np.column_stack((df['lon'].values, df['lat'].values))
    lonmin, lonmax = lonrange
    latmin, latmax = latrange

    lons = np.arange(lonmin, lonmax, step)
    lats = np.arange(latmin, latmax, step)

    xintrp, yintrp = np.meshgrid(lons, lats)
    z1 = griddata(coordinates0, df['slope'].values,
                  (xintrp, yintrp), method=method)
    xintrp = np.array(xintrp, dtype=np.float32)
    yintrp = np.array(yintrp, dtype=np.float32)

    z2 = z1[~np.isnan(z1)]

    cmapExtreme = np.max([np.abs(z2.min()), np.abs(z2.max())])

    da = xr.DataArray(z1, dims=("lat", "long"), coords={
                      "long": lons, "lat": lats},)

    return da, cmapExtreme


def plot_linear_trend_on_map(df, maplonrange=(120., 122.1), maplatrange=(21.8, 25.6), intrp_lonrange=(120., 122.), intrp_latrange=(21.8, 25.6), outfig="Maps/slope-plot.png", frame=["a1f0.25", "WSen"], cmap='jet', step=0.01):
    '''
    Plot the interpolated linear trend values along with the original data points on a geographical map using PyGMT

    :param df: Pandas dataframe containing the columns `lon`, `lat` and `slope`
    :param maplonrange: longitude min/max of the output map
    :param maplatrange: latitude min/max of the output map
    :param intrp_lonrange: longitude min/max for the interpolation of data
    :param intrp_latrange: latitude min/max for the interpolation of data
    :param step: resolution of the interpolation
    :param cmap: colormap for the output map
    :param frame: frame of the output map. See PyGMT docs for details
    :param outfig: output figure name with extension, e.g., `slope-plot.png`
    '''
    da, cmapExtreme = compute_interpolation(
        df, lonrange=intrp_lonrange, latrange=intrp_latrange, step=step)

    minlon, maxlon = maplonrange
    minlat, maxlat = maplatrange

    # Visualization
    fig = pygmt.Figure()

    pygmt.makecpt(
        cmap=cmap,
        series=f'{-cmapExtreme}/{cmapExtreme}/{step}',
        #     series='0/5000/100',
        continuous=True
    )

    # #plot high res topography
    fig.grdimage(
        region=[minlon, maxlon, minlat, maxlat],
        grid=da,
        projection='M4i',
        interpolation='l'
    )

    # plot coastlines
    fig.coast(
        region=[minlon, maxlon, minlat, maxlat],
        shorelines=True,
        water="#add8e6",
        frame=frame,
        area_thresh=1000
    )

    pygmt.makecpt(
        cmap='jet',
        series=f'{-cmapExtreme}/{cmapExtreme}/{step}',
        #     series='0/5000/100',
        continuous=True
    )

    # plot data points
    fig.plot(
        x=df['lon'].values,
        y=df['lat'].values,
        style='i0.2i',
        color=df['slope'].values,
        cmap=True,
        pen='black',
    )

    # Plot colorbar
    # Default is horizontal colorbar
    fig.colorbar(
        frame='+l"Linear Trend (mm)"'
    )

    # save figure as pdf
    fig.savefig(f"{outfig}", crop=True, dpi=300)

    print(f"Figure saved at {outfig}")


if __name__ == '__main__':
    pass

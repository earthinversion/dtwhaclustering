"""

Plot linear trend values on a geographical map

:author: Utpal Kumar, Institute of Earth Sciences, Academia Sinica
"""

import pandas as pd
import numpy as np
from scipy.interpolate import griddata
import xarray as xr
import pygmt
from pathlib import Path
import os


def compute_interpolation(df, method='nearest', lonrange=(120., 122.), latrange=(21.8, 25.6), step=0.01):
    '''
    Interpolate linear trend values using Scipy's griddata and returns xarray object

    :param df: pandas dataframe containing the columns `lon`, `lat` and `slope`
    :type df: pandas.DataFrame
    :param method: Method of interpolation. One of {'linear', 'nearest', 'cubic'}. For more, see `scipy.interpolate.griddata`
    :type method: str
    :param lonrange: minimum and maximum values of the longitude to be interpolated
    :type lonrange: tuple
    :param latrange: minimum and maximum values of the latitude to be interpolated
    :type latrange: tuple
    :param step: stepsize to interpolate data spatially
    :type step: float
    :return: `xarray.DataArray` of dims, and coords ("lat", "long"), maximum of the absolute interpolated array values
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


def plot_linear_trend_on_map(df, maplonrange=(120., 122.1), maplatrange=(21.8, 25.6), intrp_lonrange=(120., 122.),
                             intrp_latrange=(21.8, 25.6), outfig="Maps/slope-plot.png", frame=["a1f0.25", "WSen"],
                             cmap='jet', step=0.01, stn_labels=None, justify='left',
                             labelfont="6p,Helvetica-Bold,black", offset="5p/-5p", markerstyle="i10p", defpen='1p,black',
                             stn_labels_color='black', rand_justify=False, water_color='skyblue'):
    '''
    Plot the interpolated linear trend values along with the original data points on a geographical map using PyGMT

    :param df: Pandas dataframe containing the columns `lon`, `lat` and `slope`
    :type df: pandas.DataFrame
    :param maplonrange: longitude min/max of the output map
    :type maplonrange: tuple
    :param maplatrange: latitude min/max of the output map
    :type maplatrange: tuple
    :param intrp_lonrange: longitude min/max for the interpolation of data
    :type intrp_lonrange: tuple
    :param intrp_latrange: latitude min/max for the interpolation of data
    :type intrp_latrange: tuple
    :param step: resolution of the interpolation
    :type step: float
    :param cmap: colormap for the output map
    :type cmap: str
    :param frame: frame of the output map. See PyGMT docs for details
    :type frame: list
    :param outfig: output figure name with extension, e.g., `slope-plot.png`
    :type outfig: str
    :param water_color: color of the water, default: skyblue
    :type water_color: str
    :param stn_labels: label the station names. Only the stations provided will be labeled
    :type stn_labels: array
    :param justify: justification of the station labels - 'left', 'right'
    :type justify: str
    :param rand_justify: randomly decide the justification for each labels
    :type rand_justify: boolean
    :param markerstyle: marker style and size
    :type markerstyle: str
    :return: None


    .. code-block:: python

        kwargs={
            "labelfont":"16p,Helvetica-Bold,black",
            "justify":'right',
            "offset":"-10p/-1p",
            "maplonrange":(119.8, 122.1), 
            "maplatrange":(21.8, 25.6),
            'markerstyle': 'i16p',
            'rand_justify':True,
            'water_color': 'gray'
        }
        comp = "U"
        all_labels = {}
        all_labels[comp] = ['PKGM','YMSM', 'VR02', 'HUWE', 'LNCH','CHEN','TUNH']

        slopeFile=f'stn_slope_res_{comp}.txt'
        df = pd.read_csv(slopeFile, names=['stn','lon','lat','slope'], delimiter='\s+')
        figname = f"{outloc}/slope-plot_{comp}.png"
        plot_linear_trend_on_map(df, outfig=figname,stn_labels=all_labels[comp], **kwargs)
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
        water=water_color,
        frame=frame,
        area_thresh=1000
    )

    pygmt.makecpt(
        cmap='jet',
        series=f'{-cmapExtreme}/{cmapExtreme}/{step}',
        #     series='0/5000/100',
        continuous=True
    )
    stnvalues = df["stn"].values
    lonvalues = df["lon"].values
    latvalues = df["lat"].values

    # plot data points
    fig.plot(
        x=df['lon'].values,
        y=df['lat'].values,
        style=markerstyle,
        color=df['slope'].values,
        cmap=True,
        pen=defpen
    )

    def get_justify(justify):
        if justify == 'left':
            justifystr = "TL"
        else:
            justifystr = "TR"
        return justifystr

    justifystr = get_justify(justify)

    if stn_labels:
        for istn in np.arange(0, len(stnvalues)):
            # print(istn, stnvalues[istn])
            if stnvalues[istn] not in stn_labels:
                continue
            else:
                # plot east coast stations in color
                fig.plot(
                    x=lonvalues[istn],
                    y=latvalues[istn],
                    style=markerstyle,
                    color=stn_labels_color,
                    pen=defpen,
                )
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

    # Plot colorbar
    # Default is horizontal colorbar
    fig.colorbar(
        frame='+l"Linear Trend (mm)"'
    )

    # save figure as pdf
    fig.savefig(f"{outfig}", crop=True)
    if not ".pdf" in outfig:
        dirname = os.path.dirname(outfig)
        outfig0 = Path(outfig).stem
        outfigpdf = os.path.join(dirname, f"{outfig0}.pdf")
        fig.savefig(outfigpdf, crop=True)

    # print(f"Figure saved at {outfig}")


if __name__ == '__main__':
    pass

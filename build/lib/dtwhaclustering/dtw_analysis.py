"""
Classes and functions for the DTW analysis and plotting maps and figures (`dtw_analysis`)
This module is built around the dtaidistance package for the DTW computation and scipy.cluster

:author: Utpal Kumar, Institute of Earth Sciences, Academia Sinica
:note: See `dtaidistance <https://dtaidistance.readthedocs.io/en/latest/usage/dtw.html>`_ for details on HierarchicalTree, dtw, dtw_visualisation
"""

from dtaidistance.clustering import HierarchicalTree
import pandas as pd
import matplotlib
from dtaidistance import dtw
from dtaidistance import dtw_visualisation as dtwvis
import numpy as np
import matplotlib.pyplot as plt
from scipy.cluster import hierarchy
from scipy.cluster.hierarchy import fcluster
import pygmt
from scipy.interpolate import griddata
import xarray as xr
import seaborn as sns
import tqdm
import sys
import os
import concurrent.futures


# from dtaidistance.clustering.hierarchical import HierarchicalTree
# from dtaidistance import clustering

fontsize = 26
fontsize0 = 20
plt.rc('font', size=fontsize)  # controls default text size
plt.rc('axes', titlesize=fontsize)  # fontsize of the title
plt.rc('axes', labelsize=fontsize)  # fontsize of the x and y labels
plt.rc('xtick', labelsize=fontsize-10)  # fontsize of the x tick labels
plt.rc('ytick', labelsize=fontsize-10)  # fontsize of the y tick labels
plt.rc('legend', fontsize=fontsize)  # fontsize of the legend


# matplotlib.rc('font', **font)
plt.rcParams["figure.figsize"] = (12, 6)
# plt.style.use('seaborn')
plt.style.use('default')

# to edit text in Illustrator
matplotlib.rcParams['pdf.fonttype'] = 42


kwargs_default = {
    'fontsize': fontsize,
    'figsize': (10, 8)
}


class dtw_signal_pairs:
    def __init__(self, s1, s2, labels=['s1', 's2']):
        '''
        Analyze the DTW between a pair of signal

        :param s1: signal 1
        :type s1: array
        :param s2: signal 2
        :type s2: array
        :param labels: signal labels
        :type labels: array of strings
        '''
        self.s1 = np.array(s1, dtype=np.double)
        self.s2 = np.array(s2, dtype=np.double)
        if (self.s1.shape[0] == 0) or (self.s2.shape[0] == 0):
            raise ValueError(
                f"Empty time series array {self.s1.shape[0]} & {self.s2.shape[0]}")
        self.labels = labels

    def plot_signals(self, figname=None, figsize=(12, 6)):
        '''
        Plot the signals

        :param figname: output figure name
        :type figname: str
        :param figsize: output figure size
        :type figsize: tuple
        :return: figure and axes object
        '''
        fig, ax = plt.subplots(2, 1, figsize=figsize)
        ax[0].plot(self.s1, color="C0", lw=1)
        ax[0].set_ylabel(self.labels[0], fontsize=kwargs_default['fontsize'])

        ax[1].plot(self.s2, color="C1", lw=1)
        ax[1].set_ylabel(self.labels[1], fontsize=kwargs_default['fontsize'])
        fig.align_ylabels(ax)
        if figname:
            plt.savefig(figname, bbox_inches='tight', dpi=300)
            plt.close()
            fig, ax = None, None

        return fig, ax

    def compute_distance(self, pruning=True, best_path=False):
        '''
        Returns the DTW distance

        :param pruning: prunes computations by setting max_dist to the Euclidean upper bound
        :type pruning: boolean
        '''
        if not best_path:
            if pruning:
                # prunes computations by setting max_dist to the Euclidean upper bound
                distance = dtw.distance_fast(
                    self.s1, self.s2, use_pruning=True)
            else:
                distance = dtw.distance(self.s1, self.s2)
        else:
            _, path = dtw.warping_paths(
                self.s1, self.s2, window=None, psi=None)
            best_path = dtw.best_path(path)
            distance = path[best_path[-1][0], best_path[-1][1]]

        return distance

    def compute_warping_path(self, windowfrac=None, psi=None, fullmatrix=False):
        '''
        Returns the DTW path

        :param windowfrac: Fraction of the signal length. Only allow for shifts up to this amount away from the two diagonals.
        :param psi: Up to psi number of start and end points of a sequence can be ignored if this would lead to a lower distance
        :param full_matrix: The full matrix of all warping paths (or accumulated cost matrix) is built
        '''
        if fullmatrix:
            if windowfrac:
                window = int(windowfrac * np.min([len(self.s1), len(self.s2)]))
            else:
                window = None
            d, path = dtw.warping_paths(
                self.s1, self.s2, window=window, psi=psi)
        else:
            path = dtw.warping_path(self.s1, self.s2)

        return path

    def plot_warping_path(self, figname=None, figsize=(12, 6)):
        '''
        Plot the signals with the warping paths

        :param figname: output figure name
        :type figname: str
        :param figsize: output figure size
        :type figsize: tuple
        '''

        distance = self.compute_distance()

        fig, ax = dtwvis.plot_warping(
            self.s1, self.s2, self.compute_warping_path())
        ax[0].set_ylabel(self.labels[0], fontsize=kwargs_default['fontsize'])
        ax[1].set_ylabel(self.labels[1], fontsize=kwargs_default['fontsize'])
        fig.set_figwidth(figsize[0])
        fig.set_figheight(figsize[1])
        fig.align_ylabels(ax)
        if figname:
            plt.savefig(figname, bbox_inches='tight')
            plt.close()
            fig, ax = None, None
        return distance, fig, ax

    def plot_matrix(self, windowfrac=0.2, psi=None, figname=None, shownumbers=False, showlegend=True):
        '''
        Plot the signals with the DTW matrix

        :param figname: output figure name
        :type figname: str
        '''
        path = self.compute_warping_path(
            windowfrac=windowfrac, psi=psi, fullmatrix=True)

        fig, ax = dtwvis.plot_warpingpaths(
            self.s1, self.s2, path, shownumbers=shownumbers, showlegend=showlegend)
        ax[0].set_ylabel(self.labels[0], fontsize=kwargs_default['fontsize'])
        ax[1].set_ylabel(self.labels[1], fontsize=kwargs_default['fontsize'])
        if figname:
            plt.savefig(figname, bbox_inches='tight')
            plt.close()
            fig, ax = None, None
        return fig, ax


def plot_signals(matrix, labels=[], figname=None, plotpdf=True, figsize=kwargs_default['figsize'], ylabelsize=kwargs_default['fontsize'], color=None):
    '''
    Plot signals

    :param color: list of colors. If None then matplotlib defaults color sequence will be used
    :param figname: output figure name
    :type figname: str
    :param figsize: output figure size
    :type figsize: tuple
    '''
    if color is None:
        colors = [f"C{i}" for i in range(matrix.shape[0])]
    else:
        colors = color

    fig, ax = plt.subplots(
        nrows=matrix.shape[0], sharex=True, figsize=figsize)
    if len(labels) == 0:
        _labels = []
        for i in range(matrix.shape[0]):
            ax[i].plot(matrix[i, :], color=colors[i])
            lab = f"S{i+1}"
            ax[i].set_ylabel(lab, fontsize=ylabelsize)
            _labels.append(lab)
    if len(labels) == matrix.shape[0]:
        try:
            for i in range(matrix.shape[0]):
                ax[i].plot(matrix[i, :], color=colors[i])
                ax[i].set_ylabel(labels[i], fontsize=ylabelsize)

        except Exception as e:
            print(e)
    else:
        labels = _labels
        for iaxx, axx in enumerate(ax):
            axx.set_ylabel(f"{labels[iaxx]}",
                           fontsize=ylabelsize)
    fig.align_ylabels(ax)
    if figname:
        plt.savefig(figname, bbox_inches='tight')
        if plotpdf:
            figname_pdf = ".".join(figname.split(".")[:-1])+".pdf"
            ax.figure.savefig(figname_pdf,
                              bbox_inches='tight')
        plt.close()
        fig, ax = None, None

    return fig, ax


def plot_cluster(lons, lats, figname=None, figsize=(10, 10), plotpdf=True, labels=[], markersize=20):
    '''
    :param figname: output figure name
    :type figname: str
    :param figsize: output figure size
    :type figsize: tuple
    '''
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    if len(labels) == 0:
        clusterIdx = 0
        for ilonlat, (loncluster, latcluster) in enumerate(zip(lons, lats)):
            lab = f'S{clusterIdx+1}'
            if lab in labels:
                ax.plot(loncluster, latcluster, 'o',
                        color=f"C{clusterIdx}", ms=markersize)
            else:
                ax.plot(loncluster, latcluster, 'o',
                        color=f"C{clusterIdx}", ms=markersize, label=lab)
                labels.append(lab)
            if (ilonlat+1) % 3 == 0:
                clusterIdx += 1
    else:
        clusterIdx = 0
        for ilonlat, (loncluster, latcluster) in enumerate(zip(lons, lats)):
            lab = f'Cluster {clusterIdx+1}'
            if lab in labels:
                ax.plot(loncluster, latcluster, 'o',
                        color=f"C{clusterIdx}", ms=markersize)
            else:
                ax.plot(loncluster, latcluster, 'o',
                        color=f"C{clusterIdx}", ms=markersize, label=lab)
                labels.append(lab)

    plt.legend(fontsize=kwargs_default['fontsize'], bbox_to_anchor=(
        1.05, 1), loc='upper left')
    plt.xticks(fontsize=kwargs_default['fontsize'])
    plt.yticks(fontsize=kwargs_default['fontsize'])
    if figname:
        plt.savefig(figname, bbox_inches='tight')
        if plotpdf:
            figname_pdf = ".".join(figname.split(".")[:-1])+".pdf"
            ax.figure.savefig(figname_pdf,
                              bbox_inches='tight')
        plt.close()
        fig, ax = None, None
    return fig, ax


def shuffle_signals(matrix, labels=[], plot_signals=False, figsize=(10, 12), figname=None, plotpdf=True):
    '''
    :param figname: output figure name
    :type figname: str
    :param figsize: output figure size
    :type figsize: tuple
    '''
    ind_rand_perm = np.random.permutation(matrix.shape[0])
    shuffled_matrix = matrix[ind_rand_perm, :]

    if plot_signals:
        fig, ax = None, None
        fig, ax = plt.subplots(
            nrows=matrix.shape[0], sharex=True, figsize=figsize)
        for i, randidx in enumerate(ind_rand_perm):
            ax[i].plot(shuffled_matrix[i, :], color=f"C{i}")
            ax[i].set_ylabel(
                f"S{randidx}", fontsize=kwargs_default['fontsize'])
        fig.align_ylabels(ax)
        if figname:
            plt.savefig(figname, bbox_inches='tight')
            if plotpdf:
                figname_pdf = ".".join(figname.split(".")[:-1])+".pdf"
                ax.figure.savefig(figname_pdf,
                                  bbox_inches='tight')
            plt.close()

        return ind_rand_perm, shuffled_matrix, fig, ax
    return ind_rand_perm, shuffled_matrix


def noise_robustness_test(clean_df, scale=0.1):

    def addnoise(df, scale=0.1):
        df = df.copy()
        for i in range(df.shape[1]):
            df.iloc[:, i] = np.random.normal(loc=df.iloc[:, i], scale=scale)
        return df

    # print(clean_df.head(1))
    noisydf = addnoise(clean_df, scale=scale)
    # print(noisydf.head(1))
    time_series = clean_df.values.transpose()
    model1 = HierarchicalTree(
        dists_fun=dtw.distance_matrix_fast, show_progress=False, dists_options={'window': int(0.25*time_series.shape[0])})

    model1.fit(time_series)
    linkage_matrix = model1.linkage
    lidxs_orig, ridxs_orig = [], []
    for lidx, ridx, _, _ in linkage_matrix:
        lidxs_orig.append(lidx)
        # ridxs_orig.append(ridx)

    time_series_noisy = noisydf.values.transpose()
    model2 = HierarchicalTree(
        dists_fun=dtw.distance_matrix_fast, show_progress=False, dists_options={'window': int(0.25*time_series_noisy.shape[0])})
    model2.fit(time_series_noisy)
    linkage_matrix2 = model2.linkage
    lidxs_noisy, ridxs_noisy = [], []
    for lidx, ridx, _, _ in linkage_matrix2:
        lidxs_noisy.append(lidx)
        # ridxs_noisy.append(ridx)

    lidxs_diff = set(lidxs_noisy) - set(lidxs_orig)
    # ridxs_diff = set(ridxs_noisy) - set(ridxs_orig)
    percent_change_in_dendro = (len(lidxs_diff)/clean_df.shape[1])*100
    return percent_change_in_dendro


class dtw_clustering:
    def __init__(self, matrix, labels=[], longitudes=[], latitudes=[]):
        '''
        :param matrix: matrix of type numpy array, shaped with row as different signals, and column as the values
        :param labels: Labels for each signals in the matrix
        :param longitudes: geographical x location of the signals in the matrix
        :param latitudes: geographical y location of the signals in the matrix
        '''
        self.matrix = matrix
        self.shuffled_matrix = []
        self.labels = labels
        self.linkage_matrix = None
        if len(longitudes) > 0 and len(latitudes) > 0 and len(longitudes) == len(latitudes) == self.matrix.shape[0]:
            self.longitudes = longitudes
            self.latitudes = latitudes
        else:
            self.longitudes = None
            self.latitudes = None

    def plot_signals(self, figname=None, figsize=(10, 6), fontsize=kwargs_default['fontsize']):
        '''
        :param figname: output figure name
        :type figname: str
        :param figsize: output figure size
        :type figsize: tuple
        '''
        labels = self.labels
        fig, ax = plt.subplots(
            nrows=self.matrix.shape[0], sharex=True, figsize=figsize)
        for i in range(self.matrix.shape[0]):
            ax[i].plot(self.matrix[i, :], color=f"C{i}")
            if len(labels) == 0:
                ax[i].set_ylabel(f"S{i}", fontsize=fontsize)
        if len(labels):
            try:
                for iaxx, axx in enumerate(ax):
                    axx.set_ylabel(f"{labels[iaxx]}", fontsize=fontsize)
            except Exception as e:
                print(e)
                for i in range(self.matrix.shape[0]):
                    if len(labels) == 0:
                        ax[i].set_ylabel(f"S{i}", fontsize=fontsize)
        if figname:
            plt.savefig(figname, bbox_inches='tight')
            plt.close()
            fig, ax = None, None
        return fig, ax

    def plot_cluster_xymap(self, dtw_distance="optimal", figname=None, xlabel='x', ylabel='y', colorbar=True,
                           colorbarstep=1, scale=2, fontsize=kwargs_default['fontsize'], markersize=12, axesfontsize=20,
                           xtickstep=1, tickfontsize=20, edgecolors='k', cmap='jet', linewidths=2, cbarsize=18):
        '''
        Plot the cluster points in a rectangular coordinate system

        :param dtw_distance: use `dtw_distance` value to obtain the cluster division. If `optimal` then the optimal `dtw_distance` will be calculated
        :type dtw_distance: str or float
        :param figname: output figure name
        :type figname: str
        :param colorbar: plot colorbar
        :type colorbar: bool
        :param colorbarstep: step for the colorbar
        :type colorbarstep: int
        :param scale: figure scale
        :type scale: int
        :param markersize: cluster points size
        :param edgecolors: marker edge color
        :param cmap: colormap for the cluster points
        :type cmap: str

        :return: figure, axes
        '''

        kwargs = {
            'edgecolors': edgecolors,
            's': 10*markersize,
            'cmap': cmap,
            'linewidths': linewidths,
        }

        if self.longitudes is not None:
            Z = self.get_linkage()
            if dtw_distance == "optimal":
                max_d, dtw_distance = self.optimum_cluster_elbow()
            clusters = fcluster(Z, dtw_distance, criterion='distance')
            cluster_ticks = list(set(clusters))
            lenx = scale*int(self.longitudes.max()-self.longitudes.min())
            leny = scale*int(self.latitudes.max()-self.latitudes.min())
            figsize = (lenx, leny)

            fig, ax = plt.subplots(figsize=figsize)
            # plot points with cluster dependent colors

            cax = ax.scatter(self.longitudes, self.latitudes,
                             c=clusters, **kwargs)

            ax.set_xlabel(xlabel, fontsize=axesfontsize)
            ax.set_ylabel(ylabel, fontsize=axesfontsize)
            ax.set_xticklabels(np.arange(int(self.longitudes.min(
            ))-1, int(self.longitudes.max())+1, xtickstep), fontsize=tickfontsize)
            ax.set_yticklabels(np.arange(int(self.latitudes.min(
            ))-1, int(self.latitudes.max())+1, xtickstep), fontsize=tickfontsize)
            if colorbar:
                cbar = fig.colorbar(
                    cax, ticks=cluster_ticks[::colorbarstep])
                cbar.ax.tick_params(labelsize=cbarsize)
                cbar.set_label('Clusters', fontsize=fontsize)
            plt.subplots_adjust(wspace=0.01)

            if figname:
                plt.savefig(figname, bbox_inches='tight')
                plt.close()
                fig, ax = None, None
            return fig, ax
        else:
            print("Input the x and y coords")

    def plot_cluster_geomap(self,
                            dtw_distance="optimal",
                            minlon=None,
                            maxlon=None,
                            minlat=None,
                            maxlat=None,
                            figname="dtw_cluster.pdf",
                            colorbar=True,
                            colorbarstep=1,
                            doffset=1,
                            dpi=720,
                            topo_data='@earth_relief_15s',
                            plot_topo=False,
                            markerstyle='c0.3c',
                            cmap_topo='topo',
                            cmap_data='jet',
                            projection="M4i",
                            topo_cpt_range='-8000/8000/1000',
                            landcolor="#666666"
                            ):
        '''
        Plot the cluster points on a geographical map

        :param figname: output figure name
        :type figname: str
        :param colorbar: plot colorbar
        :type colorbar: bool
        :param colorbarstep: step for the colorbar
        :type colorbarstep: int
        :param cmap: colormap for the cluster points
        :type cmap: str
        :param projection: map projection
        :type projection: str
        :param topo_data: topographic data resolution, see `pygmt docs <https://docs.generic-mapping-tools.org/latest/datasets/remote-data.html#global-earth-relief-grids>`_ for details
        :type topo_data: str
        :param topo_cpt_range: min/max/step for topographic color
        :param landcolor: color for land region
        :type landcolor: str
        :param dpi: output figure resolution
        :type dpi: int
        '''
        if self.longitudes is not None:
            Z = self.get_linkage()
            if dtw_distance == "optimal":
                max_d, dtw_distance = self.optimum_cluster_elbow()
            clusters = fcluster(Z, dtw_distance, criterion='distance')
            cluster_ticks = list(set(clusters))

            if minlon is None and maxlon is None and minlat is None and maxlat is None:
                minlon = self.longitudes.min()-doffset
                maxlon = self.longitudes.max()+doffset
                minlat = self.latitudes.min()-doffset
                maxlat = self.latitudes.max()+doffset
            fig = pygmt.Figure()
            # Plot the earth relief grid on Cylindrical Stereographic projection, masking land areas
            fig.basemap(
                region=[minlon, maxlon, minlat, maxlat], projection=projection, frame=True)
            if plot_topo:
                pygmt.makecpt(
                    cmap=cmap_topo,
                    series=topo_cpt_range,
                    continuous=True
                )
                fig.grdimage(
                    grid=topo_data,
                    region=[minlon, maxlon, minlat, maxlat],
                    shading=True,
                    frame=True
                )
                fig.coast(
                    region=[minlon, maxlon, minlat, maxlat],
                    projection=projection,
                    shorelines=True,
                    frame=True
                )
            else:
                fig.coast(land=landcolor, shorelines=True)
            # Plot the sampled bathymetry points using circles (c) of 0.15 cm
            # Points are colored using elevation values (normalized for visual purposes)
            pygmt.makecpt(
                cmap=cmap_data,
                series=f'{min(clusters)}/{max(clusters)}/{colorbarstep}',
                continuous=True
            )
            fig.plot(
                x=self.longitudes,
                y=self.latitudes,
                style=markerstyle,
                pen='black',
                cmap=True,
                color=clusters,
            )
            if colorbar:
                # Plot colorbar
                fig.colorbar(
                    frame='+l"Clusters"'
                )
            # fig.show()
            fig.savefig(figname, crop=True, dpi=dpi)
        else:
            print("Input the x and y coords")

    def _compute_interpolation(self, clusters, lonrange=(120., 122.), latrange=(21.8, 25.6), gridstep=0.01):
        '''
        Compute the spatial interpolation of cluster points for plotting
        '''
        coordinates0 = np.column_stack((self.longitudes, self.latitudes))
        lonmin, lonmax = lonrange
        latmin, latmax = latrange
        step = gridstep
        lons = np.arange(lonmin, lonmax, step)
        lats = np.arange(latmin, latmax, step)

        xintrp, yintrp = np.meshgrid(lons, lats)
        z1 = griddata(coordinates0, clusters,
                      (xintrp, yintrp), method='nearest')
        xintrp = np.array(xintrp, dtype=np.float32)
        yintrp = np.array(yintrp, dtype=np.float32)

        z2 = z1[~np.isnan(z1)]

        cmapminmax = [np.abs(z2.min()), np.abs(z2.max())]

        da = xr.DataArray(z1, dims=("lat", "long"), coords={
            "long": lons, "lat": lats},)

        return da, cmapminmax

    def plot_cluster_geomap_interpolated(self, dtw_distance="optimal",
                                         lonrange=(120., 122.),
                                         latrange=(21.8, 25.6),
                                         gridstep=0.01,
                                         figname="dtw_cluster_interp.pdf",
                                         minlon=None,
                                         maxlon=None,
                                         minlat=None,
                                         maxlat=None,
                                         markerstyle='c0.3c',
                                         dpi=720,
                                         doffset=1,
                                         plot_data=True,
                                         plot_intrp=True):
        '''
        :param dtw_distance: use `dtw_distance` value to obtain the cluster division. If `optimal` then the optimal `dtw_distance` will be calculated
        :type dtw_distance: str or float
        :param lonrange: minimum and maximum of longitude values for interpolation
        :type lonrange: tuple
        :param latrange: minimum and maximum of latitude values for interpolation
        :type latrange: tuple
        :param gridstep: step size for interpolation
        :type gridstep: float
        '''
        Z = self.get_linkage()
        if dtw_distance == "optimal":
            max_d, dtw_distance = self.optimum_cluster_elbow()
        clusters = fcluster(Z, dtw_distance, criterion='distance')
        da, cmapminmax = self._compute_interpolation(clusters,
                                                     lonrange=lonrange, latrange=latrange, gridstep=gridstep)

        if minlon is None and maxlon is None and minlat is None and maxlat is None:
            minlon = self.longitudes.min()-doffset
            maxlon = self.longitudes.max()+doffset
            minlat = self.latitudes.min()-doffset
            maxlat = self.latitudes.max()+doffset

        frame = ["a1f0.25", "WSen"]
        # Visualization
        fig = pygmt.Figure()
        if plot_intrp:
            pygmt.makecpt(
                cmap='jet',
                series=f'{cmapminmax[0]}/{cmapminmax[1]}/1',
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
        if plot_data:
            pygmt.makecpt(
                cmap='jet',
                series=np.asarray(
                    [cmapminmax[0], cmapminmax[1], 1], dtype=int),

                continuous=True
            )

            fig.plot(
                x=self.longitudes,
                y=self.latitudes,
                style=markerstyle,
                pen='black',
                cmap=True,
                color=clusters,
            )

        # Plot colorbar
        # Default is horizontal colorbar
        fig.colorbar(
            frame=["x+lClusters"],
            position="JMR+o1c/0c",

        )

        # save figure as pdf
        fig.savefig(f"{figname}", crop=True, dpi=dpi)

    def compute_distance_accl(self, clusterMatrix=None):
        if clusterMatrix is None:
            linkage_matrix = self.get_linkage()
        else:
            linkage_matrix = self.get_linkage(clusterMatrix=clusterMatrix)

        dist_sort, lchild, rchild = [], [], []
        for lnk in linkage_matrix:
            left_child = lnk[0]
            right_child = lnk[1]
            dist_left_right = lnk[2]
            lchild.append(left_child)
            rchild.append(right_child)
            dist_sort.append(dist_left_right)

        dist_sort = np.array(dist_sort)
        dist_sort_id = np.argsort(dist_sort)
        # print(dist_sort)
        df_cv = pd.DataFrame()
        df_cv["level"] = np.arange(dist_sort.shape[0], 0, -1)
        df_cv["distance"] = dist_sort
        accl = df_cv["distance"].diff().diff()  # double derivative

        df_accl = pd.DataFrame()
        df_accl["level"] = np.arange(dist_sort.shape[0]+1, 1, -1)
        df_accl["accln"] = accl
        return df_cv, df_accl

    def optimum_cluster_elbow(self, minmax=False, plotloc=False):
        '''
        Gives the optimum number of clusters required to express the maximum difference in the similarity using the elbow method
        '''

        df_cv, df_accl = self.compute_distance_accl()
        idx = df_accl['accln'].argmax()
        opt_cluster = df_accl.loc[idx, 'level']
        df_cv_subset = df_cv[df_cv['level'] == opt_cluster]
        opt_distance = df_cv_subset['distance'].values[0]
        if plotloc:
            df_cv_subset2 = df_cv[df_cv['level'] == opt_cluster-1]
            opt_distance_next = df_cv_subset2['distance'].values[0]
            optdist_plotloc = opt_distance+(opt_distance_next-opt_distance)/2
            return opt_cluster, opt_distance, optdist_plotloc
        if minmax:
            mindist, maxdist = df_cv["distance"].min(), df_cv["distance"].max()
            return opt_cluster, opt_distance, (mindist, maxdist)

        return opt_cluster, opt_distance

    def plot_optimum_cluster(self,
                             max_d=None,
                             figname=None,
                             figsize=(10, 6),
                             plotpdf=True,
                             xlabel="Number of Clusters",
                             ylabel="Distance",
                             accl_label='Curvature',
                             accl_color="C10",
                             dist_label="DTW Distance",
                             dist_color="k",
                             xlim=None,
                             legend_outside=True,
                             fontsize=kwargs_default['fontsize'],
                             xlabelfont=kwargs_default['fontsize'],
                             ylabelfont=kwargs_default['fontsize']):
        '''
        :param xlim: x limits of the plot e.g., [1,2]
        :type xlim: list
        '''
        df_cv, df_accl = self.compute_distance_accl()

        fig, ax = plt.subplots(figsize=figsize)
        ax.plot(df_cv["level"], df_cv["distance"],
                "o-", color=dist_color, label=dist_label)

        ax.plot(
            df_accl["level"],
            df_accl["accln"],
            "o-",
            color=accl_color,
            label=accl_label,
        )

        if max_d is None:
            max_d, opt_distance = self.optimum_cluster_elbow()

        ax.axvline(x=max_d, label="Optimum cluster", ls="--")

        ax.set_xlabel(xlabel, fontsize=xlabelfont)
        ax.set_ylabel(ylabel, fontsize=ylabelfont)
        plt.subplots_adjust(wspace=0.01)
        fig.align_xlabels()

        if legend_outside:
            plt.legend(fontsize=fontsize, bbox_to_anchor=(
                1.05, 1), loc='upper left')
        else:
            plt.legend(fontsize=fontsize)

        plt.xticks(np.arange(df_cv["level"].min()-1, df_cv["level"].max()+1, 1
                             ).astype(int), fontsize=xlabelfont)
        plt.yticks(fontsize=ylabelfont)
        if xlim is not None:
            plt.xlim(xlim)

        if figname:
            plt.savefig(figname, bbox_inches='tight')
            if plotpdf:
                figname_pdf = ".".join(figname.split(".")[:-1])+".pdf"
                ax.figure.savefig(figname_pdf,
                                  bbox_inches='tight')
            plt.close()
            fig, ax = None, None
        return fig, ax

    def compute_distance_matrix(self, compact=True, block=None):
        ds = dtw.distance_matrix_fast(
            self.matrix, compact=compact, block=block)
        return ds

    def compute_cluster(self, clusterMatrix=None, window='constrained', windowfrac=0.25):
        '''
        :param compute_cluster: data matrix to cluster
        :return: model, cluster_idx
        '''
        window = int(windowfrac*self.matrix.shape[0])
        if clusterMatrix is None:
            clusterMatrix = self.matrix
        if window == "constrained":
            model = HierarchicalTree(
                dists_fun=dtw.distance_matrix_fast, dists_options={'window': window}, show_progress=True)
        else:
            model = HierarchicalTree(
                dists_fun=dtw.distance_matrix_fast, dists_options={}, show_progress=True)
        cluster_idx = model.fit(clusterMatrix)
        return model, cluster_idx

    def _put_labels(self, R, addlabels=False):
        stn_sort = []
        if addlabels:
            for stn_idx in R["leaves"]:
                stn_sort.append(self.labels[stn_idx])
        else:
            for stn_idx in R["leaves"]:
                stn_sort.append(f"S{stn_idx}")
        return stn_sort

    def get_linkage(self, clusterMatrix=None):
        if clusterMatrix is None:
            if self.linkage_matrix is None:
                model, _ = self.compute_cluster()
                self.linkage_matrix = model.linkage
            return self.linkage_matrix

        else:
            model, _ = self.compute_cluster(clusterMatrix=clusterMatrix)
            return model.linkage

    def compute_dendrogram(self, color_thresh=None):
        R = hierarchy.dendrogram(
            self.linkage_matrix, color_threshold=color_thresh, no_plot=True)
        return R

    def plot_dendrogram(self,
                        figname=None,
                        figsize=(20, 8),
                        xtickfontsize=kwargs_default['fontsize'],
                        labelfontsize=kwargs_default['fontsize'],
                        xlabel="3-D Stations",
                        ylabel="DTW Distance",
                        truncate_p=None,
                        distance_threshold=None,
                        annotate_above=0,
                        plotpdf=True,
                        leaf_rotation=0):
        '''
        :param truncate_p: show only last truncate_p out of all merged branches
        '''

        if distance_threshold == "optimal":
            opt_cluster, opt_distance = self.optimum_cluster_elbow()
            distance_threshold = opt_distance
        # R = hierarchy.dendrogram(linkage_matrix, no_plot=True)
        self.linkage_matrix = self.get_linkage()
        fig, ax = plt.subplots(1, 1, figsize=figsize, sharey=True)
        truncate_args = {}
        if truncate_p:
            if truncate_p > len(self.linkage_matrix)-1:
                truncate_p = len(self.linkage_matrix)-1
            truncate_args = {"truncate_mode": 'lastp',
                             "p": truncate_p, "show_contracted": True,
                             'show_leaf_counts': False}
        # highest color allowd
#         print(distance_threshold)
        R = hierarchy.dendrogram(
            self.linkage_matrix, color_threshold=distance_threshold, ax=ax, **truncate_args, leaf_rotation=leaf_rotation)
        if annotate_above > 0:
            for i, d, c in zip(R['icoord'], R['dcoord'], R['color_list']):
                x = 0.5 * sum(i[1:3])
                y = d[1]
                if y > annotate_above:
                    ax.plot(x, y, 'o', c=c)
                    ax.annotate('%.3g' % (y), (x, y), xytext=(0, -5),
                                textcoords='offset points',
                                va='top', ha='center', fontsize=xtickfontsize)

        if not truncate_p:
            if len(self.labels):  # if labels for the nodes are given
                try:
                    stn_sort = self._put_labels(R, addlabels=True)
                except:
                    stn_sort = self._put_labels(R, addlabels=False)
            else:
                stn_sort = self._put_labels(R, addlabels=False)
            # print(stn_sort)
            ax.set_xticks(np.arange(5, len(R["ivl"]) * 10 + 5, 10))
            ax.set_xticklabels(stn_sort, fontsize=xtickfontsize)
        ax.set_ylabel(ylabel, fontsize=labelfontsize)
        ax.set_xlabel(xlabel, fontsize=labelfontsize)
        plt.xticks(fontsize=xtickfontsize)
        plt.yticks(fontsize=xtickfontsize)
        _, opt_distance, opt_distance_plotloc = self.optimum_cluster_elbow(
            plotloc=True)
        ax.axhline(y=distance_threshold, c='k', lw=2)
#         ax.axhline(y=opt_distance_plotloc, c='k', lw=2)
        # if max_d:
        #     ax.axhline(y=max_d, c='k', lw=2)

        if figname:
            plt.savefig(figname, bbox_inches='tight')
            if plotpdf:
                figname_pdf = ".".join(figname.split(".")[:-1])+".pdf"
                ax.figure.savefig(figname_pdf,
                                  bbox_inches='tight')
                # plt.savefig(figname_ps, bbox_inches='tight')
            plt.close()
            return
        return fig, ax

    def compute_cut_off_inconsistency(self, t=None, depth=2, criterion='inconsistent', return_cluster=False):
        '''
        Calculate inconsistency statistics on a linkage matrix following `scipy.cluster.hierarchy.inconsistent`. It compares each cluster merge's height `h` to the average `avg` and normalize it by the standard deviation `std` formed over the depth previous levels

        :param t: threshold to apply when forming flat clusters. See scipy.cluster.hierarchy.fcluster for details
        :type t: scalar
        :param depth: The maximum depth to perform the inconsistency calculation
        :type depth: int
        :return: maximum inconsistency coefficient for each non-singleton cluster and its children; the inconsistency matrix (matrix with rows of avg, std, count, inconsistency); cluster
        '''
        from scipy.cluster.hierarchy import inconsistent, maxinconsts
        from scipy.cluster.hierarchy import fcluster
        self.linkage_matrix = self.get_linkage()
        incons = inconsistent(self.linkage_matrix, depth)
        maxincons = maxinconsts(self.linkage_matrix, incons)
        cluster = None
        if return_cluster:
            if t is None:
                t = np.median(maxincons)

            cluster = fcluster(self.linkage_matrix, t=t, criterion=criterion)
        return maxincons, incons, cluster

    def plot_hac_iteration(self,
                           figname=None,
                           figsize=(10, 8),
                           xtickfontsize=kwargs_default['fontsize'],
                           labelfontsize=kwargs_default['fontsize'],
                           xlabel="Iteration #",
                           ylabel="DTW Distance",
                           plot_color="C0",
                           plotpdf=True):

        self.linkage_matrix = self.get_linkage()
        xvals = np.arange(0, len(self.linkage_matrix))
        distance_vals = [row[2] for row in self.linkage_matrix]
        fig, ax = plt.subplots(figsize=figsize)
        ax.plot(xvals, distance_vals, '--.', color=plot_color)
        ax.set_ylabel(ylabel, fontsize=labelfontsize)
        ax.set_xlabel(xlabel, fontsize=labelfontsize)

        if figname:
            plt.savefig(figname, bbox_inches='tight')
            if plotpdf:
                figname_pdf = ".".join(figname.split(".")[:-1])+".pdf"
                ax.figure.savefig(figname_pdf,
                                  bbox_inches='tight')
            plt.close()
            fig, ax = None, None
        return fig, ax

    def _smoothsegment(self, seg, Nsmooth=50):
        '''
        From https://stackoverflow.com/questions/51936574/how-to-plot-scipy-hierarchy-dendrogram-using-polar-coordinates
        seg[1]        seg[2]
        +-------------+
        |             |
        + seg[0]      + seg[3]
        '''
        return np.concatenate([[seg[0]], np.linspace(seg[1], seg[2], Nsmooth), [seg[3]]])

    def plot_polar_dendrogram(self, figsize=(20, 20), normfactor=None,
                              Nyticks=7, gap=0.05, Nsmooth=100, linewidth=1, xtickfontsize=20, ytickfontsize=20,
                              plotstyle="seaborn", figname=None, plotpdf=True, gridcolor=None,
                              gridstyle='--', gridwidth=1, tickfontweight='bold', distance_threshold=None):
        '''
        Plot polar dendrogram of the clustering result

        :param figsize: figure size
        :param normfactor: (optional) normalization factor for the log spacing between the yticks, -np.log(dcoord+normfactor)
        :param Nyticks: number of y ticks
        :param gap: gap for the yticks
        :param plotstyle: matplotlib plot style, `plotstyle` by default
        :param distance_threshold: str, float. threshold for the coloring of branches. If str="optimal", then the optimal number of clusters based on elbow method will be used
        '''

        self.linkage_matrix = self.get_linkage()
        if distance_threshold == "optimal":
            opt_cluster, opt_distance = self.optimum_cluster_elbow()
            distance_threshold = opt_distance
        R = self.compute_dendrogram(color_thresh=distance_threshold)
        icoord = np.asarray(R["icoord"], dtype=float)
        dcoord = np.asarray(R["dcoord"], dtype=float)
        colorarray = np.asarray(R['color_list'])
        labels = self.labels

        def comp_log_dcoord(dcoord, normfactor):
            return -np.log(dcoord+normfactor)

        if normfactor is None:
            normfactor = 10*((dcoord.max()-dcoord.min())/len(dcoord))

        dcoordlog = comp_log_dcoord(dcoord, normfactor)
        # avoid a wedge over the radial labels

        imax = icoord.max()
        imin = icoord.min()
        icoord = ((icoord - imin)/(imax - imin)*(1-gap) + gap/2)*2*np.pi
        if gridcolor is not None:
            plt.style.library[plotstyle]['grid.color'] = gridcolor
        plt.style.library[plotstyle]['grid.linestyle'] = gridstyle
        plt.style.library[plotstyle]['grid.linewidth'] = gridwidth
        with plt.style.context(plotstyle):

            fig = plt.figure(figsize=figsize)
            ax = fig.add_subplot(111, polar=True)
            # ax.tick_params(axis='x', which='major',
            #                labelsize='large', labelcolor='k', rotation=45)

            for xs, ys, cs in zip(icoord, dcoordlog, colorarray):
                xs = self._smoothsegment(xs, Nsmooth=Nsmooth)
                ys = self._smoothsegment(ys, Nsmooth=Nsmooth)
                ax.plot(xs, ys, color=cs, lw=linewidth)
            ax.spines['polar'].set_visible(False)
            ax.set_rlabel_position(0)
            Nxticks = len(icoord)+1
            xticks = np.linspace(gap/2, 1-gap/2, Nxticks)
            ax.set_xticks(xticks*np.pi*2)

            ytickslog = np.linspace(
                dcoordlog.min(), dcoordlog.max(), Nyticks)

            _, opt_distance, opt_distance_plotloc = self.optimum_cluster_elbow(
                plotloc=True)
            opt_tick = comp_log_dcoord(opt_distance_plotloc, normfactor)
            # print(opt_tick)
            # print(ytickslog[:-1])
            # ax.axhline(y=opt_tick, c='k', lw=2)
            ytick_list = ytickslog[:-1]
            # ytick_list = np.sort(np.append(ytick_list, opt_tick))

            ax.set_yticks(ytick_list)
            ytickslabels = np.e**(-ytick_list)-normfactor
            ytickslabels = [f"{ylab:.1f}" for ylab in ytickslabels]
            ax.set_yticklabels(
                ytickslabels, fontsize=ytickfontsize, rotation=90)

            if labels is None:
                ax.set_xticklabels(np.round(np.linspace(imin, imax, Nxticks)).astype(
                    int), fontsize=xtickfontsize)
            else:
                try:
                    stn_sort = self._put_labels(R, addlabels=True)
                except:
                    stn_sort = self._put_labels(R, addlabels=False)

                # rotate labels
                plt.gcf().canvas.draw()
                angles = xticks*np.pi*2
                angles[np.cos(angles) < 0] = angles[np.cos(angles) < 0] + np.pi
                angles = np.rad2deg(angles)
                for label, angle, stn in zip(ax.get_xticklabels(), angles, stn_sort):
                    x, y = label.get_position()
                    lab = ax.text(x, y, stn, transform=label.get_transform(),
                                  ha=label.get_ha(), va=label.get_va(), fontsize=xtickfontsize, fontweight=tickfontweight)
                    lab.set_rotation(angle)
                ax.set_xticklabels([])
                # ax.axhline(y=opt_tick, c='k', lw=2)
            # plt.grid(color=gridcolor, linestyle='-', linewidth=2, zorder=10)
            if figname:
                plt.savefig(figname, bbox_inches='tight')
                if plotpdf:
                    figname_pdf = ".".join(figname.split(".")[:-1])+".pdf"
                    ax.figure.savefig(figname_pdf,
                                      bbox_inches='tight')
                plt.close()

    def _compute_significance(self, orig_df):
        def shuffle(df, n=1, axis=0):
            df = df.copy()
            for _ in range(n):
                df.apply(np.random.shuffle, axis=axis)
            return df

        df = shuffle(orig_df.copy())

        time_series = df.values.transpose()
        model = HierarchicalTree(
            dists_fun=dtw.distance_matrix_fast, show_progress=False, dists_options={'window': int(0.25*time_series.shape[0])})
        model.fit(time_series)
        linkage_matrix = model.linkage
        dist_sort = np.array([lnk[2] for lnk in linkage_matrix])

        df_cv = pd.DataFrame()
        df_cv["level"] = np.arange(dist_sort.shape[0], 0, -1)
        df_cv["distance"] = dist_sort
        accl = df_cv["distance"].diff().diff()  # double derivative

        df_accl = pd.DataFrame()
        df_accl["level"] = np.arange(dist_sort.shape[0]+1, 1, -1)
        df_accl["accln"] = accl

        return df_cv, df_accl

    def significance_test(self, numsimulations=10,
                          outfile="pickleFiles/dU_accl_sim_results.pickle",
                          fresh_start=False):
        orig_df = pd.DataFrame(self.matrix.transpose())
        if os.path.exists(outfile):
            if fresh_start:
                os.remove(outfile)
                sim_results_df = pd.DataFrame()
                iff = 0
            else:
                sim_results_df = pd.read_pickle(outfile)
                # print(sim_results_df.head())
                iff = sim_results_df.shape[1]
        else:
            sim_results_df = pd.DataFrame()
            iff = 0
        max_curv_dict = []

        tasks = []
        with concurrent.futures.ThreadPoolExecutor() as executor:
            print(f"Simulation started...")
            for i in range(numsimulations):
                tasks.append(executor.submit(
                    self._compute_significance, orig_df))

            for ff in concurrent.futures.as_completed(tasks):
                df_cv, df_accl = ff.result()
                sim_results_df[f"accl_{iff}"] = df_accl["accln"].values
                curVal = df_accl["accln"].abs()

                iff += 1
                print(f"Finished simulation {iff}")

        for icol, col in enumerate(sim_results_df.columns):
            curVal = sim_results_df[col].abs()
            _dict = {}
            _dict['idxmax'] = curVal.idxmax()
            _dict['idx'] = icol
            max_curv_dict.append(_dict)

        max_curv_df = pd.DataFrame(max_curv_dict)
        max_curv_df.set_index('idx', inplace=True)
        sim_results_df.to_pickle(outfile)

        return sim_results_df, max_curv_df

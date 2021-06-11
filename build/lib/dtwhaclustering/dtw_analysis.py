"""
dtw.dtw_analysis
----------------
This module is built around the dtaidistance package for the DTW computation and scipy.cluster

:author: Utpal Kumar, Institute of Earth Sciences, Academia Sinica
:note: See https://dtaidistance.readthedocs.io/en/latest/usage/dtw.html for details on dtaidistance package
"""

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


import pandas as pd

# from dtaidistance.clustering.hierarchical import HierarchicalTree
# from dtaidistance import clustering
from dtaidistance.clustering import HierarchicalTree

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
plt.style.use('ggplot')

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
        '''
        self.s1 = np.array(s1, dtype=np.double)
        self.s2 = np.array(s2, dtype=np.double)
        self.labels = labels

    def plot_signals(self, figname=None):
        '''
        Plot the signals
        '''
        fig, ax = plt.subplots(2, 1)
        ax[0].plot(self.s1, color="C0", lw=1)
        ax[0].set_ylabel(self.labels[0], fontsize=kwargs_default['fontsize'])

        ax[1].plot(self.s2, color="C1", lw=1)
        ax[1].set_ylabel(self.labels[1], fontsize=kwargs_default['fontsize'])

        if figname:
            plt.savefig(figname, bbox_inches='tight', dpi=300)
            plt.close()
            fig, ax = None, None

        return fig, ax

    def compute_distance(self, pruning=True, best_path=False):
        '''
        Returns the DTW distance
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

    def plot_warping_path(self, figname=None):
        '''
        Plot the signals with the warping paths
        '''

        distance = self.compute_distance()

        fig, ax = dtwvis.plot_warping(
            self.s1, self.s2, self.compute_warping_path())
        ax[0].set_ylabel(self.labels[0], fontsize=kwargs_default['fontsize'])
        ax[1].set_ylabel(self.labels[1], fontsize=kwargs_default['fontsize'])

        if figname:
            plt.savefig(figname, bbox_inches='tight')
            plt.close()
            fig, ax = None, None
        return distance, fig, ax

    def plot_matrix(self, windowfrac=0.2, psi=None, figname=None, shownumbers=False, showlegend=True):
        '''
        Plot the signals with the DTW matrix
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


def plot_signals(matrix, labels=[], figname=None, plotpdf=True):
    fig, ax = plt.subplots(
        nrows=matrix.shape[0], sharex=True, figsize=kwargs_default['figsize'])
    _labels = []
    for i in range(matrix.shape[0]):
        ax[i].plot(matrix[i, :], color=f"C{i}")
        if len(labels) == 0:
            lab = f"S{i}"
            ax[i].set_ylabel(lab, fontsize=kwargs_default['fontsize'])
            _labels.append(lab)
    if len(labels):
        try:
            for iaxx, axx in enumerate(ax):
                axx.set_ylabel(f"{labels[iaxx]}",
                               fontsize=kwargs_default['fontsize'])
        except Exception as e:
            print(e)
            for i in range(matrix.shape[0]):
                if len(labels) == 0:
                    ax[i].set_ylabel(
                        f"S{i}", fontsize=kwargs_default['fontsize'])
    if figname:
        plt.savefig(figname, bbox_inches='tight')
        if plotpdf:
            figname_pdf = ".".join(figname.split(".")[:-1])+".pdf"
            ax.figure.savefig(figname_pdf,
                              bbox_inches='tight')
        plt.close()
        fig, ax = None, None

    if len(labels) == 0:
        return fig, ax, np.array(_labels)

    return fig, ax


def plot_cluster(lons, lats, figname=None, figsize=(10, 10), plotpdf=True):
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    labels = []
    clusterIdx = 0
    for ilonlat, (loncluster, latcluster) in enumerate(zip(lons, lats)):
        lab = f'Cluster {clusterIdx}'
        if lab in labels:
            ax.plot(loncluster, latcluster, 'o', color=f"C{clusterIdx}", ms=20)
        else:
            ax.plot(loncluster, latcluster, 'o',
                    color=f"C{clusterIdx}", ms=20, label=lab)
            labels.append(lab)
        if (ilonlat+1) % 3 == 0:
            clusterIdx += 1

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

        if figname:
            plt.savefig(figname, bbox_inches='tight')
            if plotpdf:
                figname_pdf = ".".join(figname.split(".")[:-1])+".pdf"
                ax.figure.savefig(figname_pdf,
                                  bbox_inches='tight')
            plt.close()

        return ind_rand_perm, shuffled_matrix, fig, ax
    return ind_rand_perm, shuffled_matrix


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

    def plot_cluster_xymap(self, dtw_distance, figname=None, xlabel='x', ylabel='y', colorbar=True, colorbarstep=1, scale=2):
        if self.longitudes is not None:
            Z = self.get_linkage()
            clusters = fcluster(Z, dtw_distance, criterion='distance')
            cluster_ticks = list(set(clusters))
            lenx = scale*int(self.longitudes.max()-self.longitudes.min())
            leny = scale*int(self.latitudes.max()-self.latitudes.min())
            figsize = (lenx, leny)

            fig, ax = plt.subplots(figsize=figsize)
            # plot points with cluster dependent colors

            cax = ax.scatter(self.longitudes, self.latitudes,
                             c=clusters, cmap='jet', s=120)

            ax.set_xlabel(xlabel, fontsize=kwargs_default['fontsize'])
            ax.set_ylabel(ylabel, fontsize=kwargs_default['fontsize'])
            if colorbar:
                cbar = fig.colorbar(
                    cax, ticks=cluster_ticks[::colorbarstep])
                cbar.set_label('Clusters', fontsize=kwargs_default['fontsize'])
            plt.subplots_adjust(wspace=0.01)

            if figname:
                plt.savefig(figname, bbox_inches='tight')
                plt.close()
                fig, ax = None, None
            return fig, ax
        else:
            print("Input the x and y coords")

    def plot_cluster_geomap(self,
                            dtw_distance,
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
                            cmap_data='jet'
                            ):
        if self.longitudes is not None:
            Z = self.get_linkage()
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
                region=[minlon, maxlon, minlat, maxlat], projection="M4i", frame=True)
            if plot_topo:
                pygmt.makecpt(
                    cmap=cmap_topo,
                    series='-8000/8000/1000',
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
                    projection='M4i',
                    shorelines=True,
                    frame=True
                )
            else:
                fig.coast(land="#666666", shorelines=True)
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
            # Plot colorbar
            fig.colorbar(
                frame='+l"Clusters"'
            )
            # fig.show()
            fig.savefig(figname, crop=True, dpi=dpi)
        else:
            print("Input the x and y coords")

    def _compute_interpolation(self, clusters, lonrange=(120., 122.), latrange=(21.8, 25.6), gridstep=0.01):

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

    def plot_cluster_geomap_interpolated(self, dtw_distance,
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
        Z = self.get_linkage()
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
                series=[cmapminmax[0], cmapminmax[1], 1],
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

    def compute_distance_accl(self):
        linkage_matrix = self.get_linkage()

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
        accl = df_cv["distance"].diff().diff()

        df_accl = pd.DataFrame()
        df_accl["level"] = np.arange(dist_sort.shape[0]+1, 1, -1)
        df_accl["accln"] = accl
        return df_cv, df_accl

    def optimum_cluster_elbow(self):
        '''
        Gives the optimum number of clusters required to express the maximum difference in the similarity using the elbow method
        '''

        df_cv, df_accl = self.compute_distance_accl()
        idx = df_accl['accln'].argmax()
        opt_cluster = df_accl.loc[idx, 'level']
        df_cv_subset = df_cv[df_cv['level'] == opt_cluster]
        opt_distance = df_cv_subset['distance'].values[0]

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
                             xlim=None):
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

        ax.set_xlabel(xlabel, fontsize=kwargs_default['fontsize'])
        ax.set_ylabel(ylabel, fontsize=kwargs_default['fontsize'])
        plt.subplots_adjust(wspace=0.01)
        fig.align_xlabels()

        plt.legend(fontsize=kwargs_default['fontsize'], bbox_to_anchor=(
            1.05, 1), loc='upper left')
        plt.xticks(fontsize=kwargs_default['fontsize'])
        plt.yticks(fontsize=kwargs_default['fontsize'])
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

    def compute_cluster(self, clusterMatrix=None):
        if clusterMatrix is None:
            clusterMatrix = self.matrix

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

    def get_linkage(self):
        if self.linkage_matrix is None:
            model, cluster_idx = self.compute_cluster()
            self.linkage_matrix = model.linkage

        return self.linkage_matrix

    def compute_dendrogram(self):
        R = hierarchy.dendrogram(
            self.linkage_matrix, no_plot=True)
        return R

    def plot_dendrogram(self,
                        figname=None,
                        figsize=(20, 8),
                        xtickfontsize=kwargs_default['fontsize'],
                        labelfontsize=kwargs_default['fontsize'],
                        xlabel="3-D Stations",
                        ylabel="DTW Distance",
                        truncate_p=None,
                        max_d=None,
                        annotate_above=0,
                        plotpdf=True,
                        leaf_rotation=0):
        '''
        :param truncate_p: show only last truncate_p out of all merged branches
        '''
        if max_d:
            color_thresh = max_d

        else:
            color_thresh = None

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

        R = hierarchy.dendrogram(
            self.linkage_matrix, color_threshold=color_thresh, ax=ax, **truncate_args, leaf_rotation=leaf_rotation)
        if annotate_above > 0:
            for i, d, c in zip(R['icoord'], R['dcoord'], R['color_list']):
                x = 0.5 * sum(i[1:3])
                y = d[1]
                if y > annotate_above:
                    ax.plot(x, y, 'o', c=c)
                    ax.annotate("%.3g" % y, (x, y), xytext=(0, -5),
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

        if max_d:
            ax.axhline(y=max_d, c='k')

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
        Calculate inconsistency statistics on a linkage matrix following scipy.cluster.hierarchy.inconsistent
        It compares each cluster merge's height h to the average avg and normalize it by the standard deviation std formed over the depth previous levels
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

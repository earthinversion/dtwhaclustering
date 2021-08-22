"""
Least square modeling of GPS displacements for seasonality, tidal, co-seismic jumps.

:author: Utpal Kumar, Institute of Earth Sciences, Academia Sinica
"""


import pandas as pd
import numpy as np
from scipy.optimize import least_squares
import matplotlib.pyplot as plt
from .analysis_support import toYearFraction
import matplotlib
from tqdm import tqdm
# from tqdm.notebook import tqdm
import os
import concurrent.futures
import sys

# default matplotlib parameters
import matplotlib
font = {'family': 'Times',
        'weight': 'bold',
        'size': 22}

matplotlib.rc('font', **font)
plt.style.use('ggplot')

# to edit text in Illustrator
# matplotlib.rcParams['pdf.fonttype'] = 42


class LSQmodules:
    def __init__(self, dUU, sel_eq_file=None, station_loc_file="helper_files/stn_loc.txt", comp="U", figdir="LSQOut", periods=(13.6608, 14.7653, 27.5546, 182.62, 365.26, 6793.836)):
        '''
        Perform least square modeling of the time series in dUU

        :param dUU: Pandas dataframe containing the time series for the modeling. Should have time information as pandas datetime format in the index
        :type dUU: pandas.DataFrame
        :param sel_eq_file: File containing the earthquake origin times e.g., `2009,1,15` with header info, e.g. `year_val,month_val,date_val`
        :type sel_eq_file: str
        :param station_loc_file: File containing the station location info, e.g., `DAWU,120.89004,22.34059`, with header `stn,lon,lat`
        :type station_loc_file: str
        :param comp: Component name of the data provided
        :type comp: str
        :param figdir: Output directory for the figures, if requested
        :type figdir: str
        :param periods: Periods for the tidal and seasonal signals. e.g., (13.6608, 14.7653, 27.5546, 182.62, 365.26, 18.6). All in days, 6793.836 days == 18.6 years.
        :type periods: tuple
        '''
        self.dUU = dUU
        self.comp = comp
        # convert time to decimal year
        year = []
        for dd in self.dUU.index:
            year.append(round(toYearFraction(dd), 5))

        self.xval = np.array(year)

        # Periods in year for removal of tidal and seasonal signals
        yr = 365.26
        P1 = periods[0]/yr
        P2 = periods[1]/yr
        P3 = periods[2]/yr
        P4 = periods[3]/yr
        P5 = yr/yr
        P6 = periods[5]/yr  # in year
        self.periods = np.array([pp for pp in [P1, P2, P3, P4, P5, P6]])
        if sel_eq_file and os.path.exists(sel_eq_file):
            # selected earthquakes for the removal using least squares method
            dftmp = pd.read_csv(sel_eq_file)
            tmp = []
            for yr, mn, dt in zip(dftmp['year_val'].values, dftmp['month_val'].values, dftmp['date_val'].values):
                if yr < 10:
                    yr = '0{}'.format(yr)
                if mn < 10:
                    mn = '0{}'.format(mn)
                tmp.append('{}-{}-{}'.format(yr, mn, dt))
            # print(tmp)
            evs = pd.DatetimeIndex(tmp)  # string to pandas datetimeIndex
            # converting the events to year fraction
            events = []
            for ee in evs:
                ee_frac = round(toYearFraction(ee), 5)
                if self.xval.min() < ee_frac < self.xval.max():
                    events.append(ee_frac)

            self.events = events
        else:
            self.events = []

        # read station information
        self.stnloc = pd.read_csv(station_loc_file, header=None,
                                  sep='\s+', names=['stn', 'lon', 'lat'])
        self.stnloc.set_index('stn', inplace=True)
        del events, evs, dftmp, year

        self.stn_slope_file = f'stn_slope_res_{self.comp}.txt'
        if os.path.exists(self.stn_slope_file):
            os.remove(self.stn_slope_file)

        self.figdir = figdir
        os.makedirs(self.figdir, exist_ok=True)

    # defining the jump function
    def jump(self, t, t0):
        '''
        heaviside step function

        :param t: time data
        :type t: list
        :param t0: earthquake origin time
        '''
        o = np.zeros(len(t))
        ind = np.where(t == t0)[0][0]
        o[ind:] = 1.0
        return o

    def compute_lsq(self, plot_results=False, remove_trend=True, remove_seasonality=True, remove_jumps=True, plotformat=None):
        '''
        Compute the least-squares model using multithreading

        :param plot_results: plot the final results
        :type plot_results: boolean
        :param remove_trend: return the time series after removing the linear trend 
        :type remove_trend: boolean
        :param remove_seasonality: return the time series after removing the seasonal signals
        :type remove_seasonality: boolean
        :param remove_jumps: return the time series after removing the co-seismic jumps
        :type remove_jumps: boolean
        :param plotformat: plot format of the output figure, e.g. "png". "pdf" by default.
        :type plotformat: str
        '''
        def all_jumps(t, *cc):
            '''
            aggregate all jumps
            '''
            out = cc[0]*self.jump(t, self.events[0])
            for idx, ccval in enumerate(cc[1:]):
                eidx = idx+1
                out += ccval*self.jump(t, self.events[eidx])

            return out

        # defining the function for the removal of trend, seasonal, tidal and co-seismic signals

        def lsqfun(coeff, t, y):
            '''
            least squares function
            '''
            return coeff[0] + coeff[1] * t \
                + coeff[2] * np.cos(2*np.pi * t/self.periods[0]) + coeff[3] * np.sin(2*np.pi * t/self.periods[0]) \
                + coeff[4] * np.cos(2*np.pi * t/self.periods[1]) + coeff[5] * np.sin(2*np.pi * t/self.periods[1]) \
                + coeff[6] * np.cos(2*np.pi * t/self.periods[2]) + coeff[7] * np.sin(2*np.pi * t/self.periods[2]) \
                + coeff[8] * np.cos(2*np.pi * t/self.periods[3]) + coeff[9] * np.sin(2*np.pi * t/self.periods[3]) \
                + coeff[10] * np.cos(2*np.pi * t/self.periods[4]) + coeff[11] * np.sin(2*np.pi * t/self.periods[4]) \
                + all_jumps(t, *coeff[12:len(self.events)+12]) - y

        # defining the intial values
        x0 = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0,
                       0.0, 0.0, 0.0]+[1.0]*len(self.events))

        # function for regenerating the data after removal of signals
        def gen_data(t, a, b, a1, b1, a2, b2, a3, b3, a4, b4, a5, b5, *cc):
            '''
            output model of least squares
            '''
            y = a + b * t+a1 * np.cos(2*np.pi * t/self.periods[0])+b1 * np.sin(2*np.pi * t/self.periods[0])\
                + a2 * np.cos(2*np.pi * t/self.periods[1])+b2 * np.sin(2*np.pi * t/self.periods[1])\
                + a3 * np.cos(2*np.pi * t/self.periods[2])+b3 * np.sin(2*np.pi * t/self.periods[2])\
                + a4 * np.cos(2*np.pi * t/self.periods[3])+b4 * np.sin(2*np.pi * t/self.periods[3])\
                + a5 * np.cos(2*np.pi * t/self.periods[4])+b5 * np.sin(2*np.pi * t/self.periods[4])\
                + all_jumps(t, *cc)
            return y

        def _computelsq(stn, x0, yval1):
            '''
            Compute the lsq, and residuals
            '''
            try:
                res_lsq = least_squares(lsqfun, x0, args=(self.xval, yval1))
                trendU = res_lsq.x[0]+self.xval*res_lsq.x[1]

                seasonalityU = res_lsq.x[2] * np.cos(2*np.pi * self.xval/self.periods[0])+res_lsq.x[3] * np.sin(2*np.pi * self.xval/self.periods[0])+res_lsq.x[4] * np.cos(2*np.pi * self.xval/self.periods[1])+res_lsq.x[5] * np.sin(2*np.pi * self.xval/self.periods[1])+res_lsq.x[6] * np.cos(
                    2*np.pi * self.xval/self.periods[2])+res_lsq.x[7] * np.sin(2*np.pi * self.xval/self.periods[2])+res_lsq.x[8] * np.cos(2*np.pi * self.xval/self.periods[3])+res_lsq.x[9] * np.sin(2*np.pi * self.xval/self.periods[3])+res_lsq.x[10] * np.cos(2*np.pi * self.xval/self.periods[4])+res_lsq.x[11] * np.sin(2*np.pi * self.xval/self.periods[4])

                jumpsU = all_jumps(
                    self.xval, *res_lsq.x[12:len(self.events)+12])
                residual = yval1
                residual_label = "Residual"

                if remove_trend or remove_seasonality or remove_jumps:
                    residual_label += " after removing"
                    # remove trend, seasonality, jumps
                    if remove_trend:
                        residual_label += " trend"
                        residual -= trendU

                    if remove_seasonality:
                        residual_label += " seasonality"
                        residual -= seasonalityU

                    if remove_jumps:
                        residual_label += " jumps"
                        residual -= jumpsU

                resdU = np.array(residual)
                mresdU = np.mean(resdU)
                stdresdU = np.std(resdU)

                # removing outliers (> 3 std) with the mean of data
                maxStd = 3
                final_residual = []
                for xU in resdU:
                    if (mresdU - maxStd*stdresdU < xU < mresdU + maxStd*stdresdU):
                        final_residual.append(xU)
                    else:
                        final_residual.append(mresdU)

                with open(self.stn_slope_file, 'a') as ff:
                    ff.write('{} {:.5f} {:.5f} {:.2f}\n'.format(
                        stn, self.stnloc.loc[stn, 'lon'], self.stnloc.loc[stn, 'lat'], res_lsq.x[1]))

            except KeyboardInterrupt:
                sys.quit()

            return stn, yval1, final_residual, res_lsq, trendU, seasonalityU, jumpsU, residual_label

        tasks = []
        final_dU = pd.DataFrame(index=self.dUU.index)

        with concurrent.futures.ThreadPoolExecutor() as executor:
            for i, stn in enumerate(self.dUU.columns):
                stn = stn.split("_")[0]
                yval1 = np.array(self.dUU.iloc[:, i])
                tasks.append(executor.submit(
                    _computelsq, stn, x0, yval1))

            pbar = tqdm(concurrent.futures.as_completed(tasks))
            pbar.set_description(f"Computation started for {self.comp}...")
            for tt in pbar:
                stn, yval1, final_residual, res_lsq, trendU, seasonalityU, jumpsU, residual_label = tt.result()
                pbar.set_description(f"Finished LSQ ({self.comp}): {stn}")
                colval = f"{stn}_{self.comp}"
                if plot_results:
                    # plot fig
                    mindisp, maxdisp = yval1.min(), yval1.max()
                    pbar.set_description(f"Plotting LSQ ({self.comp}): {stn}")
                    fig, ax = plt.subplots(5, 1, figsize=(10, 6), sharex=True)
                    ax[0].plot(self.xval, yval1, "bo", markersize=1,
                               label=f"{stn}_{self.comp}")
                    ax[0].plot(self.xval, gen_data(self.xval, *res_lsq.x),
                               'k', lw=1, label='Least squares fit')
                    ax[2].set_ylabel("Displacement (mm)", color='k')
                    ax[0].set_title(stn)
                    ax[0].legend(loc=2)

                    ax[1].plot(self.xval, trendU, 'k', lw=1,
                               label='Trend, slope: {:.2f}'.format(res_lsq.x[1]))
                    # ax[1].set_ylabel("Amp", color='k')
                    ax[1].legend(loc=2)

                    ax[2].plot(self.xval, seasonalityU, 'k',
                               lw=0.5, label='Seasonality')
                    # ax[2].set_ylabel("Amp", color='k')
                    ax[2].legend(loc=2)

                    ax[3].plot(self.xval, jumpsU, 'k', lw=1,
                               label='Co Seismic Jumps')
                    # ax[3].set_ylabel("Amp", color='k')
                    ax[3].legend(loc=2)

                    ax[4].plot(self.xval, final_residual,
                               'k', lw=0.5, label=residual_label)
                    # ax[4].set_ylabel("Displacement", color='k')
                    # ax[4].set_ylabel("Amp", color='k')
                    ax[4].legend(loc=2)

                    for axx in ax:
                        axx.set_ylim([mindisp, maxdisp])
                    if plotformat:
                        try:
                            plt.savefig(os.path.join(
                                self.figdir, f'time_series_{stn}_{self.comp}.{plotformat}'), bbox_inches='tight', dpi=200)
                        except Exception as e:
                            print(sys.exc_info())
                    else:
                        plt.savefig(os.path.join(
                            self.figdir, f'time_series_{stn}_{self.comp}.pdf'), bbox_inches='tight')
                    plt.close("all")

                final_dU[colval] = final_residual
                del stn, yval1, final_residual, res_lsq, trendU, seasonalityU, jumpsU
        return final_dU


def lsqmodeling(dUU, dNN, dEE, stnlocfile,  plot_results=True, remove_trend=False, remove_seasonality=True, remove_jumps=False, sel_eq_file="helper_files/selected_eqs_new.txt", figdir="LSQOut"):
    '''
    Least square modeling for the three component time series

    :param dUU: Vertical component pandas dataframe time series
    :type dUU: pandas.DataFrame
    :param dNN: North component pandas dataframe time series
    :type dNN: pandas.DataFrame
    :param dEE: East component pandas dataframe time series
    :type dEE: pandas.DataFrame
    :param plot_results: plot the final results
    :type plot_results: boolean
    :param remove_trend: return the time series after removing the linear trend 
    :type remove_trend: boolean
    :param remove_seasonality: return the time series after removing the seasonal signals
    :type remove_seasonality: boolean
    :param remove_jumps: return the time series after removing the co-seismic jumps
    :type remove_jumps: boolean
    :param sel_eq_file: File containing the earthquake origin times e.g., `2009,1,15` with header info, e.g. `year_val,month_val,date_val`
    :type sel_eq_file: str
    :param stnlocfile: File containing the station location info, e.g., `DAWU,120.89004,22.34059`, with header `stn,lon,lat`
    :type stnlocfile: str
    :return: Pandas dataframe corresponding to the vertical, north and east components e.g., final_dU, final_dN, final_dE 
    :rtype: pandas.DataFrame, pandas.DataFrame, pandas.DataFrame


    .. code-block:: python

        from dtwhaclustering.leastSquareModeling import lsqmodeling
        final_dU, final_dN, final_dE = lsqmodeling(dUU, dNN, dEE,stnlocfile="helper_files/stn_loc.txt",  plot_results=True, remove_trend=False, remove_seasonality=True, remove_jumps=False)

    '''
    ################################################
    final_dU, final_dN, final_dE = None, None, None

    lsqmod_U = LSQmodules(dUU, sel_eq_file=sel_eq_file,
                          station_loc_file=stnlocfile, comp="U", figdir=figdir)
    final_dU = lsqmod_U.compute_lsq(
        plot_results=plot_results, remove_trend=remove_trend, remove_seasonality=remove_seasonality, remove_jumps=remove_jumps)
    del lsqmod_U

    lsqmod_N = LSQmodules(dNN, sel_eq_file=sel_eq_file,
                          station_loc_file=stnlocfile, comp="N", figdir=figdir)
    final_dN = lsqmod_N.compute_lsq(plot_results=plot_results, remove_trend=remove_trend,
                                    remove_seasonality=remove_seasonality, remove_jumps=remove_jumps)
    del lsqmod_N

    lsqmod_E = LSQmodules(dEE, sel_eq_file=sel_eq_file,
                          station_loc_file=stnlocfile, comp="E", figdir=figdir)
    final_dE = lsqmod_E.compute_lsq(plot_results=plot_results, remove_trend=remove_trend,
                                    remove_seasonality=remove_seasonality, remove_jumps=remove_jumps)
    del lsqmod_E

    return final_dU, final_dN, final_dE

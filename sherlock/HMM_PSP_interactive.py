##########################
# Scheme using Hidden markov models for Establishing Reiteratively a List Of Candidate period-spacings with liKelihood (SHERLOCK)
#
# beta version
# Joey Mombarg - Oct 2023
##########################
import matplotlib
matplotlib.use("TkAgg")
import numpy as np
import matplotlib.pyplot as plt

import matplotlib.colors as mcolors
import matplotlib.cm as cm

from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes, inset_axes
import glob, re
import pandas as pd
import pickle, h5py
import scipy
from scipy import interpolate

#from multiprocessing import Pool
#from scipy.signal import savgol_filter

from matplotlib.backend_bases import PickEvent
from matplotlib.widgets import Button

fontsize = 14
matplotlib.rc('xtick', labelsize=fontsize)
matplotlib.rc('ytick', labelsize=fontsize)

'''
Parameters set by user.
'''
# Azimuthal order to search for. m = -1 retrograde, m = 0 zonal, m = 1 prograde.
m = 1
# Maximum number of skipped modes in the pattern before terminating.
max_skip = 2
# Maximum number of modes towards smaller periods before terminating.
max_modes_smaller_periods = 35
# Maximum number of modes towards larger periods before terminating.
max_modes_larger_periods  = 35
# KIC number
KIC = '06352430' #'07760680' #
# Work directory, no trailing slash.
WORK_DIR = '/Users/joey/Documents/Projects/IGW_mixing/SHERLOCK'


def strip(text):
    try:
        return text.strip()
    except AttributeError:
        return text

def KIC_li(KIC):
    if len(KIC) == 7:
        return '00' + KIC
    elif len(KIC) ==  8:
        return '0' + KIC

#with open('/Users/joeym/Documents/Projects/Paper_III/Data_sets/Output_training_set_nodiff_All_Stars_n91-15_Zext.pkl', 'rb') as f:
#    x = pickle.load(f)


with open(f'{WORK_DIR}/grids/gyre_per_l1m{m}_ext.pkl', 'rb') as f:
    x = pickle.load(f)

deltaP_all_min = deltaP_all_max = 0

def get_deltaP_sel(n, deltaP_obs1):
    '''
    Get the difference of period-spacing of a specific radial order.

    -- Input --
    n: radial order
    deltaP_obs1: previous observed difference in period-spacing

    -- Output --
    deltaP_sel: expected difference in period spacings (P_n - P_n+1) and (P_n+1 - P_n+2) in seconds, given deltaP_obs1. p(deltaP_2 | deltaP_1)
    deltaP_all: expected difference in period spacings for the entire grid. p(deltaP)
    '''
    global x
    dP0 = np.array(x[str(n-1)])   - np.array(x[str(n)])
    dP1 = np.array(x[str(n)])   - np.array(x[str(n+1)])
    dP2 = np.array(x[str(n+1)]) - np.array(x[str(n+2)])
    # Previous period-spacing
    deltaP_prev = (dP1 - dP0)*86400
    # Next periods-spacing
    deltaP_ = (dP2 - dP1)*86400
    deltaP_ = np.array(deltaP_)
    ddp = np.abs(deltaP_prev - deltaP_obs1)
    deltaP_sort = deltaP_[np.argsort(ddp)]
    # Select 500 previous period-spacings that are closest to the observed one.
    deltaP_sel_ = deltaP_sort[0:500]
    # Filter nans in the grid.
    deltaP_sel = [x_ for x_ in deltaP_sel_ if not np.isnan(x_)]
    deltaP_all = [x_ for x_ in deltaP_ if not np.isnan(x_)]
    return deltaP_sel, deltaP_all

def deltaP_expected(deltaP_obs1, skipped_radial_order):
    '''
    Interpolate the distribution of differences in period spacing, and compute the most probable one.
    Also compute a search window based on the min/max value that is 0.001*p_max, where p_max is the highest probability.
    Lastely, integrate the probability distribution to get the normalization factor.

    -- Input --
    deltaP_obs1: observed difference in period-spacings DeltaP_2 - DeltaP_1.
    skipped_radial_order: was a radial skipped in the pattern?

    -- Output --
    deltaP_min/max: search window.
    max_prob: most likely difference in period spacing.
    p_trans_ipol: interpolator for the PDF of differences in period-spacings.
    norm: inverse of the integral of the PDF.
    '''
    global x
    deltaP_sel = []
    deltaP_all = []

    for n in np.arange(2,98,1):
        deltaP_sel_, deltaP_all_ = get_deltaP_sel(n,deltaP_obs1)
        deltaP_sel.extend(deltaP_sel_)
        deltaP_all.extend(deltaP_all_)

    deltaP_sel = np.array(deltaP_sel)
    deltaP_all = np.array(deltaP_all)

    # If a radial order is skipped, we take the estimate over the entire grid, i.e. compute p(deltaP) instead of p(deltaP_2 | deltaP_1).
    if skipped_radial_order:
        deltaP_sel = deltaP_all

    bin_low = np.min(deltaP_sel)
    bin_up  = np.max(deltaP_sel)


    PDF_bin, bins_edge = np.histogram(deltaP_sel, bins=10000, density=True, range = [bin_low, bin_up])
    bins = bins_edge[1:] + 0.5*(bins_edge[:-1] - bins_edge[1:])
    max_prob = bins[np.argmax(PDF_bin)]

    p_trans_ipol = interpolate.interp1d(bins, PDF_bin, kind='quadratic', fill_value='extrapolate')


    xx = np.linspace(np.min(bins), np.max(bins), 10000)
    p_trans_pos = p_trans_ipol(xx)
    #p_trans_pos[p_trans_pos <= 0] = 0.
    norm = 1/np.trapz(p_trans_pos)

    p_scale = np.array(p_trans_ipol(xx)/np.max(p_trans_ipol(xx)))
    xmax = xx[np.argmax(p_scale)]
    #deltaP_min = np.max(xx[(p_scale < 10**-3) & (xx < xmax)]) #-3
    #deltaP_max = np.min(xx[(p_scale < 10**-3) & (xx > xmax)]) #-3
    # This cutoff value works well for the SPB grid. Make the plots below to verify.
    deltaP_min = np.min(xx[p_scale > 10**-3])
    deltaP_max = np.max(xx[p_scale > 10**-3])

    if False:
        fig, ax = plt.subplots()
        ax.plot(xx, np.log10(p_trans_ipol(xx)/np.max(p_trans_ipol(xx))), color = 'k')
        ax.vlines([deltaP_min, deltaP_max], ymin = 0, ymax = 1, color = 'grey')
        ax.vlines([max_prob], ymin = 0, ymax = 1, color = 'g')
        ax.axhline(-3, color ='grey', linestyle ='dashed')
        ax.set_xlabel(r'$\Delta P_1 - \Delta P_2$', fontsize = 14)
        ax.set_ylabel(r'$\log P/P_{\rm max}$', fontsize = 14)

    return deltaP_min, deltaP_max, max_prob, p_trans_ipol, norm

def p_emis(A_potential, in_log):
    '''
    Compute the emission probability.
    '''
    if in_log:
        A_potential = 10**A_potential
    return A_potential/np.sum(A_potential)


def read_frequency_list(KIC, combinations_included = True):
    '''
    Read frequency list from Van Beeck et al. (2022) for a given KIC number, picking their extraction strategy with the highest f_sv factor.

    -- Input --
    KIC: KIC number without 'KIC' prefix
    combinations_included: also include combination frequencies?

    -- Output --
    P_obs: observed periods in days.
    pe: uncertainties on periods
    A: amplitudes
    ae: uncertainties on amplitutes.
    phase: phases in rad.
    phe: uncertainties on phases.
    nonlin_id: non-linear mode ID, e.g. 'freq1+freq2'.
    '''
    strat =  pd.read_csv('/Users/joey/Documents/Projects/IGW_mixing/SHERLOCK/best_strategy.txt', names = ['KIC', 'strategy'], sep=' ', dtype = str)
    KIC_list = np.array(strat['KIC'])
    strategy = strat['strategy'][np.where(KIC_list == KIC)[0][0]]
    df = pd.read_csv(f'/Users/joey/Documents/Projects/IGW_mixing/SHERLOCK/forJoey/amplitudes_frequencies_phases_KIC0{KIC}_strategy_{strategy}.asc', sep = '\t', header = 9, names = ['freq', 'sigma_freq', 'ampl', 'sigma_ampl', 'phase', 'sigma_phase', 'nr', 'nonlin_id'])
    if not combinations_included:
        i = 0
        while 1:
            nonlin_id = df['nonlin_id'][i]
            if '+' in nonlin_id or '-' in nonlin_id or '*' in nonlin_id:
                break
            else:
                i += 1
        n = i-1
        freq = np.array(df['freq'][0:n])
        fe = np.array(df['sigma_freq'][0:n])
        A = np.log10(np.array(df['ampl'][0:n]))
        ae = np.array(df['sigma_ampl'][0:n])
        phase = np.array(df['phase'][0:n])
        phe = np.array(df['sigma_phase'][0:n])
        nonlin_id = np.array(df['nonlin_id'][0:n])
    else:
        freq = np.array(df['freq'])
        fe = np.array(df['sigma_freq'])
        A = np.log10(np.array(df['ampl'])/np.min(np.array(df['ampl'])))
        ae = np.array(df['sigma_ampl'])
        phase = np.array(df['phase'])
        phe = np.array(df['sigma_phase'])
        nonlin_id = np.array(df['nonlin_id'])

    P_obs = 1/freq
    pe = fe/freq**2
    return P_obs, pe, A, ae, phase, phe, nonlin_id

P_obs, pe, A, ae, phase, phe, nonlin_id = read_frequency_list(KIC, combinations_included = True)

initial_period_index = []
psp_dict = {}

fig, ax = plt.subplots(3,1, sharex=True, figsize =(8,6))
ax[0].set_ylim(0,3)
lines = []
for i in range(len(P_obs)):
    line_ = ax[0].vlines(x=P_obs[i], ymin=0, ymax=A[i], color = 'k', picker = True)
    lines.append(line_)
    #ax[0].text(P_obs[i], 0, str(i), color ='k', fontsize = 10, ha  = 'center', va = 'top' )

# Define a function to handle the pick event
def on_pick(event):
    global initial_period_index
    if event.artist in lines:
        ind = lines.index(event.artist)
        if ind not in initial_period_index:
            initial_period_index.append(ind)
            event.artist.set_color('red')
            fig.canvas.draw_idle()
        else:
            initial_period_index.remove(ind)
            event.artist.set_color('k')
            fig.canvas.draw_idle()

# Connect the pick event to the handler function
fig.canvas.mpl_connect('pick_event', on_pick)

# Run SHERLOCK when the button is clicked
def button_search(event):
    '''
    If the search button is clicked, the SHERLOCK algorithm will start searching for a period-spacing pattern
    based on the suggested initial three periods. This is done once towards larger periods, and once towards lower periods.
    '''
    global initial_period_index, pe, nonlin_id, P_obs, A, psp_dict

    ip1 = initial_period_index[0]
    ip2 = initial_period_index[1]
    ip3 = initial_period_index[2]

    print(f"Selected points: {initial_period_index}")

    # Sort the periods such that the order in which you click the initial periods is not important.
    P_ini_sort = np.sort([P_obs[ip1], P_obs[ip2], P_obs[ip3]])
    P_obs1_ini = P_obs1 = P_ini_sort[0]
    P_obs2_ini = P_obs2 = P_ini_sort[1]
    P_obs3_ini = P_obs3 = P_ini_sort[2]
    print(P_obs1_ini, P_obs2_ini, P_obs3_ini)

    ax[2].vlines(x=P_obs1, ymin=0, ymax=1, color = 'r')
    ax[2].vlines(x=P_obs2, ymin=0, ymax=1, color = 'r')
    ax[2].vlines(x=P_obs3, ymin=0, ymax=1, color = 'r')

    pattern = []
    restart = []
    direction   = []
    uncertainty = []
    probability_tot   = []
    probability_trans = []
    probability_emis  = []
    nonlin_id_arr = []
    pattern.extend([P_obs1, P_obs2, P_obs3])
    # Keep track of the direction (towards lower/higher periods) and properly order them afterwards.
    direction.extend(['i', 'i', 'i'])
    uncertainty.extend([pe[ip1], pe[ip2], pe[ip3]])
    # We set the probabilities of the manually selected modes to 1.
    probability_tot.extend([1.,1.,1.])
    probability_trans.extend([1.,1.,1.])
    probability_emis.extend([1.,1.,1.])
    nonlin_id_arr.extend([nonlin_id[ip1][1:-1], nonlin_id[ip2][1:-1], nonlin_id[ip3][1:-1]])
    skipped_radial_order = False
    go_on = 0
    skipped = 0

    # Compute the search window based on the entire grid of models.
    deltaP_all_min, deltaP_all_max, _, _, _ = deltaP_expected(np.nan, True)

    #To the right, larger periods.
    print('To the right...')
    while go_on < max_modes_larger_periods:
        DeltaP_obs1 = np.abs(P_obs1 - P_obs2)*86400
        DeltaP_obs2 = np.abs(P_obs2 - P_obs3)*86400
        deltaP_obs1 = DeltaP_obs1 - DeltaP_obs2

        # Compute the expected difference in period-spacing for the next period, and the search interval.
        percen5, percen95, deltaP_exp, p_trans_ipol, norm  = deltaP_expected(deltaP_obs1, skipped_radial_order)
        # Compute the most probable period spacing, and the expected minimum and maximum period spacing.
        DeltaP_exp  = DeltaP_obs2 - deltaP_exp
        DeltaP_up   = DeltaP_obs2 - deltaP_all_min
        DeltaP_low  = DeltaP_obs2 - deltaP_all_max

        DeltaP_all  =  (P_obs - P_obs3)*86400
        # Keep only periods with spacings within the search interval and positive spacings.
        get = np.array([DeltaP_low < DeltaP_all]) & np.array([DeltaP_all < DeltaP_up]) & np.array([DeltaP_all > 0])
        if np.sum(get) > 0:
            P_potential = P_obs[get[0]]
            np.set_printoptions(precision=16)
            print('pobs', P_potential)

            A_potential = A[get[0]]
            print('Aobs', A_potential)

            pe_potential = pe[get[0]]
            # Compute the emission probabilities of all candidates.
            p_emission  = p_emis(A_potential, in_log = True)
            DeltaP_potential = (P_potential - P_obs3)*86400
            deltaP_obs2 = DeltaP_obs1 - DeltaP_potential
            p_transition = norm * p_trans_ipol(deltaP_obs2)
            # Interpolation can give probabilities slightly smaller than 0. Set these to 0.
            p_transition[p_transition < 0] = 0

            sig = np.array([p_emission > 0.]) & np.array([p_transition > 0])

            if np.sum(sig) > 0:
                # Compute the total probability and normalize.
                p_total = p_transition * p_emission
                p_total /= np.sum(p_total)

                P_next = P_potential[np.argmax(p_total)]
                print(f'{go_on} New period found right at {P_next} with probability {np.max(p_total)} (p_trans = {p_transition[np.argmax(p_total)]}, p_emis = {p_emission[np.argmax(p_total)]})')
                print(nonlin_id[P_obs == P_next][0])
                pattern.extend([P_next])
                uncertainty.extend([pe_potential[np.argmax(p_total)]])
                probability_tot.extend([np.max(p_total)])
                probability_trans.extend([p_transition[np.argmax(p_total)]])
                probability_emis.extend([p_emission[np.argmax(p_total)]])
                nonlin_id_arr.extend([nonlin_id[P_obs == P_next][0][1:-1]])
                ax[0].axvspan((DeltaP_low/86400.)+P_obs3, (DeltaP_up/86400.)+P_obs3, ymin = -go_on, ymax = 10, facecolor='g', alpha=0.5)
                ax[0].vlines(x=P_obs3+(DeltaP_exp/86400.), ymin=0, ymax=np.max(A), color = 'g', linestyle = 'dashed')
                ax[0].vlines(x=P_next, ymin=0, ymax=A[P_obs == P_next], color = 'r')
                ax[2].vlines(x=P_next, ymin=0, ymax=np.max(p_total), color = 'k')
                ax[0].text(P_next, A[P_obs == P_next], str(go_on+1), color ='r', fontsize = 10, ha  = 'center' )
                skipped_radial_order = False

            else:
                P_next = P_obs3 + (DeltaP_exp/86400.)
                ax[0].vlines(x=P_next, ymin=0, ymax=np.max(A), color = 'grey', linestyle = 'dashed')
                print(f'{go_on} skipping radial order because of low emission/transition probability')
                pattern.extend([np.nan])
                uncertainty.extend([np.nan])
                probability_tot.extend([np.nan])
                probability_trans.extend([np.nan])
                probability_emis.extend([np.nan])
                nonlin_id_arr.extend(['missing'])

                skipped_radial_order = True
                skipped += 1
        else:
            P_next = P_obs3 + (DeltaP_exp/86400.)
            ax[0].vlines(x=P_next, ymin=0, ymax=np.max(A), color = 'grey', linestyle = 'dashed')
            ax[0].axvspan((DeltaP_low/86400.)+P_obs3, (DeltaP_up/86400.)+P_obs3, facecolor='grey', alpha=0.5)
            print((DeltaP_low/86400.)+P_obs3, (DeltaP_exp/86400.)+P_obs3,  (DeltaP_up/86400.)+P_obs3)
            print(f'{go_on} skipping radial order')
            skipped_radial_order = True
            skipped += 1
            pattern.extend([np.nan])
            uncertainty.extend([np.nan])
            probability_tot.extend([np.nan])
            probability_trans.extend([np.nan])
            probability_emis.extend([np.nan])
            nonlin_id_arr.extend(['missing'])

        direction.extend('r')

        P_obs1 = P_obs2
        P_obs2 = P_obs3
        P_obs3 = P_next

        go_on += 1
        if skipped > max_skip:
            print('Skipped too many modes. Stopping.')
            break

    P_obs1 = P_obs1_ini
    P_obs2 = P_obs2_ini
    P_obs3 = P_obs3_ini

    skipped_radial_order = False
    go_on = 0
    skipped = 0

    ## To the left, to smaller periods.
    print('To the left...')
    while go_on < max_modes_smaller_periods:
        DeltaP_obs1 = np.abs(P_obs1 - P_obs2)*86400
        DeltaP_obs2 = np.abs(P_obs2 - P_obs3)*86400
        deltaP_obs1 = DeltaP_obs1 - DeltaP_obs2
        percen5, percen95, deltaP_exp, p_trans_ipol, norm  = deltaP_expected(deltaP_obs1, skipped_radial_order)
        DeltaP_exp  = DeltaP_obs1 + deltaP_exp
        DeltaP_low  = DeltaP_obs1 + deltaP_all_min
        DeltaP_up   = DeltaP_obs1 + deltaP_all_max


        DeltaP_all  =  (P_obs1 - P_obs)*86400
        get = np.array([DeltaP_low < DeltaP_all]) & np.array([DeltaP_all < DeltaP_up]) & np.array([DeltaP_all > 0])
        if np.sum(get) > 0:
            P_potential = P_obs[get[0]]
            A_potential = A[get[0]]
            pe_potential = pe[get[0]]

            p_emission  = p_emis(A_potential, in_log = True)
            DeltaP_potential = (P_obs1 - P_potential)*86400
            deltaP_obs2 =  DeltaP_potential - DeltaP_obs1
            p_transition = norm * p_trans_ipol(deltaP_obs2)

            print(p_transition)
            p_transition[p_transition < 0] = 0

            sig = np.array([p_emission > 0.0]) & np.array([p_transition > 0])
            if np.sum(sig) > 0:
                p_total = p_transition * p_emission
                p_total /= np.sum(p_total)

                P_next = P_potential[np.argmax(p_total)]
                print(f'{go_on} New period found left at {P_next} with probability {np.max(p_total)} (p_trans = {p_transition[np.argmax(p_total)]}, p_emis = {p_emission[np.argmax(p_total)]})')
                print(nonlin_id[P_obs == P_next][0])

                pattern.extend([P_next])
                uncertainty.extend([pe_potential[np.argmax(p_total)]])
                probability_tot.extend([np.max(p_total)])
                probability_trans.extend([p_transition[np.argmax(p_total)]])
                probability_emis.extend([p_emission[np.argmax(p_total)]])
                nonlin_id_arr.extend([nonlin_id[P_obs == P_next][0][1:-1]])

                ax[0].vlines(x=P_next, ymin=0, ymax=A[P_obs == P_next], color = 'r')
                ax[0].text(P_next, A[P_obs == P_next], str(go_on+1), color ='r', fontsize = 10, ha  = 'center' )
                ax[2].vlines(x=P_next, ymin=0, ymax=np.max(p_total), color = 'k')
                ax[0].axvspan(P_obs1-(DeltaP_low/86400.), P_obs1-(DeltaP_up/86400.), ymin = -go_on, ymax = 10,facecolor='g', alpha=0.5)
                ax[0].vlines(x=P_obs1-(DeltaP_exp/86400.), ymin=0, ymax=np.max(A), color = 'g', linestyle = 'dashed')
                skipped_radial_order = False
            else:
                P_next = P_obs1 - (DeltaP_exp/86400.)
                ax[0].vlines(x=P_next, ymin=0, ymax=np.max(A), color = 'grey', linestyle = 'dashed')
                ax[0].axvspan(P_obs1-(DeltaP_low/86400.), P_obs1-(DeltaP_up/86400.), facecolor='grey', alpha=0.5)
                print(f'{go_on} skipping radial order because of low emission/transition probability')
                skipped+=1
                pattern.extend([np.nan])
                uncertainty.extend([np.nan])
                probability_tot.extend([np.nan])
                probability_trans.extend([np.nan])
                probability_emis.extend([np.nan])
                nonlin_id_arr.extend(['missing'])
                skipped_radial_order = True

        else:
            P_next = P_obs1 - (DeltaP_exp/86400.)
            ax[0].vlines(x=P_next, ymin=0, ymax=np.max(A), color = 'grey', linestyle = 'dashed')
            ax[0].axvspan(P_obs1-(DeltaP_low/86400.), P_obs1-(DeltaP_up/86400.), facecolor='grey', alpha=0.5)
            print(f'{go_on} skipping radial order')
            pattern.extend([np.nan])
            uncertainty.extend([np.nan])
            probability_tot.extend([np.nan])
            probability_trans.extend([np.nan])
            probability_emis.extend([np.nan])
            nonlin_id_arr.extend(['missing'])
            skipped+=1
            skipped_radial_order = True

        direction.extend('l')

        P_obs3 = P_obs2
        P_obs2 = P_obs1
        P_obs1 = P_next

        go_on += 1
        if skipped > max_skip:
            print('Skipped too many modes. Stopping.')
            break
    #pattern = np.array(np.sort(pattern))
    pattern           = np.array(pattern)
    direction         = np.array(direction)
    uncertainty       = np.array(uncertainty)
    probability_tot   = np.array(probability_tot)
    probability_trans = np.array(probability_trans)
    probability_emis  = np.array(probability_emis)
    nonlin_id_arr     = np.array(nonlin_id_arr)

    # Stitch the patterns towards lower periods and towards higher periods together.
    pattern           = np.array(list(np.flip(pattern[direction == 'l'])) + list(pattern[direction != 'l']))
    uncertainty       = np.array(list(np.flip(uncertainty[direction == 'l'])) + list(uncertainty[direction != 'l']))
    probability_tot   = np.array(list(np.flip(probability_tot[direction == 'l'])) + list(probability_tot[direction != 'l']))
    probability_trans = np.array(list(np.flip(probability_trans[direction == 'l'])) + list(probability_trans[direction != 'l']))
    probability_emis  = np.array(list(np.flip(probability_emis[direction == 'l'])) + list(probability_emis[direction != 'l']))
    nonlin_id_arr     = np.array(list(np.flip(nonlin_id_arr[direction == 'l'])) + list(nonlin_id_arr[direction != 'l']))
    # Plot the found period-spacing pattern.
    for i in range(len(pattern)-2):
        if i == -1:
            if np.isnan(pattern[1]):
                continue
            else:
                ax[1].plot(pattern[i], (pattern[i+1] - pattern[i])*86400, '-o', color = 'k')
        else:
            if np.isnan(pattern[i]) or np.isnan(pattern[i+1]) or np.isnan(pattern[i+2]):
                continue
            else:
                dP1 = (pattern[i+1] - pattern[i])*86400
                dP2 = (pattern[i+2] - pattern[i+1])*86400
                ax[1].plot([pattern[i], pattern[i+1]], [dP1, dP2] , '-o', color = 'k')

    dp_ = (pattern[1:] - pattern[:-1])*86400
    ax[1].plot(pattern[:-1], dp_, '--o', color = 'k', label = 'HMM')

    ax[0].set_xlim(0.9*np.nanmin(pattern), 1.1*np.nanmax(pattern))

    psp_dict['pattern']           = pattern
    psp_dict['uncertainty']       = uncertainty
    psp_dict['total_prob']        = probability_tot
    psp_dict['transmission_prob'] = probability_trans
    psp_dict['emission_prob']     = probability_emis
    psp_dict['initial_periods_indices'] = [ip1, ip2, ip3]
    psp_dict['nonlin_id']         = nonlin_id_arr



# Create a button widget and add it to the plot
button_ax = plt.axes([0.25, 0.93, 0.1, 0.05])  # Define the button's position and size
button = Button(button_ax, 'Search')  # Create the button
button.on_clicked(button_search)  # Connect the button to the callback function



def reset_selections(event):
    '''
    Reset button to restart if the initial suggestion was not OK.
    '''
    global ax, initial_period_index, fig, lines
    for ind in initial_period_index:
        lines[ind].set_color('k')
    initial_period_index.clear()
    fig.canvas.draw_idle()
    ax[0].clear()
    ax[1].clear()
    ax[2].clear()
    lines.clear()
    for i in range(len(P_obs)):
        line_ = ax[0].vlines(x=P_obs[i], ymin=0, ymax=A[i], color = 'k', picker = True)
        lines.append(line_)  # Re-add lines
        fig.canvas.mpl_connect('pick_event', on_pick)
    ax[0].set_xlim(0,4)
    print('=== RESET ===')

# Create a "Reset Selections" button widget and add it to the plot
reset_button_ax = plt.axes([0.75, 0.93, 0.1, 0.05])  # Define the button's position and size
reset_button = Button(reset_button_ax, 'Reset')  # Create the button
reset_button.on_clicked(reset_selections)  # Connect the button to the callback function

def save_selections(event):
    '''
    Button to save the found period-spacing pattern to a dictionary. The following quantities are saved.
    pattern: periods of the modes in the pattern (days).
    uncertainty: observational errors on the periods of the modes.
    total_prob: total probability = p_trans*p_emis
    transmission_prob: transmission probability
    emission_prob: emission probability
    initial_periods_indices: indices of the three initial modes
    nonlin_id_arr: Non-linear mode ID (i.e. combination frequency)
    '''
    global psp_dict, fig, KIC, strategy, m, WORK_DIR
    fig.savefig(f'{WORK_DIR}/PSP_KIC0{KIC}_strategy_{strategy}_l1m{m}_test.png', dpi = 300)

    with open(f'{WORK_DIR}/PSP_KIC0{KIC}_strategy_{strategy}_l1m{m}_test.pkl', 'wb') as f:
       pickle.dump(psp_dict, f)
    print(f'Pattern saved as PSP_KIC0{KIC}_strategy_{strategy}_l1m{m}_test.pkl')

save_button_ax = plt.axes([0.50, 0.93, 0.1, 0.05])  # Define the button's position and size
save_button = Button(save_button_ax, 'Save')  # Create the button
save_button.on_clicked(save_selections)  # Connect the button to the callback function
fig.canvas.draw_idle()

ax[2].set_xlabel(r'$P\,[d]$', fontsize = fontsize)
ax[1].set_ylabel(r'$\Delta P\,[s]$', fontsize = fontsize)
ax[2].set_ylabel(r'$p_{\rm tot}$', fontsize = fontsize)
ax[0].set_ylabel(r'$\log A$', fontsize = fontsize)
ax[0].set_ylim(0, 1.1*np.max(A))
ax[2].set_ylim(0, 1.05)
ax[0].set_xlim(0,4)
ax[0].set_title(f'm = {m}')

plt.show(block = False)

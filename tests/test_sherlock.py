# Set of unit tests for SHERLOCK. The expected results of the tests are based on the grid gyre_per_l1m1_ext.pkl and star KIC09020774.
# To test, run 'pytest'.
# Joey Mombarg - 18 oct 2023


import pickle
import numpy as np
from sherlock import HMM_PSP_interactive as sh



def test_get_deltaP_sel():
    '''
    Test if the first five selected period spacings for a radial order are correct.
    '''
    n = 10
    deltaP_obs1 = 110.96743226179274
    deltaP_sel_expected = [111.91977582555879, 134.8216292514113, 180.49011280735243, 107.75430222686282, 96.774763465395]
    deltaP_all_expected = [174.78563501697337, 199.70701450480954, 227.1833281078692, -33.1995429704147, 293.65859240724143]
    deltaP_sel, deltaP_all = sh.get_deltaP_sel(n,deltaP_obs1)
    #assert deltaP_sel_expected == deltaP_sel[0:5]
    #assert deltaP_all_expected == deltaP_all[0:5]

    assert all ([abs(a-b)< 1E-10 for a,b in zip(deltaP_sel_expected, deltaP_sel[0:5])])
    assert all ([abs(a-b)< 1E-10 for a,b in zip(deltaP_all_expected, deltaP_all[0:5])])


def test_deltaP_expected():
    '''
    Test if the computed search window based on the entire grid and the expected difference in period spacing are correct.
    '''
    deltaP_all_min, deltaP_all_max, deltaP_exp, _, norm = sh.deltaP_expected(np.nan, True)
    deltaP_all_min_expected  = -552.0937563763036
    deltaP_all_max_expected  = 1064.282513540753
    deltaP_exp_expected = 2.47885505967497
    norm_expected  = 2.254370697315008

    assert np.abs(deltaP_all_min - deltaP_all_min_expected) < 1E-10
    assert np.abs(deltaP_all_max - deltaP_all_max_expected) < 1E-10
    assert np.abs(deltaP_exp - deltaP_exp_expected) < 1E-10
    assert np.abs(norm - norm_expected) < 1E-10

def test_find_next_period():
    '''
    Test if the next period found from the set of candidates is correct, and if the probabilities are correct.
    '''
    P_obs1_ini = P_obs1 = 0.5261158088986102
    P_obs2_ini = P_obs2 =  0.5432528658241979
    P_obs3_ini = P_obs3 =  0.5591055774689777

    skipped_radial_order = False
    go_on = 0
    skipped = 0

    # Compute the search window based on the entire grid of models.
    deltaP_all_min, deltaP_all_max, _, _, _ = sh.deltaP_expected(np.nan, True)

    DeltaP_obs1 = np.abs(P_obs1 - P_obs2)*86400
    DeltaP_obs2 = np.abs(P_obs2 - P_obs3)*86400
    deltaP_obs1 = DeltaP_obs1 - DeltaP_obs2

    # Compute the expected difference in period-spacing for the next period, and the search interval.
    percen5, percen95, deltaP_exp, p_trans_ipol, norm  = sh.deltaP_expected(deltaP_obs1, skipped_radial_order)
    # Compute the most probable period spacing, and the expected minimum and maximum period spacing.
    DeltaP_exp  = DeltaP_obs2 - deltaP_exp
    DeltaP_up   = DeltaP_obs2 - deltaP_all_min
    DeltaP_low  = DeltaP_obs2 - deltaP_all_max


    P_potential = np.array([0.5681374400945814, 0.5746361846637587, 0.564255594968133, 0.5720535495869257])
    A_potential = np.array([0.6779919598227506, 0.8452006804965032, 0.4684026337411909, 0.339648036683893])
    # Compute the emission probabilities of all candidates.
    p_emission  = sh.p_emis(A_potential, in_log = True)
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

    print(P_next)
    P_next_expected = 0.5746361846637587
    p_transition_expected = 0.0018011258929610085
    p_emission_expected = 0.41448940939460216
    assert np.abs(P_next - P_next_expected) < 1E-10
    assert np.abs(p_transition[np.argmax(p_total)] - p_transition_expected) < 1E-10
    assert np.abs(p_emission[np.argmax(p_total)] - p_emission_expected) < 1E-10

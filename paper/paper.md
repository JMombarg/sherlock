---
title: 'SHERLOCK: Scheme using Hidden markov models for Establishing Reiteratively a List Of Candidate period-spacings with liKelihood'
tags:
  - Python
  - astronomy
  - stellar astrophysics  
  - asteroseismology
authors:
  - name: Joey S. G. Mombarg
    orcid: 0000-0002-9901-3113
    affiliation: 1

affiliations:
 - name: IRAP, Université de Toulouse, CNRS, UPS, CNES, 14 avenue Édouard Belin, F-31400 Toulouse, France
   index: 1
date: 1 November 2023
bibliography: paper.bib
---

# Context

Recent space-borne missions such as the NASA *Kepler* and TESS mission have provided light curves of thousands of pulsating stars, and future missions like the ESA PLATO mission are expected to increase this number even more. The so-called gravity (g)-mode pulsators are particularly interesting to study the internal stellar structure, as the periods of g~modes excited in core-hydrogen burning stars are mostly sensitive to the physics in the deep interior. These g-mode pulsators are typically modelled with a forward-modelling approach to infer fundamental stellar properties such as the mass, age, and rotation frequency (e.g. @Pedersen2021;@Mombarg2021). Starting from a processed light curve, frequencies are extracted via iterative prewhitening. This yields a list of frequencies and their amplitudes of the g-mode pulsations, but also combination frequencies and possibly spurious ones. The stellar oscillations are described by spherical harmonics with three quantum numbers; the spherical degree, azimuthal order, and radial order. In order to compare the observed frequencies with models, the pulsation modes and the corresponding quantum numbers need to be identified. For g-mode pulsators, this is typically done by exploiting a relation between the pulsation periods of consecutive radial order and equal ($\ell, m$) [@Tassoul1980;@Miglio2008]. In period space, the so-called period-spacings (difference in period between consecutive radial orders) of g-mode pulsations should follow a pattern. The next step after iterative prewhitening is the search for period-spacing patterns in the frequency list is often non-trivial and is often done by a trained asteroseismologist. Yet, this means that the found period-spacing patterns are somewhat subjective, which can have consequences for the fundamental stellar parameters in the modelling. Automated routines to search for period-spacing patterns exist [@Li2020;@Garcia2022], but these do not account for any shifts in the pulsation periods due to the effect of a gradient in the mean molecular weight building up as the star evolves, or mode coupling. This can result in missing pulsation modes that are particularly interesting to test the theory of stellar structure. `SHERLOCK` is an open-source `Python` routine designed to extract to most probable period-spacing pattern based on predictions of a grid of stellar pulsation models.
Moreover, it provides a metric of confidence how likely a selected period is actually part of the pattern. This way in the modelling a lower priority can be assigned to pulsation periods that are more likely to have been mistaken.

# Methology

The tool presented here is based on Hidden Markov Models (HMMs). A HMM is defined by a sequence of hidden states $h_i$, and visible states, where the index $i$ indicates the time step or iteration. HMM are for example used to correct typos make by the accidental pressing of an adjacent key instead. In this analogy, the hidden states $h_i$ are the intended letters and the visible states $v_i$ are the pressed keys. Given state $h_i$, the probability for a state $h_{i+1}$ is given by, $p(h_{i+1} | h_i)$,
and a referred to as the transition probability. This transition probability in this analogy can be determined by training the model on a dictionary to determine the probability of the next letter given the previous one. At each time step, the HMM emits a visible state $v_i$ with probability, $P(v_i | h_i)$, which is referred to as the emission probability. This probability determines how likely it is to press a specific key, given an intended letter. Translating this example to the search for period-spacing patterns, we can think of the transmission probability as the probability of a period-spacing $\Delta P_{i}$ belonging to the pattern according to theory, given a previous period-spacing $\Delta P_{i-1}$. This probability is determined from a grid of stellar pulsation models that provides probability distributions. The `SHERLOCK` algorithm works with differences in period-spacings,
\begin{equation}
    \delta(\Delta P)_i = \Delta P_i - \Delta P_{i+1} = (P_{i+1} - P_i) - (P_{i+2} - P_{i+1}),
\end{equation}
where $P_i < P_{i+1} < P_{i+2}$. For each radial order in the grid, we compute $\delta (\Delta P)_{1}$ and $\delta (\Delta P)_{2}$. For an observed $\delta(\Delta P)_{1, \rm obs}$, we select the 500 values of $\delta(\Delta P)_1$ that are closest to the observed one. Next, we compute distribution of the values of $\delta(\Delta P)_2$, that is, the expected next expected difference in period-spacing. This distribution is interpolated to make a PDF from which the transition probability is computed.
The emission probability is based on the number of periods in the list that falls within a confidence interval, where periods with higher amplitudes are favoured over lower amplitude periods, as lower amplitude periods are more likely to be spurious. This probability is defined as,
\begin{equation}
    p_{{\rm emis}, i} = \frac{A_i}{\sum_j A_j},
\end{equation}
where $A_i$ is the amplitude of period $P_i$, and the index $j$ runs over all potential periods that fall within the search window. The search window (or confidence interval) is taken to be the most extreme values for which $p_{\rm emis}/{\rm max}(p_{\rm emis}) > 0.001$. An example of the distribution of this quantity is shown in \autoref{fig:prob_distr}.

The algorithm is initiated with the periods of three consecutive radial orders, provided by the user. `SHERLOCK` will then compute a search window, defined as the minimum and maximum period for which the probability is larger than 0.001. Then, for each candidate in the search window a total probability is computed,
\begin{equation}
    p_{{\rm total}, i} = p_{{\rm emis},i} p_{{\rm trans}, i},
\end{equation}
and is normalised such that $\sum_i p_{{\rm total}, i} = 1$. The period with the highest total probability is then selected and the next iteration starts.

![Normalised interpolated emission probabilty for a difference in period-spacings $\Delta P_{1} - \Delta P_{2}$, computed from a grid of stellar pulsation models. The green tick indicates the expected value, the grey ticks the search window, based on the cutoff in $\log p/p_{\rm max}$ as shown by the horizontal grey dashed line. \label{fig:prob_distr}](figures/Probability_distribution.png){ width=80% }

# Stellar models for training

In order to calculate the transmission probability, a grid of `GYRE 6.0` [@Townsend2013] pulsation models was constructed to compute the periods of the stellar eigenmodes, specifically for dipole ($\ell = 1$) modes with $m \in [-1, 0, 1]$ and for radial orders $n_{\rm pg} = -100$ to $-1$. Since the probabilites are based on differences in periods-spacing, period-spacing patterns of higher spherical degree can also be found without needing the extent the grid, as these modes behave the same way as dipole modes with the same sign for $m$. A uniform rotation profile is assumed, where the a randomly picked rotation rate between 0 and 0.6 times the critical rotation rate is picked for each stellar equilibrium model. The stellar equilibrium models are computed with `MESA r22.11.1` [@Paxton2011;@Paxton2013;@Paxton2015;@Paxton2018;@Paxton2019;@Jermyn2023], where the mass is sampled according to a Sobol sequence between 3 and 9$\,{\rm M}_{\rm \odot}$. The chemical diffusion coefficient due to core boundary mixing (overshoot) is described by an exponential decaying function where coefficient related to the width of the zone, $f_{\rm ov}$, is sampled in a similar way between values of 0.005 and 0.035. The chemical mixing in the radiative envelope is modelled by mixing induced due to internal gravity waves [e.g. @Varghese2023].

# Performance

To demonstrate to performance of `SHERLOCK`, we use the well-studied gravity-mode pulsator KIC7760680 [see e.g. @Papics2015]. Earlier studies that have manually searched for a period-spacing pattern in pulsation spectrum of this stars found a pattern of 36 consecutive radial orders. Furthermore, the dips found in the period-spacing pattern is indicative a chemical gradient near the core. \autoref{fig:gui} shows the pattern found by `SHERLOCK`, where the red colored periods are the initial three consecutive radial order suggested by the user. Once the user has found a possible start of a pattern, it takes `SHERLOCK` less than 2 seconds to complete it. For this star, initial suggestions of periods labeled (2,3,4) or (6,7,8) in the top panel of \autoref{fig:gui}, result in the same final pattern found. However, when starting for example from the periods (4,3,2) around 1 day, the same pattern is not recovered. Therefore, the initial periods should ideally be three high-amplitude periods with similar spacing. In any case, an incorrect suggested start will result in an incorrect final result that is not always trivial to notice, but typically the longest possible pattern is the correct one. The bottom panel, showing the total probabilty for each selected period, indicates that for example mode 2 toward periods shorter than the initial ones, and modes (5,6) towards periods larger than the initial ones, could have possibly been mistaken as other candidates with high probability are present within the search window. Therefore, when performing forward modelling of this star's period-spacing pattern, these modes can be assigned less weight according to the total probability given by `SHERLOCK`.

![Example of the interface and output of `SHERLOCK`. Top panel: Periods and amplitudes of the inserted frequency list. Periods colored red were selected by `SHERLOCK` to be part of the period-spacing pattern, assuming azimuthal order $m = 1$. The green shaded areas inicate the search windows and the dashed line the expected location (in grey if no period was found). Middle panel: The period vs period-spacing of the selected modes. Bottom panel: Total probability for each selected mode. Red colored modes indicate the three initial modes suggested by the user. \label{fig:gui}](figures/PSP_KIC007760680_strategy_5_l1m1_test.png){ width=95% }

# Acknowledgements
 The research leading to these results has received funding the French Agence Nationale de la Recherche (ANR), under grant MASSIF (ANR-21-CE31-0018-02). The author thanks Mathias Michielsen for his advice on unit tests and `Poetry`.

# References

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

# Introduction

Recent space-borne missions such as the NASA Kepler and TESS mission have provided light curves of many pulsating stars, and future missions like ESA's PLATO mission are expected to increase this number even more. The so-called gravity (g)-mode pulsators are particularly interesting to study the internal stellar structure, as the periods of g~modes excited in core-hydrogen burning stars are mostly sensitive to the physics in the deep interior. These g-mode pulsators are typically modelled with a forward-modelling approach to infer fundamental stellar properties such as the mass, age, and rotation velocity [@Pedersen2020, @Mombarg2021]. Starting from a processed light curve, frequencies are extracted via iterative prewhitening. This yields a list of frequencies and their amplitudes of the g-mode pulsations, but also combination frequencies and possibly spurious ones. The stellar oscillations are described by spherical harmonics with three quantum numbers; the spherical degree, azimuthal order, and radial order. In order to compare the observed frequencies with models, the pulsation modes and the corresponding quantum numbers need to be identified. For g-mode pulsators, this is typically done by exploiting a relation between the pulsations period of consecutive radial order [@Tassoul1980, @Miglio2008], and equal ($\ell, m$). In period space, the so-called period-spacings of g-mode pulsations should follow a pattern. The next step is the search for period-spacing patterns in the frequency list is often non-trivial and is often done by a trained asteroseismologist. Yet, this means that the found period-spacing patterns are somewhat subjective, which can have consequences for the fundamental stellar parameters in the modelling. Automated routines to search for period-spacing patterns exist [@Li2020, @Garcia2022], but these do not account for any shifts in the pulsation periods due to the effect of a gradient in the mean molecular weight building up as the star evolves, or mode coupling. This can result in missing pulsation modes that are particularly interesting to test the theory of stellar structure. `SHERLOCK` is an open-source Python routine designed to extract to most probable period-spacing pattern based on predictions of a grid of stellar pulsation models.
Moreover, it provides a metric of confidence how likely a selected period is actually part of the pattern. This way in the modelling a lower priority can be assigned to pulsation periods that are more likely to have been mistaken.

# Methology

The tool presented here is based on Hidden Markov Models (HMMs). A HMM is defined by a sequence of hidden states $h_i$, and visible states, where the index $i$ indicates the time step or iteration. HMM are for example used to correct typos make by the accidental pressing of an adjacent key instead. In this analogy, the hidden states $h_i$ are the intended letters and the visible states $v_i$ are the pressed keys. Given state $h_i$, the probability for a state $h_{i+1}$ is given by, $P(h_{i+1} | h_i)$,
and a referred to as the transition probability. This transition probability in this analogy can be computed by training on a dictionary to determine the probability of the next letter given the previous one. At each time step, the HMM emits a visible state $v_i$ with probability, $P(v_i | h_i)$, which is referred to as the emission probability. This probability determines how likely it is to press a specific key, given an intended letter. Translating this example to the search for period-spacing patterns, we can think of the transmission probability as the probability of a period-spacing $\Delta P_{i}$ belonging to the pattern according to theory, given a previous period-spacing $\Delta P_{i-1}$. This probability is determined from a grid of stellar pulsation models that provides probability distributions. The \sherlock algorithm works with differences in period-spacings,
\begin{equation}
    \delta(\Delta P)_i = \Delta P_i - \Delta P_{i+1} = (P_{i+1} - P_i) - (P_{i+2} - P_{i+1}),
\end{equation}
where $P_i < P_{i+1} < P_{i+2}$. For each radial order in the grid, we compute $\delta(\Delta P)_1 $ and $ \delta(\Delta P)_2 $. For an observed $\delta(\Delta P)_{1, \rm obs} $, we select the 500 values of $\delta(\Delta P)_1$ that are closest to the observed one. Next, we compute distribution of the values of $\delta(\Delta P)_2$, that is, the expected next expected difference in period-spacing. This distribution is interpolated to make a PDF from which the transition probability is computed.
The emission probability is based on the number of periods in the list that falls within a 95\% confidence interval, where periods with higher amplitudes are favoured over lower amplitude periods, as lower amplitude periods are more likely to be spurious. This probability is defined as,
\begin{equation}
    P_{{\rm emis}, i} = \frac{A_i}{\sum_j A_j},
\end{equation}
where $A_i$ is the amplitude of period $P_i$, and the index $j$ runs over all potential periods that fall within the search window.

The algorithm is initiated with the periods of three consecutive radial orders, provided by the user. \sherlock will then compute a search window defined as the minimum and maximum period for which the probability is larger than 0.001. Then, for each candidate in the search window a total probability is computed,
\begin{equation}
    P_{{\rm total}, i} = P_{{\rm emis},i} P_{{\rm trans}, i},
\end{equation}
and is normalised such that $\sum_i P_{{\rm total}, i} = 1$. The period with the highest total probability is then selected and the next iteration starts.

# Stellar models

In order to calculate the transmission probability, a grid of `GYRE` pulsation models was constructed to compute the periods of the stellar eigenmodes, specifically for dipole ($\ell = 1$) modes with $m \in [-1, 0, 1]$ and for radial orders $n_{\rm pg} = -100$ to $-1$. Since the probabilites are based on differences in periods-spacing, period-spacing patterns of higher spherical degree can also be found without needing the extent the grid, as these modes behave the same way as dipole modes with the same sign for $m$. A uniform rotation profile is assumed, where the a randomly picked rotation rate between 0 and 0.6 times the critical rotation rate is picked for each stellar equilibrium model. The stellar equilibrium models are computed with `MESA`, where the mass is sampled according to a Sobol sequence between 3 and 9\Msun. The chemical diffusion coefficient due to core boundary mixing (overshoot) is described by an exponential decaying function where coefficient related to the width of the zone, $f_{\rm ov}$, is sampled in a similar way between values of 0.005 and 0.035. The chemical mixing in the radiative envelope is modelled by mixing induced due to internal gravity waves (IGWs).

#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 26 15:20:41 2018

@author: casasorozco
"""

from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator

plt.style.use('~/Dropbox/phd_degree/general/python/daniel_thesis.mplstyle')


def profiles(k_1, k_m1, t_span, a_zero=1, b_zero=0, derivative_flag=False):

    a_profile = a_zero/(k_1 + k_m1) * \
        (k_1 * np.exp(-(k_1 + k_m1)*t_span) + k_m1)

    b_profile = a_zero - a_profile

    derivatives = []

    if derivative_flag:

        da_dk_1 = -a_zero * (
                np.exp(-timespan*(k_1 + k_m1)) *
                (1 - k_1 * timespan) / (k_1 + k_m1) -
                (k_1*np.exp(-timespan*(k_1 + k_m1)) + k_m1)/(k_1 + k_m1)**2)



        da_dk_m1 = -a_zero * (
                (1 - k_1 * timespan * np.exp(timespan*(-k_1 - k_m1)))/(k_1 + k_m1) - \
                (k_1 * np.exp(timespan*(-k_1 - k_m1)) + k_m1)/(k_1 + k_m1)**2
                )

        derivatives.append(da_dk_1)
        derivatives.append(da_dk_m1)

    return a_profile, b_profile, derivatives


if __name__ == '__main__':

    timespan = np.linspace(0, 0.5)

    k_1 = 3
    k_m1 = 3

    a_values, b_values, der = profiles(k_1, k_m1, timespan,
                                       derivative_flag=True)

    # ---------- Plot
    # States
    fig, axis = plt.subplots()

    axis.plot(timespan, a_values)
    axis.plot(timespan, b_values, '--')

    axis.legend(('[A]', '[B]'), loc='best')
    axis.xaxis.set_minor_locator(AutoMinorLocator(2))
    axis.yaxis.set_minor_locator(AutoMinorLocator(2))

    fig.savefig('../img/saltelli_example_profiles.pdf', bbox_inches='tight')
    # Derivatives
    fig_der, axis_der = plt.subplots(2, 1, figsize=(4.5, 5.8))

    ## raw
    axis_der[0].plot(timespan, abs(der[0]))
    axis_der[0].plot(timespan, abs(der[1]), '--')

    axis_der[0].legend((r'$\frac{\partial [A]}{\partial k_1}$',
                        r'$\frac{\partial [A]}{\partial k_{-1}}$'), loc='best')

    ## standarized
    sd_k1 = 0.3
    sd_km1 = 1

    std_a_squared = der[0]**2 * sd_k1**2 + der[1]**2 * sd_km1**2

    sens_k1 = sd_k1/np.sqrt(std_a_squared) * der[0]
    sens_km1 = sd_km1/np.sqrt(std_a_squared) * der[1]

    axis_der[1].plot(timespan, abs(sens_k1))
    axis_der[1].plot(timespan, abs(sens_km1), '--')

    axis_der[1].legend((r'$S^{\sigma}_{k_1} = \frac{\sigma_{k_1}}{\sigma_{[A]}} \frac{\partial [A]}{\partial k_1}$',
                        r'$S^{\sigma}_{k_{-1}} = \frac{\sigma_{k_{-1}}}{\sigma_{[A]}} \frac{\partial [A]}{\partial k_{-1}}$'),
                       loc='best')

    for ax in axis_der:
        ax.xaxis.set_minor_locator(AutoMinorLocator(2))
        ax.yaxis.set_minor_locator(AutoMinorLocator(2))

    fig_der.savefig('../img/saltelli_example_der.pdf', bbox_inches='tight')



#    scatter_ids = range(0, len(timespan), len(timespan)//4)






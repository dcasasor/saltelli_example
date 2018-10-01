#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Sep 29 12:08:37 2018

@author: casasorozco
"""

from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator
from example_saltelli import profiles


def scatter_plot(ids, k_label):
        fig_scatter, ax_scatter = plt.subplots(2, 2, sharex=True)
        ax_scatter_flat = ax_scatter.flatten()

        if k_label is 'k_1':
            factor = k1_normal
        else:
            factor = km1_normal

        for num, ind in enumerate(ids):
            ax_scatter_flat[num].scatter(factor, a_montecarlo[:, ind],
                                         facecolors='None', edgecolors='k',
                                         lw=0.4, s=2)
            ax_scatter_flat[num].text(
                    0, 1.04, 'time = %.3f' % timespan[ind],
                    transform=ax_scatter_flat[num].transAxes,
                    fontsize=8)

            ax_scatter_flat[num].xaxis.set_minor_locator(AutoMinorLocator(2))
            ax_scatter_flat[num].yaxis.set_minor_locator(AutoMinorLocator(2))

        fig_scatter.text(0.5, 0, '$%s$' % k_label)
        fig_scatter.text(0, 0.5, '$[A]$', rotation=90)

        return fig_scatter, ax_scatter


if __name__ == '__main__':

    # Data
    k_1 = 3
    k_m1 = 3

    sd_k1 = 0.3
    sd_km1 = 1

    timespan = np.linspace(0, 0.5)

    num_samples = 256

    # Create random samples
    k1_normal = np.random.randn(num_samples) * sd_k1 + 3
    km1_normal = np.random.randn(num_samples) * sd_km1 + 3

    # Create model responses
    a_montecarlo = np.zeros(shape=(num_samples, len(timespan)))
    b_montecarlo = np.zeros_like(a_montecarlo)

    fig_mcarlo, ax_mcarlo = plt.subplots(figsize=(4.5, 2.81))

    for ind, constants in enumerate(zip(k1_normal, km1_normal)):
        a_profile, b_profile, _ = profiles(t_span=timespan, *constants)

        a_montecarlo[ind] = a_profile
        b_montecarlo[ind] = b_profile

        # Plot
        ax_mcarlo.plot(timespan, a_profile, 'k', alpha=0.1)
        ax_mcarlo.plot(timespan, b_profile, 'b', alpha=0.1)

    # Plot
    fig_mcarlo.savefig(
            '../img/saltelli_example_mc_profiles.pdf', bbox_inches='tight')

    scatter_ids = (6, 14, 24, 49)

    fig_scat_k1, ax_scat_k1 = scatter_plot(scatter_ids, 'k_1')
    fig_scat_km1, ax_scat_km1 = scatter_plot(scatter_ids, 'k_{-1}')

    fig_scat_k1.savefig(
            '../img/saltelli_example_scatter_k1.pdf', bbox_inches='tight')

    fig_scat_km1.savefig(
            '../img/saltelli_example_scatter_km1.pdf', bbox_inches='tight')

    # --------------- Sensitivity using linear regression ([A] vs k_1 and k_m1)


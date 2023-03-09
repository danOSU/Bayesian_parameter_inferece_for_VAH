#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar  5 18:10:12 2022

@author: ozgesurer
"""
import scipy.stats as sps
import numpy as np

xlimits = np.array([[10, 30],
                    [-0.7, 0.7],
                    [0.5, 1.5],
                    [0, 1.7],
                    [0.3, 2],
                    [0.135, 0.165],
                    [0.13, 0.3],
                    [0.01, 0.2],
                    [-2, 1],
                    [-1, 2],
                    [0.01, 0.25],
                    [0.12, 0.3],
                    [0.025, 0.15],
                    [-0.8, 0.8],
                    [0.3, 1]])
xlb = xlimits[:, 0]
xub = xlimits[:, 1]
# beta distribution a = b = l
l=1.02
class prior_VAH:
    def lpdf(theta):
        return (sps.beta.logpdf(theta[:, 0], l, l, xlb[0], xub[0] - xlb[0]) +  # Pb_Pb [10, 30]
                              sps.beta.logpdf(theta[:, 1], l, l, xlb[1], xub[1] - xlb[1]) + # Mean [-0.7, 0.7]
                              sps.beta.logpdf(theta[:, 2], l, l, xlb[2], xub[2] - xlb[2]) + # Width [0.5, 1.5]
                              sps.beta.logpdf(theta[:, 3], l, l, xlb[3], xub[3] - xlb[3]) + # Dist [0, 1.7**3]
                              sps.beta.logpdf(theta[:, 4], l, l, xlb[4], xub[4] - xlb[4]) + # Flactuation [0.3, 2]
                              sps.beta.logpdf(theta[:, 5], l, l, xlb[5], xub[5] - xlb[5]) + # Temp [0.135, 0.165]
                              sps.beta.logpdf(theta[:, 6], l, l, xlb[6], xub[6] - xlb[6]) + # Kink [0.13, 0.30]
                              sps.beta.logpdf(theta[:, 7], l, l, xlb[7], xub[7] - xlb[7]) + # eta_s [0.01, 0.2]
                              sps.beta.logpdf(theta[:, 8], l, l, xlb[8], xub[8] - xlb[8]) + # slope_low [-2, 1]
                              sps.beta.logpdf(theta[:, 9], l, l, xlb[9], xub[9] - xlb[9]) + # slope_high [-1, 2]
                              sps.beta.logpdf(theta[:, 10], l, l, xlb[10], xub[10] - xlb[10]) + # max [0.01, 0.25]
                              sps.beta.logpdf(theta[:, 11], l, l, xlb[11], xub[11] - xlb[11]) + # Temp_peak [0.12, 0.30]
                              sps.beta.logpdf(theta[:, 12], l, l, xlb[12], xub[12] - xlb[12]) + # Width_peak [0.025, 0.150]
                              sps.beta.logpdf(theta[:, 13], l, l, xlb[13], xub[13] - xlb[13]) + # Asym_peak [-0.8, 0.8]
                              sps.beta.logpdf(theta[:, 14], l, l, xlb[14], xub[14] - xlb[14])).reshape((len(theta), 1)) # R [0.3, 1]


    def rnd(n):
        return np.vstack((sps.beta.rvs(l, l, xlb[0], xub[0] - xlb[0], size=n), # 0
                          sps.beta.rvs(l, l, xlb[1], xub[1] - xlb[1], size=n), # 1
                          sps.beta.rvs(l, l, xlb[2], xub[2] - xlb[2], size=n), # 2
                          sps.beta.rvs(l, l, xlb[3], xub[3] - xlb[3], size=n), # 3
                          sps.beta.rvs(l, l, xlb[4], xub[4] - xlb[4], size=n), # 4
                          sps.beta.rvs(l, l, xlb[5], xub[5] - xlb[5], size=n), # 5
                          sps.beta.rvs(l, l, xlb[6], xub[6] - xlb[6], size=n), # 6
                          sps.beta.rvs(l, l, xlb[7], xub[7] - xlb[7], size=n), # 7
                          sps.beta.rvs(l, l, xlb[8], xub[8] - xlb[8], size=n), # 8
                          sps.beta.rvs(l, l, xlb[9], xub[9] - xlb[9], size=n), # 9
                          sps.beta.rvs(l, l, xlb[10], xub[10] - xlb[10], size=n), # 10
                          sps.beta.rvs(l, l, xlb[11], xub[11] - xlb[11], size=n), # 11
                          sps.beta.rvs(l, l, xlb[12], xub[12] - xlb[12], size=n), # 12
                          sps.beta.rvs(l, l, xlb[13], xub[13] - xlb[13], size=n), # 13
                          sps.beta.rvs(l, l, xlb[14], xub[14] - xlb[14], size=n))).T
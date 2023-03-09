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
class prior_VAH:
    def lpdf(theta):
        return (sps.uniform.logpdf(theta[:, 0], xlb[0], xub[0] - xlb[0]) +  # Pb_Pb [10, 30]
                              sps.uniform.logpdf(theta[:, 1], xlb[1], xub[1] - xlb[1]) + # Mean [-0.7, 0.7]
                              sps.uniform.logpdf(theta[:, 2], xlb[2], xub[2] - xlb[2]) + # Width [0.5, 1.5]
                              sps.uniform.logpdf(theta[:, 3], xlb[3], xub[3] - xlb[3]) + # Dist [0, 1.7**3]
                              sps.uniform.logpdf(theta[:, 4], xlb[4], xub[4] - xlb[4]) + # Flactuation [0.3, 2]
                              sps.uniform.logpdf(theta[:, 5], xlb[5], xub[5] - xlb[5]) + # Temp [0.135, 0.165]
                              sps.uniform.logpdf(theta[:, 6], xlb[6], xub[6] - xlb[6]) + # Kink [0.13, 0.30]
                              sps.uniform.logpdf(theta[:, 7], xlb[7], xub[7] - xlb[7]) + # eta_s [0.01, 0.2]
                              sps.uniform.logpdf(theta[:, 8], xlb[8], xub[8] - xlb[8]) + # slope_low [-2, 1]
                              sps.uniform.logpdf(theta[:, 9], xlb[9], xub[9] - xlb[9]) + # slope_high [-1, 2]
                              sps.uniform.logpdf(theta[:, 10], xlb[10], xub[10] - xlb[10]) + # max [0.01, 0.25]
                              sps.uniform.logpdf(theta[:, 11], xlb[11], xub[11] - xlb[11]) + # Temp_peak [0.12, 0.30]
                              sps.uniform.logpdf(theta[:, 12], xlb[12], xub[12] - xlb[12]) + # Width_peak [0.025, 0.150]
                              sps.uniform.logpdf(theta[:, 13], xlb[13], xub[13] - xlb[13]) + # Asym_peak [-0.8, 0.8]
                              sps.uniform.logpdf(theta[:, 14], xlb[14], xub[14] - xlb[14])).reshape((len(theta), 1)) # R [0.3, 1]


    def rnd(n):
        return np.vstack((sps.uniform.rvs(xlb[0], xub[0] - xlb[0], size=n), # 0
                          sps.uniform.rvs(xlb[1], xub[1] - xlb[1], size=n), # 1
                          sps.uniform.rvs(xlb[2], xub[2] - xlb[2], size=n), # 2
                          sps.uniform.rvs(xlb[3], xub[3] - xlb[3], size=n), # 3
                          sps.uniform.rvs(xlb[4], xub[4] - xlb[4], size=n), # 4
                          sps.uniform.rvs(xlb[5], xub[5] - xlb[5], size=n), # 5
                          sps.uniform.rvs(xlb[6], xub[6] - xlb[6], size=n), # 6
                          sps.uniform.rvs(xlb[7], xub[7] - xlb[7], size=n), # 7
                          sps.uniform.rvs(xlb[8], xub[8] - xlb[8], size=n), # 8
                          sps.uniform.rvs(xlb[9], xub[9] - xlb[9], size=n), # 9
                          sps.uniform.rvs(xlb[10], xub[10] - xlb[10], size=n), # 10
                          sps.uniform.rvs(xlb[11], xub[11] - xlb[11], size=n), # 11
                          sps.uniform.rvs(xlb[12], xub[12] - xlb[12], size=n), # 12
                          sps.uniform.rvs(xlb[13], xub[13] - xlb[13], size=n), # 13
                          sps.uniform.rvs(xlb[14], xub[14] - xlb[14], size=n))).T
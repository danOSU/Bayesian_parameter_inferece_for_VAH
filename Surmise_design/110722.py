#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov  7 23:00:35 2022

@author: ozgesurer
"""

import os, sys
from tabnanny import verbose
#os.chdir('/users/PAS0254/dananjaya/VAH_SURMISE/emulation')
#sys.path.append('/users/PAS0254/dananjaya/surmise')
import dill as pickle
import numpy as np
import time
from split_data import generate_split_data
from surmise.emulation import emulator
from surmise.calibration import calibrator
from plotting import *
from priors_beta import prior_VAH
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import pyximport
pyximport.install(setup_args={"include_dirs":np.get_include()},
                  reload_support=True)

####################################################
# Note: Total emu time: 390 sec.  Total cal time: 500
####################################################

seconds_st = time.time()

# Note: Here we can create different funcs to split data into training and test
drop_obs_group = ['fluct']
drop_subset = [f'{oob}_{cen}' for og in drop_obs_group for oob in obs_groups[og] for cen in obs_cent_list['Pb-Pb-2760'][oob]]
print(f'We drop these observables {drop_subset}')
f_train, f_test, theta_train, theta_test, sd_train, sd_test, y, thetanames = generate_split_data(drop_list=drop_subset)
print(f'Number of model inputs {theta_train.shape[1]}')
# Perform a closure test or not. If True will use pseudo-experimental data to test the accuracy of the inference. 
closure = False
# Combine all for calibration
if closure == False:
    fcal = np.concatenate((f_train, f_test), axis=0)
    thetacal = np.concatenate((theta_train, theta_test), axis=0)
    sdcal = np.concatenate((sd_train, sd_test), axis=0)
else:
    # Get 9 of the most accurate simulations as pseudo-experimental data. So pick from last 50 simulation points.
    # last 75 simulation data has 1600 events per design (ignoring failure rates)
    np.random.seed(19)
    y_df_array = []
    closure_param_array = []
    while len(y_df_array)<9:
        num = np.random.randint(0,50,1)
        #print(f'Using {75-num} simulation as psuedo-experimental data')
        closure_params = theta_train[-1*num,:]
        # check if dmin parameter is in the correced range of [0,1.7]
        if closure_params[0,3] > 1.7:
            continue
        y_temp = np.vstack([f_train[-1*num,:], np.square(sd_train[-1*num,:])])
        y = pd.DataFrame(y_temp, index=['mean', 'variance'])
        print(closure_params)
        f_train = np.delete(f_train, -1*num, 0)
        sd_train = np.delete(sd_train, -1*num, 0)
        theta_train = np.delete(theta_train, -1*num, 0)
        y_df_array.append(y)
        closure_param_array.append(closure_params.flatten())
    closure_param_array = np.array(closure_param_array)
    print(f'Shape of closure param array {closure_param_array.shape}')
    fcal = np.concatenate((f_train, f_test), axis=0)
    thetacal = np.concatenate((theta_train, theta_test), axis=0)
    sdcal = np.concatenate((sd_train, sd_test), axis=0)


print(f'Shape of the emulation data set {thetacal.shape}')
x_np = np.arange(0, fcal.shape[1])[:, None]
x_np = x_np.astype('object')

##########################################################
# Note: Pick method_name = 'PCGPwM' or 'PCGPR' or 'PCSK'
##########################################################

method_name = 'PCSK'
is_train = False
if closure==False:
    emu_path = 'VAH_' + method_name + '.pkl' 
else:
    emu_path = 'VAH_' + method_name + '_closure_'  + '.pkl'
    np.save(f'multiple_closure_parameters_{method_name}', closure_param_array)
        
prior_min = [10, -0.7, 0.5, 0, 0.3, 0.135, 0.13, 0.01, -2, -1, 0.01, 0.12, 0.025, -0.8, 0.3]
prior_max = [30, 0.7, 1.5, 1.7, 2, 0.165, 0.3, 0.2, 1, 2, 0.25, 0.3, 0.15, 0.8, 1]
prior_dict = {'min': prior_min, 'max': prior_max}

if (os.path.exists(emu_path)) and (is_train==False):
    print('Saved emulators exists and overide is prohibited')
    with open(emu_path, 'rb') as file:
        emu_tr = pickle.load(file)    
else:
    print('training emulators')
    if method_name == 'PCGPwM':
        emu_tr = emulator(x=x_np,
                          theta=thetacal,
                          f=fcal.T,
                          method='PCGPwM',
                          args={'epsilon': 0.05})
        
    elif method_name == 'PCGPR':
        emu_tr = emulator(x=x_np,
                          theta=thetacal,
                          f=fcal.T,
                          method='PCGPR',
                          args={'epsilon': 0.02,
                                'prior': prior_dict})
    elif method_name == 'PCSK':
        #Scale pt_fluc uncertainity
        #l_in=index['pT_fluct']
        #print(l_in)
        #sdcal[:,index['pT_fluct'][0]:-1] =  10* sdcal[:,index['pT_fluct'][0]:-1]
        #sdcal = np.sqrt(np.absolute(sdcal))
        emu_tr = emulator(x=x_np,
                          theta=thetacal,
                          f=fcal.T,
                          method='PCSK',
                          args={'numpcs': 12, # this does not mean full errors
                                'simsd': np.absolute(sdcal.T),
                                'verbose': 1})


    if (is_train==True) or not(os.path.exists(emu_path)):
        with open(emu_path, 'wb') as file:
            pickle.dump(emu_tr, file)


seconds_end = time.time()
print('Total emu time:', seconds_end - seconds_st)


seconds_st = time.time()

y_mean = np.array(y.iloc[0])
obsvar = np.array(y.iloc[1])
from smt.sampling_methods import LHS
# code to create a new design
# define limits
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

# obtain sampling object
sampling = LHS(xlimits=xlimits)
num = 5000
x = sampling(num)
print(x.shape)

# convert data into data frame
df = pd.DataFrame(x, columns = ['Pb_Pb',
                                'Mean',
                                'Width',
                                'Dist',
                                'Flactutation',
                                'Temp',
                                'Kink',
                                'eta_s',
                                'Slope_low',
                                'Slope_high',
                                'Max',
                                'Temp_peak',
                                'Width_peak',
                                'Asym_peak',
                                'R'])

#obsvar =  #0.01*y_mean + np.square(y_sd) #np.maximum(0.00001, 0.2*y_mean)

def loglik(obsvar, emu, theta, y, x):
 

    # Obtain emulator results
    emupredict = emu.predict(x, theta)
    emumean = emupredict.mean()

    try:
        emucov = emupredict.covx()
        is_cov = True
    except Exception:
        emucov = emupredict.var()
        is_cov = False

    p = emumean.shape[1]
    n = emumean.shape[0]
    y = y.reshape((n, 1))

    loglikelihood = np.zeros((p, 1))

    for k in range(0, p):
        m0 = emumean[:, k].reshape((n, 1))

        # Compute the covariance matrix
        if is_cov is True:
            s0 = emucov[:, k, :].reshape((n, n))
            CovMat = s0 + np.diag(np.squeeze(obsvar))
        else:
            s0 = emucov[:, k].reshape((n, 1))
            CovMat = np.diag(np.squeeze(s0)) + np.diag(np.squeeze(obsvar))

        # Get the decomposition of covariance matrix
        CovMatEigS, CovMatEigW = np.linalg.eigh(CovMat)

        # Calculate residuals
        resid = m0 - y

        CovMatEigInv = CovMatEigW @ np.diag(1/CovMatEigS) @ CovMatEigW.T
        loglikelihood[k] = float(-0.5 * resid.T @ CovMatEigInv @ resid -
                                 0.5 * np.sum(np.log(CovMatEigS)))

    return loglikelihood


import random


theta = np.array(df)
loglikelihood = loglik(obsvar, emu_tr, theta, y_mean, x_np)
maxid = np.argmax(loglikelihood)
theta_sc = xlimits[:,1] - xlimits[:,0]

continuing = True
theta_curr = theta[maxid]

iterator = 0
in_id = []
out_id = list(np.arange(0, num))
in_id.append(maxid)
out_id.remove(maxid)

n_select = 29
p = theta.shape[1]

for j in range(n_select):
    iterator += 1
    best_obj = -np.inf
    best_id = -5
    for o_id in out_id:
        dist = np.sqrt(np.sum(((theta[o_id, :] - theta[in_id, :]) / theta_sc)**2, axis=1))
        ll_cand = 1/(2*p)*(loglikelihood[o_id])
        ll_i = 1/(2*p)*(loglikelihood[in_id])
        inner_metric = ll_cand + ll_i + np.log(dist)[:, None]
        cand_value = min(inner_metric)

        if cand_value > best_obj:
            best_obj = cand_value
            best_id = o_id

    in_id.append(best_id)
    out_id.remove(best_id)


plt.hist(loglikelihood[in_id])
plt.show()

plt.hist(loglikelihood[out_id])
plt.show()

theta_in = pd.DataFrame(theta[in_id, :])
theta_out = pd.DataFrame(theta[out_id, :])
theta_in['data'] = 'in'
#theta_out['data'] = 'out'
frames = [theta_in]
frames = pd.concat(frames)
sns.pairplot(frames, hue='data', diag_kind="hist")
plt.show()

theta_in = pd.DataFrame(np.round(theta[in_id, :], 4), columns = ['Pb_Pb',
                                                                 'Mean',
                                                                 'Width',
                                                                 'Dist',
                                                                 'Flactutation',
                                                                 'Temp',
                                                                 'Kink',
                                                                 'eta_s',
                                                                 'Slope_low',
                                                                 'Slope_high',
                                                                 'Max',
                                                                 'Temp_peak',
                                                                 'Width_peak',
                                                                 'Asym_peak',
                                                                 'R'])
theta_in.to_csv(r'add_design_110822.txt', header=True, index=None, sep=' ', mode='a')

import os, sys
from tabnanny import verbose
os.chdir('/users/PAS0254/dananjaya/VAH_SURMISE/emulation')
sys.path.append('/users/PAS0254/dananjaya/surmise')
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
is_train = True
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

####################################################
# CALIBRATOR
####################################################

calibrate = True


if closure==False:
    cal_path = 'VAH_' + method_name + '_calibrator_PTMC' + '.pkl' 
    if (os.path.exists(cal_path)) and (calibrate==False):
        print('Saved Calibrators exists and overide is prohibited')
        with open(cal_path, 'rb') as file:
            cal = pickle.load(file)    
    else:
        y_mean = np.array(y.iloc[0])
        obsvar = np.array(y.iloc[1])
        print(obsvar < 10**(-6))
        print(obsvar[obsvar < 10**(-6)])
        #obsvar[obsvar < 10**(-6)] = 10**(-6)
        #l_in=index['pT_fluct']
        #print('Changing the observable variance for pT_fluct')
        #obsvar[l_in[0]:l_in[1]] = 2*obsvar[l_in[0]:l_in[1]]
        if calibrate:
    #       cal = calibrator(emu=emu_tr,
    #                        y=y_mean,
    #                        x=x_np,
    #                        thetaprior=prior_VAH,
    #                        method='directbayeswoodbury',
    #                        args={'sampler': 'PTLMC',
    #                              'numtemps': 100,
    #                              'numchain': 50,
    #                              'maxtemp': 100,
    #                              'sampperchain': 5000},
    #                        yvar=obsvar)
            cal = calibrator(emu=emu_tr,
                            y=y_mean,
                            x=x_np,
                            thetaprior=prior_VAH,
                            method='directbayeswoodbury',
                            args={'sampler': 'PTMC',
                                    'nburnin' : '1000',
                                    'ndim' : '15',
                                    'niterations' : '5000' ,
                                    'ntemps' : '500',
                                    'nthin' : '10',
                                    'nwalkers' : '100' ,
                                    'nthreads' : '28',
                                    'Tmax' : '1000'},
                            yvar=obsvar)
                            
                            
        
        
        if (calibrate==True) or not(os.path.exists(cal_path)):
            with open(cal_path, 'wb') as file:
                pickle.dump(cal, file)
    
    seconds_end = time.time()
    print('Total cal time:', seconds_end - seconds_st)      

# If false, do not try to find the MAP value and load it from a saved file.
from plotting import *
from priors import xlimits
from priors import prior_VAH

find_map_param = True
from scipy import optimize
if find_map_param == True:
    bounds=[(a,b) for (a,b) in zip(xlimits[:,0],xlimits[:,1])]
    x0=[(a+b)/2 for (a,b) in zip(xlimits[:,0],xlimits[:,1])] 
    #x0=[(a+b)/4 for (a,b) in zip(xlimits[:,0],xlimits[:,1])] 
    print(bounds)
    #rslt = optimize.differential_evolution(lambda x: -1*cal.theta.lpdf(theta=np.array(x).reshape(-1,15)).flatten(),
    #                                        bounds=bounds,
    #                                       maxiter=100000,
    #                                        tol=1e-9,
    #                                        disp=True)
    minimizer_kwargs = {"method": "L-BFGS-B", "bounds": bounds,"tol":1e-20}
    rslt=optimize.basinhopping(lambda x: -cal.theta.lpdf(theta=np.array(x).reshape(-1,15)).flatten()
                                ,x0,niter=1000,minimizer_kwargs=minimizer_kwargs, T=1)
    MAP = rslt.x
    print(MAP)
    np.save(cal_path+'_MAP', MAP)
    #print(MAP)
else:
    MAP = np.load(cal_path+'_MAP.npy')

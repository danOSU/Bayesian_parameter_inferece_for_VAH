import numpy as np
import pandas as pd
from smt.sampling_methods import LHS
import seaborn as sns
from datetime import datetime

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from surmise.emulation import emulator
from surmise.calibration import calibrator
import scipy.stats as sps
from scipy import stats

import pyximport
pyximport.install(setup_args={"include_dirs":np.get_include()},
                  reload_support=True)

df_mean = pd.read_csv('mean_for_200_sliced_200_events_design', index_col=0)
df_sd = pd.read_csv('sd_for_200_sliced_200_events_design', index_col=0)

df_mean_test = pd.read_csv("mean_for_50_sliced_200_events_test_design", index_col=0)
df_sd_test = pd.read_csv("sd_for_50_sliced_200_events_test_design", index_col=0)

df_mean.shape
df_sd.shape

design = pd.read_csv('sliced_VAH_090321.txt', delimiter = ' ')
design.head()
design.shape

design_validation = pd.read_csv('sliced_VAH_090321_test.txt', delimiter = ' ')

colnames = design.columns

#drop tau_initial parameter for now because we keep it fixed
design = design.drop(labels='tau_initial', axis=1)
design.shape

design_validation = design_validation.drop(labels='tau_initial', axis=1)
colnames = colnames[0:-1]

# Read the experimental data
exp_data = pd.read_csv('PbPb2760_experiment', index_col=0)
y_mean = exp_data.to_numpy()[0, ]
y_sd = exp_data.to_numpy()[1, ]

# Get the initial 200 parameter values
theta = design.head(200)
theta.head()

theta_validation = design_validation.iloc[0:50]
theta_validation.shape

plt.scatter(theta.values[:,0], df_mean.values[:,0])
plt.show()

fig, axis = plt.subplots(3, 5, figsize=(10, 10))
theta.hist(ax=axis)
plt.show()

colname_exp = exp_data.columns
#colname_sim = df_mean.columns
#colname_theta = theta.columns

# Gather what type of experimental data do we have.
exp_label = []
x = []
j = 0
x_id = []
for i in exp_data.columns:
    words = i.split('[')
    exp_label.append(words[0]+'_['+words[1])
    if words[0] in x:
        j += 1
    else:
        j = 0
    x_id.append(j)
    x.append(words[0])


# Only keep simulation data that we have corresponding experimental data
df_mean = df_mean[exp_label]
df_sd = df_sd[exp_label]

df_mean_test = df_mean_test[exp_label]
df_sd_test = df_sd_test[exp_label]

df_mean.head()

selected_observables = exp_label[0:-32]

x_np = np.column_stack((x[0:-32], x_id[0:-32]))
x_np = x_np.astype('object')
#x_np[:, 1] = x_np[:, 1].astype(int)
y_mean = y_mean[0:-32]
y_sd = y_sd[0:-32]

print(f'Last item on the selected observable is {selected_observables[-1]}')

df_mean = df_mean[selected_observables]
df_sd = df_sd[selected_observables]

df_mean_test = df_mean_test[selected_observables]
df_sd_test = df_sd_test[selected_observables]

print(f'Shape of the constrained simulation output {df_mean.shape}')

# Remove bad designs

drop_index = np.array([19, 23, 31, 32, 71, 91, 92, 98, 129, 131, 146, 162, 171, 174, 184, 190, 194, 195, 198])
drop_index_vl = np.array([29, 35, ])
theta = theta.drop(index=drop_index)
theta.head()

theta_validation = theta_validation.drop(index=drop_index_vl)
theta_validation.head()

df_mean = df_mean.drop(index=drop_index)
df_sd = df_sd.drop(index=drop_index)

df_mean_test = df_mean_test.drop(index=drop_index_vl)
df_sd_test = df_sd_test.drop(index=drop_index_vl)

df_mean.shape
theta.shape
theta.head()


# Remove nas
theta_np = theta.to_numpy()
f_np = df_mean.to_numpy()

theta_test = theta_validation.to_numpy()
f_test = df_mean_test.to_numpy()
#theta_np = theta_np[-which_nas, :]
#f_np = f_np[-which_nas, :]
f_np = np.transpose(f_np)
f_test = np.transpose(f_test)

# Observe simulation outputs in comparison to real data
fig, axis = plt.subplots(4, 2, figsize=(15, 15))
j = 0
k = 0
uniquex = np.unique(x_np[:, 0])
for u in uniquex:
    whereu = u == x_np[:, 0]
    for i in range(f_np.shape[1]):
        axis[j, k].plot(x_np[whereu, 1].astype(int), f_np[whereu, i], zorder=1, color='grey')
    axis[j, k].scatter(x_np[whereu, 1].astype(int), y_mean[whereu], zorder=2, color='red')
    axis[j, k].set_ylabel(u)
    if j == 3:
        j = 0
        k += 1
    else:
        j += 1

f_test = np.log(f_test) #np.sqrt(f_test) #np.log(f_test + 1)
f_np = np.log(f_np) #np.sqrt(f_np) #np.log(f_np + 1)
# Build an emulator 
emu_tr = emulator(x=x_np, 
                   theta=theta_np, 
                   f=f_np, 
                   method='PCGPwM',
                   args={'epsilon': 0.01})

# code to create a new design
# define limits
xlimits = np.array([[10, 30],
                    [-0.7, 0.7],
                    [0.5, 1.5],
                    [0, 1.7**3],
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
num = 500
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

obsvar = np.maximum(0.00001, 0.2*y_mean)

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
threshold = np.sort(loglikelihood)[20] # find 10th highest likelihood as a threshodl to accept
continuing = True
theta_curr = theta[0]
idtheta = 0
iterator = 0
idlist = []
temp = 3000 # temperature
alpha = 0.99 # to adjust decreasing temperature
n_select = 100

def generate_neighbor(idlist, theta):
    # finds the most distant neighbor of thetas in idlist
    dmax = 0
    idmax = 0
    for i in range(len(theta)):
        th = theta[i, :]
        d = 0
        for j in range(len(idlist)):
            dist = th - theta[j, :]
            d += np.sum(dist**2)
        
        if (d > dmax) & (i not in idlist):
            dmax = d
            idmax = i
            
    return idmax

no_badacc = 0
   
while continuing:
    
    if loglikelihood[idtheta] >= threshold:
        # accept if it has high likelihood
        idlist.append(idtheta)
    else:
        # if the likelihood is not high enough, accept is with decreasing prob
        diff = loglikelihood[idtheta] - threshold
        
        # generate rndm variable
        rndm = random.random()
        
        # compute acceptance probability
        pro = np.exp(diff/temp)

        if rndm < pro:
            idlist.append(idtheta)
            no_badacc += 1
            print(loglikelihood[idtheta])

    idtheta = generate_neighbor(idlist, theta) #np.random.randint(500, size=1)[0]
        
    temp *= alpha
    
    if len(idlist) >= n_select:
        continuing = False
            
            
theta_selected = theta[idlist, :]        
        
        
        
        
    
    
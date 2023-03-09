import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from smt.sampling_methods import LHS
from surmise.emulation import emulator
from surmise.calibration import calibrator
import scipy.stats as sps
from scipy import stats
import seaborn as sns

# Initial data set of 300 points with 200 events
df_mean = pd.read_csv('../simulation_data/mean_for_300_sliced_200_events_design', index_col=0)
df_sd = pd.read_csv('../simulation_data/sd_for_300_sliced_200_events_design', index_col=0)
design = pd.read_csv('../design_data/sliced_VAH_090321.txt', delimiter = ' ')
design = design.drop(labels='tau_initial', axis=1)
colnames = design.columns
theta = design.head(300)
drop_index = np.array([19, 23, 31, 32, 71, 91, 92, 98, 129, 131, 146, 162, 171, 174, 184, 190, 194, 195, 198])
theta = theta.drop(index=drop_index)
df_mean = df_mean.drop(index=drop_index)
df_sd = df_sd.drop(index=drop_index)

# A batch of 90 points with 800 events 
df_mean_b0 = pd.read_csv('../simulation_data/mean_for_90_add_batch0_800_events_design', index_col=0)
df_sd_b0 = pd.read_csv('../simulation_data/sd_for_90_add_batch0_800_events_design', index_col=0)
design_b0 = pd.read_csv('../design_data/add_design_122421.txt', delimiter = ' ')
drop_index_b0 = np.array([10, 17, 27, 35, 49, 58])
design_b0 = design_b0.drop(index=drop_index_b0)
df_mean_b0 = df_mean_b0.drop(index=drop_index_b0)
df_sd_b0 = df_sd.drop(index=drop_index_b0)

# A batch of 90 points with 800 events 
df_mean_b1 = pd.read_csv('../simulation_data/mean_for_90_add_batch1_800_events_design', index_col=0)
df_sd_b1 = pd.read_csv('../simulation_data/sd_for_90_add_batch1_800_events_design', index_col=0)
design_b1 = pd.read_csv('../design_data/add_design_122721.txt', delimiter = ' ')
drop_index_b1 = np.array([0, 3, 6, 16, 18, 20, 26, 33, 37, 41, 48])
design_b1 = design_b1.drop(index=drop_index_b1)
df_mean_b1 = df_mean_b1.drop(index=drop_index_b1)
df_sd_b1 = df_sd.drop(index=drop_index_b1)

# A batch of 90 points with 800 events 
df_mean_b2 = pd.read_csv("../simulation_data/mean_for_90_sliced_test_design_800_events_design", index_col=0)
df_sd_b2 = pd.read_csv("../simulation_data/sd_for_90_sliced_test_design_800_events_design", index_col=0)
design_b2 = pd.read_csv('../design_data/sliced_VAH_090321_test.txt', delimiter = ' ')
design_b2 = design_b2.drop(labels='tau_initial', axis=1)
design_b2 = design_b2.iloc[0:90]
drop_index_vl = np.array([29, 35, 76, 84, 87])
design_b2 = design_b2.drop(index=drop_index_vl)
df_mean_b2 = df_mean_b2.drop(index=drop_index_vl)

# Read the experimental data
exp_data = pd.read_csv('../PbPb2760_experiment', index_col=0)
y_mean = exp_data.to_numpy()[0, ]
y_sd = exp_data.to_numpy()[1, ]
colname_exp = exp_data.columns

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
df_mean_b0 = df_mean_b0[exp_label]
df_mean_b1 = df_mean_b1[exp_label]
df_mean_b2 = df_mean_b2[exp_label]

selected_observables = exp_label[0:-32]

x_np = np.column_stack((x[0:-32], x_id[0:-32]))
x_np = x_np.astype('object')
y_mean = y_mean[0:-32]
y_sd = y_sd[0:-32]

#print(f'Last item on the selected observable is {selected_observables[-1]}')

df_mean = df_mean[selected_observables]
df_mean_b0 = df_mean_b0[selected_observables]
df_mean_b1 = df_mean_b1[selected_observables]
df_mean_b2 = df_mean_b2[selected_observables]

print(df_mean.shape)
print(df_mean_b0.shape)
print(df_mean_b1.shape)
print(df_mean_b2.shape)

print('Total obs.:', df_mean.shape[0] + df_mean_b0.shape[0] + df_mean_b1.shape[0] + df_mean_b2.shape[0])

print(theta.shape)
print(design_b0.shape)
print(design_b1.shape)
print(design_b2.shape)


frames = [df_mean, df_mean_b0, df_mean_b1, df_mean_b2]
feval = pd.concat(frames)

frames = [theta, design_b0, design_b1, design_b2]
theta = pd.concat(frames)

# Final training data
theta = theta.to_numpy()
feval = feval.to_numpy()

print('tr. shape (feval):', feval.shape)
print('tr. shape (feval):', theta.shape)

feval = np.transpose(feval)


# Observe simulation outputs in comparison to real data
fig, axis = plt.subplots(4, 2, figsize=(15, 15))
j = 0
k = 0
uniquex = np.unique(x_np[:, 0])
for u in uniquex:
    whereu = u == x_np[:, 0]
    for i in range(feval.shape[1]):
        axis[j, k].plot(x_np[whereu, 1].astype(int), feval[whereu, i], zorder=1, color='grey')
    axis[j, k].scatter(x_np[whereu, 1].astype(int), y_mean[whereu], zorder=2, color='red')
    axis[j, k].set_ylabel(u)
    if j == 3:
        j = 0
        k += 1
    else:
        j += 1
plt.show()

# Build an emulator
emu_tr = emulator(x=x_np,
                   theta=theta,
                   f=feval,
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

obsvar =  0.01*y_mean + np.square(y_sd) #np.maximum(0.00001, 0.2*y_mean)

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

n_select = 74
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
theta_in.to_csv(r'add_design_020522.txt', header=True, index=None, sep=' ', mode='a')

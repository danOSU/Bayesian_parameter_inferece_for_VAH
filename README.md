This repo contains python codes used for the project [Bayesian calibration of viscous anisotropic hydrodynamic simulations of heavy-ion collisions](https://arxiv.org/abs/2302.14184]). 

The plots in the paper have been generated using the notebook `emulation_and_calibration/Analysis and plotting after calibration.ipynb`

If you are looking to understand the emulation and calibration process for this project (using *Surmise*) please take a look at `emulation_and_calibration/cal_surmise_PCSK_PTMC.py`

The trained emulators for VAH heavy-ion collision simulations are `emulation_and_calibration/VAH_PCSK.pkl`

The calibrator object from Surmise with MCMC chains from ptemce is  `emulation_and_calibration/VAH_PCSK_calibrator_PTMC.pkl`

`SBATCH` script that run surmise calibration at the ohio super computer (Owens) is `cal_exp_run_super_computer`

The **active learning** sequential design scripts can be found in `Surmise_design`

The model parameter values are in `design_data` and corresponding simulation data is in `simulation_data`. The map between the two can be found in `simulation_data/map_sim_to_design`

The VAH emulators are avilable online via the [VAH gadget](https://danosu-visualization-vah-streamlit-widget-wq49dw.streamlit.app/). 

If you want to run the VAH simulation using singularity in a cluster and generate your own simulation data for relativistic heavy ion collisions look at the [vah_argonne repo](https://github.com/danOSU/vah_argonne/tree/main).


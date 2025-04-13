April 13, 2025

- cascade:
	- Despite what I said yesterday, it appears that I was using gamma=0.99 w/ the cascade architecture as well.
	- Looking at the plot of value_loss and value_loss_0.99 these appear almost identical, but the value_loss_0.99 is
	  consistently occasionally higher. This is really unexpected so it makes me think there might be a problem.
- sr
  - The results from yesterday's run look like there's instability. I'm wondering if this has to do with the underlying
  features shifting for the SR.
  - I think I have the equations for the SR and value correct, but I should double check.
    - confirmed correct. checked w/ Barreto, A., Munos, R., Schaul, T., & Silver, D. (2017). Successor Features for
    Transfer in Reinforcement Learning. Conference on Neural Information Processing Systems (NeurIPS), 4055–4065.

- General thought, I don't understand how other developers seem to write code in this way where everything is just
 written in a single flow - I want to split things up so they can be easily tested with unit tests.


April 12, 2025


- baseline and cascade both appear to run
- time to choose some environments to test on.
		- I'm going to try acrobot swingup. Actually though, I started a run for this, but just used the default parameters.
		I'll need to check what the parameters used by cleanRL are.
- cascade:
	- There is clearly an error here - the losses for each head are identical in the plots.
	- clip_vloss is already a way of addressing variance in the value loss.
	- PPO does a lot of things to try and address the variability already, so I'm not sure this is the right algorithm to
	  compare against... or maybe that means it is.
	- I'm thinking there might be an error in the calculation of the return targets.
		I wasn't using the gammas, just the single target gamma for all the returns.
	- At present the standard, single-head advantage is being used to train the policy -> this should be corrected
	- TODO: I think the baseline is running w/ gamma=0.99, but I think I only did gamma 0.95 for the cascade runs.cascade
- sr:
	- refactored to be compatible with the new format
	- runs, but not yet certain if it is correct, walked through the code again, but :shrug:


# static

- WANDB_API_KEY should be set in the environment
- CleanRL baselines: https://wandb.ai/openrlbenchmark/openrlbenchmark/reportlist
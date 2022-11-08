# ActiveInference
Simulations of 'Go-No-Go' and 'Explore-Exploit' tasks using Active Inference, as published in the paper

 - Sales AC, Friston KJ, Jones MW, Pickering AE, Moran RJ. Locus Coeruleus tracking of prediction errors optimises cognitive flexibility: An Active Inference model. PLoS Comput Biol. 2019 Jan 4;15(1):e1006267. doi: 10.1371/journal.pcbi.1006267. 

The Active Inference code in this folder was adapted from the original script spm_mdp_vb.m written by Karl Friston, and available as part of the SPM codebase at 
https://www.fil.ion.ucl.ac.uk/spm/software/spm12/

The scripts Example_ExploreExploit.m and Example_GoNoGo.m were used to run the simulations described in the paper (with parameter modifications for each variation). The structure of each task is described in detail in the paper.These scripts will produce an 'MDP' file, which is a struct containing information about each trial of the game (including the values of all parameters/beliefs at each timepoint). An example plotting file is provided for the Go-No-Go task with visualisations of outcomes, including the prediction errors experienced by the agent. The remaining files are dependencies, and must be on the MATLAB path when the games are run.

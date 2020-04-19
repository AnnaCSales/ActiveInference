function [MDP_OUT] = Example_GoNoGo(df_passed, n_trials)

%-------------------------------------------------------------------------
% This is a simple game in which the agent starts at a neutral position (location 1)
% moves to get a cue (location 2) and then can either return to (1) if the cue
% predicts no reward or move to (location 3) if the cue does predict a
% reward. Learning proceeds via the A and D matrices in this game.

% Inputs: 
% - value of model decay parameter, denoted df. Either fixed (df>0) or flexible (df=0), in
% which case decay parameter will be set in relation to the state-action
% prediction error. 
% - n_trials = number of trials to run.

% Returns: a completed MDP structure.

% Anna Sales, University of Bristol, 2018.

% The first action is compulsory and represents the move to get a cue (2).
% After that there are only two possible actions available (1) or (3),
% giving 2  possible policies only. 

% States
% 1=at location 1, reward-cue coming up this trial ('go' trial)
% 2= at location 1, unrewarded-cue coming up this trial  ('no go' trial)
% 3=at location 2, cue=go
% 4=at location 2 cue = no go
% 5= at location 3, reward received
% 6= at location 3, no reward received

% Observations
% 1 = at 1 (no info on context)
% 2 = at 2, cue = go
% 3 = at 2, cue =no go
% 4 = at 3, reward present
% 5 = at 3, reward not present
%---------------------------------------------------------------------------

rng('shuffle')
  
% outcome probabilities: A
%--------------------------------------------------------------------------
% We start by specifying the probabilistic mapping from hidden states
% to outcomes.
%--------------------------------------------------------------------------

A_ENV=[1 1 0 0 0 0;0 0  1  0  0  0;0 0  0  1  0 0; 0 0 0 0 1 0;0 0 0 0 0 1   ];

A =   [1 1 0 0 0 0;0 0 0.5 0.5 0 0;0 0 0.5 0.5 0 0; 0 0 0 0 1 0;0 0 0 0 0 1   ];

%make it naive to the game:
a=5*A; 
%The higher the multiplier, the slower the learning as priors are adjusted
%by numbers <1 on each trial. A very high order number would correspond to
%a very well learnt prior.


% controlled transitions: B{u}
%--------------------------------------------------------------------------
% Next, we have to specify the probabilistic transitions of hidden states
% under each action or control state. 
%--------------------------------------------------------------------------
B{1}  = [1 0 1 0 0 0 ; 0 1 0 1 0 0; 0 0 0 0 0 0; 0 0 0 0 0 0; 0 0 0 0 1 0; 0 0 0 0 0 1]; %% Go to location 1  
B{2}  = [0 0 0 0 0 0;  0 0 0 0 0 0; 1  0 1 0 0 0; 0 1 0 1 0 0; 0 0 0 0 1 0; 0 0 0 0 0 1]; %%  Go to cue, p=prob of oddball  
B{3}  = [1 0 0 0 0 0 ; 0 1 0 0 0 0;0 0 0 0 0 0; 0  0 0 0 0 0; 0 0 1 0 1 0; 0 0 0 1 0 1]; %% Go to dispenesr

B_ENV=B;  

% priors: (utility) C
%--------------------------------------------------------------------------
% Finally, we have to specify the prior preferences in terms of log
% probabilities. Here the agent likes being rewarded, but dislikes
% attempting to get a reward but failing.
%--------------------------------------------------------------------------
c  = 4;
C  = [0 0 0 c -0.5*c]';

% now specify prior beliefs about initial state, in terms of counts
%--------------------------------------------------------------------------
d = [1 1 0 0 0 0]';  %agent believes Go and NoGo contexts are equally likely.
 

% allowable policies  
%--------------------------------------------------------------------------
V = [ 2   2    
      1   3 ];  
  
% MDP Structure - this will be used to generate arrays for multiple trials
%==========================================================================
mdp.V = V;                    % allowable policies
mdp.A = A  ;                  % observation model
mdp.B = B;                    % transition probabilities
mdp.C = C;                    % preferred states
mdp.d = d; 
mdp.a = a;
mdp.s = 2;                     % initial state - we start the agent in the unrewarded context, location 1
mdp.A_ENV = A_ENV;             % real world (fixed) A and B matrices.  
mdp.B_ENV = B_ENV;
mdp.Ni=15;                     % number of iterative updates
mdp.alpha = 1;                 % precision of action selection
mdp.beta  = 1;                  % inverse precision of policy selection
mdp.counter=0;
mdp.SAPEall=[];
if df_passed==0
    mdp.df_set=[];
else
    mdp.df_set=df_passed;
end
mdp.anew=[];

%setup 'go' trials:
%distribute so that rewarded cue is rare and distributed at random throughout the trials. 
n         =n_trials;                % number of trials
percent_pos=10;                %percentage of 'Go' trials.
ff=randperm(n); 
oddballs=ff( 1:round(n*(percent_pos/100))  );  %indices of go trials.
mdp.oddballs=oddballs;

rev_time=150;                  %trial # at which contexts reverse, if needed. Leave as any number > n if not needed.
mdp.rev_time=rev_time; 
%setup the MDP.
MDP       = mdp;
[MDP(1:n)] = deal(mdp); %sets MDP up with a struct array of individual MDP structures.


for i = 1:n
    
    if ismember(i, oddballs)    
        
        [MDP(i).s] = 1;
    else
        [MDP(i).s] = 2;
    end

end

for i = 1:n
    
    if i>rev_time     %if a reversal is needed, change the environment to switch the cues. 
        
        MDP(i).A_ENV=[1 1 0 0 0 0;0 0  0  1  0  0;0 0  1  0  0 0; 0 0 0 0 1 0;0 0 0 0 0 1   ];

    end
end

% Solve to generate data, return in a completed MDP structure.
%==========================================================================
MDP_OUT = spm_MDP_VB_LC(MDP); 

end
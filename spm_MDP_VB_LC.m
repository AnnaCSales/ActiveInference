function [MDP] = spm_MDP_VB_LC(MDP)

% A simplified version of spm_MDP_VB (see https://www.fil.ion.ucl.ac.uk/spm/)
% amended to include calculation of state-action prediction error and model
% decay. Also includes 'environmental' A and B matrices (A_ENV, B_ENV), 
% which represent the 'real' environment and are used to generate 
% observations for the agent / work out the agent's actual location in the
% real world. 

% Original code spm_MDP_VB  Copyright (C) 2005 Wellcome Trust Centre for
% Neuroimaging; Karl Friston

% Amendments:  Anna Sales 2018, University of Bristol.

% Inputs: an vector of structs, with one struct per trial. MDP structs hold
% all the parameters relevant for the model during one trial, e.g. a,b,d
% parameters, states, policies etc.

% Returns: the same struct, but with a complete record of all calculations,
% updates, actions and observations during each trial.
 
%% if there are multiple trials ensure that parameters are updated
%--------------------------------------------------------------------------

if length(MDP) > 1

    for i = 1:length(MDP)
       
        % update concentration parameters
        %------------------------------------------------------------------
        if i > 1
            try  MDP(i).a = OUT(i - 1).a; end
            try  MDP(i).b = OUT(i - 1).b; end
            try  MDP(i).c = OUT(i - 1).c; end
            try  MDP(i).d = OUT(i - 1).d; end
            try  MDP(i).beta = OUT(i - 1).beta; end
            try  MDP(i).SAPEall= OUT(i-1).SAPEall; end
        end
        
        % solve this trial (send this MDP down to the main code below)
        %------------------------------------------------------------------
        OUT(i) = spm_MDP_VB_LC(MDP(i));  
        
    end
    MDP = OUT; 
    return
end

% get preliminaries needed to start the trial - e.g. initial values of state 
% and location, current versions of A,B,D based on a,b,d, values of precision. 
[V, T, No, Np, Ns, Nu, A , qA, B, qB, rB, sB, d, qD, Vo, H, alpha, beta, s, o, P, x, X, u ,a, qbeta, gu, A_ENV, B_ENV ]=  MDP_prelims(MDP);

%% solve
%==========================================================================
Ni    = MDP.Ni;                         % number of VB iterations
xn    = zeros(Ni,Ns,T,T,Np) + 1/Ns; % history of state updates
un    = zeros(Np,T*Ni);             %    policy updates
p     = 1:Np;   % number of allowable policies


for t = 1:T    %Do updates, pick action, get observations for every time point in the trial
    
    if t>1
      pol_OK = ismember(V(t - 1,:),a(t - 1));  %include allowable policies only
      [~,p]=find(pol_OK);
    end
     
    % Get state updates over all times in task (tau) past and future
    % Variational updates (hidden states) under sequential policies
    %======================================================================
    F     = zeros(Np,T);
    %NB 'G'  the expected free energy of policies over future time points,
    %is denoted by 'Q' in this code.
    
    for k = p  % State updates for each policy, over each time point. 

        x(:,:,k)      = spm_softmax(log(x(:,:,k))/2);  %reset. 
   
        for i = 1:Ni % Do Ni iterations of the state update equations (and calculate F / components of Q at the same time)
                 
            px    = x(:,:,k);  % store state probabilities for each time, for each policy
           
            for j = 1:T 
                
                % current state
                %----------------------------------------------------------
                qx   = log(x(:,j,k));
                
         
                % transition probabilities 
                %------------------------------------------------------
                if k > Np
                    fB  = sH;
                    bB  = rH;
                else
                    if j > 1, fB = sB{V(j - 1,k)}; end
                    if j < T, bB = rB{V(j    ,k)}; end
                end
                
                
                % evaluate free energy and gradients (v = dFdx)
                %----------------------------------------------------------
                v    = qx;
                if j <= t, v = v - qA(o(j),:)';           end
                if j == 1, v = v - qD;                    end
                if j >  1, v = v - log(fB*x(:,j - 1,k));  end
                vF = v; 
                if j <  T, v = v - log(bB*x(:,j + 1,k));  end 
                
                            
                % free energy and belief updating
                %----------------------------------------------------------
                F(k,j)  = -x(:,j,k)'*vF;    %% Free energy of policies at each time point (F(pi,tau))
                px(:,j) = spm_softmax(qx - v/Ni); %%  update equation for states.
            end
            
            % hidden state updates
            %--------------------------------------------------------------
            x(:,:,k) = px  ;  %probs of states (rows) over each policy (k, sheets), over each time (cols)     
            
        end        
    
    end
    
    %%  Get expected (future) FE over policies (negative path integral of free energy of policies (Q)
    %======================================================================
    Q     = zeros(Np,T);

    for k = p  %for each policy
        
        for j = 1:T        
            qx     = A*x(:,j,k);
            Q(k,j) = qx'*(Vo(:,j) - log(qx)) + H*x(:,j,k); %Expected free energy of k-th policy at time t=j   
        end
    end

    % Calculate Q, F as sum over time - total free energy in past/future. 
    F     = sum(F,2);
    Q     = sum(Q,2);    
   
    %% Get policy probability and precision, pi / beta and gamma
       
    for i = 1:Ni
        
        % policy (u)
        %------------------------------------------------------------------
        
        qu = spm_softmax( gu(t)*Q(p) + F(p));  %pi, probability of each policy
        pu = spm_softmax( gu(t)*Q(p)); % pi_0 
        v     = qbeta - beta + (qu - pu)'*Q(p);   %update equation for beta
        
        % precision (gu) 
     
        qbeta = qbeta - v/2; %% UPDATE = OLD BETA + ERROR
        gu(t) = 1/qbeta;
       
        u(p,t)  = qu;  %store history of values of policy prob. 

    end
  
    
    % Bayesian model averaging of hidden states over policies

    for i = 1:T
        X(:,i) = squeeze(x(:,i,:))*u(:,t);
        X_t(:,i,t)=X(:,i);
    end
    
    % Calculate the state-action prediction error as a KL divergence
    % between successive BMA distributions.
    
    if t>1
        St_lg_change=log(X_t(:,:,t))-log(X_t(:,:,t-1)); 
        SAPE(t-1)=sum(sum( X_t(:,:,t) .*St_lg_change));
    end
      
    % action selection and observations

    if t < T
        
        % posterior expectations about (remaining) actions (q)
       
        if numel(p) > 1              
            q = unique(V(t,p(1:end ))); %make sure if only picks allowable actions.
        else
            q = V(t,p);
        end
        
        v     = log(A*X(:,t + 1));
        
        for j = q
            qo     = A*B{j}*X(:,t);
            P(j,t) = (v - log(qo))'*qo + 16; 
        end
        
        % action selection
 
        P(:,t) = spm_softmax(alpha*P(:,t));
        [~,a(t)]=max(P(:,t));  %deterministic. 
          
        % Use environment matrices to work out where agent ends up in the
        % real world.
      
        s(t + 1) = find(rand < cumsum(B_ENV{a(t)}(:,s(t))),1);

        
        % Use environmental matrices to get an observation from the real
        % world.

         o(t + 1) = find(rand < cumsum(A_ENV(:,s(t + 1))),1);
  
        % save outcome and state sampled
        %------------------------------------------------------------------
        gu(1,t + 1)   = gu(t);
        
    end
    
end

%% End of trial. Now do updates to concentration parameters.

%Calculate model decay factor (here denoted 'df') based on logistic function using prediction errors

% These are mean values worked out for GNG / EE tasks from 100 trials with
% df =16.
if T==3   %GNG 
    mean=1;
elseif T==2  %EE
    mean=1.8;
end

if isfield(MDP, 'df_set') && length(MDP.df_set)==1  %If we're forcing the agent to use a fixed df.
    df=MDP.df_set ;
    df_settings.vals=df;    %store the values of df that the MDP used.
else   %Or use SAPE to calculate it from a logistic function.
    min_d=2; 
    max_d=32;
    grad_d=8; 
    df=logist(max(SAPE), grad_d,max_d, min_d, mean);   
    df_settings.vals=[grad_d,max_d, min_d, mean];     %store the values of df that the MDP used.
end
%%

for t = 1:T
   
% update concentration parameters - use model decay calculated above.
%----------------------------------------------------------------------
    if isfield(MDP, 'a')
        decay=zeros(size(MDP.a));
    end
    
    if isfield(MDP,'a')
        i        = MDP.a > 0; 
        da       = sparse(o(t),1,1,No,1)*X(:,t)';     
        dec_weights=repmat(da(o(t), :), No,1)  ;
        dec_weights(~i)=0;   %don't change things that are already 0.
        mask=true(size(MDP.a)); 
        mask(o(t), :)=0; %only want to decay elements in row for observation seen, as per outer product in update equation 
        MDP.a(mask) = MDP.a(mask)- dec_weights(mask).*( (MDP.a(mask) - 1)/df);  %decay
        MDP.a(i) = MDP.a(i) + da(i);     %increment
    end
    
   
    if isfield(MDP,'b') && t > 1  
    
        for k = 1:Np
            v           = V(t - 1,k);
            i           = MDP.b{v} > 0;
            db          = u(k,t - 1)*x(:,t,k)*x(:,t - 1,k)';
            MDP.b{v}(i) = MDP.b{v}(i) + db(i) - (MDP.b{v}(i) -1)/df;         
        end
    end

     
end

% initial hidden states:
%--------------------------------------------------------------------------
if isfield(MDP,'d')
    i        = MDP.d > 0;
    MDP.d(i) = MDP.d(i) + X(i,1) - (MDP.d(i) - 1)/df  ;  
end
 

%% assemble results and place in MDP structure

MDP.P   = P;              % probability of action at time 1,...,T - 1
MDP.Q   = x;              % conditional expectations over N hidden states
MDP.X   = X;              % Bayesian model averages
MDP.X_t=X_t;              % BMA at each time point over all times in a trial.
MDP.R   = u;              % conditional expectations over policies
MDP.o   = o;              % outcomes at 1,...,T
MDP.s   = s;              % states at 1,...,T
MDP.u   = a;              % action at 1,...,T 
MDP.SAPEall=[MDP.SAPEall, SAPE];
MDP.w   = gu;             % posterior expectations of precision (policy)
MDP.C   = Vo;             % utility
MDP.A=A;                  % this is the A matrix that has been used throughout this MDP (next time it'll be updated from a)
MDP.A_ENV=A_ENV;          % enivornmental A_ENV
MDP.Ni=Ni;               % number of iterations
MDP.SAPE=SAPE;            % state action prediction error based on changes to BMA states
MDP.df=df;               % decay factor used in trial
MDP.beta=1/gu(T);       % carry forward beta
MDP.dfsettings=df_settings;
MDP.B=B;

function A = spm_norm(A)      % normalisation of a probability transition matrix (columns)
A  = A*diag(1./sum(A,1));
end
%%

function [y] = logist(x, k, max, min, mean)  %for calculating the decay factor df %higher grad = sharper
max=max-min;
y=min+(max./ (1+exp(-k*(-(x-mean)))));
end

end
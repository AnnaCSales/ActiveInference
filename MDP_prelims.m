function [V, T, No, Np, Ns, Nu,A , qA, B, qB, rB, sB, d, qD, Vo, H, alpha, beta, s, o, P, x, X, u ,a, qbeta, gu, A_ENV, B_ENV ]=  MDP_prelims(MDP)

% sort out preliminaries in separate code here to make main AI code less lengthly  
 
% set up and preliminaries - get things out of the MDP structure that the
% AI code will use.

%==========================================================================
V   = MDP.V;                        % allowable policies (T - 1,Np)

% numbers of transitions, policies and states
%--------------------------------------------------------------------------
T   = size(MDP.V,1) + 1;           % number of transitions
Np  = size(MDP.V,2);                % number of allowable policies
Ns  = size(MDP.B{1},1)    ;          % number of hidden states
Nu  = size(MDP.B,2);                % number of hidden controls

% some small constants to keep the code stable, and stop it trying to take
% logs of zero.

p0  = exp(-8);                      % smallest probability 
q0  = 1/16;                         % smallest probability


% parameters of generative model and policies
%==========================================================================

A  = MDP.A + p0;            %this A is the default one assigned to the MDP by the game code's 'deal' - update later.
No = size(A,1);            % number of outcomes

A_ENV = MDP.A_ENV ;                 % this doesn't need p0 added on as it never gets log'd etc. Doing that will allow impossible things to happen...
A          = spm_norm(A);           % normalise 
A_ENV      = spm_norm(A_ENV);       % normalise
 
% parameters (concentration parameters): a and A
%--------------------------------------------------------------------------
if isfield(MDP,'a')       %overrides the above for A if a is provided  (enables learning on A)
    qA = MDP.a + q0;  
    qA = psi(qA ) - ones(No,1)*psi(sum(qA));    
    
    %Make the log probabilities produce normalised probability
    %distros: 
    qA=log(spm_softmax(qA));
    A=spm_norm(spm_softmax(qA));
    
else
    qA = log(spm_norm(A));
end
 
% transition probabilities (priors)
%--------------------------------------------------------------------------
for i = 1:Nu  %one B matrix for each policy
    
    B{i} = MDP.B{i} + p0;  %as above for A, will be overwritten if b is enabled
    B{i} = spm_norm(B{i});
       
    B_ENV{i} = MDP.B_ENV{i};
    B_ENV{i} = spm_norm(B_ENV{i});
        
    % parameters (concentration parameters): b and B
    %----------------------------------------------------------------------
    if isfield(MDP,'b')  %learning on b
        
        b     = MDP.b{i} + q0;
        sB{i} = spm_norm(b );
        rB{i} = spm_norm(b');
        qB{i} = psi(b) - ones(Ns,1)*psi(sum(b));               
        qB{i}=log(spm_softmax(qB{i}));      
        B{i}=spm_norm(spm_softmax(qB{i}));

    else
        b     = MDP.B{i} + p0;
        sB{i} = spm_norm(b );
        rB{i} = spm_norm(b');
        qB{i} = log(b);
    end

end
 
 
% priors over initial hidden states - d and D
%--------------------------------------------------------------------------
if isfield(MDP,'d')  
    d  = MDP.d + q0;  
    qD = psi(d) - ones(Ns,1)*psi(sum(d)); 
    qD=log(spm_softmax(qD));
elseif isfield(MDP,'D')
    d  = MDP.D + q0;
    qD = log(spm_norm(d));
else
    d  = ones(Ns,1);
    qD = psi(d) - ones(Ns,1)*psi(sum(d));
end

%% - ---------------------------------------------------------------
% prior preferences (log probabilities) : C
%--------------------------------------------------------------------------
try
    Vo = MDP.C;

catch
    Vo = zeros(No,1);  %set flat if not provided.
end

% assume constant preferences, if only final states are specified
%--------------------------------------------------------------------------
if size(Vo,2) ~= T
    Vo = Vo(:,end)*ones(1,T);
end

Vo    = log(spm_softmax(Vo));
H     = sum(spm_softmax(qA).*qA);  %-H as defined in the paper
                               % precision defaults
%--------------------------------------------------------------------------
try, alpha = MDP.alpha;  catch, alpha = 16; end  
try, beta  = MDP.beta;   catch, beta  = 1;  end
 
% initial states and outcomes
%--------------------------------------------------------------------------
try
    s = MDP.s(1);                   % initial state (index)
catch
    s = 1;
end

try
    o = MDP.o(1);                   % initial outcome (index)
catch
    o = find(rand < cumsum(A_ENV(:,s)),1);
end

P  = zeros(Nu,T - 1);               % posterior beliefs about control
x  = zeros(Ns,T,Np) + 1/Ns;         % expectations of hidden states | policy
X  = zeros(Ns,T);                   % expectations of hidden states
u  = zeros(Np,T - 1);               % expectations of policy
a  = zeros(1, T - 1);               % action (index)
    
% initialise priors over states
%--------------------------------------------------------------------------
for k = 1:Np   
    x(:,1,k) = spm_softmax(qD);
    
end
 
% expected rate parameter
%--------------------------------------------------------------------------
qbeta = beta;                       % initialise rate parameters
gu    = zeros(1,T)  + 1/qbeta;      % posterior precision (policy)
 

function A = spm_norm(A)
% normalisation of a probability transition matrix (columns)
%--------------------------------------------------------------------------
A  = A*diag(1./sum(A,1));

function A = spm_norm(A)

%%
% normalisation of a probability transition matrix (columns)
%--------------------------------------------------------------------------
A  = A*diag(1./sum(A,1));
end


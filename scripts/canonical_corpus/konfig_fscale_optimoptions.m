function o = optimoptions(varargin)
% Minimal optimoptions shim for the KONFIGF oracle: MATLAB's call passes
% 'UseParallel',true (a pure speed hint for FD gradients) and
% 'DiffMinChange',1e-5. Octave optim has no such field; map to optimset with
% tight tolerances so the unconstrained minimize runs to convergence.
  o = optimset('TolFun', 1e-10, 'TolX', 1e-12, ...
               'MaxIter', 2000, 'MaxFunEvals', 80000, ...
               'Display', 'off');
end

function modScheme = getModulationScheme(qm)
%getModulationScheme Return the modulation scheme based on the modulation order
%
%   Note: This is an internal undocumented function and its API and/or
%   functionality may change in subsequent releases.
%
%   QM = nr5g.internal.getModulationScheme(QM) returns the modulation
%   scheme based on the modulation order specified by QM.

%   Copyright 2024 The MathWorks, Inc.

% Modulation scheme and corresponding bits/symbol
fullmodlist = ["pi/2-BPSK", "BPSK", "QPSK", "16QAM", "64QAM", "256QAM", "1024QAM"];
modSchemeBits = [1 1 2 4 6 8 10];
modScheme = fullmodlist((qm == modSchemeBits)); % Get modulation scheme
end

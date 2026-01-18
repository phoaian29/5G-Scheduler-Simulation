function harqProcess = nrUpdateHARQProcess(harqProcess,ncw)
%nrUpdateHARQProcess Update HARQ process
%
%   HARQPROC = hUpdateHARQProcess(HARQPROC,NCW) updates the HARQ process,
%   HARQPROC, redundancy version values based on block errors to the next
%   value from the RVSequence. If there are no errors, it resets this to
%   the starting value. NCW is the number of codewords for the process,
%   which must be 1 or 2.
%
%   See also nrNewHARQProcesses.
%
%   Note: This is an internal undocumented class and its API and/or
%   functionality may change in subsequent releases.

%   Copyright 2022-2024 The MathWorks, Inc.

%#codegen

    % Update HARQ process redundancy version (RV)
    if any(harqProcess.blkerr)
        L = size(harqProcess.RVSequence,2);
        for cwIdx = 1:ncw
            if harqProcess.blkerr(cwIdx)
                harqProcess.RVIdx(cwIdx) = mod(harqProcess.RVIdx(cwIdx),L)+1; % 1-based indexing
                harqProcess.RV(cwIdx) = harqProcess.RVSequence(harqProcess.RVIdx(cwIdx));
            else % no error => reset
                harqProcess.RVIdx(cwIdx) = 1;
                harqProcess.RV(cwIdx) = harqProcess.RVSequence(1);
            end
        end        
    else % no error => reset
        harqProcess.RVIdx(:) = 1;
        harqProcess.RV(:) = harqProcess.RVSequence(1);
    end
end
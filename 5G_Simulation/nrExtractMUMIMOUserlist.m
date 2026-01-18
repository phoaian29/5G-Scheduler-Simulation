function mumimoUEs = nrExtractMUMIMOUserlist(muMIMOConfigDL, schedulerInput, mcsInfo, isSRSApplicable)
%extractMUMIMOUserlist Extract the MU-MIMO candidate UEs
%
%   Note: This is an internal undocumented function and its API and/or
%   functionality may change in subsequent releases.
%
%   MUMIMOUES = nr5g.internal.nrExtractMUMIMOUserlist(MUMIMOCONFIGDL,...
%   SCHEDULERINPUT, MCSINFO, ISSRSAPPLICABLE) returns a logical array
%   indicating MU-MIMO capable UEs those meet the contraints specified in
%   the MUMIMOCONFIGDL structure. i.e., UEs with non-zero buffer and meets
%   MinNumRBs, MinCQI for CSI-RS-based downlink MU-MIMO and MinSINR
%   for SRS-based downlink MU-MIMO.
%
%   MUMIMOCONFIGDL structure contains the contraints that the user pairing
%   algorithm needs for determining the MU-MIMO candidacy among UEs.
%
%   SCHEDULERINPUT structure contains the fields which scheduler
%   needs for selecting the UE for resource allocation.
%
%   MCSINFO is the MCS table used for downlink. It contains the mapping of
%   MCS indices with Modulation and Coding schemes.
%
%   ISSRSAPPLICABLE is the flag used to indicate if the CSI measurement
%   type is SRS, where a logical 'true' represents the SRS is applicable.
%
%   MUMIMOUES is a logical array and determines if the UE is
%   eligible for MU-MIMO pairing, where a logical 'true' represents
%   the corresponding UE is MU-MIMO capable.

%   Copyright 2024 The MathWorks, Inc.

mumimoUEs = zeros(size(schedulerInput.eligibleUEs, 2), 1);
mcs = schedulerInput.mcsRBG(:, 1)';
nPRB = muMIMOConfigDL.MinNumRBs;
nREPerPRB = 12*schedulerInput.numSym;

% Find MU-MIMO capable UEs
for index = 1:size(schedulerInput.eligibleUEs, 2)
    % Calculate served bits for number of RBs equal to MinNumRBs
    nlayers = schedulerInput.selectedRank(index);
    infoMCS = mcsInfo(mcs(index) + 1, :);
    modSchemeBits = infoMCS(1); % Bits per symbol for modulation scheme
    modScheme = nr5g.internal.getModulationScheme(modSchemeBits);
    codeRate = infoMCS(2)/1024;
    servedBits = nrTBS(modScheme, nlayers, nPRB, nREPerPRB, codeRate);

    % MU-MIMO candidacy
    if(schedulerInput.bufferStatus(index) > servedBits)
        %Apply filtering rule
        if ~isSRSApplicable
            cqi = schedulerInput.cqiRBG(:, 1)';
            if (cqi(index) >= muMIMOConfigDL.MinCQI)
                % logical Array
                mumimoUEs(index) = 1;
            end
        else
            if ~isempty(schedulerInput.SINRs{index}) && (schedulerInput.SINRs{index} >= muMIMOConfigDL.MinSINR)
                % logical Array
                mumimoUEs(index) = 1;
            end
        end
    end
end
end
function [allottedUEs, allottedRBs, pairedStatus, W, pairedUEsMcs] = nrUserPairingRAT1(muMIMOConfigDL, ueContext, schedulerInput, allottedRBCount, availableRBs, mcsInfo, activeUEs, userPairingMatrix, isSRSApplicable)
%nrUserPairingRAT1 User pairing algorithm for resource allocation type 1.
%
%   Note: This is an internal undocumented function and its API and/or
%   functionality may change in subsequent releases.
%
%   [ALLOCATEDUES, ALLOTTEDRBS, PAIREDSTATUS, W, PAIREDUESMCS] = nr5g.internal.nrUserPairingRAT1(...
%   MUMIMOCONFIGDL, UECONTEXT, SCHEDULERINPUT, ALLOTTEDRBCOUNT, ...
%   AVAILABLERBS, MCSINFO, ACTIVEUES, USERPAIRINGMATRIX, ISSRSAPPLICABLE)
%   returns allocated UEs, RBs, pairing status, updated precoding matrices,
%   and updated MCS after pairing.
%
%   MUMIMOCONFIGDL structure contains the constraints that the user pairing
%   algorithm needs for determining the MU-MIMO candidacy among UEs.
%
%   UECONTEXT contains the context of all the connected UEs.
%   This is a vector of nr5g.internal.nrUEContext of length equal to
%   the number of UEs connected to the gNB. Value at index 'i' stores
%   the context of a UE with RNTI 'i'.
%
%   SCHEDULERINPUT structure contains the fields that the scheduler
%   needs for selecting the UE for resource allocation.
%
%   ALLOTTEDRBCOUNT is an array which list number of RBs
%   allocated by SU-MIMO scheduling of the UEs.
%
%   AVAILABLERBS is total number of RBs available for the
%   transmission.
%
%   MCSINFO is the MCS table used for downlink. It contains the mapping of
%   MCS indices with Modulation and Coding schemes.
%
%   ACTIVEUES is an array of UEs RNTI which are scheduled by the
%   base scheduler.
%
%   USERPAIRINGMATRIX is the precomputed orthogonality matrix for all UEs
%   based on the CSI type II reports. This is used for the CSI-RS-based
%   user pairing.
%
%   ISSRSAPPLICABLE is the flag used to indicate if the CSI measurement
%   type is SRS, where a logical 'true' represents the SRS is applicable.
%
%   ALLOTTEDUES is a column vector of UEs RNTI which are scheduled by the
%   user pairing algorithm.
%
%   ALLOTTEDRBS is a column vector which list number of RBs
%   allocated by the user pairing.
%
%   PAIREDSTATUS is a column vector stores which primary UE has been paired.
%
%   W is a cell array contains the updated precoding matrices of the allotted
%   users, based on pairing. It also contains the precoding matrices of
%   all other active UEs.
%
%   PAIREDUESMCS contains the updated MCS indices of the allotted users,
%   based on pairing. It also contains the MCS indices of all other active
%   UEs.

%   Copyright 2024 The MathWorks, Inc.

activeUEsInfo = activeUEs;
numActiveUEs = size(activeUEsInfo, 2);
[allottedUEs, allottedRBs, pairedStatus] = deal(zeros(numActiveUEs, 1));
totalRBsAllocated = 0;
numAllottedUEs = 1;
% Get MU-MIMO capable UEs indication
isMumimoUE = nr5g.internal.nrExtractMUMIMOUserlist(muMIMOConfigDL, schedulerInput, mcsInfo, isSRSApplicable);
W = schedulerInput.W;
pairedUEsMcs = schedulerInput.mcsRBG(:,1);
% Loop until all active UEs are scheduled as SU-MIMO or MU-MIMO
% candidates
while (numAllottedUEs <= numActiveUEs)
    selectedUE = activeUEsInfo(1); % Primary User
    index = find(selectedUE == schedulerInput.eligibleUEs, 1);
    selectedUERank = schedulerInput.selectedRank(index);
    pairedUEs = selectedUE;
    updatedPairedUEsMcs = schedulerInput.mcsRBG(index, 1);
    rbUEs = allottedRBCount(activeUEs == selectedUE);

    % If selected UE is MU-MIMO capable UE
    if isMumimoUE(index)
        isMumimoUE(index) = 0;
        mumimoUEs = schedulerInput.eligibleUEs(isMumimoUE == 1);
        mumimoUEs = intersect(mumimoUEs, activeUEsInfo(activeUEsInfo ~= selectedUE));

        % check for UEs that can be paired with selected UE
        if mumimoUEs
            % Use the SRS-based user pairing if SRS is the CSI measurement
            % type.
            if isSRSApplicable
                [pairedUEs, V, updatedPairedUEsMcs] = nr5g.internal.nrSelectPairedUEsSRS(schedulerInput, muMIMOConfigDL, selectedUE, mumimoUEs);
                % Update the precodeing matrices for the paired UEs.
                [~, idxPairedUEs] = ismember(pairedUEs, schedulerInput.eligibleUEs);
                W(idxPairedUEs) = V;
            elseif ~isempty(userPairingMatrix)
                % Use the CSI-RS-based user pairing for the default CSI
                % measurement type.
                selectedUEsMcs = schedulerInput.mcsRBG(index, 1);
                [pairedUEs, updatedPairedUEsMcs] = nr5g.internal.nrSelectPairedUEsCSIRS(schedulerInput, selectedUE, selectedUEsMcs, selectedUERank, mumimoUEs, muMIMOConfigDL, userPairingMatrix, ueContext);
            end
        end
        pairedUEs = pairedUEs(pairedUEs ~= 0);
        [~,indices] = intersect(activeUEs, pairedUEs);
        % Calculate total RBs associated with this pairing
        rbPairedUEs = sum(allottedRBCount(indices));
        % Check the UE which has the maximum RB requirement of all
        % paired UE.
        maxRBs = max(schedulerInput.rbRequirement(pairedUEs));
        % Determine if MU-MIMO increases the RB allocation otherwise go with SU-MIMO
        if rbPairedUEs <= maxRBs && rbPairedUEs >= muMIMOConfigDL.MinNumRBs
            rbUEs = rbPairedUEs;
        else
            % SU-MIMO
            [~, idxPairedUEs] = ismember(pairedUEs, schedulerInput.eligibleUEs);
            W(idxPairedUEs) = schedulerInput.W(idxPairedUEs);
            updatedPairedUEsMcs = schedulerInput.mcsRBG(index, 1);
            pairedUEs = selectedUE;
        end
    end

    totalRBsAllocated = totalRBsAllocated+rbUEs;
    numPairedUEs = numel(pairedUEs);
    % For paired users update information on allocated RBs,
    % allocated UEs and paired status.
    for idx = 1:numPairedUEs
        % update status if we have paired UEs
        if numPairedUEs > idx
            pairedStatus(numAllottedUEs) = 1;
        end
        allottedRBs(numAllottedUEs) = min(schedulerInput.rbRequirement(pairedUEs(idx)),rbUEs);
        allottedUEs(numAllottedUEs) = pairedUEs(idx);
        pairedUEsMcs(numAllottedUEs) = updatedPairedUEsMcs(idx);
        activeUEsInfo = activeUEsInfo(activeUEsInfo ~= pairedUEs(idx));
        isMumimoUE(pairedUEs(idx)) = 0;
        numAllottedUEs = numAllottedUEs+1;
    end
    if (availableRBs <= totalRBsAllocated)
        % Remove trailing zeros from allottedUEs for further operations.
        allottedUEs = allottedUEs(allottedUEs ~= 0);
        return;
    end
end
end

function [pairedUEs, pairedUEsMcs] = nrSelectPairedUEsCSIRS(schedulerInput, selectedUE, selectedUEMcs, selectedUERank, mumimoUEs, muMIMOConfigDL, userPairingMatrix, ueContext)
%nrSelectPairedUEs User pairing for CSR-RS-based downlink MU-MIMO.
%
%   Note: This is an internal undocumented function and its API and/or
%   functionality may change in subsequent releases.
%
%   [PAIREDUES, PAIREDUESMCS] = nr5g.internal.nrSelectPairedUEsCSIRS(...,
%   SCHEDULERINPUT, SELECTEDUE, SELECTEDUEMCS, SELECTEDUERANK,...
%   MUMIMOUES, MUMIMOCONFIGDL, USERPAIRINGMATRIX, UECONTEXT) returns list
%   of paired users and corresponding MCS values.
%
%   SCHEDULERINPUT structure contains the fields which scheduler
%   needs for selecting the UE for resource allocation.
%
%   SELECTEDUE is primary UE for which the method selects
%   corresponding paired UEs.
%
%   SELECTEDUEMCS is the primary UE's MCS index.
%
%   SELECTEDUERANK is the primary UE's rank i.e., the number of
%   transmission layers.
%
%   MUMIMOUES is the list of UEs which are eligible for MU-MIMO
%   pairing.
%
%   MUMIMOCONFIGDL structure contains the constraints that the user pairing
%   algorithm needs for determining the MU-MIMO candidacy among UEs.
%
%   USERPAIRINGMATRIX is the precomputed orthogonality matrix for all UEs
%   based on the CSI type II reports. This is used for the CSI-RS-based
%   user pairing.
%
%   UECONTEXT contains the context of all the connected UEs.
%   This is a vector of nr5g.internal.nrUEContext objects of length equal
%   to the number of UEs connected to the gNB. Value at index 'i' stores
%   the context of a UE with RNTI 'i'.
%
%   PAIREDUES is the list of UEs that are orthogonal. It also
%   contains the primary UE.
%
%   PAIREDUESMCS is the list of UEs MCS that are orthogonal. It
%   also contains primary UE's MCS.

%   Copyright 2024 The MathWorks, Inc.

% User configurable constraints MaxNumUsersPaired and MaxNumLayers are used
% to limit the number of paired UEs.
maxNumUsersPaired = muMIMOConfigDL.MaxNumUsersPaired;
maxNumLayers = muMIMOConfigDL.MaxNumLayers;

pairedUEs = zeros(maxNumUsersPaired, 1);
pairedUEsMcs = zeros(maxNumUsersPaired, 1);

% Assign primary user ID and MCS to the paired UE information
pairedUEsMcs(1) = selectedUEMcs;
pairedUEs(1) = selectedUE;

% Separate orthogonal matrix and the first column which contains UE indices
userPairingMatrixIndices = userPairingMatrix(:,1);
userPairingMatrix = userPairingMatrix(:,2:end);

ueIndices = (selectedUE == userPairingMatrixIndices);
% Get corresponding orthogonality information for the selected UE indices.
pairingInfo = unique(userPairingMatrix(ueIndices,:));
pairingInfo = pairingInfo(pairingInfo ~= 0);
% Filter UEs based on the RB and CQI restrictions i.e. MU-MIMO UEs
eligiblePairedUEs = intersect(pairingInfo, mumimoUEs);
counter = 1;
numLayersPaired = 0;
% Recursive search for paired UEs
for ueIdx = 1:numel(eligiblePairedUEs)
    ueIndex = eligiblePairedUEs(ueIdx) == userPairingMatrixIndices;
    pairingSuccess = true;
    % Recursive search for orthogonality with all UE already paired
    for idx = 1:nnz(pairedUEs)
        orthogonalRows = (userPairingMatrix(ueIndex,:) == pairedUEs(idx));
        rowSize = size(orthogonalRows,1);
        numLayersOrthogonal = nnz(orthogonalRows)/rowSize;
        csiMeasurementPairedUEs = ueContext(pairedUEs(idx)).CSIMeasurementDL.CSIRS.PMISet.i1(1:2);
        csiMeasurementeligiblePairedUEs = ueContext(eligiblePairedUEs(ueIdx)).CSIMeasurementDL.CSIRS.PMISet.i1(1:2);
        orthogonalBeams = all(csiMeasurementPairedUEs == csiMeasurementeligiblePairedUEs);
        if (numLayersOrthogonal ~= selectedUERank) || (orthogonalBeams == 0)
            pairingSuccess = false;
        end
    end
    % Pairing is successful
    if pairingSuccess == true
        counter = counter+1;
        numLayersPaired = numLayersPaired + schedulerInput.selectedRank(ueIdx);
        if numLayersPaired > maxNumLayers

            return;
        end
        pairedUEs(counter) = eligiblePairedUEs(ueIdx);
        index = find(schedulerInput.eligibleUEs == pairedUEs(counter), 1);
        pairedUEsMcs(counter) = schedulerInput.mcsRBG(index, 1);
    end
    if nnz(pairedUEs) == maxNumUsersPaired
        return;
    end
end
end

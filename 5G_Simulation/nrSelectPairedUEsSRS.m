function [pairedUEs, precoder, pairedUEsMcs] = nrSelectPairedUEsSRS(schedulerInput, muMIMOConfigDL, selectedUE, mumimoUEs)
%nrSelectPairedUEsSRS User pairing for SRS-based downlink MU-MIMO.
%
%   Note: This is an internal undocumented function and its API and/or
%   functionality may change in subsequent releases.
%
%   [PAIREDUES, PRECODER, PAIREDUESMCS] = nr5g.internal.nrSelectPairedUEsSRS(...
%   SCHEDULERINPUT, MUMIMOCONFIGDL, SELECTEDUE, MUMIMOUEs) returns the
%   paired UEs, the corresponding precoding matrices, and the MCS for
%   each paired UE.
%
%   SCHEDULERINPUT structure contains the fields which scheduler
%   needs for selecting the UE for resource allocation.
%
%   MUMIMOCONFIGDL structure contains the constraints that the user pairing
%   algorithm needs for determining the MU-MIMO candidacy among UEs.
%
%   SELECTEDUE is the RNTI of the initially selected UE.
%
%   MUMIMOUES contains an array of UEs available for MU-MIMO pairing.
%
%   PAIREDUES contains the list of RNTIs of the UEs selected for pairing.
%
%   PRECODER cell array contains the precoding matrices for the paired UEs.
%
%   PAIREDUESMCS contains an array of MCS indices for each paired UE. It
%   also contains primary UE's MCS.

%   Copyright 2024 The MathWorks, Inc.

% Reference
% [1] K. Ko and J. Lee, "Determinant Based Multiuser MIMO Scheduling with
% Reduced Pilot Overhead," 2011 IEEE 73rd Vehicular Technology Conference
% (VTC Spring), Budapest, Hungary, 2011.

% User configurable constraints MaxNumUsersPaired and MaxNumLayers are used
% to limit the number of paired UEs.
maxNumUsersPaired = muMIMOConfigDL.MaxNumUsersPaired;
maxNumLayers = muMIMOConfigDL.MaxNumLayers;

% Extract the rank-reduced wideband channel matrices and noise variance
H = schedulerInput.channelMatrix;
nVar = schedulerInput.nVar;
% Number of transmit antennas at gNB
M = size(H{selectedUE == schedulerInput.eligibleUEs}, 2);

% Initialize paired UEs with the primary selected UE
pairedUEs = selectedUE;
pairedUEsMcs = schedulerInput.mcsRBG(schedulerInput.eligibleUEs == pairedUEs, 1);
precoder = schedulerInput.W(schedulerInput.eligibleUEs == pairedUEs);

remainingUEs = mumimoUEs;
if remainingUEs
    % Initialize the combined channel matrix of the selected UEs with the
    % primary UE's channel matrix
    HSelectedSet = H{selectedUE == schedulerInput.eligibleUEs};

    % Compute initial sum rate, precoding matrix, and post MMSE equalized
    % SINRs.
    [sumRate, precoder, sinr] = computeSumRate(H(selectedUE == schedulerInput.eligibleUEs), nVar);

    % Initialize orthogonal projection matrix
    X = eye(M) - HSelectedSet' / (HSelectedSet * HSelectedSet') * HSelectedSet;

    % Iteratively select additional UEs for pairing
    for i = 1:maxNumUsersPaired-1
        bestUser = selectedUE;
        selectedDet = -Inf;
        % Evaluate each potential UE in the remaining UEs set for pairing
        for n = remainingUEs
            Hn = H{n == schedulerInput.eligibleUEs};
            detValue = det(Hn * X * Hn');
            if detValue > selectedDet
                selectedDet = detValue;
                bestUser = n;
            end
        end
        % Attempt to add the best user to the current set
        pairedUEsTemp = union(pairedUEs, bestUser, "stable");
        % Check if maximum number of layers has been exceeded
        if size(HSelectedSet, 1) >= maxNumLayers
            break;
        end
        chanIndex = arrayfun(@(m) find(schedulerInput.eligibleUEs == m, 1), pairedUEsTemp);
        [sumRateTemp, precoderTemp, sinrTemp] = computeSumRate(H(chanIndex), nVar);

        % Check if adding the user improves sum rate
        if sumRateTemp <= sumRate
            break;
        else
            % Update the combined channel matrix by concatenating the channel
            % matrix of the newly selected UE.
            HSelectedSet = cat(2, HSelectedSet.', H{bestUser == schedulerInput.eligibleUEs}.').';
            % Update the projection matrix X, the paired UEs set, sum rate,
            % precoding matrices, SINR, and the remaining UEs set after
            % successful pairing.
            X = eye(M) - HSelectedSet' / (HSelectedSet * HSelectedSet') * HSelectedSet;
            pairedUEs = pairedUEsTemp;
            sumRate = sumRateTemp;
            precoder = precoderTemp;
            sinr = sinrTemp;
            remainingUEs(remainingUEs == bestUser) = [];
        end
    end
end

% If the paired list contains just the primary UE, reuse the SU-MIMO precoder and MCS
if ~isscalar(pairedUEs)
    % Initialize carrier and PDSCH configuration for MCS computation
    carrier = nrCarrierConfig;
    carrier.NSizeGrid = size(schedulerInput.channelQuality, 2);
    l2smSRS = nr5g.internal.L2SM.initialize(carrier);
    pdschConfig = nrPDSCHConfig;
    pdschConfig.PRBSet = (0:carrier.NSizeGrid-1);
    pdschConfig.DMRS.DMRSPortSet = [];
    % Compute MCS for each paired UE
    pairedUElength = numel(pairedUEs);
    pairedUEsMcs = zeros(pairedUElength, 1);
    k = 1;
    for i = 1:pairedUElength
        numLayers = size(precoder{i}, 1);
        sinrPerUELayers = sinr(k:k+numLayers-1);
        pdschConfig.NumLayers = numLayers;
        pairedUEsMcs(i) = nr5g.internal.computeMCS(l2smSRS, carrier, pdschConfig, sinrPerUELayers, 'qam256');
        k = k + numLayers;
    end
end
end

function [R, W, sinr] = computeSumRate(H, nVar)
%computeSumRate computes the sum rate, precoder matrices, and SINR.
%
%   [R, W, SINR] = computeSumRateSLS(H, NVAR) returns the sum rate,
%   precoding matrices computed using block diagonalization (BD), and SINR
%   for a given set of channel matrices and noise variance.
%
%   H is a cell array of channel matrices for each UE.
%   NVAR is the noise variance.
%
%   R is the computed sum rate.
%   W is a cell array of precoding matrices for each UE.
%   SINR is the post MMSE equalized Signal-to-Interference-plus-Noise
%   Ratio for each layer.

% Get the umber of UEs
K = numel(H);

% Compute block diagonal beamforming weights
ns = cellfun(@(x) size(x, 1), H)';
Hcell = cellfun(@transpose, H, 'UniformOutput', false);
Wg = blkdiagbfweights(Hcell, ns);

% Preallocate the cell array W for beamforming weights
W = cell(1, K);

% Extract the precoding matrices per UE from the combined block diagonal
% beamforming weights, Wg
startIdx = 1;
for k = 1:K
    N = ns(k);
    W{k} = normalizeMatrix(Wg(startIdx:startIdx+N-1, :), N);
    startIdx = startIdx + N;
end

% Concatenate channel matrices to form the combined channel matrix
Htot = [Hcell{:}];
V = Wg.';

% Compute post MMSE equalized SINR by using the precoded channel matrix
sinr = nr5g.internal.nrPrecodedSINR(Htot.', nVar, V);

% Compute the sum rate using the SINR values
R = sum(log2(1 + sinr));
end

% Normalize the matrix
function A = normalizeMatrix(A,nLayers)

A = diag(1 ./ (sqrt(nLayers*diag(A*A')))) * A;

end
function [Rank,TPMI,MCSIndex,CSIInfo] = nrULCSIMeasurements(carrier,srs,pusch,Hest,nVar,mcsTable,bandSize)
% nrULCSIMeasurements provides rank, transmit precoding matrix indicator(TPMI) and index of
% modulation and coding scheme using SRS for PUSCH transmission
%
%   Note: This is an internal undocumented function and its API and/or
%   functionality may change in subsequent releases.
%
%   [RANK,TPMI,MCSINDEX,CSIINFO] = nrULCSIMeasurements(CARRIER,SRS,PUSCH,HEST,NVAR,MCSTABLE,BANDSIZE)
%   returns the uplink channel rank RANK, TPMI using uplink codebook as
%   mentioned in TS 38.211 Table 6.3.1.5-1 to 6.3.1.5-7, modulation and
%   coding scheme index MCSINDEX, for the specified carrier configuration
%   CARRIER, sounding reference signal configuration SRS, PUSCH
%   configuration PUSCH, uplink channel grid HEST, noise variance NVAR,
%   type of mcs table MCSTABLE and specified subband size BANDSIZE. HEST is
%   the uplink channel extracted from SRS based channel estimates.
%
%   CARRIER is a carrier-specific configuration object as described in
%   <a href="matlab:help('nrCarrierConfig')">nrCarrierConfig</a>
%   with properties:
%
%   SubcarrierSpacing     - Subcarrier spacing in kHz
%   CyclicPrefix          - Cyclic prefix (CP) type
%   NSlot                 - Absolute slot number
%   NFrame                - Absolute system frame number
%   NSizeGrid             - Size of the carrier resource grid in terms of
%                           number of resource blocks (RBs)
%   NStartGrid            - Start of carrier resource grid relative to
%                           common resource block 0 (CRB 0)
%
%   SRS is an SRS-specific configuration object as described in
%   <a href="matlab:help('nrSRSConfig')">nrSRSConfig</a>.
%
%   PUSCH is the physical uplink shared channel configuration object as
%   described in <a href="matlab:help('nrPUSCHConfig')">nrPUSCHConfig</a>.
%
%   HEST is the uplink channel estimation matrix. It is of size
%   K-by-L-by-nRxAnts-by-nTxAnts, where K is the number of subcarriers in
%   the carrier resource grid, L is the number of orthogonal frequency
%   division multiplexing (OFDM) symbols spanning one slot, nRxAnts is the
%   number of receive antennas, and nTxAnts is the number of transmit
%   antennas.
%
%   MCSTABLE is PUSCH MCS Tables as defined in TS 38.214 section 6.1.4.1 .
%   MCSTABLE string input must be one of the following set
%   {'qam64','qam64LowSE','qam256','qam64','qam64LowSE'} and each
%   member represents PUSCH MCS tables as defined below :
%   If transform precoding is disabled
%   'qam64'      - MCS index table corresponding to TS 38.214 Table 5.1.3.1-1
%   'qam256'     - MCS index table corresponding to TS 38.214 Table 5.1.3.1-2
%   'qam64LowSE' - MCS index table corresponding to TS 38.214 Table 5.1.3.1-3
%   If transform precoding is enabled
%   'qam64'    - MCS index table corresponding to TS 38.214 Table 6.1.4.1-1
%   'qam64LowSE' - MCS index table corresponding to TS 38.214 Table 6.1.4.1-2
%
%   BANDSIZE is the number of physical resource blocks considered as band
%   for reporting a TPMI.
%
%   RANK is a scalar indicating the optimal number of transmission layers
%   to maximize the spectral efficiency for the given channel and noise
%   variance conditions. Its value ranges from 1 to 4.
%
%   TPMI is transmit precoding matrix computed based on matrix of size
%   numSubbands-by-1.
%
%   MCSINDEX is modulation and coding scheme index calculated using
%   MCSTable and it is of size 1-by-numCodewords.
%
%   CSIINFO is an output structure with information about SINR values and
%   SUBBANDINDICES. SINR represents the sinr values per resource element in
%   the resource grid for the selected precoder matrix from the codebook
%   and is a complex matrix output of size SRSINDICES-by-RANK.
%   SUBBANDINDICES is a matrix that contains the start and ending
%   subcarrier indices for every subband, for computing the corresponding
%   subband TPMI.

%   Copyright 2024 The MathWorks, Inc.

% Validate input arguments
fcnName = 'nrULCSIMeasurements';
% Validate 'Hest'
K = carrier.NSizeGrid;
L = carrier.SymbolsPerSlot;
numSRSPorts = srs.NumSRSPorts;
validateattributes(Hest,{'double','single'},{'size',[K*12 L NaN numSRSPorts]},fcnName,'Hest');
% Validate nVar
validateattributes(nVar,{'double','single'},{'scalar','real','nonnegative','finite'},fcnName,'NVAR');

% Get the MCS table values
mcsTableValues = getMCSTable(mcsTable,pusch);

% Get the MaxRank
nRxAnts = size(Hest,3);
maxRank = min(numSRSPorts,nRxAnts);
% Initialize overhead and blerThreshold for MCS calculations 
overhead = 0;
blerThreshold = 0.1;
% Initialize mcs cell and efficiency array
mcsUL = cell(maxRank,1);
efficiency = zeros(maxRank,1);

% Get the SRS indices to get channel estimates at SRS locations
srsIndices = nrSRSIndices(carrier,srs);
indInSlot = srsIndices(srsIndices<carrier.SymbolsPerSlot*12*carrier.NSizeGrid);
[srs_k,srs_l] = ind2sub([carrier.NSizeGrid*12,carrier.SymbolsPerSlot], indInSlot);
srs_lUnique = unique(srs_l);

Hsymb = zeros(size(Hest,1),numel(srs_lUnique),size(Hest,3),size(Hest,4));
for i = 1:numel(srs_lUnique)
    % Get the channel coefficients in the slot
    Hsymb(srs_k,i,:,:) = Hest(srs_k,srs_lUnique(i),:,:);
end

% Initialize l2sm object for MCS calculation
l2smSRS = nr5g.internal.L2SM.initialize(carrier);

% Compute the spectral efficiency for all possible ranks
if (nVar ~= 0 && ~isempty(srsIndices))
    for ranks = 1:maxRank

        % Compute PMI for each rank
        [pmiArray(:,ranks),~,csiInfo(ranks).subbandIndices,csiInfo(ranks).sinr] = nr5g.internal.nrPMISelect(ranks, Hsymb, nVar, bandSize);

        % Compute MCS for each rank
        pusch.NumLayers = ranks;
        sinrdB = pow2db(abs(csiInfo(ranks).sinr+eps(csiInfo(ranks).sinr)));
        nanIndices = isnan(sinrdB);
        if (all(nanIndices))
            return;
        end
        sinrdB = sinrdB(~any(nanIndices,2),:);
        [l2smSRS, mcsUL{ranks}, mcsInfo(ranks)] = nr5g.internal.L2SM.cqiSelect(l2smSRS,carrier,pusch,overhead,sinrdB,mcsTableValues(:,2:3),blerThreshold);

        % Compute spectral efficiency for each rank
        ncw = pusch.NumCodewords;
        cwLayers = floor((ranks + (0:ncw-1)) / ncw);
        % Add 1 to MCS index as it starts from zero in MCS table
        SpecEff = mcsTableValues(mcsUL{ranks}(1,:)+1,4);
        efficiency(ranks) = cwLayers .* (1 - mcsInfo(ranks).TransportBLER(1,:)) * SpecEff;
    end
    % Select the rank, mcs and TPMI based on efficiency
    [~,Rank] = max(efficiency);
    TPMI = pmiArray(:,Rank);
    MCSIndex = mcsUL{Rank};
    CSIInfo.SINR = csiInfo(Rank).sinr;
    CSIInfo.SubbandIndices = csiInfo(Rank).subbandIndices;
else
    Rank = NaN;
    MCSIndex = NaN;
    TPMI = NaN;
    CSIInfo = struct();
end
end

function MCSTableValues = getMCSTable(tableName,pusch)
% MCSTABLEVALUES = getMCSTable(TABLENAME,PUSCH) returns uplink the MCS
% table values for the given MCS table input and pusch configuration
persistent tables tranformPrecodingFlag;

if (isempty(tables) || ~isequal(tranformPrecodingFlag,pusch.TransformPrecoding))
    tranformPrecodingFlag = pusch.TransformPrecoding;
    mcsTableClass = nrPUSCHMCSTables;
    if tranformPrecodingFlag
        props = ["TransformPrecodingQAM64Table","TransformPrecodingQAM64LowSETable"];
    else
        props = ["QAM64Table","QAM256Table","QAM64LowSETable"];
    end
    numProps = numel(props);
    for i = 1:numProps
        tmpTable = mcsTableClass.(props(i));
        % l2sm accepts cqitable transmit code rate (tcr) values as tcr/1024
        tmpArray = [tmpTable.MCSIndex tmpTable.Qm (tmpTable.TargetCodeRate)*1024 tmpTable.SpectralEfficiency];
        tables{i} = tmpArray(~isnan(tmpArray(:,3)),:);
    end
end
if tranformPrecodingFlag
    tabNames = ["qam64","qam64LowSE"];
else
    tabNames = ["qam64","qam256","qam64LowSE"];
end
MCSTable = tables(strcmpi(tableName,tabNames));
coder.internal.errorIf(isempty(MCSTable),'nr5g:nrULCSIMeasurements:InvalidMCSTable');
MCSTableValues = MCSTable{1};
end
function [DLRank,Wp,MCSIndex,Hprg,effectiveSINR] = nrSRSDLChannelMeasurements(carrier,srs,pdsch,Hest,nVar,mcsTable,PRGSize,enablePRGLevelMCS)
% nrSRSDLChannelMeasurements provides rank, precoder and index of
% modulation and coding scheme using SRS for PDSCH transmission in TDD
% systems using reciprocity
%
%   Note: This is an internal undocumented function and its API and/or
%   functionality may change in subsequent releases.
%
%   [DLRANK,WP,MCSINDEX,HPRG,EFFECTIVESINR] = nrSRSDLChannelMeasurements(CARRIER,SRS,PDSCH,HEST,NVAR,MCSTABLE,PRGSIZE,ENABLEPRGLEVELMCS)
%   returns the downlink channel rank DLRANK, downlink zero-forcing based
%   precoder WP and modulation and coding scheme index MCSINDEX, for the
%   specified carrier configuration CARRIER, sounding reference signal
%   configuration SRS, PDSCH configuration PDSCH, downlink channel grid
%   HEST, noise variance NVAR, type of mcs table MCSTABLE and specified
%   precoding resource block group bundle size PRGSIZE. HEST is the
%   downlink channel extracted from SRS based channel estimates using
%   reciprocity. The input "ENABLEPRGLEVELMCS" activates the calculation of
%   the MCS Index and effective downlink SINR for every PRG. When this option 
%   is turned on, the "MCSINDEX" will consists of wideband MCS in the initial
%   row, with ubsequent rows representing each individual PRG level MCS and
%   the "EFFECTIVESINR" will consists of wideband effective downlink SINR 
%   in the initial row, with ubsequent rows representing each individual 
%   PRG level effective downlink SINRs.
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
%   PDSCH is the physical downlink shared channel configuration object as
%   described in <a href="matlab:help('nrPDSCHConfig')">nrPDSCHConfig</a>.
%
%   HEST is the downlink channel estimation matrix. It is of size
%   K-by-L-by-nRxAnts-by-nTxAnts, where K is the number of subcarriers in
%   the carrier resource grid, L is the number of orthogonal frequency
%   division multiplexing (OFDM) symbols spanning one slot, nRxAnts is the
%   number of receive antennas, and nTxAnts is the number of transmit
%   antennas.
%
%   MCSTABLE is PDSCH MCS Tables as defined in TS 38.214 Table 5.1.3.1-1 to
%   5.1.3.1-4. MCSTABLE string input must be one of the following set
%   {'qam64','qam256','qam64LowSE','qam1024'} and each member represents PDSCH
%   MCS tables as defined below :
%
%   'qam64'      - MCS index table 1,
%                     corresponding to TS 38.214 Table 5.1.3.1-1
%   'qam256'     - MCS index table 2,
%                     corresponding to TS 38.214 Table 5.1.3.1-2
%   'qam64LowSE' - MCS index table 3,
%                     corresponding to TS 38.214 Table 5.1.3.1-3
%   'qam1024'    - MCS index table 4,
%                     corresponding to TS 38.214 Table 5.1.3.1-4
%
%   PRGSIZE is the PRG bundle size (2, 4, or [] to indicate 'wideband').
%
%   DLRANK is a scalar which gives the best possible number of transmission
%   layers for the given channel and noise variance conditions. It is in
%   the range 1 to 8.
%
%   WP is an array of size NLAYERS-by-nTxAnts-by-NPRG, where NPRG is the
%   number of PRGs in the carrier resource grid (see <a
%   href="matlab:help('nrPRGInfo')">nrPRGInfo</a>). Wp defines a separate
%   precoding matrix of size NLAYERS-by-nTxAnts for each PRG.
%
%   MCSINDEX output is a 2-dimensional matrix of size 1-by-numCodewords
%   when ENABLEPRGLEVELMCS input is 'false' and (NPRG+1)-by-numCodewords
%   when ENABLEPRGLEVELMCS is 'true'. The first row consists of 'Wideband'
%   MCSIndex value in both the cases of ENABLEPRGLEVELMCS input and rest of
%   the rows corresponds to each PRG when ENABLEPRGLEVELMCS is true.
%   numCodewords is the number of codewords.
%
%   HPRG output is downlink channel matrix averaged across the resource
%   elements present in a PRG occupied by SRS. It is an array of size
%   DLRANK-by-nTxAnts-by-numPRG. numPRG is number of PRG.
%
%   EFFECTIVESINR is a 2-dimensional matrix of size 1-by-numCodewords
%   when ENABLEPRGLEVELMCS input is 'false' and (NPRG+1)-by-numCodewords
%   when ENABLEPRGLEVELMCS is 'true'. The first row consists of 'Wideband'
%   effective downlink SINR value in both the cases of ENABLEPRGLEVELMCS 
%   input and rest of the rows corresponds to each PRG when ENABLEPRGLEVELMCS 
%   is true. numCodewords is the number of codewords.
%
%   Copyright 2024 The MathWorks, Inc.

% Validate input arguments
fcnName = 'nrSRSDLChannelMeasurements';
% Validate 'Hest'
K = carrier.NSizeGrid;
L = carrier.SymbolsPerSlot;
numSRSPorts = srs.NumSRSPorts;
validateattributes(Hest,{'double','single'},{'size',[K*12 L numSRSPorts NaN]},fcnName,'Hest');
% Validate nVar
validateattributes(nVar,{'double','single'},{'scalar','real','nonnegative','finite'},fcnName,'NVAR');

prgInfo = nrPRGInfo(carrier,PRGSize);
indices = nrSRSIndices(carrier,srs);

if (~isempty(indices) && any(Hest(:)))
    nTxAnts = size(Hest,4);
    % Average channel matrix across REs in each RB containing SRS
    dlChannelMatrix = reduceChannelGrid(carrier,Hest,indices);
    % Get the MCS table
    mcsTableValues =  getMCSTable(mcsTable);
    % maxRank is minimum of number of SRS ports and base station transmit
    % antennas
    maxRank = min(numSRSPorts,nTxAnts);
    % Initialize an array to calculate spectral efficiency
    efficiency = zeros(maxRank,1);
    wPrecoderCell = cell(maxRank,1);
    dlChannelCell = cell(maxRank, 1);
    mcsDL = cell(maxRank,1);
    tranportBler = cell(maxRank,1);
    effectiveSINRDL = cell(maxRank,1);
    % Initialize l2sm object for MCS calculation
    l2smSRS = nr5g.internal.L2SM.initialize(carrier);

    % Compute the spectral efficiency for all possible ranks
    for ranks = 1:maxRank
        % Compute ZF precoder for each possible rank
        wPrecoder = zeros(ranks, nTxAnts, prgInfo.NPRG);
        dlChannel = zeros(ranks, nTxAnts, prgInfo.NPRG);
        for prg = 1:prgInfo.NPRG
            % Find RBs corresponding to this PRG
            RB = (find(prgInfo.PRGSet==prg));
            % Average the channel across RBs belonging to PRG
            Havg = reshape(mean(dlChannelMatrix(RB + carrier.NStartGrid,:,:,:),1),[],nTxAnts);
            % Select the layers that are less correlated and the number of
            % selected layers should be equal to rank
            triagMatrix = abs(tril(corrcoef(Havg'),-1));
            [corrValue,~] = max(triagMatrix);
            [~,corrCols] = sort(corrValue);
            dlChannel(:, :, prg) = Havg(corrCols(1:ranks),:);
            % Calculate ZF precoder
            wp = pinv(dlChannel(:, :, prg));
            wPrecoder(:,:, prg) = normalizeMatrix(wp.',ranks);
        end
        wPrecoderCell{ranks} = wPrecoder;
        dlChannelCell{ranks} = dlChannel;
        if(any(isnan(wPrecoder(:))))
            % If precoder has NaN avoid MCS calculations
            efficiency(ranks) = NaN;
            mcsDL{ranks} = NaN;
        else
            pdsch.NumLayers = ranks;
            pdsch.DMRS.DMRSPortSet = [];
            % Compute MCS for each rank
            [l2smSRS,mcsDL{ranks},tranportBler{ranks},effectiveSINRDL{ranks}] = computeMCS(l2smSRS,dlChannelMatrix,...
                carrier,pdsch,wPrecoder,nVar,prgInfo,mcsTableValues(:,2:3),enablePRGLevelMCS);

            % Compute spectral efficiency for each rank
            ncw = pdsch.NumCodewords;
            cwLayers = floor((ranks + (0:ncw-1)) / ncw);
            % Add 1 to MCS index as it starts from zero in MCS table
            SpecEff = mcsTableValues(mcsDL{ranks}(1,:)+1,4);
            efficiency(ranks) = cwLayers .* (1 - tranportBler{ranks}) * SpecEff;
        end
    end

    % Select the rank, mcs and Wp based on efficiency
    [~,DLRank] = max(efficiency);
    Wp = wPrecoderCell{DLRank};
    MCSIndex = mcsDL{DLRank};
    Hprg = dlChannelCell{DLRank};
    effectiveSINR = effectiveSINRDL{DLRank};
else
    DLRank = NaN;
    Wp = NaN;
    MCSIndex = NaN;
    Hprg = NaN;
    effectiveSINR = NaN;
end
end

function channelMatrix = reduceChannelGrid(carrier,Hest,indices)
% Compute the channel to RB level granularity
NumRx = size(Hest,3);
NumTx = size(Hest,4);
NCRB = carrier.NSizeGrid + carrier.NStartGrid;
estChannelGridReduced = NaN([NCRB 1 NumRx NumTx]);
L = carrier.SymbolsPerSlot;
K = carrier.NSizeGrid * 12;

% Get the indices available in the slot
indInSlot = indices(indices<carrier.SymbolsPerSlot*12*carrier.NSizeGrid);
[srs_k,srs_l] = ind2sub([K,L], indInSlot);
symb = unique(srs_l);
% Average channel estimate across the REs and symbols spanned by the SRS
for rbInd=1:NCRB
    srs_rb = (srs_k>(rbInd-1)*12) & (srs_k<(rbInd)*12+1);
    estChannelGridReduced(rbInd,1,:,:) = mean(Hest(srs_k(srs_rb),symb,:,:),[1,2]);
end

% Replace NaNs with nearest non-NaN values, to provide channel
% estimates in RBs outside of SRS bandwidth
nanArray = isnan(estChannelGridReduced);
nanidx = find(nanArray);
nonnanidx = find(~nanArray);
replaceidx = arrayfun(@(x)nonnanidx(find(abs(nonnanidx-x)==min(abs(nonnanidx-x)),1)),nanidx);
estChannelGridReduced(nanidx) = estChannelGridReduced(replaceidx);
channelMatrix = estChannelGridReduced;
end

function [l2smSRS,mcs,transportBler,effectiveSINRDL] = computeMCS(l2smSRS,H,carrier,PDSCHConfiguration,wPrecoder,nVar,prgInfo,mcsTableValues,enablePRGLevelMCS)
% Compute the DL MCS
overhead = 0;
blerThreshold = 0.1;
Hperm = permute(H,[3 4 1 2]);
wPrecoderPRG = wPrecoder(:,:,prgInfo.PRGSet);
sinr = nr5g.internal.nrPrecodedSINR(Hperm,nVar,permute(wPrecoderPRG,[2 1 3]));

% Remove NaN sinr values for MCS calculations
SINRDB = 10*log10(sinr+eps(sinr));
nonnan = ~any(isnan(SINRDB),2);
if ~any(nonnan,'all')
    return;
end
SINRDB = SINRDB(nonnan,:);

[l2smSRS, mcs(1,:),mcsInfo(1,:)] = nr5g.internal.L2SM.cqiSelect(l2smSRS,carrier,PDSCHConfiguration,overhead,SINRDB,mcsTableValues,blerThreshold);
effectiveSINRDL(1,:) = mcsInfo(1, :).EffectiveSINR;

% Calculate PRG level MCS
if (enablePRGLevelMCS && prgInfo.NPRG>1)
    for i = 1:prgInfo.NPRG
        [l2smSRS, mcs(i+1,:), mcsInfo(i+1,:)] = nr5g.internal.L2SM.cqiSelect(l2smSRS,carrier,PDSCHConfiguration,overhead,pow2db(sinr(prgInfo.PRGSet==i,:)),mcsTableValues,blerThreshold);
        effectiveSINRDL(i+1,:) = mcsInfo(i+1,:).EffectiveSINR;
    end
end
transportBler = mcsInfo(1).TransportBLER(1,:);
end


function MCSTableValues = getMCSTable(tableName)

persistent tables;

if isempty(tables)
    mcsTableClass = nrPDSCHMCSTables;
    props = ["QAM64Table","QAM256Table","QAM64LowSETable","QAM1024Table"];
    numProps = numel(props);
    for i = 1:numProps
        tmpTable = mcsTableClass.(props(i));
        % l2sm accepts cqitable transmit code rate(tcr) values as tcr/1024
        tmpArray = [tmpTable.MCSIndex tmpTable.Qm (tmpTable.TargetCodeRate)*1024 tmpTable.SpectralEfficiency];
        tables{i} = tmpArray(~isnan(tmpArray(:,3)),:);
    end
end
tabNames = ["qam64","qam256","qam64LowSE","qam1024"];
MCSTable = tables(strcmpi(tableName,tabNames));
coder.internal.errorIf(isempty(MCSTable),'nr5g:nrSRSDLChannelMeasurements:InvalidMCSTable');
MCSTableValues = MCSTable{1};
end

% Normalize the matrix
function A = normalizeMatrix(A,nLayers)

A = diag(1 ./ (sqrt(nLayers*diag(A*A')))) * A;

end

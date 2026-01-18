%L2SM NR link to system mapping interface
%   NR link to system mapping interface, which implements abstraction of
%   the NR PHY.
%
%   Note: This is an internal undocumented function and its API and/or
%   functionality may change in subsequent releases.

%   The PHY abstraction models two aspects of NR PHY behavior:
%   * BLER of downlink data (PDSCH / DL-SCH) and uplink data (PUSCH / 
%     UL-SCH) transmission and reception
%   * CQI selection 
%
%   The downlink/uplink BLER model consists of the following steps:
%      1. Received bit mutual information rate (RBIR) effective SINR
%         mapping (ESM), which calculates the effective SINR based on
%         instantaneous SINRs and the symbol modulation order.
%      2. HARQ processing, including effective code rate (ECR) calculation
%         and effective SINR calculation accounting for HARQ history.
%      3. Determination of code block segment (CBS) size and number of code
%         blocks C for the TBS for the current HARQ process.
%      4. Effective SINR to code block BLER mapping: a pre-calculated SINR
%         to BLER table is selected for the current symbol modulation
%         order, ECR and CBS size, and the code block BLER is looked up
%         using the effective SINR.
%      5. The transport block BLER is calculated based on the code block
%         BLER and number of code blocks C. Decoding of the current
%         transport block is considered to have failed if a random variable
%         selected uniformly in the open interval (0,1) is lower than the
%         transport block BLER.
%
%   Step 1 is described in IEEE 802.11-14/0811r2 "Overview on RBIR-based
%   PHY Abstraction". Steps 2, 4 and 5 are described in S. Lagen, K.
%   Wanuga, H. Elkotby, S. Goyal, N. Patriciello and L. Giupponi, "New
%   Radio Physical Layer Abstraction for System-Level Simulations of 5G
%   Networks," ICC 2020 - 2020 IEEE International Conference on
%   Communications (ICC), 2020, pp. 1-7. Step 3 is described in 3GPP
%   TS 38.212 Section 5.2.2.
%
%   The CQI selection model consists of the following steps:
%   * The effective SINR is calculated as described in step 1 above.
%   * Tables from step 4 above are extracted for each MCS (i.e. each symbol
%     modulation order and ECR combination) corresponding to the entries of
%     a specified CQI table.
%   * The code block BLER for each CQI index is looked up from the tables
%     using the effective SINR, and the transport block BLER is calculated
%     as described in step 5 above.
%   * The reported CQI is established by following the procedure in TS
%     38.214 Secton 5.2.2.1 using the BLERs from the previous step.

%   Copyright 2021-2025 The MathWorks, Inc.

classdef L2SM

    methods (Static)

        % -----------------------------------------------------------------
        % Constructor
        % -----------------------------------------------------------------

        % Create variables used to store PHY abstraction state
        % syntax:
        % l2sm = L2SM(carrier);                   % for CSI-RS or SRS
        % l2sm = L2SM(...,nProcesses,nCodewords); % for PDSCH or PUSCH
        function l2sm = initialize(varargin)

            l2sm = initializeL2SM(varargin{:});

        end

        % -----------------------------------------------------------------
        % Link Quality Model (LQM) and Link Performance Model (LPM)
        % -----------------------------------------------------------------

        function [l2sm,SINRs,varargout] = linkQualityModel(varargin)

            if (nargin==6)

                % Get channel estimates in a subset of PHY channel / signal
                % locations
                [l2sm,sig,varargout{1:nargout-2}] = nr5g.internal.L2SM.prepareLQMInput(varargin{:});

                % No interference
                intf = [];

            elseif (nargin==3)

                % Get channel estimates for the signal of interest and
                % the interferers
                [l2sm,sig,intf] = deal(varargin{:});

                % Project the interferers into the allocation used by the
                % signal of interest
                intf = projectInterference(l2sm.GridSize,sig,intf);

            else % nargin==1

                % If only one input argument is provided, reset LQM cache
                l2sm = varargin{1};
                [l2sm,~,~,~,varargout{1:nargout-2}] = getChannelEstimates(l2sm);
                SINRs = [];
                return;

            end

            % Calculate SINRs using IEEE 802.11 methodology
            SINRs = calculateSINR(sig,intf);

        end

        function [l2sm,sig,varargout] = prepareLQMInput(l2sm,carrier,chsig,estChannelGrid,noiseEst,wtx)

            % Get channel estimates in a subset of PHY channel / signal 
            % locations
            [l2sm,hest,prgsubs,sinrsubs,varargout{1:nargout-2}] = getChannelEstimates(l2sm,carrier,chsig,estChannelGrid,wtx);
            
            % Record channel estimates and other information in a
            % convenience structure
            sig.sinrsubs = sinrsubs;
            sig.hest = hest;
            sig.prgsubs = prgsubs;
            sig.wtx = wtx;
            sig.noiseEst = noiseEst;

        end

        function [l2sm,blkerr,lpmInfo] = linkPerformanceModel(l2sm,harqInfo,pxsch,SINRs)

            % Apply HARQ history to the SINRs, and also return the total RM
            % output capacity (Gsum), number of unique LDPC coded bits
            % received (Csum), effective code rate (ECR), and number of
            % code blocks (C)
            [l2sm,harqedSINRs,Gsum,Csum,ECR,C] = applyHarqHistory(l2sm,harqInfo,pxsch,SINRs);

            % Optionally split SINRs into code block segments, so that
            % the effective SINR and code block BLER can be calculated per
            % code block
            if (l2sm.SplitCodeBlocks)
                harqedSINRs = splitCodeBlocks(harqedSINRs,C);
            end

            % Perform effective SINR mapping using IEEE 802.11 methodology
            [~,Qm] = modulationPerCodeword(pxsch);
            effectiveSINR = effectiveSINRMapping(Qm,harqedSINRs,l2sm.Alpha,l2sm.Beta);

            % Perform mapping of SINR to code block BLER
            % (lookup of AWGN tabulated SINR versus BLER results)
            trBlkSizes = harqInfo.TransportBlockSize;
            codeBLER = sinrToCodeBLER(effectiveSINR,trBlkSizes,Qm,ECR);

            % Calculate transport block BLER from code block BLER
            if (l2sm.SplitCodeBlocks)
                transportBLER = 1 - prod(1 - codeBLER,1);
            else
                transportBLER = 1 - (1 - codeBLER).^C;
            end

            % Randomly create block errors
            nCodewords = numel(trBlkSizes);
            blkerr = (rand(1,nCodewords) < transportBLER);

            % Create LPM info structure
            lpmInfo.EffectiveSINR = effectiveSINR;
            lpmInfo.TransportBlockSize = trBlkSizes;
            lpmInfo.Qm = Qm;
            lpmInfo.Gsum = Gsum;
            lpmInfo.Csum = Csum;
            lpmInfo.EffectiveCodeRate = ECR;
            lpmInfo.CodeBLER = codeBLER;
            lpmInfo.C = C;
            lpmInfo.TransportBLER = transportBLER;

        end

        % -----------------------------------------------------------------
        % HARQ Processing
        % -----------------------------------------------------------------

        % Transmitter HARQ processing
        function l2sm = txHARQ(l2sm,harqInfo,targetCodeRate,G)

            % Current target code rate is cached because it will be used in
            % the linkPerformanceModel function
            l2sm.TargetCodeRate = targetCodeRate;
            
            % Current capacity G is cached because it will be used in the
            % linkPerformanceModel function
            l2sm.G = G;

            % For each codeword
            procIdx = harqInfo.HARQProcessID + 1;
            trBlkSizes = harqInfo.TransportBlockSize;
            nCodewords = numel(trBlkSizes);
            for cwIdx = 1:nCodewords

                % If new data is required
                if harqInfo.NewData(cwIdx)

                    % Reset HARQ-related state in link to system mapping
                    l2sm.SINRs{procIdx,cwIdx} = [];
                    l2sm.Gsum(procIdx,cwIdx) = 0;
                    l2sm.RRArgs{procIdx,cwIdx} = newRRArgs();

                end

            end

        end

        % -----------------------------------------------------------------
        % CQI Selection
        % -----------------------------------------------------------------
        
        function [l2sm,cqiIndex,cqiInfo] = cqiSelect(l2sm,carrier,pxsch,xOverhead,SINRs,cqiTable,blerThreshold)
            
            % Check if the CQI table, PHY channel indices, or xOverhead
            % have changed
            newTable = (isempty(l2sm.CQI.Table) || ~isequaln(l2sm.CQI.Table,cqiTable));
            [~,pxschIndicesInfo] = getChSigIndices(carrier,pxsch);
            newIndicesInfo = (isempty(l2sm.IndicesInfo) || ~isequal(l2sm.IndicesInfo,pxschIndicesInfo));
            newXOverhead = (isempty(l2sm.CQI.XOverhead) || ~isequal(l2sm.CQI.XOverhead,xOverhead));
            
            % If the CQI table, PHY channel indices or xOverhead have
            % changed
            if (newTable || newIndicesInfo || newXOverhead)
                
                % Cache the table and xOverhead, and clear the cache of
                % PXSCH capacities
                l2sm.CQI.Table = cqiTable;
                l2sm.CQI.XOverhead = xOverhead;
                l2sm.CQI.G = [];
                
                % Cache the PHY channel indices information
                l2sm.IndicesInfo = pxschIndicesInfo;
                
                % Create all combinations of CQI indices across the
                % codewords
                l2sm = combinationsCQI(l2sm);

            end
            
            % For each CQI combination determine the effective SINR, the
            % effective code rate, the code block BLER, and the number of
            % code blocks
            [l2sm,effectiveSINR,codeBLER] = sinrToCodeBLERForCQI(l2sm,pxsch,SINRs);

            % Calculate transport block BLER from code block BLER
            C = l2sm.CQI.C;
            if (l2sm.SplitCodeBlocks)
                transportBLER = cellfun(@(x)(1 - prod(1 - x)),codeBLER);
            else
                transportBLER = 1 - (1 - codeBLER).^C;
            end
        
            % Select the CQI combination with the largest BLER less than or
            % equal to the threshold
            idx = find(all(transportBLER<=blerThreshold,2),1,'last');
            if (isempty(idx))
                % If no CQI combination meets the BLER criterion, the first
                % CQI combination is selected and the CQI for each codeword
                % is increased if possible so that the BLER criterion is
                % met for that codeword in isolation
                idx = bestPerCodewordCQI(l2sm,transportBLER,blerThreshold);
            end
            tableRow = l2sm.CQI.TableRowCombos(idx,:);
            cqiIndex = tableRow - 1;

            % Create CQI info structure
            cqiInfo.EffectiveSINR = effectiveSINR(idx,:);
            cqiInfo.TransportBlockSize = l2sm.CQI.TransportBlockSize(idx,:);
            cqiInfo.Qm = l2sm.CQI.Table(tableRow,1);
            cqiInfo.TargetCodeRate = l2sm.CQI.Table(tableRow,2) / 1024;
            cqiInfo.G = l2sm.CQI.G(idx,:);
            cqiInfo.NBuffer = l2sm.CQI.NBuffer(idx,:);
            cqiInfo.EffectiveCodeRate = l2sm.CQI.EffectiveCodeRate(idx,:);
            cqiInfo.CodeBLER = codeBLER(idx,:);
            cqiInfo.C = C(idx,:);
            cqiInfo.TransportBLER = transportBLER(idx,:);
            if (l2sm.SplitCodeBlocks)
                cqiInfo.EffectiveSINR = arrayMaxC(cqiInfo.EffectiveSINR,cqiInfo.C);
                cqiInfo.CodeBLER = arrayMaxC(cqiInfo.CodeBLER,cqiInfo.C);
            end
            
        end
        
        % -----------------------------------------------------------------

    end

end

% -------------------------------------------------------------------------
% Local functions
% -------------------------------------------------------------------------

function l2sm = initializeL2SM(carrier,nProcesses,nCodewords)

    l2sm = struct();

    % If no input arguments are provided, reset internal RNGs and return
    if (nargin==0)
        getChannelEstimates();
        getCsum();
        return;
    end

    % Create structure containing extra HARQ-related information for PHY
    % abstraction of PDSCH or PUSCH
    if (nargin>1)
        l2sm.SINRs = cell(nProcesses,nCodewords);
        l2sm.Gsum = zeros(nProcesses,nCodewords);
        l2sm.LPMCacheSize = 8;
        l2sm.RRArgs = repmat({newRRArgs()},nProcesses,nCodewords);
        l2sm.RRCacheArgs = [];
        l2sm.SoftBuffer = [];
        l2sm.CSumSequence = [];
        l2sm.G = zeros(1,nCodewords);
        l2sm.TargetCodeRate = zeros(1,nCodewords);
    end

    % Limited buffer rate matching
    l2sm.Nref = []; % no LBRM

    % RBIR tuning parameters
    l2sm.Alpha = 1.0;
    l2sm.Beta = 1.0;

    % Code block splitting parameter (enable for high Doppler channels)
    l2sm.SplitCodeBlocks = false;

    % Cache carrier and create fields for cache of grid size and previous
    % indices calculations
    l2sm.Carrier = carrier;
    l2sm.GridSize = [];
    l2sm.IndicesInfo = [];

    % Create fields for cache of CQI calculations
    l2sm.CQI.Table = [];
    l2sm.CQI.XOverhead = [];
    l2sm.CQI.TableRowCombos = [];
    l2sm.CQI.TableQms = [];
    l2sm.CQI.TableModulations = [];
    l2sm.CQI.Qm = [];
    l2sm.CQI.G = [];
    l2sm.CQI.TransportBlockSize = [];
    l2sm.CQI.C = [];
    l2sm.CQI.NBuffer = [];
    l2sm.CQI.EffectiveCodeRate = [];
    l2sm.CQI.DLSCHInfo = [];

end

function [l2sm,hest,prgsubs,sinrsubs,varargout] = getChannelEstimates(l2sm,carrier,chsig,estChannelGrid,wtx)

    persistent rs;
    if (isempty(rs) || nargin==0)
        rs = RandStream('mt19937ar','Seed',0);
    end
    if (nargin==0)
        return;
    end

    % Initialize cache of indices calculations. The cache consists of a
    % number of separate variables, each being a cell array. Therefore the
    % 'n'th cache entry consists of the 'n'th element in the cell array for
    % each of these variables
    persistent pl2sm;
    if (isempty(pl2sm) || nargin==1)
        pl2sm = struct();
        if (nargin>1 || isempty(l2sm.LQMCacheSize))
            pl2sm.LQMCacheSize = 100;
        else
            pl2sm.LQMCacheSize = l2sm.LQMCacheSize;
        end
        c = cell(1,max(1,pl2sm.LQMCacheSize));
        pl2sm.PRGSubscripts = c;
        pl2sm.CRBSubscripts = c;
        pl2sm.SINRHestIndices = c;
        pl2sm.SINRSubscripts = c;
        pl2sm.ChSig = c;
        pl2sm.HestSize = c;
    end
    if (nargin==1)
        hest = [];
        prgsubs = [];
        sinrsubs = [];
        varargout{1} = pl2sm;
        return;
    end

    % If the channel estimate input does not have fast fading information
    % (i.e. only has a single frequency-time location)
    if (all(size(estChannelGrid,[1 2])==1))
        % Perform minimal processing and return. The condition of not
        % having fast fading information can be detected in the outputs by
        % checking isempty(sinrsubs)
        hest = permute(estChannelGrid,[2 3 4 1]);
        prgsubs = 1;
        sinrsubs = [];
        return;
    end

    % Get size of carrier resource grid
    if (isempty(l2sm.GridSize))
        l2sm.GridSize = [l2sm.Carrier.NSizeGrid*12 l2sm.Carrier.SymbolsPerSlot];
    end

    % Get size of channel estimate
    nRxAnts = size(estChannelGrid,3);
    nTxAnts = size(wtx,2);
    hestSize = [l2sm.GridSize nRxAnts nTxAnts];

    % Determine if previously stored PHY channel / signal configurations
    % are different from the current configuration in a way that affects
    % the resource element allocation; if so, new indices need to be
    % calculated
    if (pl2sm.LQMCacheSize==0)
        newIndices = true;
    else
        chSigIdx = [];
        for i = 1:numel(pl2sm.ChSig)
            if (yieldEqualREs(pl2sm.ChSig{i},chsig) ...
                    && isequal(pl2sm.HestSize{i},hestSize))
                chSigIdx = i;
                break;
            end
        end
        newIndices = isempty(chSigIdx);
    end

    % If new indices need to be calculated
    if (newIndices)

        % Select an unused element of the cache, or if all are used,
        % randomly select an element to replace
        chSigIdx = find(cellfun(@isempty,pl2sm.ChSig),1);
        if (isempty(chSigIdx))
            chSigIdx = rs.randi([1 numel(pl2sm.ChSig)]);
        end

        % Create PHY channel / signal indices for all RBs
        chsigfull = maxRBAllocation(carrier,chsig);
        ind = getChSigIndices(carrier,chsigfull);

        % Store the properties of the PHY channel / signal configuration
        % which affect RE indices in an RB
        pl2sm.ChSig{chSigIdx} = propertiesAffectingREs(chsigfull);

        % For the case of CSI-RS or SRS, project the subcarrier / OFDM
        % symbol indices for any port onto all ports
        if (isa(chsigfull,"nrCSIRSConfig") || isa(chsigfull,"nrSRSConfig"))
            p = size(wtx,1);
            [~,ind] = nrExtractResources(ind,zeros([hestSize(1:2) p]));
        end

        % Subsample to get the set of indices used for SINR measurement
        siz = hestSize([1 2 4]);
        [sinrIndices,prgsubs,crbsubs] = getSINRIndices(siz,carrier,ind,wtx);

        % Cache PRG and CRB subscripts
        pl2sm.PRGSubscripts{chSigIdx} = prgsubs;
        pl2sm.CRBSubscripts{chSigIdx} = crbsubs;

        % Extract the channel estimate in the SINR measurement locations
        % and cache indices
        [hest,pl2sm.SINRHestIndices{chSigIdx}] = nrExtractResources(sinrIndices,estChannelGrid);

        % Calculate and cache SINR subscripts
        [k,l] = ind2sub(hestSize,pl2sm.SINRHestIndices{chSigIdx}(:,1));
        pl2sm.SINRSubscripts{chSigIdx} = [floor((k-1)/12)+1 l];

        % Store size of channel estimate
        pl2sm.HestSize{chSigIdx} = hestSize;

    else

        % Extract the channel estimate in the SINR measurement locations
        hest = estChannelGrid(pl2sm.SINRHestIndices{chSigIdx});

    end

    % Get the currently allocated CRBs
    crbs = getCRBs(carrier,chsig);

    % Extract the channel estimate in allocated CRBs
    isAllocCRB = any(pl2sm.CRBSubscripts{chSigIdx}==crbs,2);
    hest = hest(isAllocCRB,:,:);

    % Return PRG subscripts in allocated CRBs
    prgsubs = pl2sm.PRGSubscripts{chSigIdx}(isAllocCRB);

    % Return SINR subscripts in allocated CRBs
    sinrsubs = pl2sm.SINRSubscripts{chSigIdx}(isAllocCRB,:);

    % Return persistent L2SM state if requested
    if (nargout==5)
        varargout{1} = pl2sm;
    end

end

function SINRs = calculateSINR(sig,intf)

    hfn = @(x)permute(x.hest,[1 3 2]);
    wfn = @(x,y)permute(x.wtx(:,:,y),[3 1 2]);

    % Prepare signal of interest channel estimate, power and precoding
    % matrices
    Htxrx = hfn(sig);
    Ptxrx = 1;
    Wtx = wfn(sig,sig.prgsubs);

    % Get noise estimate
    N0 = sig.noiseEst;

    if (isempty(intf))

        % Calculate SINR with no interference
        SINRs = wireless.internal.L2SM.calculateSINRs(Htxrx,Ptxrx,Wtx,N0);

    else

        % Prepare interference channel estimates. For interferers without
        % fast fading information (i.e. only having a single frequency-time
        % location), the values are repeated to match the number of
        % frequency-time locations in the signal of interest channel
        % estimate
        Htxrx_int = arrayfun(hfn,intf,'UniformOutput',false);
        fastFading = arrayfun(@(x)~isempty(x.sinrsubs),intf);
        repfn = @(y)cellfun(@(x)repmat(x,[size(Htxrx,1) 1 1]),y,'UniformOutput',false);
        Htxrx_int(~fastFading) = repfn(Htxrx_int(~fastFading));

        % Prepare interference powers
        Nintf = numel(intf);
        Ptxrx_int = ones(Nintf,1);

        % Prepare interference precoding matrices. For interferers without
        % fast fading, the interferer precoding matrices corresponding to
        % the signal of interest precoder subscripts are selected
        p = cell(size(intf));
        p(fastFading) = {intf(fastFading).prgsubs};
        p(~fastFading) = {sig.prgsubs};
        Wtx_int = cellfun(wfn,num2cell(intf),p,'UniformOutput',false);

        % Calculate SINR with interference
        SINRs = wireless.internal.L2SM.calculateSINRs(Htxrx,Ptxrx,Wtx,N0,Htxrx_int,Ptxrx_int,Wtx_int);

    end

end

function intf = projectInterference(siz,sig,intf)

    noFastFading = reshape(arrayfun(@(x)isempty(x.sinrsubs),intf),1,[]);
    if (all(noFastFading))
        % If fast fading information is absent for all interferers, return
        % immediately as no projection is required
        return;
    end

    siz(1) = siz(1) / 12;
    indfn = @(b)sub2ind(siz,b.sinrsubs(:,1),b.sinrsubs(:,2));
    sig_ind = indfn(sig);

    % For each interferer with fast fading
    for i = find(~noFastFading)

        intf_indi = indfn(intf(i));
        prgsubs = NaN(size(sig.prgsubs));
        hest = zeros(size(sig.hest),'like',sig.hest);

        for j = 1:numel(sig_ind)

            a = find(intf_indi==sig_ind(j));

            if (~isempty(a))

                prgsubs(j) = intf(i).prgsubs(a);
                hest(j,:,:) = intf(i).hest(a,:,:);

            end

        end

        if (any(isnan(prgsubs)))

            % If any PRG is not used by the allocation, add an extra
            % all-zero precoder to 'wtx' and point the unused PRGs to it
            % via 'prgsubs'; this allows the LQM to run without error
            [nu,P,nprg] = size(intf(i).wtx);
            intf(i).wtx = cat(3,intf(i).wtx,zeros(nu,P,1,'like',intf(i).wtx));
            prgsubs(isnan(prgsubs)) = nprg + 1;
            
        end

        intf(i).prgsubs = prgsubs;
        intf(i).hest = hest;

    end

end

function [l2sm,harqedSINRs,Gsum,Csum,ECR,C] = applyHarqHistory(l2sm,harqInfo,pxsch,SINRs)

    % Layer demap current SINRs to split them between codewords
    SINRs = nrLayerDemap(SINRs);

    % Combine current SINRs with previous SINRs in this HARQ process
    procIdx = harqInfo.HARQProcessID + 1;
    [l2sm,SINRs] = combineSINRs(l2sm,procIdx,SINRs);

    % Calculate total RM output capacity 'Gsum' across HARQ transmissions
    [l2sm,Gsum] = getGsum(l2sm,procIdx);

    % Calculate number of unique LDPC coded bits 'Csum' received across
    % HARQ transmissions, and also get the number of code blocks
    [l2sm,Csum,C] = getCsum(l2sm,harqInfo,pxsch);

    % Calculate effective code rate, taking into account the incremental
    % redunancy across HARQ transmissions
    trBlkSizes = harqInfo.TransportBlockSize;
    ECR = trBlkSizes ./ Csum;

    % Adjust the SINRs to account for the effect of Chase combining; the
    % average number of Chase combined bits is Gsum / Csum
    harqedSINRs = performChaseCombining(SINRs,Gsum,Csum);
    
end

function [l2sm,SINRs] = combineSINRs(l2sm,procIdx,SINRs)

    nCodewords = numel(SINRs);
    for cwIdx = 1:nCodewords
        prevSINRs = l2sm.SINRs{procIdx,cwIdx};
        if (~isempty(prevSINRs))
            SINRs{cwIdx} = [prevSINRs; SINRs{cwIdx}];
        end
    end
    l2sm.SINRs(procIdx,:) = SINRs;

end

% Calculate total RM output capacity across HARQ transmissions
function [l2sm,Gsum] = getGsum(l2sm,procIdx)
    
    Gsum = l2sm.Gsum(procIdx,:);
    Gsum = Gsum + l2sm.G;
    l2sm.Gsum(procIdx,:) = Gsum;

end

% Calculate number of unique LDPC coded bits received across HARQ
% transmissions, and also get the number of code blocks
function [l2sm,Csum,C] = getCsum(l2sm,harqInfo,pxsch)

    persistent rs;
    if (isempty(rs) || nargin==0)
        rs = RandStream('mt19937ar','Seed',0);
    end
    if (nargin==0)
        return;
    end

    RV = harqInfo.RedundancyVersion;
    procIdx = harqInfo.HARQProcessID + 1;

    trBlkSizes = harqInfo.TransportBlockSize;
    nCodewords = numel(trBlkSizes);
    mods = modulationPerCodeword(pxsch);
    nu = layersPerCodeword(pxsch);

    TCR = l2sm.TargetCodeRate;
    G = l2sm.G;

    % Initialize cache of rate recovery soft buffer calculations. The cache
    % consists of a number of separate variables, each being a cell array.
    % Therefore the 'n'th cache entry consists of the 'n'th element in the
    % cell array for each of these variables
    if (~iscell(l2sm.SoftBuffer))
        l2sm.RRCacheArgs = repmat({newRRArgs()},1,l2sm.LPMCacheSize);
        c = cell(1,l2sm.LPMCacheSize);
        l2sm.SoftBuffer = c;
        l2sm.CSumSequence = c;
    end

    Csum = zeros(1,nCodewords);
    C = zeros(1,nCodewords);
    for cwIdx = 1:nCodewords

        % Record the current rate recovery input arguments for this HARQ
        % process and codeword
        args = {G(cwIdx) trBlkSizes(cwIdx) TCR(cwIdx) RV(cwIdx) mods{cwIdx} nu(cwIdx) l2sm.Nref};
        l2sm.RRArgs{procIdx,cwIdx} = [l2sm.RRArgs{procIdx,cwIdx}; args];

        % Select an entry of the cache that matches at least up to the
        % current RV, if such an entry exists
        rvIdx = size(l2sm.RRArgs{procIdx,cwIdx},1);
        cacheIdx = findRRCacheEntry(l2sm,procIdx,cwIdx,rvIdx,@ge);

        % If a suitable cache entry was found
        if (~isempty(cacheIdx))

            % Use the cached 'CSum' value
            Csum(cwIdx) = l2sm.CSumSequence{cacheIdx}(rvIdx);

            % Get the cached soft buffer state
            softBuffer = l2sm.SoftBuffer{cacheIdx};

        else % no suitable cache entry was found

            % Randomly select an entry to replace among the entries with
            % the shortest RV history (this includes unused entries)
            L = cellfun(@(x)size(x,1),l2sm.RRCacheArgs);
            idxs = find(L==min(L));
            idxidx = rs.randi([1 numel(idxs)]);
            cacheIdx = idxs(idxidx);

            % Copy part of another cache entry if it can be used as a
            % starting point (start with the longest possible match then
            % look at shorter lengths)
            copyFromCacheIdx = [];
            for i = rvIdx-1:-1:1
                copyFromCacheIdx = findRRCacheEntry(l2sm,procIdx,cwIdx,i,@eq);
                if (~isempty(copyFromCacheIdx))
                    % Prepare to calculate all subsequent RVs
                    firstIdx = i + 1;
                    % Copy the cache entry
                    l2sm = copyRRCacheEntry(l2sm,copyFromCacheIdx,cacheIdx);
                    break;
                end
            end

            % If part of another cached entry cannot be copied to use as a
            % starting point
            if (isempty(copyFromCacheIdx))

                % Prepare to calculate all RVs
                firstIdx = 1;
                % Clear the cache entry
                l2sm = clearRRCacheEntry(l2sm,cacheIdx);

            end

            % Get the current soft buffer state
            softBuffer = l2sm.SoftBuffer{cacheIdx};

            % Update the soft buffer state for all relevant RVs
            for i = firstIdx:rvIdx

                args = l2sm.RRArgs{procIdx,cwIdx}(i,:);
                rrfn = @(a)nrRateRecoverLDPC(ones(a{1},1),a{2:6},[],a{7});
                rr = rrfn(args);
                rr(~isfinite(rr)) = 0; % exclude filler bits

                if (isempty(softBuffer))
                    softBuffer = rr;
                else
                    softBuffer = softBuffer + rr;
                end

                % Calculate the 'Csum' value
                Csum(cwIdx) = nnz(softBuffer(:)~=0);

                % Append the new rate recovery input arguments and 'Csum'
                % value to the current cache entry, and update its soft
                % buffer state
                l2sm = updateRRCacheEntry(l2sm,cacheIdx,args,softBuffer,Csum(cwIdx));

            end

        end

        % Calculate the 'C' value
        C(cwIdx) = size(softBuffer,2);

    end

end

function args = newRRArgs()

    args = cell(0,7);

end

function cacheIdx = findRRCacheEntry(l2sm,procIdx,cwIdx,rvIdx,compfn)

    cacheIdx = [];
    p = l2sm.RRArgs{procIdx,cwIdx}(1:rvIdx,:);
    for i = 1:numel(l2sm.RRCacheArgs)
        c = l2sm.RRCacheArgs{i};
        cacheHit = compfn(size(c,1),rvIdx);
        if (cacheHit)
            c = c(1:rvIdx,:);
            for j = 1:numel(c)
                if (~isequal(c{j},p{j}))
                    cacheHit = false;
                    break;
                end
            end
            if (cacheHit)
                cacheIdx = i;
                break;
            end
        end
    end

end

function l2sm = copyRRCacheEntry(l2sm,from,to)

    l2sm.RRCacheArgs{to} = l2sm.RRCacheArgs{from};
    l2sm.SoftBuffer{to} = l2sm.SoftBuffer{from};
    l2sm.CSumSequence{to} = l2sm.CSumSequence{from};

end

function l2sm = clearRRCacheEntry(l2sm,cacheIdx)

    l2sm.RRCacheArgs{cacheIdx} = l2sm.RRCacheArgs{cacheIdx}(false,:);
    l2sm.SoftBuffer{cacheIdx} = [];
    l2sm.CSumSequence{cacheIdx} = [];

end

function l2sm = updateRRCacheEntry(l2sm,cacheIdx,args,softBuffer,Csum)

    l2sm.RRCacheArgs{cacheIdx} = [l2sm.RRCacheArgs{cacheIdx}; args];
    l2sm.SoftBuffer{cacheIdx} = softBuffer;
    l2sm.CSumSequence{cacheIdx} = [l2sm.CSumSequence{cacheIdx} Csum];

end

function outSINRs = performChaseCombining(inSINRs,Gsum,Csum)

    maxC = size(inSINRs,1);
    nCodewords = size(inSINRs,2);
    outSINRs = cell(maxC,nCodewords);
    for i = 1:maxC
        for cwIdx = 1:nCodewords
            outSINRs{i,cwIdx} = inSINRs{i,cwIdx} + 10*log10(Gsum(cwIdx) / Csum(cwIdx));
        end
    end

end

function outSINRs = splitCodeBlocks(inSINRs,C)

    nCodewords = numel(inSINRs);
    outSINRs = cell(max(C),nCodewords);
    for cwIdx = 1:nCodewords
        N = size(inSINRs{cwIdx},1);
        n = round(linspace(1,N+1,C(cwIdx)+1));
        for i = 1:C(cwIdx)
            outSINRs{i,cwIdx} = inSINRs{cwIdx}(n(i):n(i+1)-1,:);
        end
    end

end

function y = arrayMaxC(x,C)

    nCodewords = numel(x);
    y = NaN(max(C),nCodewords);
    for cwIdx = 1:nCodewords
        y(1:C(cwIdx),cwIdx) = x{cwIdx};
    end

end

function effectiveSINR = effectiveSINRMapping(Qm,SINRs,alpha,beta,G,nBuffer)

    if (nargin==6 && nBuffer<G)
        % Adjust the SINRs to account for the effect of Chase combining;
        % the average number of Chase combined bits is G / min(G,nBuffer)
        SINRs = performChaseCombining(SINRs,G,min(G,nBuffer));
    end

    nCodewords = numel(Qm);
    maxC = size(SINRs,1);
    effectiveSINR = zeros(maxC,nCodewords,'like',SINRs{1});
    for i = 1:maxC
        for cwIdx = 1:nCodewords
            effectiveSINR(i,cwIdx) = wireless.internal.L2SM.calculateEffectiveSINR(SINRs{i,cwIdx},2^Qm(cwIdx),alpha,beta);
        end
    end

end

function codeBlockBLER = sinrToCodeBLER(effectiveSINR,trBlkSizes,Qm,ECR,varargin)

    if (nargin==5)
        dlschInfos = varargin{1};
    end

    % Load AWGN table data, converting to double
    awgnTables = loadAWGNTables();

    % Limit ECR to minimum value (1/1024) and maximum value (1023/1024) 
    % expected in the AWGN tables
    ECR = max(ECR,1/1024);
    ECR = min(ECR,1023/1024);

    % Determine the integer R for which R/1024 is closest to the ECR
    R = round(ECR * 1024);

    % For each codeword
    nCodewords = numel(trBlkSizes);
    maxC = size(effectiveSINR,1);
    codeBlockBLER = zeros(maxC,nCodewords);
    for cwIdx = 1:nCodewords

        % Get base graph number (BGN) and lifting size (Zc)
        if (nargin==5)
            dlschInfo = dlschInfos(cwIdx);
        else
            dlschInfo = nrDLSCHInfo(trBlkSizes(cwIdx),R(cwIdx) / 1024);
        end
        BGN = dlschInfo.BGN;
        Zc = dlschInfo.Zc;

        % Get AWGN table for the tuple [BGN, R, Qm, Zc]
        awgnTable = getAWGNTable(awgnTables,BGN,R(cwIdx),Qm(cwIdx),Zc);

        % For each code block
        for i = 1:maxC
        
            % Interpolate the code block BLER from the effective SINR using
            % the AWGN table
            if (~isnan(effectiveSINR(i,cwIdx)))
                codeBlockBLER(i,cwIdx) = wireless.internal.L2SM.interpolatePER(effectiveSINR(i,cwIdx),awgnTable);
            end

        end

    end

end

function [mods,Qm] = modulationPerCodeword(pxsch)

    if (isa(pxsch,"nrPDSCHConfig"))
        modList = {'QPSK', '16QAM', '64QAM', '256QAM', '1024QAM'};
    else % nrPUSCHConfig
        modList = {'pi/2-BPSK','QPSK', '16QAM', '64QAM', '256QAM'};
    end
    mods = nr5g.internal.validatePXSCHModulation('L2SM',pxsch.Modulation,pxsch.NumCodewords,modList);

    nCodewords = numel(mods);
    Qm = zeros(1,nCodewords);
    for cwIdx = 1:nCodewords
        Qm(cwIdx) = nr5g.internal.getQm(mods{cwIdx});
    end

end

function cwLayers = layersPerCodeword(pxsch)

    nCodewords = pxsch.NumCodewords;
    cwLayers = floor((pxsch.NumLayers + (0:nCodewords-1)) / nCodewords);

end

function [sinrIndices,sinrPRGSubs,sinrCRBSubs] = getSINRIndices(siz,carrier,ind,wtx)

    % Get PRG subscripts, CRB subscripts and unique CRBs in PHY channel / 
    % signal indices
    ind = double(ind);
    [prgsubs,crbsubs] = nr5g.internal.prgSubscripts(siz,carrier.NStartGrid,ind,wtx);
    ucrb = unique(crbsubs).';

    % Reduce indices to a single RE in each CRB, OFDM symbol and layer.
    % Also perform the same reduction on the PRG subscript CRBs and OFDM
    % symbols, and only keep the first layer because all layers will be in
    % the same PRG
    nucrb = numel(ucrb);
    P = size(ind,2);
    sinrIndices = zeros([0 P]);
    sinrPRGSubs = zeros([0 1]);
    sinrCRBSubs = zeros([0 1]);
    pshape = @(x)reshape(x,[],P);
    for i = 1:nucrb
        
        % Get indices and PRG subscripts in the current CRB
        thiscrb = (crbsubs==ucrb(i));
        crbIndices = pshape(ind(thiscrb));
        crbPRGSubs = pshape(prgsubs(thiscrb));

        % For each OFDM symbol and layer, get indices and PRG subscripts
        % for the subcarrier closest to the middle of the CRB
        midk = ucrb(i)*12 + 6.5;
        [k,l,~] = ind2sub(siz,crbIndices);
        ul = unique(l).';
        nul = numel(ul);
        klpidx = zeros(nul,P);
        ek = abs(k - midk);
        for j = 1:nul
            ll = ul(j);
            lidx = find(l(:,1)==ll) + (0:P-1)*size(l,1);
            [~,kidx] = min(ek(lidx));
            klpidx(j,:) = lidx(kidx + (0:P-1)*size(lidx,1));
        end
        sinrIndices = [sinrIndices; crbIndices(klpidx)]; %#ok<AGROW>
        sinrPRGSubs = [sinrPRGSubs; crbPRGSubs(klpidx(:,1))]; %#ok<AGROW>
        sinrCRBSubs = [sinrCRBSubs; ones(size(klpidx,1),1)*ucrb(i)]; %#ok<AGROW>

    end

    % Sort the indices, PRG subscripts and CRB subscripts into the order in
    % which resource elements are mapped to the resource grid (across
    % subcarriers then across OFDM symbols)
    [~,idx] = sort(sinrIndices(:,1));
    sinrIndices = sinrIndices(idx,:);
    sinrPRGSubs = sinrPRGSubs(idx);
    sinrCRBSubs = sinrCRBSubs(idx);

end

function t = getAWGNTable(tables,BGN,R,Qm,Zc)

    % Get tables for the BGN
    tables = tables.data(tables.BGN==BGN);

    % Get tables for the appropriate range of R
    tables = tables.data(R>=tables.R(:,1) & R<=tables.R(:,2));

    % Get table with the desired Qm and Zc, and with code rate closest to
    % but not exceeding the desired value
    e = tables.R - R;
    i = find(e>=0,1);
    j = find(tables.Qm==Qm);
    k = find(tables.Zc==Zc);
    t = tables.data(:,:,i,j,k);

end

function y = yieldEqualREs(x,chsig)

    if (isempty(x))
        y = false;
    elseif (~strcmp(x.class,class(chsig)))
        y = false;
    else
        y = true;
        fields = fieldnames(x).';
        for i = 1:numel(fields)
            f = fields{i};
            if (~strcmp(f,'class'))
                v = x.(f);
                w = chsig.(f);
                if (isstruct(v))
                    y = yieldEqualREs(v,w);
                else
                    y = isequal(v,w);
                end
                if (~y)
                    break;
                end
            end
        end
    end

end

function chsig = maxRBAllocation(carrier,chsig)

    if (isa(chsig,"nrPDSCHConfig") || isa(chsig,"nrPUSCHConfig"))
        if (isempty(chsig.NSizeBWP))
            prbset = 0:(carrier.NSizeGrid-1);
        else
            prbset = 0:(chsig.NSizeBWP-1);
        end
        if (~(isa(chsig,"nrPUSCHConfig") && strcmpi(chsig.FrequencyHopping,"intraSlot")))
            % RE allocation depends on RB allocation, so do not update RB
            % allocation to be a max allocation
            chsig.PRBSet = prbset;
        end
    elseif (isa(chsig,"nrCSIRSConfig"))
        chsig.NumRB = carrier.NSizeGrid;
        chsig.RBOffset = 0;
    else % nrSRSConfig
        m_SRS_b = chsig.BandwidthConfigurationTable{:,2*(chsig.BSRS+1)};
        C_SRS = chsig.BandwidthConfigurationTable.C_SRS;
        chsig.CSRS = C_SRS(find(m_SRS_b<=carrier.NSizeGrid,1,'last'));
    end

end

function [ind,indInfo] = getChSigIndices(carrier,chsig)

    if (isa(chsig,"nrPDSCHConfig"))
        [ind,indInfo] = nrPDSCHIndices(carrier,chsig);
    elseif (isa(chsig,"nrPUSCHConfig"))
        [ind,indInfo] = nrPUSCHIndices(carrier,chsig);
    elseif (isa(chsig,"nrCSIRSConfig"))
        [ind,indInfo] = nrCSIRSIndices(carrier,chsig);
    else % nrSRSConfig
        [ind,indInfo] = nrSRSIndices(carrier,chsig);
    end

end

function s = propertiesAffectingREs(chsig)

    if (isa(chsig,"nrPDSCHConfig") || isa(chsig,"nrPUSCHConfig"))
        s = getFields(chsig,pxschAffectingREs(chsig));
        s.DMRS = propertiesAffectingREs(chsig.DMRS);
    elseif (isa(chsig,"nrPDSCHDMRSConfig") || isa(chsig,"nrPUSCHDMRSConfig"))
        s = getFields(chsig,dmrsAffectingREs());
    elseif (isa(chsig,"nrCSIRSConfig"))
        s = getFields(chsig,csirsAffectingREs());
    else % nrSRSConfig
        s = getFields(chsig,srsAffectingREs());
    end
    s.class = class(chsig);

end

function f = pxschAffectingREs(pxsch)

    f = {'SymbolAllocation' 'MappingType'};
    isPDSCH = isa(pxsch,"nrPDSCHConfig");
    if (isPDSCH)
        f = [f 'ReservedRE' 'ReservedPRB'];
    else % PUSCH
        f = [f 'FrequencyHopping' 'NStartBWP' 'SecondHopStartPRB' 'Interlacing' 'InterlaceIndex'];
        if (strcmpi(pxsch.FrequencyHopping,'intraSlot'))
            f = [f 'PRBSet'];
        end
    end
    f = [f 'EnablePTRS'];
    if (pxsch.EnablePTRS)
        f = [f 'RNTI' 'PTRS'];
        if (~isPDSCH) % PUSCH
            f = [f 'TransformPrecoding' 'NSizeBWP'];
        end
    end

end

function f = dmrsAffectingREs()

    f = {'DMRSAdditionalPosition','DMRSLength', ...
        'NumCDMGroupsWithoutData','DMRSTypeAPosition', ...
        'DMRSConfigurationType','DMRSPortSet','CustomSymbolSet'};

end

function f = csirsAffectingREs()

    f = {'RowNumber','SymbolLocations','SubcarrierLocations'};

end

function f = srsAffectingREs()

    f = {'NumSRSPorts','SymbolStart','NumSRSSymbols','KTC','KBarTC', ...
        'EnableStartRBHopping','StartRBIndex','Repetition', ...
        'FrequencyScalingFactor','FrequencyStart','SRSPositioning', ...
        'EnableEightPortTDM','CombOffsetHopping','CombOffsetHoppingID', ...
        'CombOffsetHoppingSubset','HoppingWithRepetition'};

end

function s = getFields(chsig,fields)

    s = struct();
    for i = 1:numel(fields)
        f = fields{i};
        s.(f) = chsig.(f);
    end

end

function crbs = getCRBs(carrier,chsig)

    opts = struct(IndexStyle='subscript',IndexBase='0based');
    if (isa(chsig,"nrPDSCHConfig"))
        if (chsig.VRBToPRBInterleaving)
            [~,indInfo] = nrPDSCHIndices(carrier,chsig,opts);
            crbs = indInfo.PRBSet;
        else
            crbs = chsig.PRBSet;
        end
    elseif (isa(chsig,"nrPUSCHConfig"))
        if (strcmpi(chsig.FrequencyHopping,'intraSlot'))
            subs = double(nrPUSCHIndices(carrier,chsig,opts));
            crbs = unique(floor(subs(:,1) / 12));
        elseif (chsig.Interlacing)
            [~,indInfo] = nrPUSCHIndices(carrier,chsig,opts);
            crbs = indInfo.PRBSet;
        else
            crbs = chsig.PRBSet;
        end
    elseif (isa(chsig,"nrCSIRSConfig"))
        crbs = chsig.RBOffset + (0:(chsig.NumRB-1));        
    else % nrSRSConfig
        [~,indInfo] = nrSRSIndices(carrier,chsig,opts);
        crbs = indInfo.PRBSet;
    end
    crbs = unique(reshape(crbs,1,[]));

end

function l2sm = combinationsCQI(l2sm)
    
    % Get unique Qm values from the CQI table
    tableQms = unique(l2sm.CQI.Table(:,1));
    tableQms = tableQms(~isnan(tableQms));

    % Get corresponding modulation strings
    mods = unique([nr5g.internal.nrPDSCHConfigBase.Modulation_Values nr5g.internal.pusch.ConfigBase.Modulation_Values]);
    allQms = cellfun(@nr5g.internal.getQm,mods);
    tableModulations = arrayfun(@(x)mods(allQms==x),tableQms);

    % Get CQI table row index combinations
    nCQI = size(l2sm.CQI.Table,1);
    nCodewords = numel(l2sm.IndicesInfo.G);
    tableRowCombos = cell(1,nCodewords);
    [tableRowCombos{:}] = ind2sub([nCQI,1],(1:nCQI^nCodewords).');
    tableRowCombos = cat(2,tableRowCombos{:});

    % Store the CQI combinations and unique Qm values and corresponding
    % modulation strings
    l2sm.CQI.TableRowCombos = tableRowCombos;
    l2sm.CQI.TableQms = tableQms;
    l2sm.CQI.TableModulations = tableModulations;

end

function [l2sm,effectiveSINR,codeBlockBLER] = sinrToCodeBLERForCQI(l2sm,pxsch,SINRs)

    % Get dimensions of outputs
    tableRowCombos = l2sm.CQI.TableRowCombos;
    nCombosCQI = size(tableRowCombos,1);
    nCodewords = numel(l2sm.IndicesInfo.G);

    % Detect if a new configuration (CQI table, PHY channel indices, or
    % xOverhead) has been provided, and if so, reinitialize relevant cache
    % entries
    newConfiguration = isempty(l2sm.CQI.G);
    if (newConfiguration)
        NL = layersPerCodeword(pxsch);
        nPRB = numel(pxsch.PRBSet);
        l2sm.CQI.Qm = NaN(nCombosCQI,nCodewords);
        l2sm.CQI.G = NaN(nCombosCQI,nCodewords);
        l2sm.CQI.TransportBlockSize = NaN(nCombosCQI,nCodewords);
        l2sm.CQI.C = NaN(nCombosCQI,nCodewords);
        l2sm.CQI.NBuffer = NaN(nCombosCQI,nCodewords);
        l2sm.CQI.EffectiveCodeRate = NaN(nCombosCQI,nCodewords);
        l2sm.CQI.DLSCHInfo = repmat(nrDLSCHInfo(1,0.5),nCombosCQI,nCodewords);
    end

    % Set up outputs
    effectiveSINR = NaN(nCombosCQI,nCodewords);
    codeBlockBLER = NaN(nCombosCQI,nCodewords);
    if (l2sm.SplitCodeBlocks)
        effectiveSINR = num2cell(effectiveSINR);
        codeBlockBLER = num2cell(codeBlockBLER);
    end

    % If a new configuration has been provided, calculate and cache values
    % that are not a functon of the incoming SINRs
    if (newConfiguration)

        % For each CQI combination
        for i = 1:nCombosCQI
    
            % Get corresponding Qm combination
            Qm = l2sm.CQI.Table(tableRowCombos(i,:),1)';
            l2sm.CQI.Qm(i,:) = Qm;
    
            % Get PXSCH capacity
            l2sm.CQI.G(i,:) = l2sm.IndicesInfo.Gd * Qm .* NL;

            % If any Qm value is valid
            if any(Qm > 0)

                % Get target code rates and transport block sizes.
                % Note that the 'modulations' and 'targetCodeRates' passed
                % to nrTBS are only for Qm > 0, but the output is assigned
                % for all codewords because nrTBS keys the number of
                % codewords from sum(NL). Any elements of
                % TransportBlockSize and EffectiveCodeRate corresponding to
                % invalid Qm are not subsequently used, because they are
                % avoided via checks on variable 'tuples' created below
                modulations = arrayfun(@(x)l2sm.CQI.TableModulations(l2sm.CQI.TableQms==x),Qm(Qm > 0));
                targetCodeRates = l2sm.CQI.Table(tableRowCombos(i,:),2).' / 1024;
                l2sm.CQI.TransportBlockSize(i,:) = nrTBS(modulations,sum(NL),nPRB,l2sm.IndicesInfo.NREPerPRB,targetCodeRates(Qm > 0),l2sm.CQI.XOverhead);

                % Get DL-SCH information
                l2sm.CQI.DLSCHInfo(i,Qm > 0) = arrayfun(@nrDLSCHInfo,l2sm.CQI.TransportBlockSize(i,Qm > 0),targetCodeRates(Qm > 0));

            end

        end

        % Get number of code blocks
        l2sm.CQI.C(:) = [l2sm.CQI.DLSCHInfo(:).C];

        % Get rate matching buffer size
        Ncb = reshape([l2sm.CQI.DLSCHInfo(:).N],size(l2sm.CQI.DLSCHInfo));
        if (~isempty(l2sm.Nref))
            Ncb = min(Ncb,l2sm.Nref);
        end
        l2sm.CQI.NBuffer = arrayfun(@nr5g.internal.ldpc.softBufferSize,l2sm.CQI.DLSCHInfo,Ncb) .* l2sm.CQI.C;

        % Get effective code rate, accounting for rate repetition occurring
        % in the first RV for code rates lower than the mother code rate
        % for the LDPC base graph
        den = min(l2sm.CQI.G,l2sm.CQI.NBuffer);
        l2sm.CQI.EffectiveCodeRate = l2sm.CQI.TransportBlockSize ./ den;

    end

    % Perform layer demapping on input SINRs
    layerSINRs = nrLayerDemap(SINRs);

    % For each codeword
    for cwIdx = 1:nCodewords

        % Two passes:
        % p=1: effective SINR calculation
        % p=2: code block BLER calculation
        for p = 1:2

            % Get the parameter tuples for all CQI combinations for the
            % current codeword and pass
            if (p==1)

                % Effective SINR calculation, tuple is [Qm G NBuffer C]
                tuples = [l2sm.CQI.Qm(:,cwIdx) l2sm.CQI.G(:,cwIdx) l2sm.CQI.NBuffer(:,cwIdx) l2sm.CQI.C(:,cwIdx)];

            else % p==2

                % Code block BLER calculation, tuple is [trBlkSizes Qm ECR]
                tuples = [l2sm.CQI.TransportBlockSize(:,cwIdx) l2sm.CQI.Qm(:,cwIdx) round(l2sm.CQI.EffectiveCodeRate(:,cwIdx)*1024)];

            end

            % Find the unique parameter tuples 'u' and their indices in the
            % overall set of parameter tuples 'ic'
            [u,~,ic] = unique(tuples,'rows');

            % For each parameter tuple not containing NaNs
            for i = find(~any(isnan(u),2)).'

                % Extract that parameter tuple
                tuple = u(i,:);

                % Using that tuple, calculate the effective SINR or code
                % block BLER for the current codeword. Map the result into
                % the elements of the output that correspond to CQIs with
                % parameters matching that tuple
                if (p==1)

                    % Get named parameters from tuple
                    Qm = tuple(1); G = tuple(2); nBuffer = tuple(3); C = tuple(4);

                    % Optionally split SINRs into code block segments, so
                    % that the effective SINR and code block BLER can be
                    % calculated per code blocks
                    if (l2sm.SplitCodeBlocks)
                        splitSINRs = splitCodeBlocks(layerSINRs(cwIdx),C);
                    else
                        splitSINRs = layerSINRs(cwIdx);
                    end

                    % Get the effective SINR, accounting for rate
                    % repetition occurring in the first RV for code rates
                    % lower than the mother code rate for the LDPC base
                    % graph
                    e = effectiveSINRMapping(Qm,splitSINRs,l2sm.Alpha,l2sm.Beta,G,nBuffer);
                    if (l2sm.SplitCodeBlocks)
                        e = {e};
                    end
                    effectiveSINR(ic==i,cwIdx) = e;

                else % p==2

                    % Get named parameters from tuple
                    trBlkSize = tuple(1); Qm = tuple(2); ECR = tuple(3) / 1024;

                    % If a new configuration has been provided, cache the
                    % DL-SCH info for the current parameter tuple (to avoid
                    % calculating it in every 'sinrToCodeBLER' call)
                    if (newConfiguration)
                        l2sm.CQI.DLSCHInfo(ic==i,cwIdx) = nrDLSCHInfo(trBlkSize,ECR);
                    end

                    % 'ic1' is the index of the first effective SINR and
                    % DL-SCH info corresponding to this parameter tuple. It
                    % does not matter which index is selected, as all the
                    % indices 'find(ic==i)' point to elements with the same
                    % effective SINR value and DL-SCH info
                    ic1 = find(ic==i,1);

                    % Calculate the code block BLER
                    e = effectiveSINR(ic1,cwIdx);
                    if (l2sm.SplitCodeBlocks)
                        e = e{1};
                    end
                    c = sinrToCodeBLER(e,trBlkSize,Qm,ECR,l2sm.CQI.DLSCHInfo(ic1,cwIdx));
                    if (l2sm.SplitCodeBlocks)
                        c = {c};
                    end
                    codeBlockBLER(ic==i,cwIdx) = c;

                end

            end

        end

    end

end

function idx = bestPerCodewordCQI(l2sm,transportBLER,blerThreshold)

    % Select the first CQI combination
    idx = 1;

    % In the case of multiple codewords 
    nCodewords = size(transportBLER,2);
    if (nCodewords > 1)

        % For each codeword
        cwIdxs = 1:nCodewords;
        for cwIdx = cwIdxs

            % Get the current CQI combination
            tableRow = l2sm.CQI.TableRowCombos(idx,:);

            % Find the rows of the CQI table that correspond to all the
            % CQIs for the current codeword and the current CQI for all the
            % other codewords
            idxAllCQIsThisCW = find(all(l2sm.CQI.TableRowCombos(:,cwIdxs~=cwIdx)==tableRow(cwIdxs~=cwIdx),2));

            % From the set of rows above, find (if it exists) the row that
            % meets the BLER criterion for the current codeword
            idxBestCQIThisCW = find(transportBLER(idxAllCQIsThisCW,cwIdx)<=blerThreshold,1,'last');

            if (~isempty(idxBestCQIThisCW))
                idx = idxAllCQIsThisCW(idxBestCQIThisCW);
            end

        end

    end

end

function t = loadAWGNTables()

    persistent awgnTables;
    if isempty(awgnTables)
        data = coder.load('nr5g/internal/L2SM.mat');
        awgnTables = data.awgnTables;
        for bgn = 1:size(awgnTables.BGN,1)
            for r = 1:size(awgnTables.data(bgn).R,1)
                x = awgnTables.data(bgn).data(r).data;
                awgnTables.data(bgn).data(r).data = double(x);
            end
        end
    end

    t = awgnTables;

end

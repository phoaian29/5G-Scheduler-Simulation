classdef nrNodeValidation
    %nrNodeValidation Implements the validations required for 5G node inputs
    %
    %   Note: This is an internal undocumented class and its API and/or
    %   functionality may change in subsequent releases.

    % Copyright 2022-2024 The MathWorks, Inc.

    methods(Static)
        function param = validateGNBInputs(gNB, param)
            %validateGNBInputs Validate the nrGNB constructor inputs

            % Allowed properties
            allowedProps = ["Name" "Position" "Velocity" "ReceiveGain" "NoiseFigure" "NumTransmitAntennas" ...
                "NumReceiveAntennas" "TransmitPower" "DuplexMode" "CarrierFrequency" ...
                "ChannelBandwidth" "SubcarrierSpacing" "NumResourceBlocks" "DLULConfigTDD" ...
                "NumHARQ" "PHYAbstractionMethod" "SRSPeriodicityUE"];
            txBandwidthConfig = nr5g.internal.MACConstants.txBandwidthConfig;
            % Convert the character vectors to strings
            paramLength = numel(param);
            for idx=1:2:paramLength
                param{idx} = string(param{idx});
                if isstring(param{idx+1})||ischar(param{idx+1})||iscellstr(param{idx+1})
                    param{idx+1} = string(param{idx+1});
                end
            end
            names = [param{1:2:end}];
            unMatchedNamesFlag = ~ismember(names, allowedProps);
            coder.internal.errorIf(sum(unMatchedNamesFlag) > 0, 'nr5g:nrNode:InvalidNVPair', names(find(unMatchedNamesFlag, 1)))

            modeIdx = find(strcmp(names, 'DuplexMode'));
            if ~isempty(modeIdx)
                validateattributes(param{2*modeIdx}, {'string','char'}, {'nonempty', 'scalartext'}, 'DuplexMode', 'DuplexMode')
                param{2*modeIdx} = validatestring(param{2*modeIdx}, nrGNB.DuplexMode_Values, mfilename, "DuplexMode");
            end
            dlulConfigTDD = gNB.DLULConfigTDD;
            % Validate TDD configuration
            if ~isempty(modeIdx) && strcmpi(param{2*modeIdx}, 'TDD')
                param{2*modeIdx} = upper(param{2*modeIdx});
                scsIdx = find(strcmp(names, 'SubcarrierSpacing'));
                if isempty(scsIdx)
                    scs = gNB.SubcarrierSpacing;
                else
                    scs = param{2*scsIdx};
                end
                idx = find(strcmp(names, 'DLULConfigTDD'));
                validSCS = [15e3 30e3 60e3 120e3];
                numerology = find(validSCS==scs, 1, 'first');
                if ~isempty(idx) && ~isempty(numerology)
                    dlulConfigTDD = param{2*idx};
                    validateattributes(dlulConfigTDD, {'struct'}, {'nonempty'}, 'DLULConfigTDD', 'DLULConfigTDD');
                    actDLULConfigTDDFields = fieldnames(dlulConfigTDD);
                    expDLULConfigTDDFields = {'DLULPeriodicity', 'NumDLSlots', 'NumULSlots', 'NumDLSymbols', 'NumULSymbols'};
                    unMatchedIdxs = ~ismember(expDLULConfigTDDFields, actDLULConfigTDDFields);
                    coder.internal.errorIf(sum(unMatchedIdxs) ~= 0, 'nr5g:nrGNB:MissingDLULConfigField', expDLULConfigTDDFields{find(unMatchedIdxs,1)})

                    % Validate the DL-UL pattern duration
                    validDLULPeriodicity{1} = { 1 2 5 10 }; % Applicable for scs = 15e3 Hz
                    validDLULPeriodicity{2} = { 0.5 1 2 2.5 5 10 }; % Applicable for scs = 30e3 Hz
                    validDLULPeriodicity{3} = { 0.5 1 1.25 2 2.5 5 10 }; % Applicable for scs = 60e3 Hz
                    validDLULPeriodicity{4} = { 0.5 0.625 1 1.25 2 2.5 5 10}; % Applicable for scs = 120e3 Hz

                    validateattributes(dlulConfigTDD.DLULPeriodicity, {'numeric'}, {'nonempty'}, 'DLULConfigTDD.DLULPeriodicity', 'DLULPeriodicity');
                    validSet = cell2mat(validDLULPeriodicity{numerology});
                    if ~ismember(dlulConfigTDD.DLULPeriodicity, validSet) % DLULPeriodicity is not valid for the specified numerology
                        formattedValidSetStr = [sprintf('{') (sprintf(repmat('%.3f, ', 1, length(validSet)-1)', validSet(1:end-1) )) sprintf('%.3f}', validSet(end))];
                        coder.internal.error('nr5g:nrGNB:InvalidDLULPeriodicity', ""+dlulConfigTDD.DLULPeriodicity, formattedValidSetStr);
                    end

                    % Number of slots in DL-UL pattern
                    numSlotsDLULPattern = dlulConfigTDD.DLULPeriodicity * (scs/15e3);

                    % Validate the number of full DL slots at the beginning of DL-UL pattern.
                    % Full DL slots must be in the range [0 numSlotsDLULPattern-1]. '0' is
                    % allowed as 'S' slot can be used for DL data
                    validateattributes(dlulConfigTDD.NumDLSlots, {'numeric'}, {'nonempty', 'integer', 'nonnegative'}, 'DLULConfigTDD.NumDLSlots', 'NumDLSlots');
                    coder.internal.errorIf(~(dlulConfigTDD.NumDLSlots <= (numSlotsDLULPattern-1)), 'nr5g:nrGNB:InvalidNumFullDLSlots', dlulConfigTDD.NumDLSlots, numSlotsDLULPattern)

                    % Validate the number of full UL slots at the end of DL-UL pattern.
                    % Full UL slots must be in the range [1 numSlotsDLULPattern-1]. '0' is
                    % not allowed as 'S' slot is not supported for sending UL data. Any UL symbol is
                    % utilized only for SRS
                    validateattributes(dlulConfigTDD.NumULSlots, {'numeric'}, {'nonempty', 'integer', 'positive'}, 'DLULConfigTDD.NumULSlots', 'NumULSlots');
                    coder.internal.errorIf(~(dlulConfigTDD.NumULSlots <= (numSlotsDLULPattern-1)),'nr5g:nrGNB:InvalidNumFullULSlots', dlulConfigTDD.NumULSlots, numSlotsDLULPattern)

                    % Validate number of DL symbols in 'S' slot
                    validateattributes(dlulConfigTDD.NumDLSymbols, {'numeric'}, {'nonempty', 'integer', 'nonnegative'}, 'DLULConfigTDD.NumDLSymbols', 'NumDLSymbols');
                    % Validate number of DL symbols
                    if (dlulConfigTDD.NumDLSymbols > 0)
                        coder.internal.errorIf((dlulConfigTDD.NumDLSymbols < 7 || dlulConfigTDD.NumDLSymbols > 14),'nr5g:nrGNB:InvalidNumDLSymbols', dlulConfigTDD.NumDLSymbols);
                    end

                    % Validate number of UL symbols in 'S' slot
                    validateattributes(dlulConfigTDD.NumULSymbols, {'numeric'}, {'nonempty', 'integer', 'scalar', '>=', 0, '<=', 1}, 'DLULConfigTDD.NumULSymbols', 'NumULSymbols');

                    % Validate sum of DL and UL symbols in 'S' slot
                    coder.internal.errorIf((dlulConfigTDD.NumDLSymbols + dlulConfigTDD.NumULSymbols > 14),'nr5g:nrGNB:InvalidSumDLULSymbols', dlulConfigTDD.NumDLSymbols + dlulConfigTDD.NumULSymbols);

                    % Validate sum of full DL and UL slots
                    if (dlulConfigTDD.NumDLSymbols==0 && dlulConfigTDD.NumULSymbols==0)
                        % If there is no 'S' slot then the sum of full DL and UL slots must match
                        % the total slots in DL-UL pattern
                        coder.internal.errorIf(~(dlulConfigTDD.NumDLSlots + dlulConfigTDD.NumULSlots == numSlotsDLULPattern), ...
                            'nr5g:nrGNB:InvalidSumFullDLULSlotsWithoutSSlot', dlulConfigTDD.NumDLSlots + dlulConfigTDD.NumULSlots, numSlotsDLULPattern)
                    else
                        % If there is an 'S' slot then the sum of full DL and UL slots be one less
                        % than the total slots in DL-UL pattern
                        coder.internal.errorIf(~(dlulConfigTDD.NumDLSlots + dlulConfigTDD.NumULSlots  == (numSlotsDLULPattern-1)), ...
                            'nr5g:nrGNB:InvalidSumFullDLULSlotsWithSSlot', dlulConfigTDD.NumDLSlots + dlulConfigTDD.NumULSlots, numSlotsDLULPattern-1);
                    end

                    % Validate that there must be some DL resources in the DL-UL pattern
                    coder.internal.errorIf((dlulConfigTDD.NumDLSlots == 0 && dlulConfigTDD.NumDLSymbols == 0), ....
                        'nr5g:nrGNB:InvalidDLResources', dlulConfigTDD.NumDLSlots, dlulConfigTDD.NumDLSymbols)
                else
                    param{end+1} = 'DLULConfigTDD';
                    dlulConfigTDD.DLULPeriodicity = dlulConfigTDD.DLULPeriodicity * (15e3/scs);
                    param{end+1} = dlulConfigTDD;
                end
            end

            % Validate PHY abstraction method
            phyTypeIdx = find(strcmp(names, 'PHYAbstractionMethod'));
            if ~isempty(phyTypeIdx)
                param{2*phyTypeIdx} = nr5g.internal.nrNodeValidation.validatePHYAbstractionMethod(param{2*phyTypeIdx});
            end

            % Cross validate ChannelBandwidth, SCS and NumResourceBlocks
            channelBandwidth = gNB.ChannelBandwidth;
            channelBandwidthIdx = find(strcmp(names, 'ChannelBandwidth'), 1);
            if ~isempty(channelBandwidthIdx)
                channelBandwidth = param{2*channelBandwidthIdx};
            end
            scs = gNB.SubcarrierSpacing;
            scsIdx = find(strcmp(names, 'SubcarrierSpacing'));
            if ~isempty(scsIdx)
                scs = param{2*scsIdx};
            end
            % Validate scs and bandwidth before cross-validating them
            if (isnumeric(scs) && isnumeric(channelBandwidth) && isscalar(scs) ....
                    && isscalar(channelBandwidth) && ismember(channelBandwidth, ...
                    [5e6 10e6 15e6 20e6 25e6 30e6 35e6 40e6 45e6 50e6 60e6 70e6 80e6 90e6 100e6 200e6 400e6]) ...
                    &&  ismember(scs,  [15e3 30e3 60e3 120e3]))
                channelBandwidthRows = txBandwidthConfig((txBandwidthConfig(:, 1) == channelBandwidth), :);
                matchingRow =  find(channelBandwidthRows(:, 2) == scs);  % Match SCS
                if channelBandwidthRows(matchingRow, 3) == 0 % Invalid combination of bandwidth and scs
                    allowedSCSRows = channelBandwidthRows(channelBandwidthRows(:,3) ~=0, :);
                    allowedSCS = allowedSCSRows(:, 2);
                    if size(allowedSCS,1)>1
                        coder.internal.errorIf(channelBandwidthRows(matchingRow, 3) == 0, 'nr5g:nrGNB:InvalidBandwidthSCSCombination', channelBandwidth, scs, ...
                            sprintf(strcat(repmat('%d, ', 1, size(allowedSCS,1)-1), ' or %d'), allowedSCS));
                    else
                        coder.internal.errorIf(channelBandwidthRows(matchingRow, 3) == 0, 'nr5g:nrGNB:InvalidBandwidthSCSCombination', channelBandwidth, scs, ...
                            allowedSCS);
                    end
                end
                numResourceBlocksIdx = find(strcmp(names, 'NumResourceBlocks'), 1);
                if ~isempty(numResourceBlocksIdx)
                    % If numResourceBlocks is configured then validate that it should not more
                    % than maximum RBs in transmission bandwidth
                    numResourceBlocks = param{2*numResourceBlocksIdx};
                    if isnumeric(numResourceBlocks) && isscalar(numResourceBlocks) &&  ....
                            floor(numResourceBlocks)==numResourceBlocks && ....
                            isfinite(numResourceBlocks) && numResourceBlocks>0
                        coder.internal.errorIf(numResourceBlocks > channelBandwidthRows(matchingRow, 3), 'nr5g:nrGNB:InvalidNumRBs', ...
                            numResourceBlocks, channelBandwidthRows(matchingRow, 3), channelBandwidth, scs);
                    end
                else
                    % Automatically calculate NumResourceBlocks from channel bandwidth and SCS
                    numResourceBlocks =  channelBandwidthRows(matchingRow, 3);
                    param{end+1} = 'NumResourceBlocks';
                    param{end+1} = numResourceBlocks;
                end
            end

            % Validate SRS periodicity
            validSRSPeriodicity = [1 2 4 5 8 10 16 20 32 40 64 80 160 320 640 1280 2560];
            srsPeriodicityIdx = find(strcmp(names, 'SRSPeriodicityUE'), 1);
            value = gNB.SRSPeriodicityUE;
            minSRSResourcePeriodicity = gNB.MinSRSResourcePeriodicity;

            if ~isempty(modeIdx) && strcmpi(param{2*modeIdx}, 'TDD')
                numSlotsDLULPattern = dlulConfigTDD.DLULPeriodicity*(scs/15e3);
                % SRS resource periodicity as minimum value such that it is at least 5
                % slots and integer multiple of numSlotsDLULPattern
                validSetTDD = validSRSPeriodicity(validSRSPeriodicity>=minSRSResourcePeriodicity & ...
                    ~mod(validSRSPeriodicity, numSlotsDLULPattern));
                minSRSResourcePeriodicity = validSetTDD(1);
            end

            if ~isempty(srsPeriodicityIdx)
                value = param{2*srsPeriodicityIdx};
                validateattributes(value, {'numeric'}, {'nonempty', 'scalar', 'integer'}, 'SRSPeriodicityUE', 'SRSPeriodicityUE');
                validSet = validSRSPeriodicity(~mod(validSRSPeriodicity,minSRSResourcePeriodicity));
                formattedValidSRSSetStr = [sprintf('{') (sprintf(repmat('%d, ', 1, length(validSet)-1)', validSet(1:end-1) )) sprintf('%d}', validSet(end))];
                % SRS periodicity must be an integer multiple of SRS resource occurrence
                coder.internal.errorIf(~ismember(value, validSet), 'nr5g:nrGNB:InvalidSRSPeriodicityUE', value, formattedValidSRSSetStr);
            end
        end

        function param = validateUEInputs(param)
            %validateUEInputs Validate the nrUE constructor inputs

            % Convert the character vectors to strings
            paramLength = numel(param);
            for idx=1:2:paramLength
                param{idx} = string(param{idx});
                if isstring(param{idx+1})||ischar(param{idx+1})||iscellstr(param{idx+1})
                    param{idx+1} = string(param{idx+1});
                end
            end

            % Allowed properties
            allowedProps = ["Name" "Position" "Velocity" "ReceiveGain" "NoiseFigure" "NumTransmitAntennas" "NumReceiveAntennas" "TransmitPower" "PHYAbstractionMethod"];
            names = [param{1:2:end}];
            unMatchedNamesFlag = ~ismember(names, allowedProps);
            coder.internal.errorIf(sum(unMatchedNamesFlag) > 0, 'nr5g:nrNode:InvalidNVPair', names(find(unMatchedNamesFlag, 1)))

            % Validate PHY abstraction method
            phyTypeIdx = find(strcmp(names, 'PHYAbstractionMethod'));
            if ~isempty(phyTypeIdx)
                param{2*phyTypeIdx} = nr5g.internal.nrNodeValidation.validatePHYAbstractionMethod(param{2*phyTypeIdx});
            end
        end

        function value = validateConnectUEInputs(name, value)
            %validateConnectUEInputs Validate the connect UE input parameters

            switch name
                case 'BSRPeriodicity'
                    if value ~= Inf % Inf is an allowed value
                        validateattributes(value, {'numeric'}, {'nonempty', 'scalar', 'integer'}, 'BSRPeriodicity', 'BSRPeriodicity');
                    end
                case 'CSIReportPeriodicity'
                    validateattributes(value, {'numeric'}, {'nonempty', 'scalar', 'positive', 'integer', 'finite'}, 'CSIReportPeriodicity', 'CSIReportPeriodicity');
                case 'FullBufferTraffic'
                    validateattributes(value, {'cell', 'char','string'}, {'nonempty', 'vector'}, 'FullBufferTraffic', 'FullBufferTraffic');
                    if isstring(value)||ischar(value)||iscellstr(value)
                        value = string(value);
                    else
                        coder.internal.error('nr5g:nrNode:InvalidStringDataType');
                    end
                case 'RLCBearerConfig'
                    coder.internal.errorIf(~isa(value, 'nrRLCBearerConfig'), 'nr5g:nrGNB:InvalidRLCBearerObject')
                    logicalChannelIDs = [value.LogicalChannelID];
                    coder.internal.errorIf(numel(logicalChannelIDs)~=numel(unique(logicalChannelIDs)), 'nr5g:nrGNB:DuplicateLogicalChannelID')
                    % Convert the RLC bearer configuration input into a column vector if it is
                    % currently in a row vector format
                    if isrow(value)
                        value = value';
                    end
                case 'CustomContext'
                    validateattributes(value, {'struct'}, {}, 'CustomContext', 'CustomContext')
                case 'CSIRSConfig'
                    if ~isempty(value)
                        validateattributes(value, {'nrCSIRSConfig'}, {'scalar'}, 'CSIRSConfig', 'CSIRSConfig');
                    end
                otherwise
                    coder.internal.error('nr5g:nrGNB:InvalidMethodArgs', name);
            end
        end

        function connConfig = validateConnectionConfig(connConfig)
            %validateConnectionConfig Validate per UE connection information

            if connConfig.CSIReportType == 1
                % Applicable for CSI type-I report
                csiReportConfig.CodebookType = 'Type1SinglePanel';
                % Restricting maximum rank to 4 as only single codeword is supported
                csiReportConfig.RIRestriction = [1 1 1 1 0 0 0 0];
            else
                % Applicable for CSI type-II report
                csiReportConfig.CodebookType = 'Type2';
                % Added clause as mentioned in 3GPP TS 38.214 5.2.2.2.3
                if (connConfig.CSIRSConfiguration.NumCSIRSPorts == 4)
                    csiReportConfig.NumberOfBeams = 2;
                else
                    % For NumCSIRSPorts > 4, NumberOfBeams(L) can be either {2, 3, 4}.
                    % Selecting L = 4.
                    csiReportConfig.NumberOfBeams = 4;
                end
                csiReportConfig.SubbandAmplitude = false;
                csiReportConfig.PhaseAlphabetSize = 4;
            end

            if ~isempty(connConfig.CSIRSConfiguration)
                % Set wideband measurement CSI-RS configuration on the full bandwidth
                csiReportConfig.NStartBWP = 0;
                csiReportConfig.NSizeBWP = connConfig.NumResourceBlocks;
                csiReportConfig.CQIMode = 'wideband';
                csiReportConfig.PMIMode = 'wideband';
                csiReportConfig.SubbandSize = 4;
                csiReportConfig.PRGSize = [];
                csiReportConfig.CodebookMode = 1;
                csiReportConfig.CodebookSubsetRestriction = [];
                csiReportConfig.i2Restriction = [];
                csiReportConfig.RIRestriction = [];
                if(connConfig.CSIRSConfiguration.NumCSIRSPorts > 2)
                    % Supported panel configurations for type 1 single
                    % panel codebooks, as defined in TS 38.214 Table
                    % 5.2.2.2.1-2. Each row contains: number of CSI-RS
                    % ports, N1, N2. Read N1, N2 from the row with
                    % first match of CSI-RS ports
                    panelConfigs = [4 2 1; 8 2 2; 8 4 1; 12 3 2; 12 6 1; 16 4 2; ...
                        16 8 1; 24 4 3; 24 6 2; 24 12 1; 32 4 4; 32 8 2; 32 16 1];
                    configIdx = find(panelConfigs(:,1) == connConfig.CSIRSConfiguration.NumCSIRSPorts);
                    csiReportConfig.PanelDimensions = panelConfigs(configIdx(1), 2:3); % Read the first index matched
                end

                % CSI-RS report period
                reportSlotOffSet = 2; % Report at least 2 slots after reception slot
                csirsTxPeriod = connConfig.CSIRSConfiguration.CSIRSPeriod;

                if strcmp(connConfig.DuplexMode, "FDD")
                    if isempty(connConfig.CSIReportPeriodicity)
                        % Automatic report periodicity and offset calculation in slots
                        connConfig.CSIReportPeriod = [csirsTxPeriod(1) mod(csirsTxPeriod(2) + reportSlotOffSet, csirsTxPeriod(1))];
                    else
                        connConfig.CSIReportPeriod = [connConfig.CSIReportPeriodicity ...
                            mod(csirsTxPeriod(2) + reportSlotOffSet, connConfig.CSIReportPeriodicity)];
                    end
                else % TDD
                    numSlotsDLULPattern = connConfig.DLULConfigTDD.DLULPeriodicity * (connConfig.SubcarrierSpacing / 15e3);
                    reportingSlotDLULPattern = mod(csirsTxPeriod(2) + reportSlotOffSet, numSlotsDLULPattern);
                    reportPeriodOffset = csirsTxPeriod(2) + reportSlotOffSet + ... % Add slot offset to ensure that it's a UL slot
                        max(0, connConfig.DLULConfigTDD.NumDLSlots + (connConfig.DLULConfigTDD.NumULSymbols == 0) - reportingSlotDLULPattern);
                    if isempty(connConfig.CSIReportPeriodicity)
                        % Automatic report periodicity and offset calculation in slots
                        connConfig.CSIReportPeriod = [csirsTxPeriod(1) mod(reportPeriodOffset, csirsTxPeriod(1))];
                    else
                        coder.internal.errorIf(mod(connConfig.CSIReportPeriodicity, numSlotsDLULPattern), ...
                            'nr5g:nrGNB:InvalidCSIReportPeriodicity', connConfig.CSIReportPeriodicity, numSlotsDLULPattern);
                        connConfig.CSIReportPeriod = [connConfig.CSIReportPeriodicity mod(reportPeriodOffset, connConfig.CSIReportPeriodicity)];
                    end
                end
                connConfig.CSIReportConfiguration = csiReportConfig;
            end

            % Validate the BSR periodicity
            validBSRPeriodicity = [1, 5, 10, 16, 20, 32, 40, 64, 80, 128, 160, 320, 640, 1280, 2560, Inf];
            coder.internal.errorIf(~ismember(connConfig.BSRPeriodicity, validBSRPeriodicity), 'nr5g:nrGNB:InvalidBSRPeriodicityScalar',connConfig.BSRPeriodicity);

            % Validate full buffer
            connConfig.FullBufferTraffic = lower(connConfig.FullBufferTraffic);
            coder.internal.errorIf(~any(strcmp(connConfig.FullBufferTraffic, ["of" "d" "u" "on" "off" "dl" "ul"])), 'nr5g:nrGNB:InvalidFullBufferStringChoiceScalar',connConfig.FullBufferTraffic);
            connConfig.FullBufferTraffic = validatestring(connConfig.FullBufferTraffic, {'on', 'off', 'dl', 'ul'}, mfilename);
        end

        function [upperLayerDataInfo, rlcEntity] = validateNVPairAddTrafficSource(obj, nvPair)
            %validateNVPairAddTrafficSource Validates the NV pairs for
            %addTrafficSource method

            % Initialize default parameters structure
            defaultParams = struct('DestinationNode', [], 'LogicalChannelID', []);

            % Validate the given Name-Value pairs and update the
            % defaultParams structure
            for idx = 1:2:numel(nvPair)
                value = nvPair{idx};
                switch value
                    case 'DestinationNode'
                        destinationNode = nvPair{idx+1};
                        coder.internal.errorIf(isempty(destinationNode), "nr5g:nrNode:NoDestination", obj.ID);
                        validateattributes(destinationNode, {'nrGNB', 'nrUE'}, {'scalar', 'nonempty'}, mfilename, 'DestinationNode');
                        cellID = destinationNode.NCellID;
                        coder.internal.errorIf(isempty(cellID) || obj.NCellID ~= cellID, "nr5g:nrNode:NoConnection", destinationNode.ID, obj.ID)
                        defaultParams.DestinationNode = destinationNode;
                    case 'LogicalChannelID'
                        logicalChannelID = nvPair{idx+1};
                        validateattributes(logicalChannelID, {'numeric'}, {'scalar', 'nonempty', 'integer', '>=', 4, '<=', 32});
                        defaultParams.LogicalChannelID = logicalChannelID;
                    otherwise
                        coder.internal.error("nr5g:nrNode:InvalidNVPair", value);
                end
            end

            upperLayerDataInfo = struct('DestinationNodeID', 0, 'LogicalChannelID', defaultParams.LogicalChannelID, 'RNTI', 0);
            if isa(obj, 'nrGNB')
                linkType = "DL";
                coder.internal.errorIf(~isa(destinationNode, 'nrUE'), "nr5g:nrNode:InvalidGNBDestination", obj.ID);
                coder.internal.errorIf(any(strcmpi(obj.FullBufferTraffic(destinationNode.RNTI),  ["on" "DL" "UL"])), "nr5g:nrNode:FullBufferEnabled");
                upperLayerDataInfo.DestinationNodeID = destinationNode.ID;
                upperLayerDataInfo.RNTI = destinationNode.RNTI;
            else
                linkType = "UL";
                if ~isempty(defaultParams.DestinationNode)
                    coder.internal.errorIf(~isa(destinationNode, 'nrGNB'), "nr5g:nrNode:InvalidUEDestination", obj.ID);
                end
                upperLayerDataInfo.DestinationNodeID = obj.GNBNodeID;
                upperLayerDataInfo.RNTI = obj.RNTI;
                coder.internal.errorIf(any(strcmpi(obj.FullBufferTraffic, ["on" "DL" "UL"])), "nr5g:nrNode:FullBufferEnabled");
            end
            if isempty(defaultParams.LogicalChannelID)
                rlcEntity = nr5g.internal.nrNodeValidation.findRLCEntityWithMinLCHID(obj, upperLayerDataInfo.RNTI);
                coder.internal.errorIf(isempty(rlcEntity), "nr5g:nrNode:NoLogicalChannel", upperLayerDataInfo.DestinationNodeID, linkType);
                upperLayerDataInfo.LogicalChannelID = rlcEntity.LogicalChannelID;
            else
                rlcEntity = nr5g.internal.nrNodeValidation.getRLCEntity(obj, upperLayerDataInfo);
                coder.internal.errorIf(isempty(rlcEntity)||isempty(rlcEntity.BufferSize), "nr5g:nrNode:InvalidTrafficMapping", upperLayerDataInfo.DestinationNodeID, upperLayerDataInfo.LogicalChannelID, linkType);
            end
        end

        function value = validateConfigureSchedulerInputs(name, value)
            %validateConfigureSchedulerInputs Validate the configure scheduler input parameters

            switch name
                case 'ResourceAllocationType'
                    validateattributes(value, {'numeric'}, {'nonempty', 'scalar', 'integer', '>=', 0, '<=', 1}, 'ResourceAllocationType', 'ResourceAllocationType');
                case 'FixedMCSIndexDL'
                    validateattributes(value, {'numeric'}, {'scalar', 'integer', '>=', 0, '<=', 27}, 'FixedMCSIndexDL', 'FixedMCSIndexDL');
                case 'FixedMCSIndexUL'
                    validateattributes(value, {'numeric'}, {'scalar', 'integer', '>=', 0, '<=', 27}, 'FixedMCSIndexUL', 'FixedMCSIndexUL');
                case 'MaxNumUsersPerTTI'
                    validateattributes(value, {'numeric'}, {'nonempty', 'integer', 'scalar', 'finite', '>=', 1}, 'MaxNumUsersPerTTI', 'MaxNumUsersPerTTI');
                case 'Scheduler'
                    if ~((isstring(value) && isscalar(value) && ismember(value,["RoundRobin", "ProportionalFair", "BestCQI"])) ||...
                            (ischar(value) && ismember(value,["RoundRobin", "ProportionalFair", "BestCQI"])) || ...
                            (isa(value, 'nrScheduler') && isscalar(value)))
                        coder.internal.error('nr5g:nrGNB:InvalidScheduler');
                    end
                case 'PFSWindowSize'
                    validateattributes(value, {'numeric'}, {'nonempty', 'integer', 'scalar', 'finite', '>=', 1}, 'PFSWindowSize', 'PFSWindowSize');
                case 'RVSequence'
                    if ~(isnumeric(value) && all(ismember(value, [0 1 2 3])) && numel(value)==numel(unique(value)))
                        coder.internal.error('nr5g:nrGNB:InvalidRVSequence');
                    end
                otherwise
                    coder.internal.error('nr5g:nrGNB:InvalidMethodArgs', name);
            end
        end

        function updatedLAConfig = validateConfigureSchedulerLAInputs(linkAdaptationConfig, defaultLAConfig)
            %validateConfigureSchedulerLAInputs Validate the configure scheduler
            %LA input parameters

            % Assign default values and return if the provided LA configuration
            % is empty
            if isempty(linkAdaptationConfig)
                updatedLAConfig = defaultLAConfig;
                return;
            end
            validateattributes(linkAdaptationConfig, {'struct'}, {}, 'linkAdaptationConfig', 'linkAdaptationConfig');
            actFields = fieldnames(linkAdaptationConfig);
            updatedLAConfig = defaultLAConfig;
            for fieldIdx = 1:numel(actFields)
                name = actFields{fieldIdx};
                value = linkAdaptationConfig.(name);
                switch name
                    case {'StepUp','StepDown'}
                        validateattributes(value, {'numeric'}, {'nonempty', 'scalar', '>=', 0, '<=', 27}, name, name);
                    case 'InitialOffset'
                        validateattributes(value, {'numeric'}, {'nonempty', 'scalar', '>=', -27, '<=', 27}, name, name);
                    otherwise
                        coder.internal.error('nr5g:nrGNB:InvalidMethodArgs', name);
                end
                updatedLAConfig.(name) = value;
            end
        end

        function value = validateConfigureSchedulerCSIMeasurementSignalDL(obj, csiMeasurementSignalDL)
            %validateConfigureSchedulerCSIMeasurementSignalDL Validate the configure scheduler
            %DL CSI Measurement Signal parameters

            % No further processing is required if the provided configuration is empty
            if isempty(csiMeasurementSignalDL)
                value = "CSI-RS";
                return;
            end
            % Validating the number of antennas, duplex mode and allowed strings
            coder.internal.errorIf(any([obj.NumTransmitAntennas]~=[obj.NumReceiveAntennas]) && strcmpi(csiMeasurementSignalDL, "SRS"), ...
                'nr5g:nrGNB:InvalidNumGNBTxRxAntennas');
            coder.internal.errorIf(any([obj.DuplexMode] == "FDD") && strcmpi(csiMeasurementSignalDL, "SRS"), ...
                'nr5g:nrGNB:InvalidDuplexModeDLTDDMeasurement');
            validatestring(csiMeasurementSignalDL, ["SRS", "CSI-RS"]);
            value = csiMeasurementSignalDL;
        end

        function updatedMUMIMOConfigDL = validateConfigureSchedulerMUMIMOInputs(obj, mumimoConfigDL, csiMeasurementSignalDL, defaultMUMIMOConfigDLCSIRS, defaultMUMIMOConfigDLSRS)
            %validateConfigureSchedulerMUMIMOInputs Validate the configure scheduler
            %MU-MIMO input parameters

            % Check number of transmit antennas
            coder.internal.errorIf(any([obj.NumTransmitAntennas] < 4), 'nr5g:nrGNB:InvalidNumTransmitAntennasForMUMIMO');

            % No further processing is required if the provided MU-MIMO configuration is empty
            if isempty(mumimoConfigDL)
                updatedMUMIMOConfigDL = [];
                return;
            end

            % Validate mumimoConfigDL is a nonempty struct
            validateattributes(mumimoConfigDL, {'struct'}, {'nonempty'}, 'MUMIMOConfigDL', 'MUMIMOConfigDL');

            % Define fields relevant to each type
            fieldsCSIRS = {'MinNumRBs', 'MaxNumUsersPaired', 'MaxNumLayers', 'SemiOrthogonalityFactor', 'MinCQI'};
            fieldsSRS = {'MinNumRBs', 'MaxNumUsersPaired', 'MaxNumLayers', 'MinSINR'};

            % Initialize updated configuration with default values
            if strcmpi(csiMeasurementSignalDL, "CSI-RS")
                updatedMUMIMOConfigDL = defaultMUMIMOConfigDLCSIRS;
            else
                updatedMUMIMOConfigDL = defaultMUMIMOConfigDLSRS;
            end

            % Get actual fields from input configuration
            actFields = fieldnames(mumimoConfigDL);

            % Initialize lists to track irrelevant fields as a cell array
            irrelevantFields = {};

            for fieldIdx = 1:numel(actFields)
                name = actFields(fieldIdx);
                value = mumimoConfigDL.(char(name));

                % Check field relevance based on measurement type
                if strcmpi(csiMeasurementSignalDL, "CSI-RS") && ismember(char(name), fieldsSRS) && ~ismember(char(name), fieldsCSIRS)
                    irrelevantFields{end+1} = char(name); % Append to cell array
                    continue; % Skip validation for irrelevant fields
                elseif strcmpi(csiMeasurementSignalDL, "SRS") && ismember(char(name), fieldsCSIRS) && ~ismember(char(name), fieldsSRS)
                    irrelevantFields{end+1} = char(name); % Append to cell array
                    continue; % Skip validation for irrelevant fields
                end

                % Validate fields
                switch char(name)
                    case 'MaxNumUsersPaired'
                        validateattributes(value, {'numeric'}, {'nonempty', 'scalar', 'integer', '>=', 2, '<=', 4}, 'MaxNumUsersPaired', 'MaxNumUsersPaired');
                    case 'SemiOrthogonalityFactor'
                        validateattributes(value, {'numeric'}, {'nonempty', 'scalar', '>=', 0, '<=', 1}, 'SemiOrthogonalityFactor', 'SemiOrthogonalityFactor');
                    case 'MinNumRBs'
                        validateattributes(value, {'numeric'}, {'nonempty', 'finite', 'scalar', 'integer', '>=', 1}, 'MinNumRBs', 'MinNumRBs');
                        coder.internal.errorIf(any([obj.NumResourceBlocks] < value), 'nr5g:nrGNB:InvalidMinNumRBsForMUMIMO', value, obj.NumResourceBlocks);
                    case 'MinCQI'
                        validateattributes(value, {'numeric'}, {'nonempty', 'scalar', 'integer', '>=', 1, '<=', 15}, 'MinCQI', 'MinCQI');
                    case 'MaxNumLayers'
                        validateattributes(value, {'numeric'}, {'nonempty', 'scalar', 'integer', '>=', 2, '<=', 16}, 'MaxNumLayers', 'MaxNumLayers');
                    case 'MinSINR'
                        validateattributes(value, {'numeric'}, {'nonempty', 'scalar', 'real', '>=', -7, '<=', 25}, 'MinSINR', 'MinSINR');
                    otherwise
                        coder.internal.error('nr5g:nrGNB:InvalidDLMUMIMOConfigField', char(name));
                end

                % Update the configuration
                updatedMUMIMOConfigDL.(char(name)) = value;
            end

            % Check if any fields are irrelevant
            if ~isempty(irrelevantFields)
                % Join the irrelevant fields into a single string with commas
                fieldList = strjoin(irrelevantFields, ', ');

                % Issue a warning based on the measurement type
                switch csiMeasurementSignalDL
                    case 'CSI-RS'
                        warning(message('nr5g:nrGNB:IrrelevantFieldsMUMIMOCSIRS', fieldList));
                    case 'SRS'
                        warning(message('nr5g:nrGNB:IrrelevantFieldsMUMIMOSRS', fieldList));
                    otherwise
                        warning(message('nr5g:nrGNB:UnknownMeasurementType', csiMeasurementSignalDL));
                end
            end
        end

        function rlcEntity = getRLCEntity(obj, upperLayerDataInfo)
            %getRLCEntity Return the RLC entity associated with the logical
            %channel for a given UE

            rnti = upperLayerDataInfo.RNTI;
            logicalChannelID = upperLayerDataInfo.LogicalChannelID;
            for rlcIdx = 1:numel(obj.RLCEntity)
                rlcEntity = obj.RLCEntity{rlcIdx};
                if (rnti == rlcEntity.RNTI) && (logicalChannelID == rlcEntity.LogicalChannelID)
                    return;
                end
            end
            rlcEntity = [];
        end
    end

    methods (Access = private, Static)
        function rlcEntity = findRLCEntityWithMinLCHID(obj, rnti)
            %findRLCEntityWithMinLCHID Return the RLC entity with the
            %smallest logical channel ID for a given UE

            logicalChannelID = 32; % Max logical channel ID for data
            rlcEntity = [];
            % Iterate through the RLC entities
            for rlcIdx = 1:numel(obj.RLCEntity)
                % Get the RLC entity and its associated logical channel ID
                currRLCEntity = obj.RLCEntity{rlcIdx};
                currLogicalChannelID = currRLCEntity.LogicalChannelID;
                % Check if the RLC entity's logical channel ID for the
                % specified UE is less than or equal to the latest min
                % logical channel ID
                if rnti == currRLCEntity.RNTI && ...
                        currLogicalChannelID <= logicalChannelID && ...
                        ~isempty(currRLCEntity.BufferSize)
                    % Update the min logical channel ID and its
                    % corresponding RLC entity
                    logicalChannelID = currLogicalChannelID;
                    rlcEntity = currRLCEntity;
                end
            end
        end

        function value = validatePHYAbstractionMethod(phyAbstractionMethod)
            validateattributes(phyAbstractionMethod, {'string','char'}, {'nonempty', 'scalartext'}, 'PHYAbstractionMethod', 'PHYAbstractionMethod')
            value = validatestring(phyAbstractionMethod, wirelessnetwork.internal.nrNode.PHYAbstraction_Values, mfilename, "PHYAbstractionMethod");
        end
    end
end

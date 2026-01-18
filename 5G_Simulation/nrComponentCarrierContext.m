classdef nrComponentCarrierContext < handle
    %nrComponentCarrierContext Carrier-specific information of a UE at gNB scheduler
    %
    %   This class reprsents the information which scheduler maintains for a
    %   connected UE on a particular carrier.
    %
    %   Note: This is an internal undocumented class and its API and/or
    %   functionality may change in subsequent releases.

    %   Copyright 2024 The MathWorks, Inc.

    properties(SetAccess=private)
        %NumTransmitAntennas Number of transmit antennas at the UE
        NumTransmitAntennas

        %NumReceiveAntennas Number of receive antennas at the UE
        NumReceiveAntennas

        %CSIRSConfiguration CSI-RS configuration of the UE
        CSIRSConfiguration

        %SRSConfiguration SRS configuration of the UE
        SRSConfiguration

        %CSIMeasurementDL Wideband DL channel measurement for the UE node based on CSI-RS and SRS
        % It is specified as a structure with these fields.
        %
        % CSIRS — A structure with these fields:
        %      RI — Rank indicator
        %      PMISet — Precoder matrix indicator (PMI) set
        %      CQI — Channel quality indicator
        %      W — Precoding matrix
        %
        % SRS — A structure with these fields:
        %       RI — Rank indicator
        %       W — Precoding matrix
        %       MCSIndex — Modulation and coding scheme index
        CSIMeasurementDL

        %CSIMeasurementUL UL CSI measurements based on SRS configured
        % It is a structure with the fields: 'RI', 'TPMI', 'MCSIndex'
        CSIMeasurementUL
    end

    properties(Hidden)
        %RetransmissionContextDL Information about downlink retransmission requirements
        % 1-by-P matrix where 'P' is the number of HARQ processes. It stores the
        % information of HARQ processes for which the reception failed at gNB. This
        % information is used for assigning downlink assignments for
        % retransmissions. 1 indicates that the reception has failed for this
        % particular HARQ process ID governed by the column index. 0 indicates that
        % the HARQ process does not have any retransmission requirement.
        RetransmissionContextDL

        %RetransmissionContextUL Information about uplink retransmission requirements
        % 1-by-P matrix where 'P' is the number of HARQ processes. It stores the
        % information of HARQ processes for which the reception failed at gNB. This
        % information is used for assigning uplink grants for retransmissions. 1
        % indicates that the reception has failed for this particular HARQ process
        % ID governed by the column index. 0 indicates that the HARQ process does
        % not have any retransmission requirement.
        RetransmissionContextUL

        %HarqProcessesDL Downlink HARQ processes context
        % 1-by-P structure array where 'P' is the number of DL HARQ
        % processes. Value at index 'i' contains the HARQ context of
        % process with HARQ ID 'i-1'
        HarqProcessesDL

        %HarqProcessesUL Uplink HARQ processes context
        % 1-by-P structure array where 'P' is the number of UL HARQ
        % processes. Value at index 'i' contains the HARQ context of
        % process with HARQ ID 'i-1'
        HarqProcessesUL

        %TBSizeDL Stores the size of transport block sent for DL HARQ processes
        % 1-by-P matrix where P is number of HARQ process. Value at index j stores
        % size of transport block sent for HARQ process index 'j'. Value is
        % 0 if DL HARQ process is free
        TBSizeDL

        %TBSizeUL Stores the size of transport block to be received for UL HARQ processes
        % 1-by-P matrix where P is number of HARQ process. Value at index j stores
        % size of transport block to be received from UE for HARQ process index
        % 'j'. Value is 0, if no UL packet is expected for HARQ process of the UE
        TBSizeUL

        %HarqNDIDL Last sent NDI value for the DL HARQ processes
        % 1-by-P logical array where 'P' is the number of HARQ processes. Values at
        % index j stores the last sent NDI for the DL HARQ process index 'j'
        HarqNDIDL

        %HarqNDIUL Last sent NDI value for the UL HARQ processes
        % 1-by-P logical array where 'P' is the number of HARQ processes. Values at
        % index j stores the last sent NDI for the UL HARQ process index 'j'
        HarqNDIUL

        %HarqStatusDL Status (free or busy) of each downlink HARQ process
        % 1-by-P cell array where 'P' is the number of HARQ processes. A
        % non-empty value at index j indicates that HARQ process is busy with value
        % being the downlink assignment for the UE with HARQ index 'j'. Empty value
        % indicates that the HARQ process is free.
        HarqStatusDL

        %HarqStatusUL Status (free or busy) of each uplink HARQ process
        % 1-by-P cell array where 'P' is the number of HARQ processes. A non-empty
        % value at index j indicates that HARQ process is busy with value being the
        % uplink grant for the UE with HARQ index 'j'. Empty value indicates that
        % the HARQ process is free.
        HarqStatusUL

        %IsBusyHARQDL Status (free or busy) of each downlink HARQ process of the UE
        % 1-by-P matrix where 'P' is the number of HARQ processes. 1 indicates busy
        % HARQ process and 0 indicates that the HARQ process is free.
        IsBusyHARQDL

        %IsBusyHARQUL Status (free or busy) of each uplink HARQ process
        % 1-by-P matrix where 'P' is the number of HARQ processes. 1 indicates busy
        % HARQ process and 0 indicates that the HARQ process is free.
        IsBusyHARQUL

        %RBGSize Size of a resource block group (RBG) in terms of number of RBs
        RBGSize

        %NumRBGs Number of RBGs in the bandwidth
        NumRBGs

        %MCSOffset Specifies the current MCS offset applied to the PXSCH transmissions.
        % It is a 1-by-2 matrix with corresponding MCS offsets for
        %DL/UL transmissions.
        MCSOffset

        %XOverheadPDSCH Additional overheads in PDSCH transmission
        XOverheadPDSCH = 0;

        %PrecodingGranularity PDSCH precoding granularity in terms of physical resource blocks (PRBs)
        PrecodingGranularity
    end

    methods(Hidden)
        function obj = nrComponentCarrierContext(param, cellConfig, schedulerConfig)
            %nrComponentCarrierContext Initialize carrier context
            %
            % PARAM is a structure including the following fields:
            % NumTransmitAntennas          - Number of UE Tx antennas
            % NumReceiveAntennas           - Number of UE Rx antennas
            % CSIRSConfiguration           - CSI-RS configuration of the UE corresponding to PrimaryCarrierIndex
            % SRSConfiguration             - SRS configuration of the UE corresponding to PrimaryCarrierIndex
            % NumHARQ                      - Number of HARQ processes
            % RBGSizeConfig                - RBG configuration to derive number of RBs in an RBG
            % InitialCQIDL                 - Initial DL CQI
            % InitialMCSIndexUL            - Initial UL MCS
            % InitialMCSOffsetDL           - Initial MCS offset (link adaptation) for DL direction
            % InitialMCSOffsetUL           - Initial MCS offset (link adaptation) for UL direction

            % CELLCONFIG is cell configuration.
            % SCHEDULERCONFIG is scheduler configuration.

            if ~isempty(param)
                % Initialize the properties
                inputParam = ["NumTransmitAntennas", "NumReceiveAntennas", "CSIRSConfiguration", "SRSConfiguration"];
                for idx=1:numel(inputParam)
                    obj.(inputParam(idx)) = param.(inputParam(idx));
                end
                numHARQ = param.NumHARQ;
                % Create retransmission context
                obj.HarqStatusDL = cell(1, numHARQ);
                obj.HarqStatusUL = cell(1, numHARQ);
                obj.RetransmissionContextDL = zeros(1, numHARQ);
                obj.RetransmissionContextUL = zeros(1, numHARQ);
                obj.IsBusyHARQDL = zeros(1, numHARQ);
                obj.IsBusyHARQUL = zeros(1, numHARQ);
                obj.HarqNDIDL = zeros(1, numHARQ);
                obj.HarqNDIUL = zeros(1, numHARQ);
                obj.TBSizeDL = zeros(1, numHARQ);
                obj.TBSizeUL = zeros(1, numHARQ);
                ncw = 1; % Only single codeword
                % Create HARQ processes context array
                obj.HarqProcessesDL = nr5g.internal.nrNewHARQProcesses(numHARQ, schedulerConfig.RVSequence, ncw);
                obj.HarqProcessesUL = nr5g.internal.nrNewHARQProcesses(numHARQ, schedulerConfig.RVSequence, ncw);

                % RBGSize configuration as 1 (configuration-1 RBG table) or 2
                % (configuration-2 RBG table) as defined in 3GPP TS 38.214 Section
                % 5.1.2.2.1. It defines the number of RBs in an RBG.
                nominalRBGSizePerBW = nr5g.internal.MACConstants.NominalRBGSizePerBW;
                rbgSizeIndex = min(find(cellConfig.NumResourceBlocks <= nominalRBGSizePerBW(:, 1), 1));
                if param.RBGSizeConfig == 1
                    obj.RBGSize = nominalRBGSizePerBW(rbgSizeIndex, 2);
                else % RBGSizeConfig is 2
                    obj.RBGSize = nominalRBGSizePerBW(rbgSizeIndex, 3);
                end
                obj.NumRBGs = ceil(cellConfig.NumResourceBlocks/obj.RBGSize);

                % Set PDSCH overhead
                if ~isempty(param.CSIRSConfiguration)
                    obj.XOverheadPDSCH = 18;
                    numCSIRSPorts = obj.CSIRSConfiguration.NumCSIRSPorts;
                else
                    numCSIRSPorts = cellConfig.NumTransmitAntennas;
                end

                obj.PrecodingGranularity = cellConfig.NumResourceBlocks;
                % CSI measurements initialization (DL and UL)
                initialRank = 1; % Initial ranks for UEs
                wp = ones(numCSIRSPorts, 1)./sqrt(numCSIRSPorts);
                csirs = struct('RI', initialRank, 'PMISet', struct('i1',[1 1 1]), ...
                    'CQI', param.InitialCQIDL, 'W', wp);
                obj.CSIMeasurementDL = struct('CSIRS', csirs, 'SRS', []);
                obj.CSIMeasurementUL = struct('RI', initialRank, 'TPMI', 0, 'MCSIndex', param.InitialMCSIndexUL);

                % Initialize link adaptation MCS offset values for the carrier. The
                % first column contains MCS offset in DL direction and second column
                % contains MCS offset in UL direction.
                obj.MCSOffset = [0 0];
                if ~isempty(schedulerConfig.LinkAdaptationConfigDL)
                    obj.MCSOffset(1) = param.InitialMCSOffsetDL;
                end
                if ~isempty(schedulerConfig.LinkAdaptationConfigUL)
                    obj.MCSOffset(2) = param.InitialMCSOffsetUL;
                end
            end
        end

        function updateChannelQualityDL(obj, channelQualityInfo, schedulerConfig)
            %updateChannelQualityDL Update downlink channel quality information for a UE
            %   UPDATECHANNELQUALITYDL(OBJ, CHANNELQUALITYINFO) updates
            %   downlink (DL) channel quality information for a UE.
            %   CHANNELQUALITYINFO is a structure with these fields: RI, PMISet, W, CQI.
            %   SCHEDULERCONFIG is a structure containing field 'LinkAdaptationConfigDL'.
            %       LinkAdaptationConfigDL - Link adaptation (LA) configuration structure for downlink transmissions.
            %                          The structure contains these fields: InitialOffset, StepUp and StepDown

            if schedulerConfig.CSIMeasurementSignalDLType && isfield(channelQualityInfo, 'SRSBasedDLMeasurements')
                % Update CSI measurement based on SRS
                obj.CSIMeasurementDL.SRS = channelQualityInfo.SRSBasedDLMeasurements;
            else
                % Update CSI measurement based on CSI-RS
                obj.CSIMeasurementDL.CSIRS.CQI = channelQualityInfo.CQI;
                obj.CSIMeasurementDL.CSIRS.RI = channelQualityInfo.RankIndicator;
                obj.CSIMeasurementDL.CSIRS.PMISet.i1 = channelQualityInfo.PMISet.i1;
                obj.CSIMeasurementDL.CSIRS.W = channelQualityInfo.W;
            end
            if ~isempty(schedulerConfig.LinkAdaptationConfigDL)
                % Reset link adaptation MCS offset value to initial offset
                obj.MCSOffset(1) = schedulerConfig.LinkAdaptationConfigDL.InitialOffset;
            end
        end

        function updateChannelQualityUL(obj, channelQualityInfo, schedulerConfig)
            %updateChannelQualityUL Update uplink channel quality information for a UE
            %   UPDATECHANNELQUALITYUL(OBJ, CHANNELQUALITYINFO) updates
            %   uplink (UL) channel quality information for a UE.
            %   CHANNELQUALITYINFO is a structure with these fields: RI, TPMI, MCSIndex.
            %   SCHEDULERCONFIG is a structure containing field 'LinkAdaptationConfigUL'.
            %       LinkAdaptationConfigUL - Link adaptation (LA) configuration structure for uplink transmissions.
            %                          The structure contains these fields: InitialOffset, StepUp and StepDow

            % Update CSI measurement
            obj.CSIMeasurementUL.MCSIndex = channelQualityInfo.MCSIndex;
            obj.CSIMeasurementUL.RI = channelQualityInfo.RankIndicator;
            obj.CSIMeasurementUL.TPMI = channelQualityInfo.TPMI;
            if ~isempty(schedulerConfig.LinkAdaptationConfigUL)
                % Reset link adaptation MCS offset value to initial offset
                obj.MCSOffset(2) = schedulerConfig.LinkAdaptationConfigUL.InitialOffset;
            end
        end

        function updateSRSPeriod(obj, srsPeriod)
            %updateSRSPeriod Update the SRS periodicity of the UE for the specified carrier

            obj.SRSConfiguration.SRSPeriod = srsPeriod;
        end
    end
end
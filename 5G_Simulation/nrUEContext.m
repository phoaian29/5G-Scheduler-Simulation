classdef nrUEContext < handle
    %nrUEContext Context of a connected user equipment (UE)
    %
    %   Note: This is an internal undocumented class and its API and/or
    %   functionality may change in subsequent releases.

    %   Copyright 2024 The MathWorks, Inc.

    properties (SetAccess = private)
        %RNTI RNTI of the connected UE
        RNTI

        %Name Node name of the connected UE
        Name

        %ID Node ID of the connected UE
        ID

        %NumTransmitAntennas Number of transmit antennas at UE
        NumTransmitAntennas

        %NumReceiveAntennas Number of receive antennas at UE
        NumReceiveAntennas

        %BearerConfigurationDL Bearer configuration for DL direction
        BearerConfigurationDL = cell(32, 1)

        %BearerConfgurationUL Bearer configuration for UL direction
        BearerConfigurationUL = cell(32, 1)

        %BufferStatusDLPerLCH Pending buffer amount in DL direction per logical channel
        % It is a 1-by-32 matrix where the columns contain the pending DL
        % buffer (in bytes) for logical channel IDs from 1 to 32
        BufferStatusDLPerLCH

        %BufferStatusULPerLCG Pending buffer amount in UL direction per logical channel group
        % It is a 1-by-8 matrix where the columns contain the pending UL
        % buffer (in bytes) for logical channel group IDs from 0 to 7
        BufferStatusULPerLCG

        %CSIRSConfiguration CSI-RS configuration of the UE
        CSIRSConfiguration

        %SRSConfiguration SRS configuration of the UE
        SRSConfiguration

        %CSIMeasurementDL Wideband DL channel measurement reported by
        % the UE node based on CSI-RS and SRS reception, specified as a
        % structure with these fields.
        %
        % CSIRS — A structure with these fields:
        %      RI — Rank indicator
        %      PMISet — Precoder matrix indications (PMI) set
        %      CQI — Channel quality indicator
        %      W — Precoding matrix
        %
        % SRS — A structure with these fields:
        %       RI — Rank indicator
        %       W — Precoding matrix
        %       MCSIndex — Modulation and coding scheme index
        CSIMeasurementDL

        %CSIMeasurementUL UL CSI measurements based on SRS
        % It is a structure with the fields: 'RI', 'TPMI', 'MCSIndex'
        CSIMeasurementUL

        %BufferStatusDL Total pending DL buffer amount for the UE
        % It represents the cumulative sum of all LCH's pending DL buffer
        % (in bytes) for a UE. It accounts for total amount of data
        % scheduled since the last RLC buffer status update
        BufferStatusDL

        %BufferStatusUL Total pending UL buffer amount for the UE
        % It represents the cumulative sum of all LCG's pending UL buffer
        % (in bytes) for a UE. It accounts for total amount of data
        % scheduled since the last BSR update
        BufferStatusUL
    end

    properties
        %CustomContext User-supplied custom context supplied for the UE
        CustomContext
    end

    properties(Access = private)
        %RBGSizeConfig RBG configuration to derive number of RBs in an RBG
        RBGSizeConfig

        %InitialMCSOffsetDL Initial MCS offset (link adaptation) for DL direction
        InitialMCSOffsetDL

        %InitialMCSOffsetUL Initial MCS offset (link adaptation) for UL direction
        InitialMCSOffsetUL
    end

    properties(Hidden)
        %NumHARQ Number of HARQ processes
        NumHARQ

        %PUSCHPreparationTime PUSCH preparation time in terms of number of symbols
        % Scheduler ensures that PUSCH grant arrives at UEs at least these
        % many symbols before the transmission time
        PUSCHPreparationTime

        %PUSCHDMRSConfigurationType PUSCH DM-RS configuration type (1 or 2)
        PUSCHDMRSConfigurationType

        %PUSCHDMRSLength PUSCH demodulation reference signal (DM-RS) length
        PUSCHDMRSLength

        %PUSCHDMRSAdditionalPosTypeA Additional PUSCH DM-RS positions for type A (0..3)
        PUSCHDMRSAdditionalPosTypeA

        %PUSCHDMRSAdditionalPosTypeB Additional PUSCH DM-RS positions for type B (0..3)
        PUSCHDMRSAdditionalPosTypeB

        %PDSCHDMRSConfigurationType PDSCH DM-RS configuration type (1 or 2)
        PDSCHDMRSConfigurationType

        %PDSCHDMRSLength PDSCH demodulation reference signal (DM-RS) length
        PDSCHDMRSLength

        %PDSCHDMRSAdditionalPosTypeA Additional PDSCH DM-RS positions for type A (0..3)
        PDSCHDMRSAdditionalPosTypeA

        %PDSCHDMRSAdditionalPosTypeB Additional PDSCH DM-RS positions for type B (0 or 1)
        PDSCHDMRSAdditionalPosTypeB

        %UEsServedDataRate Average data rata served for a UE considering PFS window
        %size. It is an 1-by-2 matrix which corresponding average data rate for
        %DL/UL transmissions
        UEsServedDataRate

        %InitialCQIDL Initial DL CQI for all carriers
        InitialCQIDL

        %InitialMCSIndexUL Initial UL MCS index for all carriers
        InitialMCSIndexUL

        %ComponentCarrier Information specific to component carriers
        % It is a object vector with number of elements equal to total number of
        % carriers configured on gNB. The meaningful objects are at the vector
        % indices given by the value of the property 'ConfiguredCarrier'. The rest
        % of the indices correspond to the gNB carriers which are not configured for this
        % UE and contain objects with empty property values
        ComponentCarrier

        %ConfiguredCarrier Carrier(s) on which UE is connected 
        % It is a vector of indices of carriers operated by gNB.
        ConfiguredCarrier
    end

    methods (Hidden)
        function obj = nrUEContext(param, cellConfig, schedulerConfig)
            %UEContext Construct an UE context object
            %
            % PARAM is a structure including the following fields:
            % RNTI                         - Radio network temporary identifier of the UE
            % UEName                       - Name of the UE
            % UEID                         - ID of the UE
            % NumCarriersGNB               - Number of carriers configured on gNB
            % PrimaryCarrierIndex          - Index of the primary carrier for the UE
            % NumTransmitAntennas          - Number of UE Tx antennas corresponding to PrimaryCarrierIndex
            % NumReceiveAntennas           - Number of UE Rx antennas corresponding to PrimaryCarrierIndex
            % CSIRSConfiguration           - CSI-RS configuration of the UE corresponding to PrimaryCarrierIndex
            % SRSConfiguration             - SRS configuration of the UE corresponding to PrimaryCarrierIndex
            % NumHARQ                      - Number of HARQ processes
            % PUSCHPreparationTime         - PUSCH preparation time required by UEs
            % PUSCHDMRSConfigurationType   - PUSCH DM-RS configuration type
            % PUSCHDMRSLength              - PUSCH DM-RS length
            % PUSCHDMRSAdditionalPosTypeA  - Additional PUSCH DM-RS positions for Type A
            % PUSCHDMRSAdditionalPosTypeB  - Additional PUSCH DM-RS positions for Type B
            % PDSCHMappingType             - PDSCH mapping type
            % PDSCHDMRSConfigurationType   - PDSCH DM-RS configuration type
            % PDSCHDMRSLength              - PDSCH DM-RS length
            % PDSCHDMRSAdditionalPosTypeA  - Additional PDSCH DM-RS positions for Type A
            % PDSCHDMRSAdditionalPosTypeB  - Additional PDSCH DM-RS positions for Type B
            % RBGSIZECONFIG                - RBG configuration to derive number of RBs in an RBG
            % InitialCQIDL                 - Initial DL CQI
            % InitialMCSIndexUL            - Initial UL MCS index
            %
            % CELLCONFIG is cell configuration.
            % SCHEDULERCONFIG is scheduler configuration.

            % Initialize the properties
            inputParam = ["RNTI", "NumTransmitAntennas", "NumReceiveAntennas", "CSIRSConfiguration", "SRSConfiguration", ...
                "NumHARQ", "PUSCHPreparationTime", "PUSCHDMRSConfigurationType", "PUSCHDMRSLength", "PUSCHDMRSAdditionalPosTypeA", "PUSCHDMRSAdditionalPosTypeB", ...
                "PDSCHDMRSConfigurationType", "PDSCHDMRSLength", "PDSCHDMRSAdditionalPosTypeA", "PDSCHDMRSAdditionalPosTypeB", "CustomContext","RBGSizeConfig", ...
                "InitialCQIDL", "InitialMCSIndexUL"];
            for idx=1:numel(inputParam)
                obj.(inputParam(idx)) = param.(inputParam(idx));
            end
            obj.ID = param.UEID;
            obj.Name = param.UEName;
            obj.BufferStatusDLPerLCH = zeros(1, 32); % 32 logical channels
            obj.BufferStatusULPerLCG = zeros(1, 8); % 8 logical channel groups
            obj.BufferStatusDL = 0;
            obj.BufferStatusUL = 0;

            % Initialize served data rate to 1. The first column contains served data rate in DL direction and
            % second column contains served data rate in UL direction
            obj.UEsServedDataRate([1 2]) = 1;

            % Read initial MCS offset for LA
            obj.InitialMCSOffsetDL = 0;
            obj.InitialMCSOffsetUL = 0;
            if ~isempty(schedulerConfig.LinkAdaptationConfigDL)
                obj.InitialMCSOffsetDL = schedulerConfig.LinkAdaptationConfigDL.InitialOffset;
            end
            if ~isempty(schedulerConfig.LinkAdaptationConfigUL)
                obj.InitialMCSOffsetUL = schedulerConfig.LinkAdaptationConfigUL.InitialOffset;
            end

            % Set up carrier context w.r.t index corresponding to primary carrier (empty
            % initialization for other carriers)
            obj.ConfiguredCarrier = param.PrimaryCarrierIndex;
            obj.ComponentCarrier = createArray(1, param.NumCarriersGNB, FillValue=nr5g.internal.nrComponentCarrierContext([]));
            primaryConnectionInfo = struct(NumTransmitAntennas=param.NumTransmitAntennas, ...
                NumReceiveAntennas=param.NumReceiveAntennas, CSIRSConfiguration=param.CSIRSConfiguration, ...
                SRSConfiguration=param.SRSConfiguration, NumHARQ=obj.NumHARQ, ...
                RBGSizeConfig=obj.RBGSizeConfig, InitialCQIDL=obj.InitialCQIDL, ...
                InitialMCSIndexUL=obj.InitialMCSIndexUL, InitialMCSOffsetDL=obj.InitialMCSOffsetDL, ...
                InitialMCSOffsetUL=obj.InitialMCSOffsetUL);
            obj.ComponentCarrier(param.PrimaryCarrierIndex) = nr5g.internal.nrComponentCarrierContext(primaryConnectionInfo, cellConfig, schedulerConfig);

            % Retain a copy of primary carrier context for these properties (for
            % backward compatibility)
            obj.CSIMeasurementDL = obj.ComponentCarrier(obj.ConfiguredCarrier).CSIMeasurementDL;
            obj.CSIMeasurementUL = obj.ComponentCarrier(obj.ConfiguredCarrier).CSIMeasurementUL;
        end

        function addSecondaryCarrier(obj, carrierIndex, connectionInfo, cellConfig, schedulerConfig)
            %addSecondaryCarrier Add specified carrier as secondary carrier for the UE

            % Add the carrier to configured carrier list. Initialize context of the carrier
            obj.ConfiguredCarrier = [obj.ConfiguredCarrier carrierIndex];
            connectionInfo = struct(NumTransmitAntennas=connectionInfo.NumTransmitAntennas, ...
                NumReceiveAntennas=connectionInfo.NumReceiveAntennas, CSIRSConfiguration=connectionInfo.CSIRSConfiguration, ...
                SRSConfiguration=connectionInfo.SRSConfiguration, NumHARQ=obj.NumHARQ, ...
                RBGSizeConfig=obj.RBGSizeConfig, InitialCQIDL=obj.InitialCQIDL, ...
                InitialMCSIndexUL=obj.InitialMCSIndexUL, InitialMCSOffsetDL=obj.InitialMCSOffsetDL, ...
                InitialMCSOffsetUL=obj.InitialMCSOffsetUL);
            obj.ComponentCarrier(carrierIndex) = nr5g.internal.nrComponentCarrierContext(connectionInfo, cellConfig, schedulerConfig);
        end

        function addBearerConfig(obj, bearerConfig)
            %addBearerConfig Add bearer configuration to UE context

            if bearerConfig.RLCEntityType == "UM" || bearerConfig.RLCEntityType == "AM"
                % For bi-directional RLC entities, add one DL config and one UL config
                obj.BearerConfigurationDL{bearerConfig.LogicalChannelID} = bearerConfig;
                obj.BearerConfigurationUL{bearerConfig.LogicalChannelID} = bearerConfig;
            elseif bearerConfig.RLCEntityType == "UMDL" % Uni-directional DL
                obj.BearerConfigurationDL{bearerConfig.LogicalChannelID} = bearerConfig;
            else % Uni-directional UL
                obj.BearerConfigurationUL{bearerConfig.LogicalChannelID} = bearerConfig;
            end
        end

        function obj = updateLCBufferStatusDL(obj, lcBufferStatus)
            %updateLCBufferStatusDL Update DL buffer status for a logical channel of the specified UE
            %
            %   updateLCBufferStatusDL(obj, LCBUFFERSTATUS) updates the
            %   DL buffer status for a logical channel of the specified UE.
            %
            %   LCBUFFERSTATUS is a structure with following two fields.
            %       LogicalChannelID - Logical channel ID
            %       BufferStatus - Pending amount in bytes for the specified logical channel of UE

            obj.BufferStatusDLPerLCH(lcBufferStatus.LogicalChannelID) = lcBufferStatus.BufferStatus;
            % Calculate the cumulative sum of all the logical channels pending buffer amount
            obj.BufferStatusDL = sum(obj.BufferStatusDLPerLCH);
        end

        function processMACControlElement(obj, pktInfo, varargin)
            %processMACControlElement Process the received MAC control element
            %
            %   processMACControlElement(OBJ, PKTINFO) processes the received MAC
            %   control element (CE) of the specified UE. This interface currently
            %   supports buffer status report (BSR) only.
            %
            %   processMACControlElement(OBJ, PKTINFO, LCGPRIORITY) processes the
            %   received MAC control element (CE) of the specified UE. This interface
            %   currently supports long truncated buffer status report (BSR) only.
            %
            %   PKTINFO - A structure with packet information
            %   LCGPRIORITY - A vector of priorities of all the LCGs of UE, used for
            %   processing long truncated BSR

            if pktInfo.Metadata.PacketType == 1 % BSR received
                bsr = pktInfo.Data;
                ulType = nr5g.internal.MACConstants.ULType;
                [lcid, payload] = nrMACPDUDecode(bsr, ulType); % Parse the BSR
                macCEInfo.LCID = lcid;
                macCEInfo.Packet = payload{1};

                % Values 59, 60, 61, 62 represents LCIDs corresponding to different BSR
                % formats as per 3GPP TS 38.321
                if(macCEInfo.LCID == 59 || macCEInfo.LCID == 60 || macCEInfo.LCID == 61 || macCEInfo.LCID == 62)
                    [lcgIDList, bufferSizeList] = nrMACBSRDecode(macCEInfo.LCID, macCEInfo.Packet);
                    % When buffer size is not 0 for the logical channel groups but the BSR is
                    % long truncated BSR, map the lcgIDList values to bufferSizeList values
                    % using priority and then set the buffer status context
                    if macCEInfo.LCID == 60
                        [~, priorityOrder] = sort(varargin{1}(lcgIDList+1));
                        lcgIDList = lcgIDList(priorityOrder);
                        numBufferSizeLCGs = size(bufferSizeList,1);
                        obj.BufferStatusULPerLCG(lcgIDList(1:numBufferSizeLCGs)+1) = bufferSizeList(:,2);
                        ulcgIDList = lcgIDList(numBufferSizeLCGs+1:end); % LCGs with unreported data

                        % Check whether the buffer status is zero for LCGs with unreported data
                        Idx = find(obj.BufferStatusULPerLCG(ulcgIDList+1) == 0);

                        % For LCGs having data and are not reported, assume the buffer size of 10
                        % bytes which is the first non zero entry in 3GPP TS 38.321 Table 6.1.3.1-2
                        if ~isempty(Idx)
                            obj.BufferStatusULPerLCG(ulcgIDList(Idx)+1) = 10;
                        end

                        % When buffer size is 0 for the logical channel groups in the BSR packet,
                        % set the buffer status context of all logical channel groups to 0 by
                        % considering that no logical channel group has data for transmission
                    elseif isempty(bufferSizeList) || bufferSizeList(2) == 0
                        obj.BufferStatusULPerLCG(:) = 0;
                    else
                        obj.BufferStatusULPerLCG(lcgIDList+1) = bufferSizeList(:,2);
                    end
                    % Calculate the cumulative sum of all LCG's pending buffer amount for a
                    % specific UE
                    obj.BufferStatusUL = sum(obj.BufferStatusULPerLCG, 2);
                end
            end
        end

        function updateChannelQualityDL(obj, channelQualityInfo, schedulerConfig)
            %updateChannelQualityDL Update downlink channel quality information for a UE
            %   UPDATECHANNELQUALITYDL(OBJ, CHANNELQUALITYINFO) updates
            %   downlink (DL) channel quality information for a UE.
            %   CHANNELQUALITYINFO is a structure with these fields: GNBCarrierIndex, RI, PMISet, W, CQI.
            %   SCHEDULERCONFIG is a structure containing field 'LinkAdaptationConfigDL'.
            %       LinkAdaptationConfigDL - Link adaptation (LA) configuration structure for downlink transmissions.
            %                          The structure contains these fields: InitialOffset, StepUp and StepDown

            carrierIndex = channelQualityInfo.GNBCarrierIndex;
            % Update the measurements for the carrier
            ccInfo = obj.ComponentCarrier(carrierIndex);
            ccInfo.updateChannelQualityDL(channelQualityInfo, schedulerConfig);

            % Update copy of CSIMeasurementDL w.r.t primary carrier context for backward compatibility
            if obj.ConfiguredCarrier(1) == carrierIndex
                obj.CSIMeasurementDL = ccInfo.CSIMeasurementDL;
            end
        end

        function updateChannelQualityUL(obj, channelQualityInfo, schedulerConfig)
            %updateChannelQualityUL Update uplink channel quality information for a UE
            %   UPDATECHANNELQUALITYUL(OBJ, CHANNELQUALITYINFO) updates
            %   uplink (UL) channel quality information for a UE.
            %   CHANNELQUALITYINFO is a structure with these fields: GNBCarrierIndex, RI, TPMI, MCSIndex.
            %   SCHEDULERCONFIG is a structure containing field 'LinkAdaptationConfigUL'.
            %       LinkAdaptationConfigUL - Link adaptation (LA) configuration structure for uplink transmissions.
            %                          The structure contains these fields: InitialOffset, StepUp and StepDown

            carrierIndex = channelQualityInfo.GNBCarrierIndex;
            % Update the measurements for the carrier
            ccInfo = obj.ComponentCarrier(carrierIndex);
            ccInfo.updateChannelQualityUL(channelQualityInfo, schedulerConfig);
            % Update copy of CSIMeasurementUL w.r.t primary carrier context for backward compatibility
            if obj.ConfiguredCarrier(1) == carrierIndex
                obj.CSIMeasurementUL = ccInfo.CSIMeasurementUL;
            end
        end

        function handleDLRxResult(obj, rxResultInfo, schedulerConfig)
            %handleDLRxResult Update the HARQ process context based on the Rx success/failure for DL packets
            % handleDLRxResult(OBJ, RXRESULTINFO, SCHEDULERCONFIG) updates the HARQ
            % process context, based on the ACK/NACK received by gNB for the DL packet.
            %
            % RXRESULTINFO is a structure with following fields.
            %   HARQID - HARQ process ID
            %   RxResult - 0 means NACK or no feedback received. 1 means ACK.
            %   GNBCARRIERINDEX -Index of the carrier
            %
            % SCHEDULERCONFIG is a scheduler configuration structure contains one of the following fields.
            %   LinkAdaptationConfigDL - Link adaptation (LA) configuration structure for downlink transmissions.
            %                            The structure contains the fields InitialOffset, StepUp and StepDown fields

            harqIndex = rxResultInfo.HARQID+1;
            rxResult = rxResultInfo.RxResult;

            ccInfo = obj.ComponentCarrier(rxResultInfo.GNBCarrierIndex); % Read the component carrier info
            isNewTx = strcmp(ccInfo.HarqStatusDL{harqIndex}.Type, 'newTx');
            % If DL link adaptation is enabled and it is a new transmission then update
            % MCS offset for the received feedback.
            if isNewTx && ~isempty(schedulerConfig.LinkAdaptationConfigDL)
                linkDir = nr5g.internal.MACConstants.DLType+1; % Downlink
                laConfig = schedulerConfig.LinkAdaptationConfigDL;
                offset = ccInfo.MCSOffset(linkDir);
                ccInfo.MCSOffset(linkDir)=nr5g.internal.getMCSIndexOffset(laConfig, offset, rxResult);
            end

            if rxResult % Rx success
                % Update the DL HARQ process context Mark the HARQ process as free
                ccInfo.HarqStatusDL{harqIndex} = [];
                ccInfo.IsBusyHARQDL(harqIndex) = 0;
                harqProcess = ccInfo.HarqProcessesDL(harqIndex);
                harqProcess.blkerr(1) = 0;
                ccInfo.HarqProcessesDL(harqIndex) = harqProcess;

                % Clear the retransmission context for the HARQ process of the UE. It would
                % already be empty if this feedback was not for a retransmission.
                ccInfo.RetransmissionContextDL(harqIndex) = 0;
            else % Rx failure or no feedback received
                harqProcess = ccInfo.HarqProcessesDL(harqIndex);
                harqProcess.blkerr(1) = 1;
                if harqProcess.RVIdx(1) == size(harqProcess.RVSequence,2)
                    % Packet reception failed for all redundancy versions. Mark the HARQ
                    % process as free. Also clear the retransmission context to not allow any
                    % further retransmissions for this packet
                    harqProcess.blkerr(1) = 0;
                    % Mark the HARQ process as free
                    ccInfo.HarqStatusDL{harqIndex} = [];
                    ccInfo.IsBusyHARQDL(harqIndex) = 0;
                    ccInfo.HarqProcessesDL(harqIndex) = harqProcess;
                    ccInfo.RetransmissionContextDL(harqIndex) = 0;
                else
                    % Update the retransmission context for the UE and HARQ process to indicate
                    % retransmission requirement
                    ccInfo.HarqProcessesDL(harqIndex) = harqProcess;
                    lastDLGrant = ccInfo.HarqStatusDL{harqIndex};
                    if lastDLGrant.RV == harqProcess.RVSequence(1) % Only store the original transmission grant's TBS
                        ccInfo.TBSizeDL(harqIndex) = lastDLGrant.TBS;
                    end
                    ccInfo.RetransmissionContextDL(harqIndex) = 1;
                end
            end
        end

        function handleULRxResult(obj, rxResultInfo, schedulerConfig)
            %handleULRxResult Update the HARQ process context based on the Rx success/failure for UL packets
            % handleULRxResult(OBJ, RXRESULTINFO, SCHEDULERCONFIG) updates the HARQ
            % process context, based on the reception success/failure of UL packets.
            %
            % RXRESULTINFO is a structure with following fields.
            %   HARQID - HARQ process ID.
            %   RxResult - 0 means Rx failure or no reception. 1 means Rx success.
            %   GNBCARRIERINDEX - Index of the carrier
            %
            %   SCHEDULERCONFIG is a scheduler configuration structure contains one of the following fields.
            %   LinkAdaptationConfigUL - Link adaptation (LA) configuration structure for uplink transmissions.
            %                            The structure contains the fields InitialOffset, StepUp and StepDown fields

            harqIndex = rxResultInfo.HARQID+1;
            rxResult = rxResultInfo.RxResult;

            ccInfo = obj.ComponentCarrier(rxResultInfo.GNBCarrierIndex); % Read the component carrier info
            isNewTx = strcmp(ccInfo.HarqStatusUL{harqIndex}.Type, 'newTx');
            % If UL link adaptation is enabled and it is a new transmission then update
            % MCS offset for the received feedback.
            if isNewTx && ~isempty(schedulerConfig.LinkAdaptationConfigUL)
                linkDir = nr5g.internal.MACConstants.ULType+1; % Uplink
                laConfig = schedulerConfig.LinkAdaptationConfigUL;
                offset = ccInfo.MCSOffset(linkDir);
                ccInfo.MCSOffset(linkDir)=nr5g.internal.getMCSIndexOffset(laConfig, offset, rxResult);
            end

            if rxResult % Rx success
                % Update the HARQ process context
                ccInfo.HarqStatusUL{harqIndex} = []; % Mark HARQ process as free
                ccInfo.IsBusyHARQUL(harqIndex) = 0;
                harqProcess = ccInfo.HarqProcessesUL(harqIndex);
                harqProcess.blkerr(1) = 0;
                ccInfo.HarqProcessesUL(harqIndex) = harqProcess;

                % Clear the retransmission context for the HARQ process of the UE. It would
                % already be empty if this reception was not a retransmission.
                ccInfo.RetransmissionContextUL(harqIndex) = 0;
            else % Rx failure or no packet received
                % No packet received (or corrupted) from UE although it was scheduled to
                % send. Store the transmission uplink grant in retransmission context,
                % which will be used while assigning grant for retransmission
                harqProcess = ccInfo.HarqProcessesUL(harqIndex);
                harqProcess.blkerr(1) = 1;
                if harqProcess.RVIdx(1) == size(harqProcess.RVSequence,2)
                    % Packet reception failed for all redundancy versions. Mark the HARQ
                    % process as free. Also clear the retransmission context to not allow any
                    % further retransmissions for this packet
                    harqProcess.blkerr(1) = 0;
                    ccInfo.HarqStatusUL{harqIndex} = []; % Mark HARQ as free
                    ccInfo.IsBusyHARQUL(harqIndex) = 0;
                    ccInfo.HarqProcessesUL(harqIndex) = harqProcess;
                    ccInfo.RetransmissionContextUL(harqIndex) = 0;
                else
                    ccInfo.HarqProcessesUL(harqIndex) = harqProcess;
                    lastULGrant = ccInfo.HarqStatusUL{harqIndex};
                    if lastULGrant.RV == harqProcess.RVSequence(1) % Only store the original transmission grant TBS
                        ccInfo.TBSizeUL(harqIndex) = lastULGrant.TBS;
                    end
                    ccInfo.RetransmissionContextUL(harqIndex) = 1;
                end
            end
        end

        function clearRetransmissionContext(obj, linkDir, harqID, gNBCarrierIndex)
            %clearRetransmissionContext Clears the retransmission context for HARQ
            %process with ID as harqID to make it ineligible for retransmission assignments
            % clearRetransmissionContext(OBJ, LINKDIR, HARQID, GNBCARRIERINDEX) Clears the
            % retransmission context for HARQ process to make it ineligible for
            % retransmission assignments
            %
            % LINKDIR - Link direction (0 means DL and 1 means UL)
            % HARQID  - HARQ process ID
            % GNBCARRIERINDEX -Index of the carrier
            
            % Clear the retransmission context for the HARQ process (Retransmission
            % context would again get set, if Rx fails again in future for this
            % retransmission assignment)
            ccInfo = obj.ComponentCarrier(gNBCarrierIndex); % Read the component carrier info
            if linkDir  % Uplink
                ccInfo.RetransmissionContextUL(harqID+1) = 0;
            else % Downlink
                ccInfo.RetransmissionContextDL(harqID+1) = 0;
            end
        end

        function updateHARQContextDL(obj, assignment)
            %updateHARQContextDL Update DL HARQ context based on the allotted DL assignment
            % updateHARQContextDL(OBJ, ASSIGNMENT) Update DL HARQ context based on the
            % scheduled assignment
            %
            % ASSIGNMENT - DL assignment scheduled

            harqIndex = assignment.HARQID+1;
            ccInfo = obj.ComponentCarrier(assignment.GNBCarrierIndex); % Read the component carrier info
            % Update downlink HARQ based on the allotted grant
            harqProcess = nr5g.internal.nrUpdateHARQProcess(ccInfo.HarqProcessesDL(harqIndex), 1);
            ccInfo.HarqProcessesDL(harqIndex) = harqProcess;
            % Mark HARQ process as busy
            ccInfo.HarqStatusDL{harqIndex} = assignment;
            ccInfo.IsBusyHARQDL(harqIndex) = 1;
            ccInfo.HarqNDIDL(harqIndex) = assignment.NDI;

            if strcmp(assignment.Type, 'reTx')
                % Clear the retransmission context for this HARQ process of the selected UE
                % to make it ineligible for retransmission assignments
                ccInfo.RetransmissionContextDL(harqIndex) = 0;
            end
        end

        function updateHARQContextUL(obj, grant)
            %updateHARQContextUL Update UL HARQ context based on the allotted grant
            % updateHARQContextUL(OBJ, GRANT) Update UL HARQ context based on the
            % scheduled grant
            %
            % GRANT - UL grant scheduled by the scheduler

            harqIndex = grant.HARQID+1;
            ccInfo = obj.ComponentCarrier(grant.GNBCarrierIndex); % Read the component carrier info
            % Update uplink HARQ based on the allotted grant
            harqProcess = nr5g.internal.nrUpdateHARQProcess(ccInfo.HarqProcessesUL(harqIndex), 1);
            ccInfo.HarqProcessesUL(harqIndex) = harqProcess;
            ccInfo.HarqStatusUL{harqIndex} = grant; % Mark HARQ process as busy
            ccInfo.IsBusyHARQUL(harqIndex) = 1;
            ccInfo.HarqNDIUL(harqIndex) = grant.NDI;

            if strcmp(grant.Type, 'reTx')
                % Clear the retransmission context for this HARQ process of the selected UE
                % to make it ineligible for retransmission assignments
                ccInfo.RetransmissionContextUL(harqIndex) = 0;
            end
        end

        function updateBufferStatusForGrants(obj, linkDir, tbs)
            %updateBufferStatusForGrants Update the buffer status by reducing the UE
            %pending buffer amount based on the scheduled grant tbs
            % updateBufferStatusForGrants(OBJ, LINKDIR, TBS) Update the buffer status
            % by reducing the UE pending buffer amount based on the scheduled grant tbs
            %
            % LINKDIR - Link direction (0 means DL and 1 means UL)
            % TBS     - Scheduled grant TBS

            if linkDir % Uplink
                obj.BufferStatusUL = max(0, obj.BufferStatusUL-tbs);
            else % Downlink
                obj.BufferStatusDL = max(0, obj.BufferStatusDL-tbs);
            end
        end

        function harqID = findFreeUEHarqProcess(obj, linkDir, gNBCarrierIndex)
            %findFreeUEHarqProcess Returns index of a free uplink or downlink HARQ process of UE, based on the link direction (UL/DL)
            % findFreeUEHarqProcess(OBJ, LINKDIR, GNBCARRIERINDEX) Returns index of a
            % free HARQ process of UE, based on the link direction (UL/DL)
            %
            % LINKDIR - Link direction (0 means DL and 1 means UL)
            % GNBCARRIERINDEX -Index of the carrier
          
            ccInfo = obj.ComponentCarrier(gNBCarrierIndex); % Read the component carrier info
            if linkDir % Uplink
                harqProcessInfo = ccInfo.IsBusyHARQUL;
            else % Downlink
                harqProcessInfo = ccInfo.IsBusyHARQDL;
            end

            freeHARQIndex = find(harqProcessInfo==0,1);
            if isempty(freeHARQIndex)
                harqID = -1;
            else
                harqID = freeHARQIndex-1;
            end
        end

        function updateSRSPeriod(obj, srsPeriod, gNBCarrierIndex)
            %updateSRSPeriod Update the SRS periodicity of the UE for the specified carrier
            % updateSRSPeriod(OBJ, SRSPERIOD, GNBCARRIERINDEX) updates the
            % SRS period for the UE over the carrier specified by
            % GNBCARRIERINDEX.
            %
            % SRSPERIOD - A vector of [srsPeriodicity slotOffset]
            % GNBCARRIERINDEX - Index of the carrier

            ccInfo = obj.ComponentCarrier(gNBCarrierIndex);
            ccInfo.updateSRSPeriod(srsPeriod);
            % Update copy of SRSConfiguration w.r.t primary carrier context for backward compatibility
            if obj.ConfiguredCarrier(1) == gNBCarrierIndex
                obj.SRSConfiguration.SRSPeriod = srsPeriod;
            end
        end

        function updateUEsServedDataRate(obj, linkDir, schedulerConfig, instantaneousDataRate)
            % updateUEsServedDataRate Update UE served data rate based on Rx
            % success/failure for DL/UL packets
            % updateUEsServedDataRate(OBJ, LINKDIR, SCHEDULERCONFIG,
            % INSTANTANEOUSDATARATE) Update UE served data rate based on Rx
            % success/failure for DL/UL packets
            %
            % LINKDIR              - Link direction (0 means DL and 1 means UL)
            % SCHEDULERCONFIG      - Scheduler configuration structure contains one of the following fields.
            %   PFSWindowSize      - Specifies the time constant (in number of subframes)
            %   of exponential moving average equation for average data rate
            %   calculation in PF scheduler.
            % INSTANTANEOUSDATARATE - Instantaneous data rate for the UE

            % Calculate the served data rate
            index = linkDir+1;
            pfsWindowSize = schedulerConfig.PFSWindowSize;

            if instantaneousDataRate % Update served data rate for new transmission
                obj.UEsServedDataRate(index) = obj.UEsServedDataRate(index)+(1/pfsWindowSize)*instantaneousDataRate;
            else
                obj.UEsServedDataRate(index) = (1-1/pfsWindowSize)*obj.UEsServedDataRate(index);
            end
        end
    end
end
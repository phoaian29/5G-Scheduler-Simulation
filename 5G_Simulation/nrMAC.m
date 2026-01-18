classdef (Abstract) nrMAC < handle
    %nrMAC Define an NR MAC base class
    %
    %   Note: This is an internal undocumented class and its API and/or
    %   functionality may change in subsequent releases.

    %   Copyright 2022-2024 The MathWorks, Inc.

    properties(SetAccess = protected)
        %MACType Type of node to which MAC Entity belongs
        % Value 1 means UE MAC and value 0 means gNB MAC
        MACType = 1;

        %NCellID Physical cell ID. Values: 0 to 1007 (TS 38.211, sec 7.4.2.1). The default value is []
        NCellID

        %NumHARQ Number of HARQ processes. The default value is []
        NumHARQ

        %SubcarrierSpacing Subcarrier spacing used
        SubcarrierSpacing

        %StatTransmittedPackets Number of MAC packets sent to PHY
        StatTransmittedPackets

        %StatTransmittedBytes Number of MAC bytes sent to PHY
        StatTransmittedBytes

        %StatRetransmittedPackets Number of retransmission MAC packets sent to PHY
        StatRetransmittedPackets

        %StatRetransmittedBytes Number of retransmission MAC bytes sent to PHY
        StatRetransmittedBytes

        %StatReceivePackets Number of successfully received packets received at MAC
        StatReceivedPackets

        %StatReceivedBytes Number of successfully received bytes received at MAC
        StatReceivedBytes

        %LCHBufferStatus An array of logical channel buffer status information
        % This property size depends on the type of device inheriting it. In case
        % of a UE, this property size becomes 1-by-MaxLogicalChannels. In case of a
        % gNB, its size is NumUEs-by-MaxLogicalChannels. Each row in the matrix
        % corresponds to a different UE and each column corresponds to a different
        % logical channel
        LCHBufferStatus
    end

    properties(SetAccess=protected, Hidden)
        %NextRxTime Next scheduled reception time for MAC (in nanoseconds)
        NextRxTime = Inf

        %LastRxTime Timestamp when MAC last instructed PHY to receive a packet (in nanoseconds)
        LastRxTime = -1;

        %CurrFrame Current running frame number
        CurrFrame = 0;

        %CurrSlot Current running slot number in the current frame
        CurrSlot = 0;

        %CurrSymbol Current running symbol in the current slot
        CurrSymbol = 0;

        %CurrDLULSlotIndex Slot index of the current running slot in the DL-UL pattern (for TDD mode)
        CurrDLULSlotIndex = 0;
    end

    properties (Access = protected)
        %NotificationFcn A function handle to inform node about the events
        NotificationFcn = []

        %RLCTxFcn Cell array of function handles to interact with RLC
        %entities to pull the RLC PDUs for transmission. For UE, it is of
        %size 1-by-MaxLogicalChannels. For gNB, it is of size
        %NumUEs-by-MaxLogicalChannels.
        RLCTxFcn

        %RLCRxFcn Cell array of function handles to interact with RLC
        %entities to push the received PDUs up the stack. For UE, it is of
        %size 1-by-MaxLogicalChannels. For gNB, it is of size
        %NumUEs-by-MaxLogicalChannels.
        RLCRxFcn

        %LogicalChannelConfig Cell array of logical channel configuration structure
        % For UE, it is of size 1-by-MaxLogicalChannels. For gNB, it is of
        % size NumUEs-by-MaxLogicalChannels. Each row in the matrix
        % corresponds to a different UE and each column corresponds to a
        % different logical channel. Each structure contains these fields:
        %
        % RNTI      - Radio network temporary identifier
        % LCID      - Logical channel identifier
        % LCGID     - Logical channel group identifier
        % Priority  - Priority of the logical channel
        % PBR       - Prioritized bit rate (in kilo bytes per second)
        % BSD       - Bucket size duration (in ms)
        LogicalChannelConfig

        %LCHBjList An array of Bj values for different logical channels
        % This property size depends on the type of device inheriting it.
        % In case of a UE device, this property size is
        % 1-by-MaxLogicalChannels. In case of a gNB, its size is
        % NumUEs-by-MaxLogicalChannels. Each row in the matrix
        % corresponds to a different UE and each column corresponds to a
        % different logical channel
        LCHBjList

        %LCGPriority An array to store the logical channel group priority for all UEs
        LCGPriority

        %ElapsedTimeSinceLastLCP An array of elapsed times (in milliseconds) since the last LCP run for the UE
        % This property size depends on the type of device inheriting it.
        % In case of a UE device, this property size is 1-by-1. In case of
        % a gNB, its size is numUEs-by-1. Each row in the matrix
        % corresponds to a different UE
        ElapsedTimeSinceLastLCP

        %NumSlotsFrame Number of slots in a 10 ms frame. Depends on the SCS used
        NumSlotsFrame

        %TxDataRequestFcn Function handle to send data to PHY
        TxDataRequestFcn

        %RxDataRequestFcn Function handle to receive data from PHY
        RxDataRequestFcn

        %DlControlRequestFcn Function handle to send DL control request to PHY
        DlControlRequestFcn

        %UlControlRequestFcn Function handle to send DL control request to PHY
        UlControlRequestFcn

        %TxOutofBandFcn Function handle to transmit out-of-band packets to receiver's MAC
        TxOutofBandFcn

        %LastRunTime Time (in nanoseconds) at which the MAC layer was invoked last time
        LastRunTime = 0;

        %PreviousSymbol Previous symbol in the current frame. This helps to
        %avoid running scheduler and performing CSI-RS/SRS related
        %operations multiple times in a symbol
        PreviousSymbol = -1;

        %SymbolEndTimesInSlot Symbol end times (in nanoseconds) in a slot
        SymbolEndTimesInSlot

        %SymbolDurationsInSlot Symbol end durations (in nanoseconds) in a slot
        SymbolDurationsInSlot

        %SlotDurationInNS Slot duration in nanoseconds
        SlotDurationInNS

        %SlotDurationInSec Slot duration in seconds
        SlotDurationInSec

        %NumSymInFrame  Number of symbols in a 10 ms frame
        NumSymInFrame

        %CSIRSConfigurationRSRP CSI-RS resource set configurations corresponding to the SSB directions
        % Array of length N-by-1 where 'N' is the maximum number of SSBs in a SSB
        % burst. Each element of the array at index 'i' corresponds to the CSI-RS
        % resource set associated with SSB 'i'. The number of CSI-RS resources in
        % each resource set is same for all configurations.
        CSIRSConfigurationRSRP

        %SRSConfiguration Sounding reference signal (SRS) resource configuration for the UEs
        % Array containing the SRS configuration information. Each element is an
        % object of type nrSRSConfig. For a UE, it is an array of size 1. For a
        % gNB, it is an array of size equal to number of UEs connected to the gNB.
        % An element at index 'i' stores the SRS configuration of UE with RNTI 'i'.
        SRSConfiguration

        %PacketStruct Empty packet structure
        PacketStruct = wirelessnetwork.internal.wirelessPacket

        %PDSCHInfoStruct PDSCH information structure
        PDSCHInfo = struct('NSlot',[],'HARQID',[],'NewData',[],'RV',[],'TargetCodeRate',[],'TBS',[],'PrecodingMatrix',[],'BeamIndex',[],'PDSCHConfig',nrPDSCHConfig);

        %PUSCHInfoStruct PUSCH information structure
        PUSCHInfo = struct('NSlot',[],'HARQID',[],'NewData',[],'RV',[],'TargetCodeRate',[],'TBS',[],'PUSCHConfig',nrPUSCHConfig('TransmissionScheme','codebook'));

        %UplinkGrantStruct Uplink grant structure (Transport block size(TBS) is added to grant for runtime optimization))
        UplinkGrantStruct = struct('SlotOffset',[],'ResourceAllocationType',[],'FrequencyAllocation',[],'StartSymbol',[],'NumSymbols',[],'MCSIndex',[],'NDI',[],'RV',[],...
            'HARQID',[],'DMRSLength',[],'MappingType',[],'NumLayers',1,'NumCDMGroupsWithoutData',[],'TPMI',[],'NumAntennaPorts',1, 'TBS', [], 'PRBSet', []);

        %DownlinkGrantStruct Downlink grant structure (TBS is added to grant for runtime optimization)
        DownlinkGrantStruct = struct('SlotOffset',[],'ResourceAllocationType',[],'FrequencyAllocation',[],'StartSymbol',[],'NumSymbols',[],'MCSIndex',[],'NDI',[],'RV',[],...
            'HARQID',[],'FeedbackSlotOffset',[],'DMRSLength',[],'MappingType',[],'NumLayers',1,'NumCDMGroupsWithoutData',[], 'TBS', [], 'PRBSet', []);

        %TimestampInfo Structure that holds the timestamp information in terms of
        % nanoseconds and equivalent representation in terms of frame number, slot
        % number and symbol number
        TimestampInfo = struct('Timestamp', 0, 'NFrame', 0, 'NSlot', 0, 'NSymbol', 0);

        %CarrierConfigUL nrCarrierConfig object for UL
        CarrierConfigUL

        %CarrierConfigDL nrCarrierConfig object for DL
        CarrierConfigDL

        %HigherLayerPacketFormat Defines the packet format of the application layer in Rx chain
        HigherLayerPacketFormat = struct(NodeID=0, Packet=[], PacketLength=0, Tags=[])

        %PacketInfo MAC packet format
        PacketInfo = struct(Packet=[], PacketLength=0, Tags=[])
    end

    properties (Constant)
        %NumSymbols Number of symbols in a slot
        NumSymbols = 14;

        %MaxLogicalChannels Maximum number of logical channels
        MaxLogicalChannels = 32;

        %MinPriorityForLCH Minimum logical channel priority value
        MinPriorityForLCH = 1;

        %MaxPriorityForLCH Maximum logical channel priority value
        MaxPriorityForLCH = 16;

        %PBR Set of valid PBR (in kBps) values for a logical channel. For
        % more details, refer 3GPP TS 38.331 information element
        %LogicalChannel-Config
        PBR = [0, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, ...
            65536, Inf];

        %BSD Set of valid BSD (in ms) values for a logical channel. For
        % more details, refer 3GPP TS 38.331 information element
        % LogicalChannel-Config
        BSD = [5, 10, 20, 50, 100, 150, 300, 500, 1000];

        %NominalRBGSizePerBW Nominal RBG size for the specified bandwidth
        % in accordance with 3GPP TS 38.214, Section 5.1.2.2.1
        NominalRBGSizePerBW = nr5g.internal.MACConstants.NominalRBGSizePerBW;

        %BSR Packet type for BSR
        BSR = 1;

        %ULGrant Packet type for uplink grant
        ULGrant = 2;

        %DLGrant Packet type for downlink grant
        DLGrant = 3;

        %PDSCHFeedback Packet type for PDSCH ACK/NACK
        PDSCHFeedback = 4;

        %CSIReport Packet type for channel state information (CSI) report of the format ri-pmi-cqi
        CSIReport = 5;

        %CSIReportRSRP Packet type for CSI report of the format cri-rsrp
        CSIReportRSRP = 6;

        %DLType Value to specify downlink direction or downlink symbol type
        DLType = nr5g.internal.MACConstants.DLType;

        %ULType Value to specify uplink direction or uplink symbol type
        ULType = nr5g.internal.MACConstants.ULType;

        %GuardType Value to specify guard symbol type
        GuardType = nr5g.internal.MACConstants.GuardType;

        %FrameDurationInNS Frame duration in nanoseconds
        FrameDurationInNS = 10e6;
    end

    methods (Access = public)
        function registerRLCInterfaceFcn(obj, rnti, logicalChannelID, rlcTxFcn, rlcRxFcn)
            %registerRLCInterfaceFcn Register the RLC entity callbacks
            %
            %   registerRLCInterfaceFcn(OBJ, RNTI, LOGICALCHANNELID,
            %   RLCTXFCN, RLCRXFCN) registers the callback function to
            %   interact with the RLC entity.
            %
            %   RNTI is radio network temporary identifier of UE.
            %
            %   LOGICALCHANNELID is logical channel identifier.
            %
            %   RLCTXFCN is a function handle to interact with RLC Tx
            %   entities.
            %
            %   RLCRXFCN is a function handle to interact with RLC Rx
            %   entities.

            if ~obj.MACType
                % In case of gNB, RNTI of UE is the cell array index to get
                % its logical channel configuration set
                lchSetIdx = rnti;
            else
                % In case of UE, 1 is the cell array index to get its
                % logical channel configuration set
                lchSetIdx = 1;
            end

            obj.RLCTxFcn{lchSetIdx, logicalChannelID} = rlcTxFcn;
            obj.RLCRxFcn{lchSetIdx, logicalChannelID} = rlcRxFcn;
        end

        function registerPhyInterfaceFcn(obj, txFcn, rxFcn, dlControlReqFcn, ulControlReqFcn)
            %registerPhyInterfaceFcn Register PHY interface functions for Tx and Rx
            %   registerPhyInterfaceFcn(OBJ, TXFCN, RXFCN, DLCONTROLREQFCN, ULCONTROLREQFCN) registers PHY
            %   interface functions at MAC for (i) Sending packets to PHY
            %   (ii) Sending Rx request to PHY at the Rx start time
            %   (iii) Sending DL control request to PHY
            %   (iv) Sending UL control request to PHY
            %
            %   TXFCN Function handle to send data to PHY.
            %
            %   RXFCN Function handle to indicate Rx start to PHY.
            %
            %   DLCONTROLREQFCN Function handle to send DL control request to PHY.
            %
            %   ULCONTROLREQFCN Function handle to send UL control request to PHY.

            obj.TxDataRequestFcn = txFcn;
            obj.RxDataRequestFcn = rxFcn;
            obj.DlControlRequestFcn = dlControlReqFcn;
            obj.UlControlRequestFcn = ulControlReqFcn;
        end

        function registerOutofBandTxFcn(obj, sendOutofBandPktsFcn)
            %registerOutofBandTxFcn Set the function handle for transmitting out-of-band packets from sender's MAC to receiver's MAC
            %
            % SENDOUTOFBANDPKTSFCN is the function handle provided by
            % hNRNode object, to be used by the MAC for transmitting
            % out-of-band packets

            obj.TxOutofBandFcn = sendOutofBandPktsFcn;
        end

        function addLogicalChannelInfo(obj, logicalChannelConfig, rnti)
            %addLogicalChannelInfo Add the logical channel information
            %
            % addLogicalChannelInfo(OBJ, LOGICALCHANNELCONFIG) adds the
            % logical channel information to the list of active logical
            % channels in the UE.
            %
            % addLogicalChannelInfo(OBJ, LOGICALCHANNELCONFIG, RNTI) adds
            % the logical channel information to the list of active logical
            % channels for the UE in gNB.
            %
            % LOGICALCHANNELCONFIG is a logical channel id, specified in
            % the range between 1 and 32, inclusive.
            %
            % RNTI is a radio network temporary identifier, specified in
            % the range between 1 and 65522, inclusive. Refer table 7.1-1
            % in 3GPP TS 38.321 version 18.1.0.

            if ~obj.MACType
                % In case of gNB, RNTI of UE is the cell array index to get
                % its logical channel configuration set
                lchSetIdx = rnti;
            else
                % In case of UE, 1 is the cell array index to get its
                % logical channel configuration set
                lchSetIdx = 1;
            end

            % Store the logical channel information
            obj.LogicalChannelConfig{lchSetIdx, logicalChannelConfig.LogicalChannelID} = logicalChannelConfig;
            % Set the logical channel group priority for a UE in the
            % gNB
            if ~obj.MACType
                if obj.LCGPriority(lchSetIdx,logicalChannelConfig.LogicalChannelGroup + 1) > logicalChannelConfig.Priority
                    obj.LCGPriority(lchSetIdx,logicalChannelConfig.LogicalChannelGroup + 1) = logicalChannelConfig.Priority;
                end
                addBearerConfig(obj.Scheduler, lchSetIdx, logicalChannelConfig);
            end
        end

        function packetInfo = constructMACPDU(obj, tbs, rnti)
            %CONSTRUCTMACPDU Construct and return a MAC PDU based on transport block size
            %
            %   PACKETINFO = CONSTRUCTMACPDU(OBJ, TBS) returns a UL MAC PDU.
            %
            %   PACKETINFO = CONSTRUCTMACPDU(OBJ, TBS, RNTI) returns a DL MAC PDU for
            %   UE identified with specified RNTI.
            %
            %   PACKETINFO is a structure with these fields.
            %     PACKET       - Array of octets in decimal format.
            %     PACKETLENGTH - Length of packet.
            %     TAGS         - Array of structures where each structure contains
            %                    these fields.
            %                    Name      - Name of the tag.
            %                    Value     - Data associated with the tag.
            %                    ByteRange - Specific range of bytes within the
            %                                packet to which the tag applies.
            %
            %   TBS Transport block size in bytes.
            %
            %   RNTI RNTI of the UE for which DL MAC PDU needs to be
            %   constructed.

            paddingSubPDU = [];

            % Run LCP and construct MAC PDU
            if nargin < 3 || isempty(rnti)
                % Construct UL MAC PDU
                [dataSubPDUList, remainingBytes] = performLCP(obj, tbs);
            else
                % Construct DL MAC PDU
                [dataSubPDUList, remainingBytes] = performLCP(obj, tbs, rnti);
            end
            if remainingBytes > 0
                paddingSubPDU = nrMACSubPDU(remainingBytes);
            end

            packetInfo = obj.PacketInfo;
            % Construct MAC PDU by concatenating subPDUs. Downlink MAC PDU constructed
            % as per 3GPP TS 38.321 Figure 6.1.2-4
            if ~isempty(dataSubPDUList)
                % Initialize packet information with the first subPDU's data
                packetInfo = dataSubPDUList(1);
                packetLength = packetInfo.PacketLength;
                % Loop through the remaining sub-PDUs in the list to aggregate
                % them
                for idx = 2:size(dataSubPDUList,2) % Start from the second subPDU
                    subPDU = dataSubPDUList(idx);
                    % Concatenate the current subPDU's packet data to the
                    % aggregated packet
                    packetInfo.Packet = [packetInfo.Packet; subPDU.Packet];
                    % Aggregate the current subPDU's tags with the previously
                    % aggregated tags, adjusting based on the updated packet
                    % length
                    packetInfo.Tags = ...
                        wirelessnetwork.internal.packetTags.aggregate(packetInfo.Tags, ...
                        packetLength, subPDU.Tags, subPDU.PacketLength);
                    % Increment the total length of the aggregated packet
                    packetLength = packetLength + subPDU.PacketLength;
                end
            end
            % Concatenate the padding subPDU's packet data to the aggregated
            % packet
            packetInfo.Packet = [packetInfo.Packet; paddingSubPDU];
            % Set the total length of the aggregated packet
            packetInfo.PacketLength = tbs;
        end
    end

    methods(Access = protected)
        function duration = durationToSymNum(obj, symNum)
            % Calculate time duration to a symbol number (w.r.t start of frame) from current
            % symbol number (w.r.t start of slot). The returned duration is in nanoseconds

            numSlots = floor(symNum/obj.NumSymbols);
            numSymbols = mod(symNum, obj.NumSymbols);
            if obj.CurrSymbol + numSymbols > obj.NumSymbols
                totalSymbolDuration = sum(obj.SymbolDurationsInSlot(obj.CurrSymbol+1:obj.NumSymbols)) + sum(obj.SymbolDurationsInSlot(1:obj.CurrSymbol+numSymbols-obj.NumSymbols));
            else
                totalSymbolDuration = sum(obj.SymbolDurationsInSlot(obj.CurrSymbol+1:obj.CurrSymbol+numSymbols));
            end
            duration = (numSlots*obj.SlotDurationInNS) + totalSymbolDuration;
        end

        function [dataSubPDUList, remainingBytes] = performLCP(obj, tbs, rnti)
            %performLCP Perform the logical channel prioritization (LCP) procedure
            %
            %   [DATASUBPDULIST, REMAININGBYTES] = performLCP(OBJ, TBS)
            %   performs the logical channel prioritization procedure in UE
            %   MAC.
            %
            %   [DATASUBPDULIST, REMAININGBYTES] = performLCP(OBJ, TBS,
            %   RNTI) performs the logical channel prioritization procedure
            %   in gNB MAC for UE identified with RNTI.
            %
            %   DATASUBPDULIST is a array of MAC subPDUs where each
            %   MAC subPDU is a structure with these fields.
            %       Packet       - Array of octets in decimal format.
            %       PacketLength - Length of packet.
            %       Tags         - Array of structures where each structure
            %                      contains these fields.
            %                      Name      - Name of the tag.
            %                      Value     - Data associated with the tag.
            %                      ByteRange - Specific range of bytes within
            %                                  the packet to which the tag
            %                                  applies.
            %
            %   REMAININGBYTES is an integer scalar, which represents the
            %   number of bytes left unused in the TBS.
            %
            %   TBS is an integer scalar, which represents the size of the
            %   MAC PDU to be constructed as per the received grant.
            %
            %   RNTI is a radio network temporary identifier. Specify the
            %   RNTI as an integer scalar between 1 and 65522, inclusive.
            %   Refer table 7.1-1 in 3GPP TS 38.321 version 18.1.0.

            % Based on the MAC type, select the logical channel
            % configuration set for the UE
            ueIdx = 1;
            if ~obj.MACType
                % In case of gNB MAC, RNTI acts as an index to get the
                % logical channel information, associated to a UE in the
                % downlink direction, from a cell array
                ueIdx = rnti;
            end

            % Identify the number of logical channels having data to
            % transmit. If no logical channel has data to transmit, abort
            % the LCP procedure
            activeLCH = nnz(obj.LCHBufferStatus(ueIdx, :));
            if activeLCH == 0
                dataSubPDUList = [];
                remainingBytes = tbs;
                return;
            end

            lcpPriorityList = cell(obj.MaxPriorityForLCH, 1);
            % Iterate through the configured logical channels
            for lchIdx = 1:numel(obj.LogicalChannelConfig(ueIdx, :))
                if isempty(obj.LogicalChannelConfig{ueIdx, lchIdx})
                    continue;
                end
                % Check if prioritized bit rate is not set to infinity
                if obj.LogicalChannelConfig{ueIdx, lchIdx}.PrioritizedBitRate ~= Inf
                    % Calculate the time elapsed since the last LCP run for
                    % the UE
                    timeElapsed = min(obj.ElapsedTimeSinceLastLCP(ueIdx, 1), obj.LogicalChannelConfig{ueIdx, lchIdx}.BucketSizeDuration);
                    % Increment the Bj by the product of time elapsed and
                    % PBR
                    obj.LCHBjList(ueIdx, lchIdx) = obj.LCHBjList(ueIdx, lchIdx) + ...
                        ceil(obj.LogicalChannelConfig{ueIdx, lchIdx}.PrioritizedBitRate * timeElapsed);
                else
                    % When the prioritized bit rate is set to infinity,
                    % update the minimum grant Bj required by the logical
                    % channel to the minimum of logical channel buffer
                    % status and remaining grant
                    obj.LCHBjList(ueIdx, lchIdx) = min(obj.LCHBufferStatus(ueIdx, lchIdx), tbs);
                end
                % Store the logical channel id in a cell array where the
                % storing index of the cell array refer its priority
                lchPriority = obj.LogicalChannelConfig{ueIdx, lchIdx}.Priority;
                lcpPriorityList{lchPriority, 1}{end+1, 1} = lchIdx;
            end

            if activeLCH == 1
                % If only one logical channel has data to transmit, then
                % skip the LCP round-1
                [dataSubPDUList, remainingBytes] = performLCPRound2(obj, tbs, ueIdx, lcpPriorityList);
            else
                % As per Section 5.4.3.1.3 of the 3GPP TS 38.321, perform
                % the LCP procedure and get the MAC SDUs from the RLC
                % entity
                [subPDUlist, remainingBytes] = performLCPRound1(obj, tbs, ueIdx, lcpPriorityList);
                [dataSubPDUList, remainingBytes] = performLCPRound2(obj, remainingBytes, ueIdx, lcpPriorityList);
                dataSubPDUList = [subPDUlist dataSubPDUList];
            end

            % Reset the elapsed time since last LCP run
            obj.ElapsedTimeSinceLastLCP(ueIdx, 1) = 0;
        end

        function csirsInfo = calculateCSIRSPeriodicity(obj, csirsResourceConfig)
            %calculateCSIRSPeriodicity Returns CSI-RS transmission/reception time information

            count = 0;
            csirsInfo = inf(1, 2);
            for idx=1:size(csirsResourceConfig,2)
                if ischar(csirsResourceConfig(idx).CSIRSPeriod)
                    if strcmp(csirsResourceConfig(idx).CSIRSPeriod, 'on')
                        % CSI-RS resource is present in all the slots
                        count = count + 1;
                        csirsInfo(count, 1) = obj.SlotDurationInNS;
                        csirsInfo(count, 2) = 0;
                    end
                elseif ~iscell(csirsResourceConfig(idx).CSIRSPeriod)
                    % CSI-RS resource is present in specific slots
                    % represented as periodicity and slot offset pairs
                    count = count + 1;
                    csirsInfo(count, 1) = max(csirsResourceConfig(idx).CSIRSPeriod) * obj.SlotDurationInNS;
                    csirsInfo(count, 2) = min(csirsResourceConfig(idx).CSIRSPeriod) * obj.SlotDurationInNS;
                else
                    % CSI-RS resources are present in a list of slots
                    % Cell array with different CSIRS periods
                    if iscell(csirsResourceConfig(idx).CSIRSType)
                        maxResourceConfigs =  numel([csirsResourceConfig(idx).CSIRSType]);
                    else
                        maxResourceConfigs = 1;
                    end
                    numCSIRSConfig = min(maxResourceConfigs, numel(csirsResourceConfig(idx).CSIRSPeriod));
                    for csirsIdx=1:numCSIRSConfig
                        if ischar(csirsResourceConfig(idx).CSIRSPeriod{csirsIdx})
                            if strcmp(csirsResourceConfig(idx).CSIRSPeriod{csirsIdx}, 'on')
                                % CSI-RS resource is present in all the slots
                                count = count + 1;
                                csirsInfo(count, 1) = obj.SlotDurationInNS;
                                csirsInfo(count, 2) = 0;
                            end
                        else
                            % CSI-RS resource is present in specific slots
                            % represented as periodicity and slot offset pairs
                            count = count + 1;
                            csirsInfo(count, 1) = max(csirsResourceConfig(idx).CSIRSPeriod{csirsIdx}) * obj.SlotDurationInNS;
                            csirsInfo(count, 2) = min(csirsResourceConfig(idx).CSIRSPeriod{csirsIdx}) * obj.SlotDurationInNS;
                        end
                    end
                end
            end
            csirsInfo =  unique(csirsInfo(1:count, :), 'rows');
        end

        function srsInfo = calculateSRSPeriodicity(obj, srsConfig)
            %calculateSRSPeriodicity Returns SRS transmission/reception time information

            srsInfo = inf(1, 2);
            count = 0;
            for idx=1:size(srsConfig,2)
                if isa(srsConfig(idx), 'nrSRSConfig')
                    if ischar(srsConfig(idx).SRSPeriod)
                        if strcmp(srsConfig(idx).SRSPeriod, 'on')
                            % SRS resource is present in all the slots
                            count = count + 1;
                            srsInfo(count, 1) = obj.SlotDurationInNS;
                            srsInfo(count, 2) = 0;
                        end
                    else
                        % SRS resource is present in specific slots
                        % represented as periodicity and slot offset pairs
                        count = count + 1;
                        srsInfo(count, 1) = max(srsConfig(idx).SRSPeriod) * obj.SlotDurationInNS;
                        srsInfo(count, 2) = min(srsConfig(idx).SRSPeriod) * obj.SlotDurationInNS;
                    end
                end
            end
            srsInfo = unique(srsInfo(1:end, :), 'rows');
        end

        function flag = checkCSIRSOccurrence(~, carrier, csirs)
            % Check if CSI-RS is supposed to be sent in current slot

            flag = false;
            % Get CSI-RS locations within a slot
            [csirslocations,csirsParams] = nr5g.internal.getCSIRSLocations(carrier,csirs);

            % Extract the following properties of carrier and cast the necessary ones to double
            absNSlot = double(carrier.NSlot);  % Absolute slot number
            nFrame = double(carrier.NFrame); % Absolute frame number
            nFrameSlot = carrier.SlotsPerFrame;  % Number of slots per frame

            % Get the relative slot number and relative frame number
            [nslot,sfn] = nr5g.internal.getRelativeNSlotAndSFN(absNSlot,nFrame,nFrameSlot);

            % Generate ZP-CSI-RS and NZP-CSI-RS indices
            numCSIRSRes = numel(csirslocations);
            for resIdx = 1:numCSIRSRes

                % Extract the slot periodicity and offset
                if isnumeric(csirsParams.CSIRSPeriod{resIdx})
                    tCSIRS = double(csirsParams.CSIRSPeriod{resIdx}(1));
                    tOffset = double(csirsParams.CSIRSPeriod{resIdx}(2));
                else
                    if strcmpi(csirsParams.CSIRSPeriod{resIdx},'on')
                        tCSIRS = 1;
                    else
                        tCSIRS = 0;
                    end
                    tOffset = 0;
                end
                % Schedule CSI-RS based on slot periodicity and offset
                if (tCSIRS ~= 0) && (mod(nFrameSlot*sfn + nslot - tOffset, tCSIRS) == 0)
                    flag = true;
                end
            end
        end
        function [dataSubPDUList, remainingBytes] = performLCPRound1(obj, remainingBytes, ueIdx, lcpPriorityList)
            %performLCPRound1 Perform 1st iteration of allocation among
            %logical channels using MAC LCP

            dataSubPDUList = [];
            % Iterate through the logical channel priority cell array,
            % from highest to lowest priority
            for priorityIdx = obj.MinPriorityForLCH:obj.MaxPriorityForLCH
                % Iterate through the logical channels which are having
                % the selected priority value
                for j = 1:numel(lcpPriorityList{priorityIdx, 1})
                    % As per Section 5.4.3.1.3 of the 3GPP TS 38.321,
                    % minimum required grant for a logical channel is 8
                    % bytes
                    if remainingBytes < 8
                        % Reset the elapsed time since last LCP run
                        obj.ElapsedTimeSinceLastLCP(ueIdx, 1) = 0;
                        return;
                    end
                    % Get the stored logical channel id
                    lchIdx = lcpPriorityList{priorityIdx, 1}{j, 1};
                    % Check if the buffer status of the logical channel
                    % is not zero
                    if obj.LCHBufferStatus(ueIdx, lchIdx) == 0
                        continue;
                    end
                    % Don't consider the logical channels whose Bj <= 0
                    if obj.LCHBjList(ueIdx, lchIdx) <= 0
                        continue;
                    end
                    % Set the grant to minimum of the remaining
                    % bytes and Bj in LCP round-1
                    grantSize = min(obj.LCHBjList(ueIdx, lchIdx), remainingBytes);

                    % Get MAC subPDUs for the SDUs received from higher layers
                    [macSubPDUs, utilizedGrant] = getMACSubPDUs(obj, ueIdx, lchIdx, grantSize, remainingBytes - grantSize);
                    dataSubPDUList = [dataSubPDUList macSubPDUs];

                    if obj.LogicalChannelConfig{ueIdx, lchIdx}.PrioritizedBitRate ~= Inf
                        % Decrement the Bj by the total number of bytes
                        % used by the logical channel
                        obj.LCHBjList(ueIdx, lchIdx) = obj.LCHBjList(ueIdx, lchIdx) - utilizedGrant;
                    end
                    % Update number of bytes left in the given grant
                    remainingBytes = remainingBytes - utilizedGrant;
                end
            end
        end

        function [dataSubPDUList, remainingBytes] = performLCPRound2(obj, remainingBytes, ueIdx, lcpPriorityList)
            %performLCPRound2 Perform second iteration of allocation among
            %logical channels using MAC LCP

            dataSubPDUList = [];
            % Iterate through the logical channel priority cell array,
            % from highest to lowest priority
            for priorityIdx = obj.MinPriorityForLCH:obj.MaxPriorityForLCH
                if numel(lcpPriorityList{priorityIdx, 1})
                    % Share the grant equally among all logical
                    % channels with equal priority
                    avgGrant = getEqualShareAmongLCH(obj, lcpPriorityList{priorityIdx, 1}, remainingBytes, ueIdx);
                end
                % Iterate through the logical channels which are having
                % the selected priority value
                for j = 1:numel(lcpPriorityList{priorityIdx, 1})
                    % As per Section 5.4.3.1.3 of the 3GPP TS 38.321,
                    % minimum required grant for a logical channel is 8
                    % bytes
                    if remainingBytes < 8
                        % Reset the elapsed time since last LCP run
                        obj.ElapsedTimeSinceLastLCP(ueIdx, 1) = 0;
                        return;
                    end
                    % Get the stored logical channel id
                    lchIdx = lcpPriorityList{priorityIdx, 1}{j, 1};
                    % Check if the buffer status of the logical channel
                    % is not zero
                    if obj.LCHBufferStatus(ueIdx, lchIdx) == 0
                        continue;
                    end
                    % Set the grant to minimum of the remaining
                    % bytes and Bj in LCP round-2
                    grantSize = min(avgGrant(j, 1), remainingBytes);

                    % Get MAC subPDUs for the SDUs received from higher layers
                    [macSubPDUs, utilizedGrant] = getMACSubPDUs(obj, ueIdx, lchIdx, grantSize, remainingBytes - grantSize);
                    dataSubPDUList = [dataSubPDUList macSubPDUs];

                    % Update number of bytes left in the given grant
                    remainingBytes = remainingBytes - utilizedGrant;
                end
            end
        end

        function [dataSubPDUList, utilizedGrant] = getMACSubPDUs(obj, ueIdx, lchIdx, grantSize, remainingTBS)
            %getMACSubPDUs Return MAC subPDUs, includes MAC header and SDU

            if obj.MACType
                % UE MAC
                linkDir = obj.ULType; % Uplink
            else
                % gNB MAC
                linkDir = obj.DLType; % Downlink
            end

            % Get SDUs from higher layers
            macSDUs = obj.RLCTxFcn{ueIdx, lchIdx}(grantSize, remainingTBS, obj.LastRunTime);

            utilizedGrant = 0;
            dataSubPDUList = macSDUs;
            % Calculate the number of bytes used by the logical channel
            for idx = 1:numel(macSDUs)
                dataSubPDUList(idx).Packet = nrMACSubPDU(linkDir, lchIdx, macSDUs(idx).Packet);
                packetLength = size(dataSubPDUList(idx).Packet,1);
                % Get the MAC header length and adjust the higher layer
                % tags
                subHeaderLength = packetLength - macSDUs(idx).PacketLength;
                dataSubPDUList(idx).Tags = wirelessnetwork.internal.packetTags.adjust(...
                    macSDUs(idx).Tags, subHeaderLength);
                dataSubPDUList(idx).PacketLength = packetLength;
                utilizedGrant = utilizedGrant + packetLength;
            end
        end

        function avgGrantSize = getEqualShareAmongLCH(obj, lchList, remainingBytes, ueIdx)
            %getEqualShareAmongLCH Assign the remaining grant among the logical channels with same priority in second round of LCP

            % Number of logical channels with the same priority
            numLCH = numel(lchList);
            % Get the buffer status of those logical channels
            bufStatusList = zeros(numLCH, 1);
            activeLCHCount = 0;
            for lchIdx = 1:numLCH
                bufStatusList(lchIdx) = obj.LCHBufferStatus(ueIdx, lchList{lchIdx});
                % Find the number of logical channels actually has data to
                % send
                if bufStatusList(lchIdx)
                    % Update the count of active logical channels
                    activeLCHCount = activeLCHCount + 1;
                end
            end
            % Initialize the assigned grant to zero for all the logical
            % channels
            avgGrantSize = zeros(numLCH, 1);
            % Check if the sum of logical channels buffer status is less
            % than grant remaining
            if sum(bufStatusList) < remainingBytes
                % Make average grant of each logical channel equal to its
                % buffer status
                avgGrantSize = bufStatusList;
            else
                % Calculate the average grant for each logical channel
                avgBytes = fix(remainingBytes/activeLCHCount);
                numBytesLeft = mod(remainingBytes, activeLCHCount);
                isNumBytesLeftLessthanRemNumLCH = false;
                % Make the resource assignment such that utilization of
                % resources is close to 100 percent
                for iter = 1:activeLCHCount % Helps in utilizing the overflown average grant of some logical channels
                    bytesFilled = 0;
                    for lchIdx = 1:numLCH % Shares the resources equally between logical channels
                        if avgGrantSize(lchIdx) == bufStatusList(lchIdx)
                            continue;
                        end
                        if isNumBytesLeftLessthanRemNumLCH && (numBytesLeft == bytesFilled)
                            % Avoid the resource allocation assignment
                            % overflow in case of number of bytes grant
                            % left is less than the logical channels
                            % requiring grant
                            avgBytes = 0;
                        end
                        bytesFilled = bytesFilled + avgBytes;
                        % Allocate average grant for each logical
                        % channel
                        avgGrantSize(lchIdx) = avgGrantSize(lchIdx) + avgBytes;
                        % Check if bytes allotted to the logical
                        % channel are more than its buffer status. If
                        % so, add the surplus bytes back to the
                        % numBytesLeft
                        if avgGrantSize(lchIdx) > bufStatusList(lchIdx)
                            % Calculate surplus amount of bytes for
                            % this logical channel from the average
                            % grant
                            numBytesLeft = numBytesLeft + (avgGrantSize(lchIdx) - bufStatusList(lchIdx));
                            avgGrantSize(lchIdx) = bufStatusList(lchIdx);
                            % Requirement of logical channel satisfied,
                            % decrement the counter of logical channels
                            % with grant requirement
                            activeLCHCount = activeLCHCount - 1;
                        end
                    end
                    % Stop sharing the grant among equally prioritized
                    % logical channels if the number of bytes remaining is
                    % 0 or isNumBytesLeftLessthanRemNumLCH flag is enabled
                    if (numBytesLeft == 0) || isNumBytesLeftLessthanRemNumLCH
                        break;
                    end
                    % If numBytesLeft is non-zero, calculate the
                    % average grant size for all the logical channels
                    % with grant requirement
                    if numBytesLeft < activeLCHCount
                        avgBytes = 1;
                        isNumBytesLeftLessthanRemNumLCH = true;
                    else
                        avgBytes = fix(numBytesLeft/activeLCHCount);
                        numBytesLeft = mod(numBytesLeft, activeLCHCount);
                    end
                end
            end
        end
    end

    methods(Abstract)
        addConnection(obj, connectionInfo)
    end
end
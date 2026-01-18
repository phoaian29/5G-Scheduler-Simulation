classdef nrUEMAC < nr5g.internal.nrMAC
    %nrUEMAC Implements UE MAC functionality
    %   The class implements the UE MAC and its interactions with RLC and PHY
    %   for Tx and Rx chains. It involves adhering to packet transmission and
    %   reception schedule and other related parameters which are received from
    %   gNB in the form of uplink (UL) and downlink (DL) assignments. Reception
    %   of uplink and downlink assignments on physical downlink control channel
    %   (PDCCH) is not modeled and they are received as out-of-band packets
    %   i.e. without using frequency resources and with guaranteed reception.
    %   Additionally, physical uplink control channel (PUCCH) is not modeled.
    %   The UE MAC sends the periodic buffer status report (BSR), PDSCH
    %   feedback, and DL channel quality report out-of-band. Hybrid automatic
    %   repeat request (HARQ) control mechanism to enable retransmissions is
    %   implemented. MAC controls the HARQ processes residing in physical
    %   layer.
    %
    %   Note: This is an internal undocumented class and its API and/or
    %   functionality may change in subsequent releases.

    %   Copyright 2022-2025 The MathWorks, Inc.

    properties (SetAccess = protected)
        % RNTI Radio network temporary identifier of a UE
        %   Specify the RNTI as an integer scalar within [1 65522]. Refer
        %   table 7.1-1 in 3GPP TS 38.321 version 18.1.0.
        RNTI

        %NumResourceBlocks Number of resource blocks (RBs)
        NumResourceBlocks

        %SchedulingType Type of scheduling (slot based or symbol based)
        % Value 0 means slot based and value 1 means symbol based.
        SchedulingType

        %DuplexMode Duplexing mode. Frequency division duplexing (FDD) or time division duplexing (TDD)
        % Value 0 means FDD and 1 means TDD.
        DuplexMode

        %DLULConfigTDD TDD configuration
        % It is a structure with following fields
        %   DLULPeriodicity - DL-UL pattern periodicity
        %   NumDLSlots - Number of full DL slots at the start of DL-UL pattern
        %   NumDLSymbols  - Number of DL symbols after full DL slots in the DL-UL pattern
        %   NumULSlots - Number of full UL slots at the end of DL-UL pattern
        %   NumULSymbols  - Number of UL symbols before full UL slots in the DL-UL pattern
        DLULConfigTDD

        %BSRPeriodicity Buffer status report periodicity in subframes
        BSRPeriodicity

        %CSIReportPeriod CSI reporting periodicity in slots as [periodicity offset]
        CSIReportPeriod

        %CSIReportPeriodicityRSRP CSI reporting (cri-l1RSRP) periodicity in slots
        CSIReportPeriodicityRSRP

        %CSIMeasurement Channel state information (CSI) measurements
        % Structure with the fields: 'RankIndicator', 'PMISet', 'CQI'.
        % RankIndicator is a scalar value to representing the rank reported by a UE.
        % PMISET has the following fields:
        %   i1 - Indicates wideband PMI (1-based). It a three-element vector in the
        %        form of [i11 i12 i13].
        %   i2 - Indicates subband PMI (1-based). It is a vector of length equal to
        %        the number of subbands or number of PRGs.
        % CQI - Array of size equal to number of RBs in the bandwidth. Each index
        % contains the CQI value corresponding to the RB index.
        % W is the precoder for the reported PMISET
        CSIMeasurement = struct('RankIndicator', [], 'PMISet', [], 'CQI', [], 'W',[]);

        %DLULSlotFormat Format of the slots in DL-UL pattern (for TDD mode)
        % N-by-14 matrix where 'N' is number of slots in DL-UL pattern.
        % Each row contains the symbol type of the 14 symbols in the slot.
        % Value 0, 1 and 2 represent DL symbol, UL symbol, guard symbol,
        % respectively
        DLULSlotFormat

        %NumDLULPatternSlots Number of slots in DL-UL pattern (for TDD mode)
        NumDLULPatternSlots

        %StatDLTransmissionRB Number of downlink resource blocks assigned
        %corresponding to new transmissions
        StatDLTransmissionRB

        %StatDLRetransmissionRB Number of downlink resource blocks assigned
        %corresponding to retransmissions
        StatDLRetransmissionRB

        %StatULTransmissionRB Number of uplink resource blocks assigned
        %corresponding to new transmissions
        StatULTransmissionRB

        %StatULRetransmissionRB Number of uplink resource blocks assigned
        %corresponding to retransmissions
        StatULRetransmissionRB
    end

    properties(Hidden)
        %XOverheadPDSCH Additional overheads in PDSCH transmission
        XOverheadPDSCH = 0;

        % MCSTableUL MCS table used for uplink
        % It contains the mapping of MCS indices with Modulation and Coding
        % schemes
        MCSTableUL

        % MCSTableDL MCS table used for downlink
        % It contains the mapping of MCS indices with Modulation and Coding
        % schemes
        MCSTableDL

        %DMRSTypeAPosition Position of DM-RS in type A transmission
        DMRSTypeAPosition (1, 1) {mustBeMember(DMRSTypeAPosition, [2, 3])} = 2;

        %PUSCHDMRSConfigurationType PUSCH DM-RS configuration type (1 or 2)
        PUSCHDMRSConfigurationType (1,1) {mustBeMember(PUSCHDMRSConfigurationType, [1, 2])} = 1;

        %PUSCHDMRSAdditionalPosTypeA Additional PUSCH DM-RS positions for type A (0..3)
        PUSCHDMRSAdditionalPosTypeA (1, 1) {mustBeMember(PUSCHDMRSAdditionalPosTypeA, [0, 1, 2, 3])} = 0;

        %PUSCHDMRSAdditionalPosTypeB Additional PUSCH DM-RS positions for type B (0..3)
        PUSCHDMRSAdditionalPosTypeB (1, 1) {mustBeMember(PUSCHDMRSAdditionalPosTypeB, [0, 1, 2, 3])} = 0;

        %PDSCHDMRSConfigurationType PDSCH DM-RS configuration type (1 or 2)
        PDSCHDMRSConfigurationType (1,1) {mustBeMember(PDSCHDMRSConfigurationType, [1, 2])} = 1;

        %PDSCHDMRSAdditionalPosTypeA Additional PDSCH DM-RS positions for type A (0..3)
        PDSCHDMRSAdditionalPosTypeA (1, 1) {mustBeMember(PDSCHDMRSAdditionalPosTypeA, [0, 1, 2, 3])} = 0;

        %PDSCHDMRSAdditionalPosTypeB Additional PDSCH DM-RS positions for type B (0 or 1)
        PDSCHDMRSAdditionalPosTypeB (1, 1) {mustBeMember(PDSCHDMRSAdditionalPosTypeB, [0, 1])} = 0;

        %UplinkTxContext Tx context used for PUSCH transmissions
        % Cell array of size 'N' where 'N' is the number of symbols in a 10
        % ms frame. At index 'i', it contains the uplink grant for a
        % transmission which is scheduled to start at symbol number 'i'
        % w.r.t start of the frame. Value at an index is empty, if no
        % uplink transmission is scheduled for the symbol
        UplinkTxContext

        %DownlinkRxContext Rx context used for PDSCH reception
        % Cell array of size 'N' where N is the number of symbols in a 10
        % ms frame. An element at index 'i' stores the downlink grant for
        % PDSCH scheduled to be received at symbol 'i' from the start of
        % the frame. If no PDSCH reception is scheduled, cell element is
        % empty
        DownlinkRxContext

        % PDSCHRxFeedback Feedback to be sent for PDSCH reception
        % N-by-3 array where 'N' is the number of HARQ process. For each
        % HARQ process, first column contains the symbol number w.r.t start
        % of 10ms frame where PDSCH feedback is scheduled to be
        % transmitted. Second column contains the feedback to be sent. The
        % third column contains time (in nanoseconds) at which
        % feedback has to be transmitted. Symbol number is -1 if no
        % feedback is scheduled for HARQ process. Feedback value 0 means
        % NACK while value 1 means ACK.
        PDSCHRxFeedback

        %HARQNDIUL Stores the last received NDI for uplink HARQ processes
        % Vector of length 'N' where 'N' is number of HARQ process. Value
        % at index 'i' stores last received NDI for the HARQ process index
        % 'i'. NDI in the UL grant is compared with this NDI to decide
        % whether grant is for new transmission or retransmission
        HARQNDIUL

        %HARQNDIDL Stores the last received NDI for downlink HARQ processes
        % Vector of length 'N' where 'N' is number of HARQ process. Value
        % at index 'i' stores last received NDI for the HARQ process index
        % 'i'. NDI in the DL grant is compared with this NDI to decide
        % whether grant is for new transmission or retransmission
        HARQNDIDL

        %RBGSize Resource block group size for BWP in terms of number of RBs
        RBGSize

        %NumRBGsUL Number of RBGs in uplink BWP
        NumRBGsUL

        %NumRBGsDL Number of RBGs in downlink BWP
        NumRBGsDL

        %SSBIdx Index of the SSB associated to the UE
        SSBIdx

        %LCGBufferStatus Logical channel group buffer status
        LCGBufferStatus = zeros(8, 1);
    end
    
    properties (Access = protected)
        %NextBSRTime Time (in nanoseconds) at which next BSR will be sent
        NextBSRTime = 0;

        %NextCSIReportTime Time (in nanoseconds) at which next CSI report will be sent
        NextCSIReportTime = Inf;

        %NextRSRPReportTime Time (in nanoseconds) at which next CSI report (cri-rsrp format) will be sent
        NextRSRPReportTime = Inf;

        %GuardDuration Guard period in the DL-UL pattern in terms of number of symbols (for TDD mode)
        GuardDuration

        %CSIRSRxInfo Information about CSI-RS reception
        % It is an array of size N-by-2 where N is the number of unique CSI-RS
        % periodicity, slot offset pairs configured for the UE.
        % Each row of the array contains CSI-RS reception periodicity (in
        % nanoseconds) and the next reception start time (in
        % nanoseconds) of the UE.
        CSIRSRxInfo = [Inf Inf];

        %SRSTxInfo Information about SRS transmission
        % It is a vector of size 2. First element in the vector contains
        % SRS transmission periodicity (in nanoseconds) and the second
        % element contains the next transmission start time (in
        % nanoseconds) of the UE.
        SRSTxInfo = [Inf Inf];

        %GNBInfo Information about the GNB to which this UE is connected
        % It is a structure with two fields.
        %   ID - Node id of the UE
        %   Name - Node Name of the UE
        GNBInfo

        %CSIMeasurementFlag Flag to indicate whether CSI is yet measured or not
        CSIMeasurementFlag = 0;

        %CSIRSConfiguration CSI-RS configuration for the UE
        CSIRSConfiguration

        %EnableAdvancedOptimization Flag to enable/disable advanced runtime optimizations
        % Enable this under the assumption that all nodes in the simulation
        % run with same SCS and the slots are fully aligned for the nodes.
        % It also assumes that channel bandwidth for all the nodes in the
        % simulation either fully overlaps or doesn't overlap at all.
        EnableAdvancedOptimization=true

        %ReceptionRBSet RB set on which the node is expecting reception
        % This helps in rejecting incoming packets as irrelevant which do not overlap with this RB set
        ReceptionRBSet

        %ReceptionEndTime Reception end time correspending to ReceptionRBSet
        ReceptionEndTime = 0;
    end

    methods
        function obj = nrUEMAC(notificationFcn)
            %nrUEMAC Construct a UE MAC object
            %
            % NOTIFICATIONFCN - It is a handle of the node's processEvents
            % method

            obj.NotificationFcn = notificationFcn;
            obj.StatTransmittedPackets = 0;
            obj.StatTransmittedBytes = 0;
            obj.StatRetransmittedPackets = 0;
            obj.StatRetransmittedBytes = 0;
            obj.StatReceivedPackets = 0;
            obj.StatReceivedBytes = 0;
            obj.StatDLTransmissionRB = 0;
            obj.StatULTransmissionRB = 0;
            obj.StatDLRetransmissionRB = 0;
            obj.StatULRetransmissionRB = 0;
            obj.PacketStruct.Type= 2; % 5G packet
            obj.PacketStruct.Metadata = struct('NCellID', [], 'RNTI', [], 'PacketType', []);
            obj.RLCTxFcn = cell(1, obj.MaxLogicalChannels);
            obj.RLCRxFcn = cell(1, obj.MaxLogicalChannels);
        end

        function nextInvokeTime = run(obj, currentTime, packets)
            %run Run the UE MAC layer operations and return next invoke time in nanoseconds
            %   NEXTINVOKETIME = run(OBJ, CURRENTTIME, PACKETS) runs the
            %   MAC layer operations and returns the next invoke time.
            %
            %   NEXTINVOKETIME is the next invoke time (in nanoseconds) for
            %   MAC.
            %
            %   CURRENTTIME is the current time (in nanoseconds).
            %
            %   PACKETS are the received packets from other nodes.

            nextInvokeTime = Inf;
            if isempty(obj.RNTI) % UE is not yet connected to gNB
                return;
            end
            % Advance the slot and symbol level timers
            elapsedTime = currentTime - obj.LastRunTime; % In nanoseconds
            if currentTime > obj.LastRunTime
                % Update the LCP timers
                obj.ElapsedTimeSinceLastLCP  = obj.ElapsedTimeSinceLastLCP  + round(elapsedTime*1e-6, 4); % In milliseconds
                obj.LastRunTime = currentTime;

                % Find the current frame number
                obj.CurrFrame = floor(currentTime/obj.FrameDurationInNS);
                numSlotsInTime = floor(currentTime/obj.SlotDurationInNS);
                % Current slot number in 10 ms frame
                obj.CurrSlot = mod(numSlotsInTime, obj.NumSlotsFrame);

                if obj.DuplexMode == 1 % TDD
                    % Current slot number in DL-UL pattern
                    obj.CurrDLULSlotIndex = mod(numSlotsInTime, obj.NumDLULPatternSlots);
                end

                % Find the current symbol in the current slot
                durationCompletedInCurrSlot = mod(currentTime, obj.SlotDurationInNS);
                obj.CurrSymbol = find(durationCompletedInCurrSlot < obj.SymbolEndTimesInSlot, 1) - 1;

                % Update timing info context
                obj.TimestampInfo.NFrame = obj.CurrFrame;
                obj.TimestampInfo.NSlot = obj.CurrSlot;
                obj.TimestampInfo.NSymbol = obj.CurrSymbol;
                obj.TimestampInfo.Timestamp = currentTime;
            end

            % Receive and process control packet
            controlRx(obj, packets);

            % Avoid running MAC operations more than once in the same symbol
            symNumFrame = obj.CurrSlot * obj.NumSymbols + obj.CurrSymbol;
            if obj.PreviousSymbol == symNumFrame && elapsedTime < obj.SlotDurationInNS/obj.NumSymbols
                nextInvokeTime = getNextInvokeTime(obj, currentTime);
                return;
            end

            if obj.TimestampInfo.Timestamp>=obj.ReceptionEndTime
                obj.ReceptionRBSet = []; % Clear reception RB set after rx end time.
            end
            % Send Tx request to PHY for transmission which is scheduled to start at current
            % symbol. Construct and send the UL MAC PDUs scheduled for
            % current symbol to PHY
            dataTx(obj);

            % Send Rx request to PHY for reception which is scheduled to start at current symbol
            dataRx(obj);

            % Send BSR, PDSCH feedback (ACK/NACK) and CQI report
            controlTx(obj);

            % Send requests to PHY for non-data receptions and
            % transmissions scheduled in this slot (currently only CSI-RS
            % and SRS are supported). Send these requests at the first
            % symbol of the slot
            % Update the next CSI-RS tx times
            idxList = find(obj.CSIRSRxInfo(:, 2) == currentTime);
            if ~isempty(idxList)
                dlControlRequest(obj);
                obj.CSIRSRxInfo(idxList, 2) = obj.CSIRSRxInfo(idxList, 1) + currentTime;
            end
            % Update the next SRS tx times
            if obj.SRSTxInfo(2) == currentTime
                obj.SRSTxInfo(2) = obj.SRSTxInfo(1) + currentTime;
                ulControlRequest(obj);
            end

            % Update the previous symbol to the current symbol in the frame
            obj.PreviousSymbol = symNumFrame;
            % Return the next invoke time for MAC
            nextInvokeTime = getNextInvokeTime(obj, currentTime);
        end

        function addConnection(obj, connectionInfo)
            %addConnection Configures the UE MAC with connection information
            %
            % CONNECTIONINFO is a structure including the following fields:
            %
            %   GNBID                       - Node id of the GNB to which this
            %                                 UE is connected
            %   RNTI                        - Radio network temporary identifier
            %                                 specified within [1, 65522]. Refer
            %                                 table 7.1-1 in 3GPP TS 38.321 version 18.1.0.
            %   NCellID                     - Physical cell ID. Values: 0 to 1007 (TS 38.211, sec 7.4.2.1)
            %   SubcarrierSpacing           - Subcarrier spacing
            %   DuplexMode                  - Duplexing mode as 'FDD' or 'TDD'
            %   BSRPeriodicity              - Periodicity for the BSR packet
            %                                 generation in terms of subframes.
            %   NumResourceBlocks           - Number of RBs in PUSCH and PDSCH bandwidth
            %   NumHARQ                     - Number of HARQ processes on UEs
            %   DLULConfigTDD               - TDD specific configuration. It is
            %                                 is a structure with following fields
            %       DLULPeriodicity         - Duration of the DL-UL pattern in ms (for TDD mode)
            %       NumDLSlots              - Number of full DL slots at the start of DL-UL pattern (for TDD mode)
            %       NumDLSymbols            - Number of DL symbols after full DL slots of DL-UL pattern (for TDD mode)
            %       NumULSymbols            - Number of UL symbols before full UL slots of DL-UL pattern (for TDD mode)
            %       NumULSlots              - Number of full UL slots at the end of DL-UL pattern (for TDD mode)
            %   SchedulingType              - Slot based scheduling (value 0) or symbol based scheduling (value 1)
            %   RBGSizeConfiguration        - RBG size configuration as 1 (configuration-1 RBG table) or 2 (configuration-2 RBG table)
            %                                  as defined in 3GPP TS 38.214 Section 5.1.2.2.1. It defines the
            %                                  number of RBs in an RBG. Default value is 1
            %   DMRSTypeAPosition            - DM-RS type A position (2 or 3)
            %   PUSCHDMRSConfigurationType   - PUSCH DM-RS configuration type (1 or 2)
            %   PUSCHDMRSAdditionalPosTypeA  - Additional PUSCH DM-RS positions for Type A (0..3)
            %   PUSCHDMRSAdditionalPosTypeB  - Additional PUSCH DM-RS positions for Type B (0..3)
            %   PDSCHDMRSConfigurationType   - PDSCH DM-RS configuration type (1 or 2)
            %   PDSCHDMRSAdditionalPosTypeA  - Additional PDSCH DM-RS positions for Type A (0..3)
            %   PDSCHDMRSAdditionalPosTypeB  - Additional PDSCH DM-RS positions for Type B (0 or 1)
            %   UETxAnts                 - Number of UE Tx antennas
            %   CSIRSConfiguration       - CSI-RS configuration information as an
            %                              object of type nrCSIRSConfig.
            %   CSIRSConfigurationRSRP   - CSI-RS resource set configurations corresponding to the SSB directions.
            %                              It is a cell array of length N-by-1 where 'N' is the number of
            %                              maximum number of SSBs in a SSB burst. Each element of the array
            %                              at index 'i' corresponds to the CSI-RS resource set associated
            %                              with SSB 'i-1'. The number of CSI-RS resources in each resource
            %                              set is same for all configurations.
            %   SSBIdx                   - Scalar representing the index of
            %                              SSB acquired by the UE during P-1 procedure as defined in 3GPP TR 38.802 Section 6.1.6.1
            %   SRSConfiguration         - SRS configuration specified as an object of type nrSRSConfig
            %   CSIReportPeriod          - CSI reporting period in slots as [periodicity offset]
            %   CSIReportPeriodicityRSRP - CSI reporting (cri-l1RSRP)
            %                              periodicity in slots
            %   InitialCQIDL             - Initial DL channel quality

            inputParam = {'RNTI', 'NCellID', 'SubcarrierSpacing', 'SchedulingType', 'NumHARQ', ...
                'CSIRSConfiguration', 'SRSConfiguration', 'NumResourceBlocks', ...
                'BSRPeriodicity', 'CSIReportPeriod'};

            for idx=1:numel(inputParam)
                obj.(inputParam{idx}) = connectionInfo.(inputParam{idx});
            end

            if ~isempty(connectionInfo.CSIRSConfiguration)
                obj.XOverheadPDSCH = 18;
            end

            if strcmp(connectionInfo.DuplexMode, 'FDD')
                obj.DuplexMode = 0;
            else
                obj.DuplexMode = 1;
            end

            if isfield(connectionInfo, 'CSIRSConfigurationRSRP')
                obj.CSIRSConfigurationRSRP = connectionInfo.CSIRSConfigurationRSRP;
            end
            if isfield(connectionInfo, 'SSBIdx')
                obj.SSBIdx = connectionInfo.SSBIdx;
            end
            if isfield(connectionInfo, 'CSIReportPeriodicityRSRP')
                obj.CSIReportPeriodicityRSRP = connectionInfo.CSIReportPeriodicityRSRP;
            end

            obj.GNBInfo  = struct('ID', connectionInfo.GNBID, 'Name', connectionInfo.GNBName);
            % All the out-of-band packets are sent directly to gNB
            obj.PacketStruct.DirectToDestination = obj.GNBInfo.ID;

            obj.MACType = 1; % UE MAC
            slotDuration = 1/(obj.SubcarrierSpacing/15); % In ms
            obj.SlotDurationInNS = slotDuration * 1e6; % In nanoseconds
            obj.SlotDurationInSec = slotDuration * 1e-3;
            obj.NumSlotsFrame = 10/slotDuration; % Number of slots in a 10 ms frame
            obj.NumSymInFrame = obj.NumSlotsFrame*obj.NumSymbols;
            % Calculate symbol end times (in nanoseconds) in a slot for the given scs
            obj.SymbolEndTimesInSlot = round(((1:obj.NumSymbols)*slotDuration)/obj.NumSymbols, 4) * 1e6;
            obj.SymbolDurationsInSlot = obj.SymbolEndTimesInSlot(1:obj.NumSymbols) - [0 obj.SymbolEndTimesInSlot(1:13)];

            % TDD specific configuration
            if obj.DuplexMode == 1 % TDD
                obj.DLULConfigTDD = connectionInfo.DLULConfigTDD;
                obj.NumDLULPatternSlots = obj.DLULConfigTDD.DLULPeriodicity/(obj.SlotDurationInNS*1e-6);

                % All the remaining symbols in DL-UL pattern are assumed to
                % be guard symbols
                obj.GuardDuration = (obj.NumDLULPatternSlots * obj.NumSymbols) - ...
                    (((obj.DLULConfigTDD.NumDLSlots + obj.DLULConfigTDD.NumULSlots)*obj.NumSymbols) + ...
                    obj.DLULConfigTDD.NumDLSymbols + obj.DLULConfigTDD.NumULSymbols);

                % Set format of slots in the DL-UL pattern. Value 0 means
                % DL symbol, value 1 means UL symbol while symbols with
                % value 2 are guard symbols
                obj.DLULSlotFormat = obj.GuardType * ones(obj.NumDLULPatternSlots, obj.NumSymbols);
                obj.DLULSlotFormat(1:obj.DLULConfigTDD.NumDLSlots, :) = obj.DLType; % Mark all the symbols of full DL slots as DL
                obj.DLULSlotFormat(obj.DLULConfigTDD.NumDLSlots + 1, 1 : obj.DLULConfigTDD.NumDLSymbols) = obj.DLType; % Mark DL symbols following the full DL slots
                obj.DLULSlotFormat(obj.DLULConfigTDD.NumDLSlots + floor(obj.GuardDuration/obj.NumSymbols) + 1, (obj.DLULConfigTDD.NumDLSymbols + mod(obj.GuardDuration, obj.NumSymbols) + 1) : end)  ...
                    = obj.ULType; % Mark UL symbols at the end of slot before full UL slots
                obj.DLULSlotFormat((end - obj.DLULConfigTDD.NumULSlots + 1):end, :) = obj.ULType; % Mark all the symbols of full UL slots as UL type
            end

            % Set the RBG size configuration (for defining number of RBs in
            % one RBG) to 1 (configuration-1 RBG table) or 2
            % (configuration-2 RBG table) as defined in 3GPP TS 38.214
            % Section 5.1.2.2.1. If it is not configured, take default
            % value as 1.
            % Calculate RBG size in terms of number of RBs
            rbgSizeIndex = min(find(obj.NumResourceBlocks <= obj.NominalRBGSizePerBW(:, 1), 1));
            if connectionInfo.RBGSizeConfiguration == 1
                obj.RBGSize = obj.NominalRBGSizePerBW(rbgSizeIndex, 2);
            else % RBGSizeConfig is 2
                obj.RBGSize = obj.NominalRBGSizePerBW(rbgSizeIndex, 3);
            end

            obj.PDSCHRxFeedback = -1*ones(obj.NumHARQ, 3);
            obj.PDSCHRxFeedback(:, 3) = Inf;
            obj.HARQNDIUL = zeros(obj.NumHARQ, 1); % Initialize NDI of each UL HARQ process to 0
            obj.HARQNDIDL = zeros(obj.NumHARQ, 1); % Initialize NDI of each DL HARQ process to 0

            % Stores uplink assignments (if any), corresponding to uplink
            % transmissions starting at different symbols of the frame
            obj.UplinkTxContext = cell(obj.NumSymInFrame, 1);

            % Stores downlink assignments (if any), corresponding to
            % downlink receptions starting at different symbols of the
            % frame
            obj.DownlinkRxContext = cell(obj.NumSymInFrame, 1);

            % Set non-zero-power (NZP) CSI-RS configuration for the UE
            if ~isempty(obj.CSIRSConfiguration)
                obj.NextCSIReportTime = obj.CSIReportPeriod(2)*obj.SlotDurationInNS; % In nanoseconds
            end

            % CSIRS-RSRP configuration for the UE
            if ~isempty(obj.CSIRSConfigurationRSRP)
                obj.NextRSRPReportTime = obj.CSIReportPeriodicityRSRP*obj.SlotDurationInNS; % In nanoseconds
            end

            % Calculate unique CSI-RS reception slots
            csirsResourceConfig = [obj.CSIRSConfiguration obj.CSIRSConfigurationRSRP];
            obj.CSIRSRxInfo = calculateCSIRSPeriodicity(obj, csirsResourceConfig);

            % Calculate unique SRS transmission time and periodicity
            obj.SRSTxInfo = calculateSRSPeriodicity(obj, obj.SRSConfiguration);

            obj.MCSTableUL = nr5g.internal.MACConstants.MCSTable;
            obj.MCSTableDL = nr5g.internal.MACConstants.MCSTable;
            obj.LCHBufferStatus = zeros(1, obj.MaxLogicalChannels);
            obj.LCHBjList = zeros(1, obj.MaxLogicalChannels);
            obj.LogicalChannelConfig = cell(1, obj.MaxLogicalChannels);
            obj.ElapsedTimeSinceLastLCP = 0;
            obj.CSIMeasurement.CQI = connectionInfo.InitialCQIDL;

            % Create carrier configuration object for UL
            obj.CarrierConfigUL = nrCarrierConfig(SubcarrierSpacing=obj.SubcarrierSpacing, NSizeGrid=obj.NumResourceBlocks);
            % Create carrier configuration object for DL
            obj.CarrierConfigDL = nrCarrierConfig(SubcarrierSpacing=obj.SubcarrierSpacing, NSizeGrid=obj.NumResourceBlocks);

            % Set the DMRS configuration
            obj.PDSCHInfo.PDSCHConfig.DMRS = nrPDSCHDMRSConfig(DMRSConfigurationType=obj.PDSCHDMRSConfigurationType, ...
                DMRSTypeAPosition=obj.DMRSTypeAPosition);
            obj.PUSCHInfo.PUSCHConfig.DMRS = nrPUSCHDMRSConfig(DMRSConfigurationType=obj.PUSCHDMRSConfigurationType, ...
                DMRSTypeAPosition=obj.DMRSTypeAPosition);

            % Fill NID and RNTI at the connection set-up time (for runtime optimization)
            obj.PDSCHInfo.PDSCHConfig.NID = obj.NCellID;
            obj.PDSCHInfo.PDSCHConfig.RNTI = obj.RNTI;
            obj.PUSCHInfo.PUSCHConfig.NID = obj.NCellID;
            obj.PUSCHInfo.PUSCHConfig.RNTI = obj.RNTI;

            if isinf(obj.BSRPeriodicity)
                % Disable BSR control element generation when BSR periodicity is infinite
                obj.NextBSRTime = Inf;
            end

            % Set the metadata for the packet
            obj.PacketStruct.Metadata.NCellID = obj.NCellID;
            obj.PacketStruct.Metadata.RNTI = obj.RNTI;
        end

        function updateBufferStatus(obj, lcBufferStatus)
            %updateBufferStatus Update the buffer status of the logical channel
            %
            %   updateBufferStatus(OBJ, LCBUFFERSTATUS) Updates the buffer
            %   status of a logical channel based on information present in
            %   LCBUFFERSTATUS object
            %
            %   LCBUFFERSTATUS - Represents an object which contains the
            %   current buffer status of a logical channel. It contains the
            %   following properties:
            %       RNTI                    - UE's radio network temporary identifier
            %       LogicalChannelID        - Logical channel identifier
            %       BufferStatus            - Number of bytes in the logical
            %                                 channel's Tx buffer

            lcgID = 0; % Default LCG ID
            for i = 1:length(obj.LogicalChannelConfig)
                if ~isempty(obj.LogicalChannelConfig{i}) && (obj.LogicalChannelConfig{i}.LogicalChannelID == lcBufferStatus.LogicalChannelID)
                    lcgID = obj.LogicalChannelConfig{i}.LogicalChannelGroup;
                    break;
                end
            end

            % Subtract from the old buffer status report of the corresponding
            % logical channel
            lcgIdIndex = lcgID + 1; % Indexing starts from 1

            % Update the buffer status of LCG to which this logical channel
            % belongs to. Subtract the current logical channel buffer
            % amount and adding the new amount
            obj.LCGBufferStatus(lcgIdIndex) = obj.LCGBufferStatus(lcgIdIndex) -  ...
                obj.LCHBufferStatus(lcBufferStatus.LogicalChannelID) + lcBufferStatus.BufferStatus;

            % Update the new buffer status
            obj.LCHBufferStatus(lcBufferStatus.LogicalChannelID) = lcBufferStatus.BufferStatus;
        end

        function rxIndication(obj, rxInfo, currentTime)
            %rxIndication Packet reception from PHY
            %   rxIndication(OBJ, MACPDU, CRC, RXINFO) receives a MAC PDU from
            %   PHY.
            %   RXINFO is a structure containing information about the
            %   reception.
            %       NodeID  - Node identifier.
            %       HARQID  - HARQ process identifier
            %       MACPDU  - It is a vector of decimal octets received from PHY.
            %       CRCFlag - It is the success(value as 0)/failure(value as 1)
            %                 indication from PHY.
            %       Tags   - Array of structures where each structure contains
            %                these fields.
            %                Name      - Name of the tag.
            %                Value     - Data associated with the tag.
            %                ByteRange - Specific range of bytes within the
            %                            packet to which the tag applies.
            %
            %   CURRENTTIME - Current time in nanoseconds.

            isRxSuccess = ~rxInfo.CRCFlag; % CRC value 0 indicates successful reception
            if isRxSuccess % Packet received is error-free
                byteOffset = 0;
                % Utilize the soft error handling feature for PDU decode failures. This
                % approach prevents hard errors upon PDU decode failures and returns empty
                % values instead
                softErrorFlag = true;
                % Parse downlink MAC PDU
                [lcidList, sduList, subHeaderLengthList] = nrMACPDUDecode(rxInfo.MACPDU, obj.DLType, softErrorFlag);

                packetInfo = obj.HigherLayerPacketFormat;
                packetInfo.NodeID = rxInfo.NodeID;
                for sduIndex = 1:numel(lcidList)
                    byteOffset = byteOffset+subHeaderLengthList(sduIndex);
                    if lcidList(sduIndex) >=4 && lcidList(sduIndex) <= 32
                        packetInfo.Packet = sduList{sduIndex};
                        packetInfo.PacketLength = length(packetInfo.Packet);
                        packetInfo.Tags = wirelessnetwork.internal.packetTags.segment(rxInfo.Tags, [byteOffset+1 ...
                            byteOffset+packetInfo.PacketLength]);
                        obj.RLCRxFcn{1, lcidList(sduIndex)}(packetInfo, currentTime);
                    end
                    byteOffset = byteOffset+length(sduList{sduIndex});
                end
                obj.PDSCHRxFeedback(rxInfo.HARQID+1, 2) = 1;  % Positive ACK
                obj.StatReceivedBytes = obj.StatReceivedBytes + length(rxInfo.MACPDU);
                obj.StatReceivedPackets = obj.StatReceivedPackets + 1;

            else % Packet corrupted
                obj.PDSCHRxFeedback(rxInfo.HARQID+1, 2) = 0; % NACK
            end
        end

        function csirsIndication(obj, varargin)
            %csirsIndication Reception of CSI measurements from PHY
            %   csirsIndication(OBJ, RANK, PMISET, CQI, W) receives the DL channel
            %   measurements from PHY, measured on the configured CSI-RS for the
            %   UE.
            %   RANK - Rank indicator.
            %   PMISET - Wideband PMI corresponding to RANK. It is a structure
            %   with field 'i11'.
            %   CQI - Wideband CQI corresponding to RANK and PMISET.
            %   W  - Precoding matrix corresponding to PMISET.
            %
            %   csirsIndication(OBJ, CRI, L1RSRP) receives the DL beam
            %   measurements from PHY, measured on the configured CSI-RS for the UE
            %   CRI - CSI-RS resource indicator. It is a scalar
            %   representing the CSI-RS resource with the highest L1-RSRP
            %   measurement.
            %   L1RSRP - Layer 1 Reference Signal Received Power(L1-RSRP). This
            %   contains the L1-RSRP measurement for the CSI-RS resource
            %   indicated by CRI.

            if nargin == 5
                obj.CSIMeasurement.RankIndicator = varargin{1};
                obj.CSIMeasurement.PMISet = varargin{2};
                obj.CSIMeasurement.CQI = varargin{3};
                obj.CSIMeasurement.W = varargin{4};
            elseif nargin == 3
                obj.CSIMeasurement.CRI = varargin{1};
                obj.CSIMeasurement.L1RSRP = varargin{2};
            end
            obj.CSIMeasurementFlag = 1;
        end

        function macStats = statistics(obj)
            %statistics Return the UE MAC statistics
            %
            %   MACSTATS = statistics(OBJ) Returns the MAC statistics of
            %   each UE at gNB MAC
            %
            %   MACSTATS - It is a structure with following fields.
            %       TransmittedPackets   - Number of packets transmitted in UL
            %                              corresponding to new transmissions
            %       TransmittedBytes     - Number of bytes transmitted in UL
            %                              corresponding to new transmissions
            %       Retransmissions      - Number of packets retransmitted in UL
            %       RetransmissionBytes  - Number of bytes retransmitted in UL
            %       ReceivedPackets      - Number of packets received in DL
            %       ReceivedBytes        - Number of bytes received in DL
            %       DLTransmissionRB     - Number of downlink resource blocks assigned
            %                              corresponding to new transmissions
            %       DLRetransmissionRB   - Number of downlink resource blocks assigned
            %                              corresponding to retransmissions
            %       ULTransmissionRB     - Number of uplink resource blocks assigned
            %                              corresponding to new transmissions
            %       ULRetransmissionRB   - Number of uplink resource blocks assigned
            %                              corresponding to retransmissions

            macStats = struct('TransmittedPackets', obj.StatTransmittedPackets, 'TransmittedBytes', ...
                obj.StatTransmittedBytes, 'ReceivedPackets', obj.StatReceivedPackets, ...
                'ReceivedBytes', obj.StatReceivedBytes, 'Retransmissions', obj.StatRetransmittedPackets, ...
                'RetransmissionBytes', obj.StatRetransmittedBytes, 'DLTransmissionRB',...
                obj.StatDLTransmissionRB, 'DLRetransmissionRB', obj.StatDLRetransmissionRB, ...
                'ULTransmissionRB', obj.StatULTransmissionRB, 'ULRetransmissionRB', obj.StatULRetransmissionRB);
        end
    end

    methods (Hidden)
        function dataTx(obj)
            %dataTx Construct and send the UL MAC PDUs scheduled for current symbol to PHY
            %
            %   dataTx(OBJ) Based on the uplink grants received in earlier,
            %   if current symbol is the start symbol of a Tx then send the UL MAC PDU to
            %   PHY.
            %

            if ~isempty(obj.UplinkTxContext)
                symbolNumFrame = obj.CurrSlot*obj.NumSymbols + obj.CurrSymbol;
                uplinkGrant = obj.UplinkTxContext{symbolNumFrame + 1};
                % If there is any uplink grant corresponding to which a transmission is scheduled at the current symbol
                if ~isempty(uplinkGrant)
                    % Construct and send MAC PDU to PHY
                    [sentPDULen, type] = sendMACPDU(obj, uplinkGrant);
                    obj.UplinkTxContext{symbolNumFrame + 1} = []; % Tx done. Clear the context

                    if strcmp(type, 'newTx') % New transmission
                        obj.StatTransmittedBytes = obj.StatTransmittedBytes + sentPDULen;
                        obj.StatTransmittedPackets = obj.StatTransmittedPackets + 1;
                    else  % Retransmission
                        obj.StatRetransmittedBytes = obj.StatRetransmittedBytes + sentPDULen;
                        obj.StatRetransmittedPackets = obj.StatRetransmittedPackets + 1;
                    end
                end
            end
        end

        function dataRx(obj)
            %dataRx Send Rx start request to PHY for the reception scheduled to start now
            %
            %   dataRx(OBJ) sends the Rx start request to PHY for the
            %   reception scheduled to start now, as per the earlier
            %   received downlink assignments.
            %

            if ~isempty(obj.DownlinkRxContext)
                downlinkGrant = obj.DownlinkRxContext{obj.CurrSlot*obj.NumSymbols + obj.CurrSymbol + 1}; % Rx context of current symbol
                if ~isempty(downlinkGrant) % If PDSCH reception is expected
                    % Calculate feedback transmission symbol number w.r.t start
                    % of 10ms frame
                    feedbackSlot = mod(obj.CurrSlot + downlinkGrant.FeedbackSlotOffset, obj.NumSlotsFrame);
                    %For TDD, the symbol at which feedback would be transmitted
                    %is kept as first UL symbol in feedback slot. For FDD, it
                    %simply the first symbol in the feedback slot
                    if obj.DuplexMode % TDD
                        feedbackSlotDLULIdx = mod(obj.CurrDLULSlotIndex + downlinkGrant.FeedbackSlotOffset, obj.NumDLULPatternSlots);
                        feedbackSlotPattern = obj.DLULSlotFormat(feedbackSlotDLULIdx + 1, :);
                        feedbackSym = (find(feedbackSlotPattern == obj.ULType, 1, 'first')) - 1; % Check for location of first UL symbol in the feedback slot
                    else % FDD
                        feedbackSym = 0;  % First symbol
                    end
                    obj.PDSCHRxFeedback(downlinkGrant.HARQID+1, 1) = feedbackSlot*obj.NumSymbols + feedbackSym; % Set symbol number for PDSCH feedback transmission

                    % Absolute feedback symbol number
                    obj.PDSCHRxFeedback(downlinkGrant.HARQID+1, 3) = downlinkGrant.FeedbackSlotOffset*obj.NumSymbols + feedbackSym;
                    rxRequestToPHY(obj, downlinkGrant); % Indicate Rx start to PHY
                    % Clear the Rx context
                    obj.DownlinkRxContext{(obj.CurrSlot * obj.NumSymbols) + obj.CurrSymbol + 1} = [];
                    obj.LastRxTime = obj.TimestampInfo.Timestamp;
                end
            end
        end

        function dlControlRequest(obj)
            % dlControlRequest Request from MAC to PHY to receive non-data DL receptions
            %
            %   dlControlRequest(OBJ) sends a request to PHY for non-data
            %   downlink receptions in the current slot. MAC sends it at
            %   the start of a DL slot for all the scheduled DL receptions
            %   in the slot (except PDSCH, which is received using dataRx
            %   function of this class).
            %

            % Check if current slot is a slot with DL symbols. For FDD (Value 0),
            % there is no need to check as every slot is a DL slot. For
            % TDD (Value 1), check if current slot has any DL symbols
            if(obj.DuplexMode == 0 || ~isempty(find(obj.DLULSlotFormat(obj.CurrDLULSlotIndex + 1, :) == obj.DLType, 1)))
                dlControlType = zeros(1, 2);
                dlControlPDUs = cell(1, 2);
                numDLControlPDUs = 0;

                if ~isempty(obj.CSIRSConfigurationRSRP)
                    csirsConfig = obj.CSIRSConfigurationRSRP;
                    % To account for consecutive symbols in CDM pattern
                    additionalCSIRSSyms = [0 0 0 0 1 0 1 1 0 1 1 1 1 1 3 1 1 3];
                    csirsSymbolRange(1) = min(csirsConfig.SymbolLocations); % First CSI-RS symbol
                    csirsSymbolRange(2) = max(csirsConfig.SymbolLocations) + ... % Last CSI-RS symbol
                        additionalCSIRSSyms(csirsConfig.RowNumber);
                    if obj.DuplexMode == 0 || all(obj.DLULSlotFormat(obj.CurrDLULSlotIndex + 1, csirsSymbolRange+1) == obj.DLType)
                        % Set carrier configuration object
                        carrier = obj.CarrierConfigDL;
                        carrier.NSlot = obj.CurrSlot;
                        carrier.NFrame = obj.CurrFrame;
                        % Check if the current slot is CSI-RS reception slot based on configured CSI-RS periodicity and offset
                        if (~isnumeric(csirsConfig.CSIRSPeriod) && csirsConfig.CSIRSPeriod == "on") || ~mod(obj.NumSlotsFrame*obj.CurrFrame + obj.CurrSlot - csirsConfig.CSIRSPeriod(2), csirsConfig.CSIRSPeriod(1))
                            numDLControlPDUs = numDLControlPDUs + 1;
                            dlControlType(numDLControlPDUs) = 0; % CSIRS PDU
                            dlControlPDUs{numDLControlPDUs} = obj.CSIRSConfigurationRSRP(obj.SSBIdx);
                        end
                    end
                end
                if ~isempty(obj.CSIRSConfiguration)
                    csirsConfig = obj.CSIRSConfiguration(1);
                    % To account for consecutive symbols in CDM pattern
                    additionalCSIRSSyms = [0 0 0 0 1 0 1 1 0 1 1 1 1 1 3 1 1 3];
                    csirsSymbolRange = zeros(2, 1);
                    csirsSymbolRange(1) = min(csirsConfig.SymbolLocations); % First CSI-RS symbol
                    csirsSymbolRange(2) = max(csirsConfig.SymbolLocations) + ... % Last CSI-RS symbol
                        additionalCSIRSSyms(csirsConfig.RowNumber);
                    % Check whether the mode is FDD OR if it is TDD then all the CSI-RS symbols must be DL symbols
                    if obj.DuplexMode == 0 || all(obj.DLULSlotFormat(obj.CurrDLULSlotIndex + 1, csirsSymbolRange+1) == obj.DLType)
                        % Set carrier configuration object
                        carrier = obj.CarrierConfigDL;
                        carrier.NSlot = obj.CurrSlot;
                        carrier.NFrame = obj.CurrFrame;
                        % Check if the current slot is CSI-RS reception slot based on configured CSI-RS periodicity and offset
                        if (~isnumeric(csirsConfig.CSIRSPeriod) && csirsConfig.CSIRSPeriod == "on") || ~mod(obj.NumSlotsFrame*obj.CurrFrame + obj.CurrSlot - csirsConfig.CSIRSPeriod(2), csirsConfig.CSIRSPeriod(1))
                            % If the next CSI-RS reception (i.e. after this one) is scheduled at least 2
                            % slots before the next reporting occurence then ignore this CSI-RS
                            currAbsoluteSlotNum = obj.CurrFrame*obj.NumSlotsFrame + obj.CurrSlot;
                            reportPeriodicity = obj.CSIReportPeriod(1);
                            reportSlotOffset = obj.CSIReportPeriod(2);
                            slotsToNextReport = reportPeriodicity - mod(currAbsoluteSlotNum-reportSlotOffset, reportPeriodicity);
                            % If upcoming CSI report occurence is less than
                            % two slots away then this CSI-RS reception
                            % measurement can only be reported in the
                            % report occurence after the upcoming report
                            % occurence
                            if slotsToNextReport < 2
                                slotsToNextReport = slotsToNextReport + reportPeriodicity;
                            end
                            if (slotsToNextReport - csirsConfig.CSIRSPeriod(1)) < 2
                                % Measure on this CSI-RS reception only if another CSI-RS reception
                                % will not happen before next report occurence
                                numDLControlPDUs = numDLControlPDUs + 1;
                                dlControlType(numDLControlPDUs) = 0; % CSIRS PDU
                                dlControlPDUs{numDLControlPDUs} = csirsConfig;
                            end
                        end
                    end
                end
                if numDLControlPDUs > 0 % Send request when there is CSI-RS transmission
                    obj.ReceptionRBSet = 0:carrier.NSizeGrid-1; % Full-bandwidth CSI-RS
                    obj.ReceptionEndTime = obj.TimestampInfo.Timestamp+obj.SlotDurationInNS; % Update rx end time
                    obj.DlControlRequestFcn(dlControlType(1:numDLControlPDUs), dlControlPDUs(1:numDLControlPDUs), obj.TimestampInfo); % Send DL control request to PHY
                    obj.LastRxTime = obj.TimestampInfo.Timestamp;
                end
            end
        end

        function ulControlRequest(obj)
            %ulControlRequest Request from MAC to PHY to send non-data UL transmissions
            %   ulControlRequest(OBJ) sends a request to PHY for non-data
            %   uplink transmission scheduled for the current slot. MAC
            %   sends it at the start of a UL slot for all the scheduled UL
            %   transmissions in the slot (except PUSCH, which is sent
            %   using dataTx function of this class).
            %

            if ~isempty(obj.SRSConfiguration)
                % Check if current slot is a slot with UL symbols. For FDD (Value 0),
                % there is no need to check as every slot is a UL slot. For
                % TDD (Value 1), check if current slot has any UL symbols
                if(obj.DuplexMode == 0 || ~isempty(find(obj.DLULSlotFormat(obj.CurrDLULSlotIndex + 1, :) == obj.ULType, 1)))
                    ulControlType = [];
                    ulControlPDUs = {};

                    srsLocations = obj.SRSConfiguration.SymbolStart : (obj.SRSConfiguration.SymbolStart+obj.SRSConfiguration.NumSRSSymbols-1); % SRS symbol locations
                    % Check whether the mode is FDD OR if it is TDD then all the SRS symbols must be UL symbols
                    if obj.DuplexMode == 0 || all(obj.DLULSlotFormat(obj.CurrDLULSlotIndex + 1, srsLocations+1) == obj.ULType)
                        % Set carrier configuration object
                        carrier = obj.CarrierConfigUL;
                        carrier.NSlot = obj.CurrSlot;
                        carrier.NFrame = obj.CurrFrame;
                        srsConfigUE = obj.SRSConfiguration;
                        % Check if the current slot is SRS transmission slot based on configured SRS periodicity and offset
                        if (~isnumeric(srsConfigUE.SRSPeriod) && srsConfigUE.SRSPeriod == "on") || ~mod(obj.NumSlotsFrame*obj.CurrFrame + obj.CurrSlot - srsConfigUE.SRSPeriod(2), srsConfigUE.SRSPeriod(1))
                            ulControlType(1) = 1; % SRS PDU
                            ulControlPDUs{1} = obj.SRSConfiguration;
                        end
                    end
                    obj.UlControlRequestFcn(ulControlType, ulControlPDUs, obj.TimestampInfo); % Send UL control request to PHY
                end
            end
        end

        function controlTx(obj)
            %controlTx Send BSR packet, PDSCH feedback and CQI report
            %   controlTx(OBJ) sends the buffer status report
            %   feedback for PDSCH receptions, and DL channel quality
            %   information. These are sent out-of-band to gNB's MAC
            %   without the need of frequency resources

            currentTime = obj.TimestampInfo.Timestamp;
            % Send BSR if its transmission periodicity reached
            if currentTime >= obj.NextBSRTime
                if obj.DuplexMode == 1 % TDD
                    % UL symbol is checked
                    if obj.DLULSlotFormat(obj.CurrDLULSlotIndex + 1, obj.CurrSymbol+1) == obj.ULType % UL symbol
                        bsrTx(obj);
                        obj.NextBSRTime = currentTime + obj.BSRPeriodicity*1e6;
                    else
                        % Find next UL symbol
                        obj.NextBSRTime = currentTime + getNextULSymbolTime(obj); % In nanoseconds
                    end
                else % For FDD, no need to check for UL symbol
                    bsrTx(obj);
                    obj.NextBSRTime = currentTime + obj.BSRPeriodicity*1e6;
                end
            end

            % Send PDSCH feedback (ACK/NACK), if scheduled
            symNumFrame = obj.CurrSlot*obj.NumSymbols + obj.CurrSymbol;
            feedback = -1*ones(obj.NumHARQ, 1);
            for harqIdx=1:obj.NumHARQ
                if obj.PDSCHRxFeedback(harqIdx, 1) == symNumFrame % If any feedback is scheduled in current symbol
                    feedback(harqIdx) = obj.PDSCHRxFeedback(harqIdx, 2); % Set the feedback (ACK/NACK)
                    obj.PDSCHRxFeedback(harqIdx, :) = [-1 -1 Inf]; % Clear the context
                end
            end
            if any(feedback ~=-1) % If any PDSCH feedback is scheduled to be sent
                % Construct packet information
                pktInfo = obj.PacketStruct;
                pktInfo.Data = feedback;
                pktInfo.Metadata.PacketType = obj.PDSCHFeedback;
                obj.TxOutofBandFcn(pktInfo); % Send the PDSCH feedback out-of-band to gNB's MAC
            end

            % Send CSI report(ri-pmi-cqi format) if the transmission periodicity has reached
            if currentTime >= obj.NextCSIReportTime
                obj.NextCSIReportTime = currentTime + obj.CSIReportPeriod(1)*obj.SlotDurationInNS; % In nanoseconds
                if obj.CSIMeasurementFlag
                    % Construct packet information
                    pktInfo = obj.PacketStruct;
                    pktInfo.Metadata.PacketType = obj.CSIReport;
                    csiReport.RankIndicator = obj.CSIMeasurement.RankIndicator;
                    csiReport.PMISet = obj.CSIMeasurement.PMISet;
                    csiReport.W = obj.CSIMeasurement.W;
                    csiReport.CQI = obj.CSIMeasurement.CQI;
                    pktInfo.Data = csiReport;
                    obj.TxOutofBandFcn(pktInfo); % Send the CSI report out-of-band to gNB's MAC
                end
            end

            % Send CSI report(CRI-RSRP format) if the transmission periodicity has reached
            if currentTime >= obj.NextRSRPReportTime
                obj.NextRSRPReportTime = currentTime + obj.CSIReportPeriodicityRSRP*obj.SlotDurationInNS; % In nanoseconds
                if isfield(obj.CSIMeasurement, 'CRI')
                    if (obj.DuplexMode == 1 && obj.DLULSlotFormat(obj.CurrDLULSlotIndex + 1, obj.CurrSymbol+1) == obj.ULType) || (obj.DuplexMode == 0)
                        criRSRPReport.CRI = obj.CSIMeasurement.CRI;
                        criRSRPReport.L1RSRP = obj.CSIMeasurement.L1RSRP;
                        % Construct packet information
                        pktInfo = obj.PacketStruct;
                        pktInfo.Data = criRSRPReport;
                        pktInfo.Metadata.PacketType = obj.CSIReportRSRP;
                        obj.TxOutofBandFcn(pktInfo); % Send the CSI report out-of-band to gNB's MAC
                        obj.NextRSRPReportTime = currentTime + obj.CSIReportPeriodicityRSRP*obj.SlotDurationInNS; % In nanoseconds
                    elseif obj.DuplexMode == 1
                        % Find next UL symbol
                        obj.NextRSRPReportTime = currentTime + getNextULSymbolTime(obj); % In nanoseconds
                    end
                end
            end
        end

        function controlRx(obj, packets)
            %controlRx Receive callback for uplink and downlink grants for this UE

            for pktIdx = 1:numel(packets)
                pktInfo = packets(pktIdx);
                if pktInfo.DirectToDestination ~= 0 && ...
                        pktInfo.Metadata.NCellID == obj.NCellID

                    if obj.RNTI ~= pktInfo.Metadata.RNTI
                        % Don't process the unintended packet
                        continue;
                    end

                    pktType = pktInfo.Metadata.PacketType;
                    switch(pktType)
                        case obj.ULGrant % Uplink grant received
                            uplinkGrant = pktInfo.Data;
                            % Store the uplink grant at the corresponding Tx start
                            % symbol. The uplink grant is later used for PUSCH
                            % transmission at the transmission time defined by
                            % uplink grant
                            numSymFrame = obj.NumSlotsFrame * obj.NumSymbols; % Number of symbols in 10 ms frame
                            txStartSymbol = mod((obj.CurrSlot + uplinkGrant.SlotOffset)*obj.NumSymbols + uplinkGrant.StartSymbol, numSymFrame);
                            % Store the grant at the PUSCH start symbol w.r.t the 10 ms frame
                            obj.UplinkTxContext{txStartSymbol + 1} = uplinkGrant;

                        case obj.DLGrant % Downlink grant received
                            downlinkGrant = pktInfo.Data;
                            % Store the downlink grant at the corresponding Rx start
                            % symbol. The downlink grant is later used for PDSCH
                            % reception at the reception time defined by
                            % downlink grant
                            numSymFrame = obj.NumSlotsFrame * obj.NumSymbols; % Number of symbols in 10 ms frame
                            rxStartSymbol = mod((obj.CurrSlot + downlinkGrant.SlotOffset)*obj.NumSymbols + downlinkGrant.StartSymbol, numSymFrame);
                            obj.DownlinkRxContext{rxStartSymbol + 1} = downlinkGrant; % Store the grant at the PDSCH start symbol w.r.t the 10 ms frame
                    end

                end
            end
        end

        function updateSRSPeriod(obj, srsPeriod)
            %updateSRSPeriod Update the SRS periodicity of UE

            obj.SRSConfiguration.SRSPeriod = srsPeriod;
            % Calculate unique SRS transmission time and periodicity
            obj.SRSTxInfo = calculateSRSPeriodicity(obj, obj.SRSConfiguration);
        end

        function flag = rxOn(obj, packet)
            %rxOn Returns whether Rx is scheduled to be on during the packet duration
            % Return value is 1 if packet is relevant, otherwise 0.

            flag = 1;
            % Check whether the packet overlaps (partially or fully) with scheduled
            % reception time of the node
            startTimeSec = packet.StartTime;
            lastRxTimeSec = obj.LastRxTime*1e-9;
            if (abs(startTimeSec - lastRxTimeSec) > 1e-9) && ...
                    ((startTimeSec - lastRxTimeSec) > (obj.SlotDurationInSec-1e-9) && ...
                    ((startTimeSec + packet.Duration) < ((obj.NextRxTime*1e-9)+1e-9)))
                flag = 0; % Rx is off on this node. Mark the packet as irrelevant.
            elseif packet.Abstraction==1 && packet.Metadata.PacketType==0 && obj.EnableAdvancedOptimization % For abstract-PHY data packet
                % If rx is on for this node then check if the incoming
                % packet is sent on overlapping RBs as the ones on which
                % this node is expecting reception. If not, mark the packet
                % as irrelevant.
                flag = 0;
                rxPRBSet = obj.ReceptionRBSet;
                packetPRBSet = packet.Metadata.PacketConfig.PRBSet;
                if isempty(rxPRBSet) || (rxPRBSet(end)<packetPRBSet(1) || packetPRBSet(end)<rxPRBSet(1))
                    % Assuming that RBs in both RB sets are sorted, packet
                    % is irrelevant if last RB in rxPRBSet is less than
                    % first RB in packetPRBSet or vice-versa.
                    return;
                end
                numRxPRB = numel(rxPRBSet);
                numPRBPacket = numel(packetPRBSet);
                for i=1:numRxPRB
                    rxRB = rxPRBSet(i);
                    for j=1:numPRBPacket
                        packetRB = packetPRBSet(j);
                        if packetRB==rxRB
                            flag=1; % Packet overlaps with reception RB
                            return;
                        elseif packetRB>rxRB
                            break;
                        end
                    end
                end
            end
        end
    end

    methods (Access = protected)
        function [pduLen, type] = sendMACPDU(obj, uplinkGrant)
            %sendMACPDU Send MAC PDU as per the parameters of the uplink grant
            % Uplink grant and its parameters were sent beforehand by gNB
            % in uplink grant. Based on the NDI received in the uplink
            % grant, either the packet in the HARQ buffer would be retransmitted
            % or a new MAC packet would be sent

            macPDU = [];
            % Populate PUSCH information to be sent to PHY, along with the
            % MAC PDU. For runtime optimization, only set a field if its
            % value is different from last PUSCH
            puschConfig = obj.PUSCHInfo.PUSCHConfig;
            ulGrantRBs = uplinkGrant.PRBSet;
            numResourceBlocks = numel(ulGrantRBs);
            obj.PUSCHInfo.PUSCHConfig.PRBSet = ulGrantRBs;
            % Get the corresponding row from the mcs table
            mcsInfo = obj.MCSTableUL(uplinkGrant.MCSIndex + 1, :);
            modSchemeBits = mcsInfo(1); % Bits per symbol for modulation scheme (stored in column 1)
            obj.PUSCHInfo.TargetCodeRate = mcsInfo(2)/1024; % Coderate (stored in column 2)
            modScheme = nr5g.internal.getModulationScheme(modSchemeBits);
            if puschConfig.Modulation ~= modScheme(1)
                obj.PUSCHInfo.PUSCHConfig.Modulation = modScheme(1);
            end
            if uplinkGrant.StartSymbol ~= puschConfig.SymbolAllocation(1) || uplinkGrant.NumSymbols ~= puschConfig.SymbolAllocation(2)
                obj.PUSCHInfo.PUSCHConfig.SymbolAllocation = [uplinkGrant.StartSymbol uplinkGrant.NumSymbols];
            end
            obj.PUSCHInfo.NSlot = obj.CurrSlot;
            obj.PUSCHInfo.HARQID = uplinkGrant.HARQID;
            obj.PUSCHInfo.RV = uplinkGrant.RV;
            if puschConfig.NumLayers ~= uplinkGrant.NumLayers
                obj.PUSCHInfo.PUSCHConfig.NumLayers = uplinkGrant.NumLayers;
            end
            if puschConfig.NumAntennaPorts ~= uplinkGrant.NumAntennaPorts
                obj.PUSCHInfo.PUSCHConfig.NumAntennaPorts = uplinkGrant.NumAntennaPorts;
            end
            obj.PUSCHInfo.PUSCHConfig.TPMI = uplinkGrant.TPMI;
            if puschConfig.MappingType ~= uplinkGrant.MappingType
                obj.PUSCHInfo.PUSCHConfig.MappingType = uplinkGrant.MappingType;
            end
            if uplinkGrant.MappingType == 'A'
                dmrsAdditonalPos = obj.PUSCHDMRSAdditionalPosTypeA;
            else
                dmrsAdditonalPos = obj.PUSCHDMRSAdditionalPosTypeB;
            end
            if puschConfig.DMRS.DMRSAdditionalPosition ~= dmrsAdditonalPos
                obj.PUSCHInfo.PUSCHConfig.DMRS.DMRSAdditionalPosition = dmrsAdditonalPos;
            end
            if puschConfig.DMRS.DMRSLength ~= uplinkGrant.DMRSLength
                obj.PUSCHInfo.PUSCHConfig.DMRS.DMRSLength = uplinkGrant.DMRSLength;
            end
            if puschConfig.DMRS.NumCDMGroupsWithoutData ~= uplinkGrant.NumCDMGroupsWithoutData
                obj.PUSCHInfo.PUSCHConfig.DMRS.NumCDMGroupsWithoutData = uplinkGrant.NumCDMGroupsWithoutData;
            end

            % Carrier configuration
            carrierConfig = obj.CarrierConfigUL;
            carrierConfig.NSlot = obj.PUSCHInfo.NSlot;

            uplinkGrantHARQId =  uplinkGrant.HARQID;
            pduLen = uplinkGrant.TBS;  % In bytes
            lastNDI = obj.HARQNDIUL(uplinkGrantHARQId+1); % Last receive NDI for this HARQ process
            if uplinkGrant.NDI ~= lastNDI
                % NDI has been toggled, so send a new MAC packet. This acts
                % as an ACK for the last sent packet of this HARQ process,
                % in addition to acting as an uplink grant
                type = 'newTx';
                % Generate MAC PDU
                macPDU = constructMACPDU(obj, pduLen);
                % Store the uplink grant NDI for this HARQ process which
                % will be used in taking decision of 'newTx' or 'reTx' when
                % an uplink grant for the same HARQ process comes
                obj.HARQNDIUL(uplinkGrantHARQId+1) = uplinkGrant.NDI; % Update NDI
                obj.StatULTransmissionRB = obj.StatULTransmissionRB + numResourceBlocks;
            else
                type = 'reTx';
                obj.StatULRetransmissionRB = obj.StatULRetransmissionRB + numResourceBlocks;
            end

            obj.PUSCHInfo.TBS = pduLen;
            obj.TxDataRequestFcn(obj.PUSCHInfo, macPDU, obj.TimestampInfo);
        end

        function rxRequestToPHY(obj, downlinkGrant)
            % Send Rx request to PHY

            % Fill information to be passed to PHY for PDSCH reception. For
            % runtime optimization, only set a field if its value is
            % different from last PDSCH
            pdschConfig = obj.PDSCHInfo.PDSCHConfig;
            dlGrantRBs = downlinkGrant.PRBSet;
            % Update only if ReceptionRBSet is not already set w.r.t
            % CSI-RS. CSI-RS is full-bandwidth hence would be superset of
            % any PDSCH RB set. This condition saves from overwriting a
            % RB superset with a smaller subset.
            if isempty(obj.ReceptionRBSet)
                obj.ReceptionRBSet = dlGrantRBs; % Populate the reception RB set for this slot
                rxEndTime = obj.TimestampInfo.Timestamp+obj.SlotDurationInNS;
                if rxEndTime>obj.ReceptionEndTime
                    obj.ReceptionEndTime = rxEndTime;
                end
            end
            numResourceBlocks = numel(dlGrantRBs);
            obj.PDSCHInfo.PDSCHConfig.PRBSet = dlGrantRBs;
            % Get the corresponding row from the mcs table
            mcsInfo = obj.MCSTableDL(downlinkGrant.MCSIndex + 1, :);
            modSchemeBits = mcsInfo(1); % Bits per symbol for modulation scheme(stored in column 1)
            obj.PDSCHInfo.TargetCodeRate = mcsInfo(2)/1024; % Coderate (stored in column 2)
            modScheme = nr5g.internal.getModulationScheme(modSchemeBits);
            if pdschConfig.Modulation ~= modScheme(1)
                obj.PDSCHInfo.PDSCHConfig.Modulation = modScheme(1);
            end
            if downlinkGrant.StartSymbol ~= pdschConfig.SymbolAllocation(1) || downlinkGrant.NumSymbols ~= pdschConfig.SymbolAllocation(2)
                obj.PDSCHInfo.PDSCHConfig.SymbolAllocation = [downlinkGrant.StartSymbol downlinkGrant.NumSymbols];
            end
            obj.PDSCHInfo.NSlot = obj.CurrSlot;
            if downlinkGrant.NumLayers ~= pdschConfig.NumLayers
                obj.PDSCHInfo.PDSCHConfig.NumLayers = downlinkGrant.NumLayers;
            end
            if downlinkGrant.MappingType ~= pdschConfig.MappingType
                obj.PDSCHInfo.PDSCHConfig.MappingType = downlinkGrant.MappingType;
            end
            if downlinkGrant.MappingType == 'A'
                dmrsAdditonalPos = obj.PDSCHDMRSAdditionalPosTypeA;
            else
                dmrsAdditonalPos = obj.PDSCHDMRSAdditionalPosTypeB;
            end
            if dmrsAdditonalPos ~= pdschConfig.DMRS.DMRSAdditionalPosition
                obj.PDSCHInfo.PDSCHConfig.DMRS.DMRSAdditionalPosition = dmrsAdditonalPos;
            end
            if downlinkGrant.DMRSLength ~= pdschConfig.DMRS.DMRSLength
                obj.PDSCHInfo.PDSCHConfig.DMRS.DMRSLength = downlinkGrant.DMRSLength;
            end
            if downlinkGrant.NumCDMGroupsWithoutData ~= pdschConfig.DMRS.NumCDMGroupsWithoutData
                obj.PDSCHInfo.PDSCHConfig.DMRS.NumCDMGroupsWithoutData = downlinkGrant.NumCDMGroupsWithoutData;
            end

            % Carrier configuration
            carrierConfig = obj.CarrierConfigDL;
            carrierConfig.NSlot = obj.PDSCHInfo.NSlot;
            carrierConfig.NFrame = obj.CurrFrame;

            obj.PDSCHInfo.TBS = downlinkGrant.TBS;  % In bytes
            if obj.HARQNDIDL(downlinkGrant.HARQID+1) ~= downlinkGrant.NDI % NDI toggled: new transmission
                obj.PDSCHInfo.NewData = 1;
                obj.StatDLTransmissionRB = obj.StatDLTransmissionRB + numResourceBlocks;
            else % Retransmission
                obj.PDSCHInfo.NewData = 0;
                obj.StatDLRetransmissionRB = obj.StatDLRetransmissionRB + numResourceBlocks;
            end

            obj.HARQNDIDL(downlinkGrant.HARQID+1) = downlinkGrant.NDI; % Update the stored NDI for HARQ process
            obj.PDSCHInfo.HARQID = downlinkGrant.HARQID;
            obj.PDSCHInfo.RV = downlinkGrant.RV;

            % Set reserved REs information. Generate 0-based
            % carrier-oriented CSI-RS indices in linear indexed form
            obj.PDSCHInfo.PDSCHConfig.ReservedRE = [];
            for csirsIdx = 1:length(obj.CSIRSConfiguration)
                csirsConfig = obj.CSIRSConfiguration(csirsIdx);
                csirsLocations = csirsConfig.SymbolLocations; % CSI-RS symbol locations
                % (Mode is FDD) OR (Mode is TDD And CSI-RS symbols are DL symbols)
                if obj.DuplexMode == 0 || all(obj.DLULSlotFormat(obj.CurrDLULSlotIndex + 1, csirsLocations+1) == obj.DLType)
                    % Check if the current slot is CSI-RS reception slot based on configured CSI-RS periodicity and offset
                    if (~isnumeric(csirsConfig.CSIRSPeriod) && csirsConfig.CSIRSPeriod == "on") || ~mod(obj.NumSlotsFrame*obj.CurrFrame + obj.CurrSlot - csirsConfig.CSIRSPeriod(2), csirsConfig.CSIRSPeriod(1))
                        obj.PDSCHInfo.PDSCHConfig.ReservedRE = [obj.PDSCHInfo.PDSCHConfig.ReservedRE ; ...
                            nrCSIRSIndices(carrierConfig, csirsConfig, 'IndexBase', '0based')]; % Reserve CSI-RS REs
                    end
                end
            end
            for idx = 1:length(obj.CSIRSConfigurationRSRP)
                csirsConfig = obj.CSIRSConfiguration(csirsIdx);
                csirsLocations = csirsConfig.SymbolLocations; % CSI-RS symbol locations
                % (Mode is FDD) OR (Mode is TDD And CSI-RS symbols are DL symbols)
                if obj.DuplexMode == 0 || all(obj.DLULSlotFormat(obj.CurrDLULSlotIndex + 1, csirsLocations+1) == obj.DLType)
                    % Check if the current slot is CSI-RS reception slot based on configured CSI-RS periodicity and offset
                    if (~isnumeric(csirsConfig.CSIRSPeriod) && csirsConfig.CSIRSPeriod == "on") || ~mod(obj.NumSlotsFrame*obj.CurrFrame + obj.CurrSlot - csirsConfig.CSIRSPeriod(2), csirsConfig.CSIRSPeriod(1))
                        obj.PDSCHInfo.PDSCHConfig.ReservedRE = [obj.PDSCHInfo.PDSCHConfig.ReservedRE ; ...
                            nrCSIRSIndices(carrierConfig, csirsConfig, 'IndexBase', '0based')]; % Reserve CSI-RS REs
                    end
                end
            end

            % Call PHY to start receiving PDSCH
            obj.RxDataRequestFcn(obj.PDSCHInfo, obj.TimestampInfo);
        end

        function bsrTx(obj)
            %bsrTx Construct and send a BSR

            % Construct BSR
            [lcid, bsr] = nrMACBSR(obj.LCGBufferStatus);

            % Generate the subPDU
            subPDU = nrMACSubPDU(obj.ULType, lcid, bsr);
            % Construct packet information
            pktInfo = obj.PacketStruct;
            pktInfo.Data = subPDU;
            pktInfo.Metadata.PacketType = obj.BSR;
            obj.TxOutofBandFcn(pktInfo); % Send the BSR out-of-band to gNB's MAC
        end

        function nextInvokeTime = getNextInvokeTime(obj, currentTime)
            %getNextInvokeTime Return the next invoke time in nanoseconds

            nextInvokeTime = Inf;
            if ~isempty(obj.RNTI) % If UE is connected to gNB
                % Find the duration completed in the current symbol
                durationCompletedInCurrSlot = mod(currentTime, obj.SlotDurationInNS);
                currSymDurCompleted = obj.SymbolDurationsInSlot(obj.CurrSymbol+1) - obj.SymbolEndTimesInSlot(obj.CurrSymbol+1) + durationCompletedInCurrSlot;

                symbolNumFrame = obj.CurrSlot*obj.NumSymbols + obj.CurrSymbol;
                totalSymbols = obj.NumSymInFrame;
                nextInvokeTime = Inf;

                % Next Tx start symbol
                nextTxSymbol = find(~cellfun('isempty',obj.UplinkTxContext(symbolNumFrame+2:totalSymbols)), 1);
                if isempty(nextTxSymbol)
                    nextTxSymbol = (totalSymbols-symbolNumFrame-1) + find(~cellfun('isempty',obj.UplinkTxContext(1:symbolNumFrame)), 1);
                end

                % Next Rx start symbol
                nextRxSymbol = find(~cellfun('isempty',obj.DownlinkRxContext(symbolNumFrame+2:totalSymbols)), 1);
                if isempty(nextRxSymbol)
                    nextRxSymbol = (totalSymbols-symbolNumFrame-1) + find(~cellfun('isempty',obj.DownlinkRxContext(1:symbolNumFrame)), 1);
                end
                nextInvokeSymbol = min([Inf nextTxSymbol nextRxSymbol]);
                if nextInvokeSymbol ~= Inf
                    nextInvokeTime = currentTime +  durationToSymNum(obj, nextInvokeSymbol) - currSymDurCompleted;
                end

                % Set next Rx time
                if ~isempty(nextRxSymbol)
                    obj.NextRxTime = currentTime + durationToSymNum(obj, nextRxSymbol) - currSymDurCompleted;
                else
                    obj.NextRxTime = Inf;
                end

                % Next PDSCH feedback Tx symbol
                pdschFeedbackTxSym = obj.PDSCHRxFeedback(:, 1);
                pdschFeedbackTxSym = pdschFeedbackTxSym(pdschFeedbackTxSym~=-1);
                nextPDSCHFeedbackTime = Inf;
                if ~isempty(pdschFeedbackTxSym)
                    nextPDSCHFeedbackTxSym = min(pdschFeedbackTxSym(pdschFeedbackTxSym>symbolNumFrame)) - symbolNumFrame;
                    if isempty(nextPDSCHFeedbackTxSym)
                        nextPDSCHFeedbackTxSym = (totalSymbols-symbolNumFrame) + min(pdschFeedbackTxSym);
                    end
                    nextPDSCHFeedbackTime = currentTime + durationToSymNum(obj, nextPDSCHFeedbackTxSym) - currSymDurCompleted;
                end

                % Next control transmission time
                controlTxStartTime = min([obj.NextRSRPReportTime obj.NextBSRTime obj.NextCSIReportTime obj.SRSTxInfo(2) nextPDSCHFeedbackTime]);
                % Next control reception time
                controlRxStartTime = min(obj.CSIRSRxInfo(:, 2));
                obj.NextRxTime = min([obj.NextRxTime controlRxStartTime]);
                nextInvokeTime = min([nextInvokeTime controlTxStartTime controlRxStartTime]);
            end
        end

        function duration = getNextULSymbolTime(obj)
            %getNextULSymbolTime Return the time to next UL symbol occurrence

            % Find the first UL symbol in the current slot
            firstULSymbol = find(obj.DLULSlotFormat(obj.CurrDLULSlotIndex + 1, obj.CurrSymbol+2:end) == obj.ULType, 1);
            if isempty(firstULSymbol) % Find the first UL symbol in the future slots
                duration = sum(obj.SymbolDurationsInSlot(obj.CurrSymbol+1:end));
                numDLULSlots = size(obj.DLULSlotFormat, 1);
                for idx=1:numDLULSlots
                    dlulSlotIndex = mod(obj.CurrDLULSlotIndex+idx, numDLULSlots);
                    firstULSymbol = find(obj.DLULSlotFormat(dlulSlotIndex+1, 1:obj.NumSymbols) == obj.ULType, 1);
                    if isempty(firstULSymbol)
                        duration = duration + obj.SlotDurationInNS;
                    else
                        duration = duration + sum(obj.SymbolDurationsInSlot(1:firstULSymbol-1));
                        break;
                    end
                end
            else % First UL symbol is present in the current slot
                duration = sum(obj.SymbolDurationsInSlot(obj.CurrSymbol+1:obj.CurrSymbol+firstULSymbol));
            end
        end
    end
end
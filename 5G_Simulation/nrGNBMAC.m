classdef nrGNBMAC < nr5g.internal.nrMAC
    %nrGNBMAC Implements gNB MAC functionality
    %   The class implements the gNB MAC and its interactions with RLC and PHY
    %   for Tx and Rx chains. Both, frequency division duplex (FDD) and time
    %   division duplex (TDD) modes are supported. It contains scheduler entity
    %   which takes care of uplink (UL) and downlink (DL) scheduling. Using the
    %   output of UL and DL schedulers, it implements transmission of UL and DL
    %   assignments. UL and DL assignments are sent out-of-band from MAC itself
    %   (without using frequency resources and with guaranteed reception), as
    %   physical downlink control channel (PDCCH) is not modeled. Physical
    %   uplink control channel (PUCCH) is not modeled too, so the control
    %   packets from UEs: buffer status report (BSR), PDSCH feedback, and DL
    %   channel state information (CSI) report are also received out-of-band.
    %   Hybrid automatic repeat request (HARQ) control mechanism to enable
    %   retransmissions is implemented. MAC controls the HARQ processes
    %   residing in physical layer
    %
    %   Note: This is an internal undocumented class and its API and/or
    %   functionality may change in subsequent releases.

    %   Copyright 2022-2025 The MathWorks, Inc.

    properties(Hidden)
        %Scheduler Scheduler object
        Scheduler

        %RxContextFeedback Rx context at gNB used for feedback reception (ACK/NACK) of PDSCH transmissions
        % N-by-P-by-K cell array where 'N' is the number of UEs, 'P' is the
        % number of symbols in a 10 milliseconds (ms) frame and K is the number of
        % downlink HARQ processes. This is used by gNB in the reception of
        % ACK/NACK from UEs. An element at index (i, j, k) in this array,
        % stores the downlink grant for the UE with RNTI 'i' where
        % 'j' is the symbol number from the start of the frame where
        % ACK/NACK is expected for UE's HARQ process number 'k'
        RxContextFeedback
    end

    properties(SetAccess = protected, Hidden)
        %NumResourceBlocks Number of resource blocks (RB) in the uplink and
        %downlink bandwidth
        NumResourceBlocks

        %DuplexModeNumber Duplexing mode (FDD or TDD)
        % Value 0 means FDD and 1 means TDD
        DuplexModeNumber

        %NumDLULPatternSlots Number of slots in DL-UL pattern (for TDD mode)
        NumDLULPatternSlots

        %DLULSlotFormat Format of the slots in DL-UL pattern (for TDD mode)
        % N-by-14 matrix where 'N' is number of slots in DL-UL pattern. Each row
        % contains the symbol type of the 14 symbols in the slot. Value 0, 1 and 2
        % represent DL symbol, UL symbol, guard symbol, respectively.
        DLULSlotFormat

        %MCSTableDL MCS table used for downlink. It contains the mapping of MCS
        %indices with Modulation and Coding schemes
        MCSTableDL

        %MCSTableUL MCS table used for uplink. It contains the mapping of MCS
        %indices with Modulation and Coding schemes
        MCSTableUL

        %DMRSTypeAPosition Position of DM-RS in type A transmission (2 or 3)
        DMRSTypeAPosition = 2;

        %PUSCHDMRSConfigurationType PUSCH DM-RS configuration type (1 or 2)
        PUSCHDMRSConfigurationType = 1;

        %PDSCHDMRSConfigurationType PDSCH DM-RS configuration type (1 or 2)
        PDSCHDMRSConfigurationType= 1;

        %PDSCHDMRSAdditionalPosTypeA Additional PDSCH DM-RS positions for type A (0..3)
        PDSCHDMRSAdditionalPosTypeA = 0;

        %PDSCHDMRSAdditionalPosTypeB Additional PDSCH DM-RS positions for type B (0 or 1)
        PDSCHDMRSAdditionalPosTypeB = 0;

        %PUSCHDMRSAdditionalPosTypeA Additional PUSCH DM-RS positions for type A (0..3)
        PUSCHDMRSAdditionalPosTypeA = 0;

        %PUSCHDMRSAdditionalPosTypeB Additional PUSCH DM-RS positions for type B (0..3)
        PUSCHDMRSAdditionalPosTypeB = 0;

        %NumPDSCHNACKs The total count of NACKs received for the transmitted PDSCHs
        NumPDSCHNACKs = 0
    end

    properties(SetAccess = protected)
        %UEs RNTIs of the UEs connected to the gNB
        UEs

        %ScheduledResources Structure containing information about scheduling event
        ScheduledResources

        %UEInfo Information about the UEs connected to the GNB
        % N-by-1 array where 'N' is the number of UEs. Each element in the
        % array is a structure with two fields:
        %   ID - Node id of the UE
        %   Name - Node Name of the UE
        UEInfo

        % CSIRSMapping A mapping of CSI-RS configuration and corresponding UEs
        % Structure array with the number of elements equal to the number
        % of CSI-RS configurations on the gNB. Each structure element has
        % two fields:
        %  CSIRSConfig - The CSI-RS configuration.
        %  RNTI - A list of UEs (identified by their RNTIs) associated with the CSIRSConfig
        CSIRSMapping
    end

    properties (Access = protected)
        %DownlinkTxContext Tx context used for PDSCH transmissions
        % N-by-P cell array where is N is number of UEs and 'P' is number of
        % symbols in a 10 ms frame. An element at index (i, j) stores the
        % downlink grant for UE with RNTI 'i' with PDSCH transmission scheduled to
        % start at symbol 'j' from the start of the frame. If no PDSCH
        % transmission scheduled, cell element is empty
        DownlinkTxContext

        %UplinkRxContext Rx context used for PUSCH reception
        % N-by-P cell array where 'N' is the number of UEs and 'P' is the
        % number of symbols in a 10 ms frame. It stores uplink resource
        % assignment details done to UEs. This is used by gNB in the
        % reception of uplink packets. An element at position (i, j) stores
        % the uplink grant corresponding to a PUSCH reception expected from
        % UE with RNTI 'i' starting at symbol 'j' from the start of the frame. If
        % there is no assignment, cell element is empty
        UplinkRxContext

        %CSIRSTxInfo Contains the information about CSI-RS transmissions
        % It is an array of size N-by-2 where N is the number of unique
        % CSI-RS periodicity, slot offset pairs configured for the UEs. Each
        % row of the array contains CSI-RS transmission periodicity (in
        % nanoseconds) and the next absolute transmission start time (in
        % nanoseconds) to the UEs.
        CSIRSTxInfo = [Inf 0]

        %SRSRxInfo Contains the information about SRS receptions
        % It is an array of size N-by-2 where N is the number of unique
        % SRS periodicity, slot offset pairs configured for the UEs. Each
        % row of the array contains SRS reception periodicity (in
        % nanoseconds) and the next absolute reception start time (in
        % nanoseconds) from the UEs.
        SRSRxInfo = [Inf 0]

        %SchedulerNextInvokeTime Time (in nanoseconds) at which scheduler will get invoked next time
        SchedulerNextInvokeTime = 0;

        %ULGrantFieldNames Contains the field names in the uplink grant
        ULGrantFieldNames;

        %ULGrantFieldNamesCount Stores the total number of field names in the uplink grant
        ULGrantFieldNamesCount;

        %DLGrantFieldNames Contains the field names in the downlink grant
        DLGrantFieldNames;

        %DLGrantFieldNames Stores the total number of field names in the downlink grant
        DLGrantFieldNamesCount;

        %NumULRBsUsed The cumulative count of resource blocks utilized
        %for uplink transmissions.
        NumULRBsUsed = 0;

        %NumDLRBsUsed The cumulative count of resource blocks utilized
        %for downlink transmissions.
        NumDLRBsUsed = 0;

        %DLSlotAvailability The proportion of time that the carrier is
        %accessible for downlink transmissions, expressed as a percentage.
        DLSlotAvailability = 1;

        %ULSlotAvailability The proportion of time that the carrier is
        %accessible for uplink transmissions, expressed as a percentage.
        ULSlotAvailability = 1;
    end

    methods
        function obj = nrGNBMAC(param, notificationFcn)
            %nrGNBMAC Construct a gNB MAC object
            %
            % PARAM is a structure including the following fields:
            %   NCellID            - Physical cell ID. Values: 0 to 1007 (TS 38.211, sec 7.4.2.1)
            %   SubcarrierSpacing  - Subcarrier spacing used
            %   NumHARQ            - Number of HARQ processes
            %   NumResourceBlocks  - Number of resource blocks
            %   DuplexMode         - Duplexing mode
            %   DLULConfigTDD      - TDD specific configuration
            %
            % NOTIFICATIONFCN - It is a handle of the node's processEvents
            % method

            obj.NotificationFcn = notificationFcn;
            obj.MACType = 0; % gNB MAC type
            obj.NCellID = param.NCellID;
            obj.SubcarrierSpacing = param.SubcarrierSpacing;
            obj.NumHARQ = param.NumHARQ;
            obj.NumResourceBlocks = param.NumResourceBlocks;
            obj.DuplexModeNumber = 0;
            obj.CSIRSMapping = repmat(struct('CSIRSConfig',[],'RNTI',[]),1,0);
            slotDuration = 1/(obj.SubcarrierSpacing/15); % In ms
            if strcmp(param.DuplexMode, "TDD")
                obj.DuplexModeNumber = 1;
                configTDD = param.DLULConfigTDD;
                numDLULPatternSlots = configTDD.DLULPeriodicity/slotDuration;
                obj.NumDLULPatternSlots = numDLULPatternSlots;
                numDLSlots = configTDD.NumDLSlots;
                obj.DLSlotAvailability = (numDLSlots+1)/numDLULPatternSlots;
                numDLSymbols = configTDD.NumDLSymbols;
                numULSlots = configTDD.NumULSlots;
                obj.ULSlotAvailability = (numULSlots+1)/numDLULPatternSlots;
                numULSymbols = configTDD.NumULSymbols;
                % All the remaining symbols in DL-UL pattern are assumed to be guard symbols
                guardDuration = (numDLULPatternSlots * 14) - ...
                    (((numDLSlots + numULSlots)*14) + ...
                    numDLSymbols + numULSymbols);

                % Set format of slots in the DL-UL pattern. Value 0, 1 and 2 means symbol
                % type as DL, UL and guard, respectively
                obj.DLULSlotFormat = obj.GuardType * ones(numDLULPatternSlots, 14);
                obj.DLULSlotFormat(1:numDLSlots, :) = obj.DLType; % Mark all the symbols of full DL slots as DL
                obj.DLULSlotFormat(numDLSlots + 1, 1 : numDLSymbols) = obj.DLType; % Mark DL symbols following the full DL slots
                obj.DLULSlotFormat(numDLSlots + floor(guardDuration/14) + 1, (numDLSymbols + mod(guardDuration, 14) + 1) : end)  ...
                    = obj.ULType; % Mark UL symbols at the end of slot before full UL slots
                obj.DLULSlotFormat((end - numULSlots + 1):end, :) = obj.ULType; % Mark all the symbols of full UL slots as UL type
            end

            obj.SlotDurationInNS = slotDuration * 1e6; % In nanoseconds
            obj.SlotDurationInSec = slotDuration * 1e-3;
            obj.NumSlotsFrame = 10/slotDuration; % Number of slots in a 10 ms frame
            obj.NumSymInFrame = obj.NumSlotsFrame*obj.NumSymbols;
            % Calculate symbol end times (in nanoseconds) in a slot for the
            % given SCS
            obj.SymbolEndTimesInSlot = round(((1:obj.NumSymbols)*slotDuration)/obj.NumSymbols, 4) * 1e6;
            % Duration of each symbol (in nanoseconds)
            obj.SymbolDurationsInSlot = obj.SymbolEndTimesInSlot(1:obj.NumSymbols) - [0 obj.SymbolEndTimesInSlot(1:13)];

            % No SRS resource
            obj.SRSRxInfo = Inf(1, 2);
            % Create carrier configuration object for UL
            obj.CarrierConfigUL = nrCarrierConfig("SubcarrierSpacing",obj.SubcarrierSpacing);

            % Resource scheduling event data
            obj.ScheduledResources = struct('CurrentTime', 0, ...
                'TimingInfo', [0 0 0], ...
                'ULGrants', struct([]), ...
                'DLGrants', struct([]));

            obj.PacketStruct.Type= 2; % 5G packet
            obj.PacketStruct.Metadata = struct('NCellID', obj.NCellID, 'RNTI', [], 'PacketType', []);

            % Store the uplink and downlink grant related information
            obj.ULGrantFieldNames = fieldnames(obj.UplinkGrantStruct);
            obj.ULGrantFieldNamesCount = numel(obj.ULGrantFieldNames);
            obj.DLGrantFieldNames = fieldnames(obj.DownlinkGrantStruct);
            obj.DLGrantFieldNamesCount = numel(obj.DLGrantFieldNames);

            % Fill NID at the set-up time (for runtime optimization)
            obj.PDSCHInfo.PDSCHConfig.NID = obj.NCellID;
            obj.PUSCHInfo.PUSCHConfig.NID = obj.NCellID;

            % Set the MCS tables as matrices
            obj.MCSTableUL = nr5g.internal.MACConstants.MCSTable;
            obj.MCSTableDL = nr5g.internal.MACConstants.MCSTable;

            obj.PDSCHInfo.PDSCHConfig.DMRS = nrPDSCHDMRSConfig(DMRSConfigurationType=obj.PDSCHDMRSConfigurationType, ...
                DMRSTypeAPosition=obj.DMRSTypeAPosition);
            obj.PUSCHInfo.PUSCHConfig.DMRS = nrPUSCHDMRSConfig(DMRSConfigurationType=obj.PUSCHDMRSConfigurationType, ...
                DMRSTypeAPosition=obj.DMRSTypeAPosition);

            % Create carrier configuration object for DL and UL
            obj.CarrierConfigUL.NSizeGrid = obj.NumResourceBlocks;
            obj.CarrierConfigDL = obj.CarrierConfigUL;
        end

        function addConnection(obj, ueInfo)
            %addConnection Configures the GNB MAC with UE connection information
            %
            % connectionInfo is a structure including the following fields:
            %
            % RNTI                     - Radio network temporary identifier
            %                            specified within [1, 65522]. Refer
            %                            table 7.1-1 in 3GPP TS 38.321 version 18.1.0.
            % UEID                     - Node ID of the UE
            % UEName                   - Node name of the UE
            % CSIRSConfiguration       - CSI-RS configuration information as an
            %                            object of type nrCSIRSConfig.
            % CSIRSConfigurationRSRP   - CSI-RS resource set configurations corresponding to the SSB directions.
            %                            It is an array of length N-by-1 where 'N' is
            %                            the number of maximum number of SSBs in a SSB
            %                            burst. Each element of the array at index 'i'
            %                            corresponds to the CSI-RS resource set
            %                            associated with SSB 'i-1'. The number of
            %                            CSI-RS resources in each resource set is same
            %                            for all configurations.
            % SRSConfiguration         - SRS configuration information specified as an object of type nrSRSConfig

            obj.UEs = [obj.UEs ueInfo.RNTI];
            nodeInfo = struct('ID', ueInfo.UEID, 'Name', ueInfo.UEName);
            obj.UEInfo = [obj.UEInfo nodeInfo];
            if numel(obj.UEs) > 1
                obj.PDSCHInfo = [obj.PDSCHInfo obj.PDSCHInfo(1)];
                obj.PDSCHInfo(end).PDSCHConfig.RNTI = ueInfo.RNTI;
                obj.PUSCHInfo = [obj.PUSCHInfo obj.PUSCHInfo(1)];
                obj.PUSCHInfo(end).PUSCHConfig.RNTI = ueInfo.RNTI;
            end

            if isfield(ueInfo, 'CSIRSConfigurationRSRP') && ~isempty(ueInfo.CSIRSConfigurationRSRP)
                obj.CSIRSConfigurationRSRP = [obj.CSIRSConfigurationRSRP ueInfo.CSIRSConfigurationRSRP];
            end

            % Append CSI-RS configuration (only if it is unique)
            if ~isempty(ueInfo.CSIRSConfiguration)
                uniqueConfiguration = true;
                for idx = 1:numel(obj.CSIRSMapping)
                    if isequal(obj.CSIRSMapping(idx).CSIRSConfig, ueInfo.CSIRSConfiguration)
                        obj.CSIRSMapping(idx).RNTI = [obj.CSIRSMapping(idx).RNTI, ueInfo.RNTI];
                        uniqueConfiguration = false;
                        break;
                    end
                end
                if uniqueConfiguration
                    % Validate that the new configuration does not conflict
                    % with existing configurations based on CSIRSPeriod
                    newPeriod = ueInfo.CSIRSConfiguration.CSIRSPeriod(1);
                    newOffset = ueInfo.CSIRSConfiguration.CSIRSPeriod(2);
                    for idx = 1:numel(obj.CSIRSMapping)
                        existingPeriod = obj.CSIRSMapping(idx).CSIRSConfig.CSIRSPeriod(1);
                        existingOffset = obj.CSIRSMapping(idx).CSIRSConfig.CSIRSPeriod(2);
                        if mod(newPeriod, existingPeriod) == 0 || mod(existingPeriod, newPeriod) == 0
                            if mod(newOffset, existingPeriod) == existingOffset || mod(existingOffset, newPeriod) == newOffset
                                coder.internal.error('nr5g:nrGNB:ConflictingCSIRSPeriodicity', newPeriod, newOffset, existingPeriod, existingOffset);
                            end
                        end
                    end
                    newConfig = struct('CSIRSConfig', ueInfo.CSIRSConfiguration, 'RNTI', ueInfo.RNTI);
                    obj.CSIRSMapping = [obj.CSIRSMapping, newConfig];
                end
            end

            obj.SRSConfiguration = [obj.SRSConfiguration ueInfo.SRSConfiguration];
            % Update the MAC context after each UE is connected
            updateMACContext(obj);
        end

        function nextInvokeTime = run(obj, currentTime, packets)
            %run Run the gNB MAC layer operations and return the next invoke time in nanoseconds
            %   NEXTINVOKETIME = run(OBJ, CURRENTTIME, PACKETS) runs the
            %   MAC layer operations and returns the next invoke time.
            %
            %   NEXTINVOKETIME is the next invoke time (in nanoseconds) for
            %   MAC.
            %
            %   CURRENTTIME is the current time (in nanoseconds).
            %
            %   PACKETS are the received packets from other nodes.

            elapsedTime = currentTime - obj.LastRunTime; % In nanoseconds
            if currentTime > obj.LastRunTime
                % Update the LCP timers
                obj.ElapsedTimeSinceLastLCP  = obj.ElapsedTimeSinceLastLCP + round(elapsedTime*1e-6, 4);
                obj.LastRunTime = currentTime;

                % Find the current frame number
                obj.CurrFrame = floor(currentTime/obj.FrameDurationInNS);
                absoluteSlotNum = floor(currentTime/obj.SlotDurationInNS);
                % Current slot number in 10 ms frame
                obj.CurrSlot = mod(absoluteSlotNum, obj.NumSlotsFrame);

                if obj.DuplexModeNumber % TDD
                    % Current slot number in DL-UL pattern
                    obj.CurrDLULSlotIndex = mod(absoluteSlotNum, obj.NumDLULPatternSlots);
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

            % Send data Tx request to PHY for transmission(s) which is(are)
            % scheduled to start at current symbol. Construct and send the
            % DL MAC PDUs scheduled for current symbol to PHY
            dataTx(obj);

            % Send data Rx request to PHY for reception(s) which is(are) scheduled to start at current symbol
            dataRx(obj);

            % Run schedulers (UL and DL) and send the resource assignment information to the UEs.
            % Resource assignments returned by a scheduler (either UL or
            % DL) is empty, if either scheduler was not scheduled to run at
            % the current time or no resource got assigned
            if currentTime == obj.SchedulerNextInvokeTime % Run scheduler at slot boundary
                resourceAssignmentsUL = runULScheduler(obj);
                resourceAssignmentsDL = runDLScheduler(obj);
                % Check if UL/DL assignments are done
                if ~isempty(resourceAssignmentsUL) || ~isempty(resourceAssignmentsDL)
                    % Construct and send UL assignments and DL assignments to
                    % UEs. UL and DL assignments are assumed to be sent
                    % out-of-band without using any frequency-time resources,
                    % from gNB's MAC to UE's MAC
                    controlTx(obj, resourceAssignmentsUL, resourceAssignmentsDL);
                end
                obj.SchedulerNextInvokeTime = obj.SchedulerNextInvokeTime + obj.SlotDurationInNS;
            end

            % Send request to PHY for:
            % (i) Non-data transmissions scheduled in this slot (currently
            % only CSI-RS supported)
            % (ii) Non-data receptions scheduled in this slot (currently
            % only SRS supported)
            %
            % Send at the first symbol of the slot for all the non-data
            % transmissions/receptions scheduled in the entire slot
            idxList = find(obj.CSIRSTxInfo(:, 2) == currentTime);
            if ~isempty(idxList)
                dlControlRequest(obj);
                % Update the next CSI-RS Tx times
                obj.CSIRSTxInfo(idxList, 2) = obj.CSIRSTxInfo(idxList, 1) + currentTime;
            end
            idxList = find(obj.SRSRxInfo(:, 2) == currentTime);
            if ~isempty(idxList)
                ulControlRequest(obj);
                % Update the next SRS Rx times
                obj.SRSRxInfo(idxList, 2) = obj.SRSRxInfo(idxList, 1) + currentTime;
            end

            % Update the previous symbol to the current symbol in the frame
            obj.PreviousSymbol = symNumFrame;
            % Return the next invoke time for MAC
            nextInvokeTime = getNextInvokeTime(obj, currentTime);
        end

        function addScheduler(obj, scheduler)
            %addScheduler Add scheduler object to MAC
            %   addScheduler(OBJ, SCHEDULER) adds the scheduler to MAC.
            %
            %   SCHEDULER Scheduler object.

            obj.Scheduler = scheduler;
        end

        function rxIndication(obj, rxInfo, currentTime)
            %rxIndication Packet reception from PHY
            %   rxIndication(OBJ, RXINFO) receives a MAC PDU from
            %   PHY.
            %   RXINFO is a structure containing information about the
            %   reception.
            %       NodeID  - Node identifier.
            %       RNTI    - Radio network temporary identifier
            %       HARQID  - HARQ process identifier
            %       MACPDU  - It is a vector of decimal octets received from PHY.
            %       CRCFlag - It is the success(value as 0)/failure(value as 1)
            %                 indication from PHY.
            %       Tags    - Array of structures where each structure contains
            %                 these fields.
            %                 Name      - Name of the tag.
            %                 Value     - Data associated with the tag.
            %                 ByteRange - Specific range of bytes within the
            %                             packet to which the tag applies.
            %
            %   CURRENTTIME - Current time in nanoseconds

            isRxSuccess = ~rxInfo.CRCFlag; % CRC value 0 indicates successful reception

            % Notify PUSCH Rx result to scheduler for updating the HARQ context
            rxResultInfo.RNTI = rxInfo.RNTI;
            rxResultInfo.RxResult = isRxSuccess;
            rxResultInfo.HARQID = rxInfo.HARQID;
            handleULRxResult(obj.Scheduler, rxResultInfo);
            if isRxSuccess % Packet received is error free
                byteOffset = 0;
                % Utilize the soft error handling feature for PDU decode failures. This
                % approach prevents hard errors upon PDU decode failures and returns empty
                % values instead
                softErrorFlag = true;
                % Parse uplink MAC PDU
                [lcidList, sduList, subHeaderLengthList] = nrMACPDUDecode(rxInfo.MACPDU, obj.ULType, softErrorFlag);

                packetInfo = obj.HigherLayerPacketFormat;
                packetInfo.NodeID = rxInfo.NodeID;
                for sduIndex = 1:numel(lcidList)
                    byteOffset = byteOffset+subHeaderLengthList(sduIndex);
                    if lcidList(sduIndex) >=4 && lcidList(sduIndex) <= 32
                        packetInfo.Packet = sduList{sduIndex};
                        packetInfo.PacketLength = length(packetInfo.Packet);
                        packetInfo.Tags = wirelessnetwork.internal.packetTags.segment(rxInfo.Tags, [byteOffset+1 ...
                            byteOffset+packetInfo.PacketLength]);
                        obj.RLCRxFcn{rxInfo.RNTI, lcidList(sduIndex)}(packetInfo, currentTime);
                    end
                    byteOffset = byteOffset+length(sduList{sduIndex});
                end
                obj.StatReceivedBytes(rxResultInfo.RNTI) = obj.StatReceivedBytes(rxResultInfo.RNTI) + length(rxInfo.MACPDU);
                obj.StatReceivedPackets(rxResultInfo.RNTI) = obj.StatReceivedPackets(rxResultInfo.RNTI) + 1;
            end
        end

        function srsIndication(obj, csiMeasurement)
            %srsIndication Reception of SRS measurements from PHY
            %   srsIndication(OBJ, csiMeasurement) receives the UL channel
            %   measurements from PHY, measured on the configured SRS for the
            %   UE.
            %   csiMeasurement - It is a structure and contains following
            %   fields
            %       RNTI - UE corresponding to the SRS
            %       RankIndicator - Rank indicator
            %       TPMI - Measured transmitted precoding matrix indicator (TPMI)
            %       MCSIndex - UL MCS index corresponding to RANK and TPMI
            %       SRSBasedDLMeasurements - SRS based DL measurements
            %       When you set the CSIMeasurementSignalDL name-value argument
            %       of configureScheduler to "CSI-RS", SRSBasedDLMeasurements
            %       will be empty. When set to "SRS", SRSBasedDLMeasurements becomes a
            %       structure with the fields 'RI', 'W', and 'MCSIndex
            %           RI - Rank indicator
            %           W - Precoding matrix
            %           MCSIndex - DL MCS index corresponding to the rank and precoder

            updateChannelQualityUL(obj.Scheduler, csiMeasurement);

            % Update the DL CSI using SRS based DL measurements
            if ~isempty(csiMeasurement.SRSBasedDLMeasurements)
                updateChannelQualityDL(obj.Scheduler, csiMeasurement);
            end
        end

        function updateBufferStatus(obj, lchBufferStatus)
            %updateBufferStatus Update DL buffer status for UEs, as notified by RLC
            %
            %   updateBufferStatus(obj, LCHBUFFERSTATUS) updates the
            %   DL buffer status for a logical channel of specified UE
            %
            %   LCHBUFFERSTATUS is the report sent by RLC. It is a
            %   structure with 3 fields:
            %       RNTI - Specified UE
            %       LogicalChannelID - ID of logical channel
            %       BufferStatus - Pending amount of data in bytes for the
            %       specified logical channel of UE.

            updateLCBufferStatusDL(obj.Scheduler, lchBufferStatus);
            obj.LCHBufferStatus(lchBufferStatus.RNTI, lchBufferStatus.LogicalChannelID) = ...
                lchBufferStatus.BufferStatus;
        end

        function macStats = statistics(obj)
            %statistics Return the gNB MAC statistics for each UE
            %
            %   MACSTATS = statistics(OBJ) Returns the MAC statistics of
            %   each UE at gNB MAC
            %
            %   MACSTATS - Nx1 array of structures, where N is the number
            %   of UEs. Each structure contains following fields.
            %       UEID                 - Node ID of the UE
            %       UEName               - Node name of the UE
            %       RNTI                 - RNTI of the UE
            %       TransmittedPackets   - Number of packets transmitted in DL
            %                              corresponding to new transmissions
            %       TransmittedBytes     - Number of bytes transmitted in DL
            %                              corresponding to new transmissions
            %       ReceivedPackets      - Number of packets received in UL
            %       ReceivedBytes        - Number of bytes received in UL
            %       Retransmissions      - Number of retransmission indications in DL
            %       RetransmissionBytes  - Number of bytes corresponding to retransmissions
            %                              retransmitted in DL

            numUEs = numel(obj.UEs);
            macStats = repmat(struct('UEID', 0, 'UEName', 0, 'RNTI', 0, 'TransmittedPackets', 0, ...
                'TransmittedBytes', 0, 'ReceivedPackets', 0, 'ReceivedBytes', 0, ...
                'Retransmissions', 0, 'RetransmissionBytes', 0), numUEs, 1);
            for ueIdx=1:numUEs
                macStats(ueIdx).UEID = obj.UEInfo(ueIdx).ID;
                macStats(ueIdx).UEName = obj.UEInfo(ueIdx).Name;
                macStats(ueIdx).RNTI = obj.UEs(ueIdx);
                macStats(ueIdx).TransmittedPackets = obj.StatTransmittedPackets(ueIdx);
                macStats(ueIdx).TransmittedBytes = obj.StatTransmittedBytes(ueIdx);
                macStats(ueIdx).ReceivedPackets = obj.StatReceivedPackets(ueIdx);
                macStats(ueIdx).ReceivedBytes = obj.StatReceivedBytes(ueIdx);
                macStats(ueIdx).Retransmissions = obj.StatRetransmittedPackets(ueIdx);
                macStats(ueIdx).RetransmissionBytes = obj.StatRetransmittedBytes(ueIdx);
            end
        end
    end

    methods (Hidden)
        function resourceAssignments = runULScheduler(obj)
            %runULScheduler Run the UL scheduler
            %
            %   RESOURCEASSIGNMENTS = runULScheduler(OBJ) runs the UL scheduler
            %   and returns the resource assignments structure array.
            %
            %   RESOURCEASSIGNMENTS is a structure array where each element is an
            %   uplink grant

            resourceAssignments = runULScheduler(obj.Scheduler, obj.TimestampInfo);
            % Set Rx context at gNB by storing the UL grants. It is set at
            % symbol number in the 10 ms frame, where UL reception is
            % expected to start. gNB uses this to anticipate the reception
            % start time of uplink packets
            for i = 1:length(resourceAssignments)
                grant = resourceAssignments(i);
                slotNum = mod(obj.CurrSlot + grant.SlotOffset, obj.NumSlotsFrame); % Slot number in the frame for the grant
                obj.UplinkRxContext{grant.RNTI, slotNum*obj.NumSymbols + grant.StartSymbol + 1} = grant;
            end
        end

        function resourceAssignments = runDLScheduler(obj)
            %runDLScheduler Run the DL scheduler
            %
            %   RESOURCEASSIGNMENTS = runDLScheduler(OBJ) runs the DL scheduler
            %   and returns the resource assignments structure array.
            %
            %   RESOURCEASSIGNMENTS is a structure array where each element is a
            %   downlink assignment

            resourceAssignments = runDLScheduler(obj.Scheduler, obj.TimestampInfo);
            % Update Tx context at gNB by storing the DL grants at the
            % symbol number (in the 10 ms frame) where DL transmission
            % is scheduled to start
            for i = 1:length(resourceAssignments)
                grant = resourceAssignments(i);
                slotNum = mod(obj.CurrSlot + grant.SlotOffset, obj.NumSlotsFrame); % Slot number in the frame for the grant
                obj.DownlinkTxContext{grant.RNTI, slotNum*obj.NumSymbols + grant.StartSymbol + 1} = grant;
            end
        end

        function dataTx(obj)
            % dataTx Construct and send the DL MAC PDUs scheduled for current symbol to PHY
            %
            % dataTx(OBJ) Based on the assignments sent earlier, if current
            % symbol is the start symbol of downlink transmissions then
            % send the DL MAC PDUs to PHY.

            symbolNumFrame = obj.CurrSlot*obj.NumSymbols + obj.CurrSymbol; % Current symbol number in the 10 ms frame
            rbOccupancyStatus = zeros(obj.NumResourceBlocks, 1);
            for rnti = 1:length(obj.UEs) % For all UEs
                downlinkAssignment = obj.DownlinkTxContext{rnti, symbolNumFrame + 1};
                % If there is any downlink assignment corresponding to which a transmission is scheduled at the current symbol
                if ~isempty(downlinkAssignment)
                    % Construct and send MAC PDU in adherence to downlink
                    % assignment properties
                    sentPDULen = sendMACPDU(obj, rnti, downlinkAssignment);
                    type = downlinkAssignment.Type;
                    % Tx done. Clear the context
                    obj.DownlinkTxContext{rnti, symbolNumFrame + 1} = [];

                    % Calculate the slot number where PDSCH ACK/NACK is
                    % expected
                    feedbackSlot = mod(obj.CurrSlot + downlinkAssignment.FeedbackSlotOffset, obj.NumSlotsFrame);

                    % For TDD, the selected symbol at which feedback would
                    % be transmitted by UE is the first UL symbol in
                    % feedback slot. For FDD, it is the first symbol in the
                    % feedback slot (as every symbol is UL)
                    if obj.DuplexModeNumber % TDD
                        feedbackSlotDLULIdx = mod(obj.CurrDLULSlotIndex + downlinkAssignment.FeedbackSlotOffset, obj.NumDLULPatternSlots);
                        feedbackSlotPattern = obj.DLULSlotFormat(feedbackSlotDLULIdx + 1, :);
                        feedbackSym = (find(feedbackSlotPattern == obj.ULType, 1)) - 1; % Check for location of first UL symbol in the feedback slot
                    else % FDD
                        feedbackSym = 0;  % First symbol
                    end

                    % Update the context for this UE at the symbol number
                    % w.r.t start of the frame where feedback is expected
                    % to be received
                    obj.RxContextFeedback{rnti, ((feedbackSlot*obj.NumSymbols) + feedbackSym + 1), downlinkAssignment.HARQID + 1} = downlinkAssignment;

                    if strcmp(type, 'newTx') % New transmission
                        obj.StatTransmittedBytes(rnti) = obj.StatTransmittedBytes(rnti) + sentPDULen;
                        obj.StatTransmittedPackets(rnti) = obj.StatTransmittedPackets(rnti) + 1;
                    else % Retransmission
                        obj.StatRetransmittedBytes(rnti) = obj.StatRetransmittedBytes(rnti) + sentPDULen;
                        obj.StatRetransmittedPackets(rnti) = obj.StatRetransmittedPackets(rnti) + 1;
                    end
                    rbOccupancyStatus(downlinkAssignment.PRBSet+1) = 1;
                end
            end
            obj.NumDLRBsUsed = obj.NumDLRBsUsed + nnz(rbOccupancyStatus);
        end

        function controlTx(obj, resourceAssignmentsUL, resourceAssignmentsDL)
            %controlTx Construct and send the uplink and downlink assignments to the UEs
            %
            %   controlTx(obj, RESOURCEASSIGNMENTSUL, RESOURCEASSIGNMENTSDL)
            %   Based on the resource assignments done by uplink and
            %   downlink scheduler, send assignments to UEs. UL and DL
            %   assignments are sent out-of-band without the need of
            %   frequency resources.
            %
            %   RESOURCEASSIGNMENTSUL is an array of structures that contains the UL
            %   resource assignments information.
            %
            %   RESOURCEASSIGNMENTSDL is an array of structures that contains the DL
            %   resource assignments information.

            scheduledResources = obj.ScheduledResources;
            scheduledResources.TimingInfo = [mod(obj.CurrFrame, 1024) obj.CurrSlot obj.CurrSymbol];
            % Construct and send uplink grants
            if ~isempty(resourceAssignmentsUL)
                scheduledResources.ULGrants = resourceAssignmentsUL;
                pktInfo = obj.PacketStruct;
                uplinkGrant = obj.UplinkGrantStruct;
                grantFieldNames = obj.ULGrantFieldNames;
                for i = 1:length(resourceAssignmentsUL) % For each UL assignment
                    grant = resourceAssignmentsUL(i);
                    for ind = 1:obj.ULGrantFieldNamesCount
                        uplinkGrant.(grantFieldNames{ind}) = grant.(grantFieldNames{ind});
                    end
                    % Construct packet information
                    pktInfo.DirectToDestination = obj.UEInfo(grant.RNTI).ID;
                    pktInfo.Data = uplinkGrant;
                    pktInfo.Metadata.PacketType = obj.ULGrant;
                    pktInfo.Metadata.RNTI = grant.RNTI;
                    obj.TxOutofBandFcn(pktInfo); % Send the UL grant out-of-band to UE's MAC
                end
            end

            % Construct and send downlink grants
            if ~isempty(resourceAssignmentsDL)
                scheduledResources.DLGrants = resourceAssignmentsDL;
                pktInfo = obj.PacketStruct;
                downlinkGrant = obj.DownlinkGrantStruct;
                grantFieldNames = obj.DLGrantFieldNames;
                for i = 1:length(resourceAssignmentsDL) % For each DL assignment
                    grant = resourceAssignmentsDL(i);
                    for ind = 1:obj.DLGrantFieldNamesCount
                        downlinkGrant.(grantFieldNames{ind}) = grant.(grantFieldNames{ind});
                    end
                    % Construct packet information and send the DL grant out-of-band to UE's MAC
                    pktInfo.DirectToDestination = obj.UEInfo(grant.RNTI).ID;
                    pktInfo.Data = downlinkGrant;
                    pktInfo.Metadata.PacketType = obj.DLGrant;
                    pktInfo.Metadata.RNTI = grant.RNTI;
                    obj.TxOutofBandFcn(pktInfo);
                end
            end

            % Notify the node about resource allocation event
            obj.NotificationFcn('ScheduledResources', scheduledResources);
        end

        function controlRx(obj, packets)
            %controlRx Receive callback for BSR, feedback(ACK/NACK) for
            % PDSCH, and CSI report. CSI report can either be of the format
            % ri-pmi-cqi or cri-l1RSRP. The ri-pmi-cqi format specifies the
            % rank indicator (RI), precoding matrix indicator (PMI) and the channel
            % quality indicator (CQI) values. the cri-rsrp format specifies the
            % CSI-RS resource indicator (CRI) and layer-1 reference signal
            % received power (L1-RSRP) values.

            for pktIdx = 1:numel(packets)
                pktInfo = packets(pktIdx);
                if packets(pktIdx).DirectToDestination ~= 0 && ...
                        packets(pktIdx).Metadata.NCellID == obj.NCellID

                    pktType = pktInfo.Metadata.PacketType;
                    rnti = pktInfo.Metadata.RNTI;
                    switch pktType
                        case obj.BSR % BSR received
                            processMACControlElement(obj.Scheduler, rnti, pktInfo, obj.LCGPriority(rnti,:));

                        case obj.PDSCHFeedback % PDSCH feedback received
                            feedbackList = pktInfo.Data;
                            symNumFrame = obj.CurrSlot*obj.NumSymbols + obj.CurrSymbol;
                            for harqIdx = 1:obj.NumHARQ % Check for all HARQ processes
                                feedbackContext =  obj.RxContextFeedback{rnti, symNumFrame+1, harqIdx};
                                if ~isempty(feedbackContext) % If any ACK/NACK expected from the UE for this HARQ process
                                    rxResult = feedbackList(feedbackContext.HARQID+1); % Read Rx success/failure result
                                    % Notify PDSCH Rx result to scheduler for updating the HARQ context
                                    rxResultInfo.RNTI = rnti;
                                    rxResultInfo.RxResult = rxResult;
                                    rxResultInfo.HARQID = harqIdx-1;
                                    handleDLRxResult(obj.Scheduler, rxResultInfo);
                                    obj.RxContextFeedback{rnti, symNumFrame+1, harqIdx} = []; % Clear the context
                                    if rxResult == 0
                                        % Update the counter for the number of NACKs received
                                        obj.NumPDSCHNACKs = obj.NumPDSCHNACKs + 1;
                                    end
                                end
                            end

                        case obj.CSIReport % CSI report received containing RI, PMI and CQI
                            csiReport = pktInfo.Data;
                            channelQualityInfo.RNTI = rnti;
                            channelQualityInfo.RankIndicator = csiReport.RankIndicator;
                            channelQualityInfo.PMISet = csiReport.PMISet;
                            channelQualityInfo.CQI = csiReport.CQI;
                            channelQualityInfo.W = csiReport.W;
                            updateChannelQualityDL(obj.Scheduler, channelQualityInfo);

                        case obj.CSIReportRSRP % CSI report received containing CRI and RSRP
                            csiReport = pktInfo.Data;
                            channelQualityInfo.RNTI = rnti;
                            channelQualityInfo.CRI = csiReport.CRI;
                            channelQualityInfo.L1RSRP = csiReport.L1RSRP;
                            updateChannelQualityDL(obj.Scheduler, channelQualityInfo);
                    end
                end
            end
        end

        function dataRx(obj)
            %dataRx Send Rx start request to PHY for the receptions scheduled to start now
            %
            %   dataRx(OBJ) sends the Rx start request to PHY for the
            %   receptions scheduled to start now, as per the earlier sent
            %   uplink grants.
            %

            if ~isempty(obj.UplinkRxContext)
                gNBRxContext = obj.UplinkRxContext(:, (obj.CurrSlot * obj.NumSymbols) + obj.CurrSymbol + 1); % Rx context of current symbol
                txUEs = find(~cellfun(@isempty, gNBRxContext)); % UEs which are assigned uplink grants starting at this symbol
                rbOccupancyStatus = zeros(obj.NumResourceBlocks,1);
                for i = 1:length(txUEs)
                    % For the UE, get the uplink grant information
                    uplinkGrant = gNBRxContext{txUEs(i)};
                    % Send the UE uplink Rx context to PHY
                    rxRequestToPHY(obj, txUEs(i), uplinkGrant);
                    obj.LastRxTime = obj.TimestampInfo.Timestamp;
                    rbOccupancyStatus(uplinkGrant.PRBSet+1) = 1;
                end
                obj.UplinkRxContext(:, (obj.CurrSlot * obj.NumSymbols) + obj.CurrSymbol + 1) = {[]}; % Clear uplink RX context
                obj.NumULRBsUsed = obj.NumULRBsUsed + nnz(rbOccupancyStatus);
            end
        end

        function updateSRSPeriod(obj, rnti, srsPeriod)
            %updateSRSPeriod Update the SRS periodicity of UE

            obj.SRSConfiguration(rnti).SRSPeriod = srsPeriod;
            obj.Scheduler.updateSRSPeriod(rnti, srsPeriod);
            % Calculate unique SRS reception time and periodicity
            obj.SRSRxInfo = calculateSRSPeriodicity(obj, obj.SRSConfiguration);
        end

        function kpiValue = kpi(obj, kpiType, linkType)
            %kpi Return the key performance indicator (KPI) value for the
            %specified KPI type.

            switch kpiType
                case "prbUsage"
                    % Calculate the number of slots based on the last
                    % runtime and slot duration.
                    numSlots = floor(obj.LastRunTime / obj.SlotDurationInNS) + 1;

                    if linkType == "DL"
                        % Calculate available Resource Blocks (RBs) for
                        % Downlink (DL).
                        availableRBs = floor(numSlots * obj.DLSlotAvailability) * obj.NumResourceBlocks;
                        % Calculate KPI value as the ratio of used DL RBs
                        % to available RBs.
                        kpiValue = (obj.NumDLRBsUsed / availableRBs) * 100;
                    else
                        % Calculate available Resource Blocks (RBs) for
                        % Uplink (UL).
                        availableRBs = floor(numSlots * obj.ULSlotAvailability) * obj.NumResourceBlocks;
                        % Calculate KPI value as the ratio of used UL RBs
                        % to available RBs.
                        kpiValue = (obj.NumULRBsUsed / availableRBs) * 100;
                    end
            end
        end

        function flag = rxOn(obj, packet)
            %rxOn Returns whether Rx is scheduled to be on during the packet duration

            flag = 1;
            % Check whether the packet overlaps (partially or fully) with scheduled
            % reception time of the node
            startTimeSec = packet.StartTime;
            lastRxTimeSec = obj.LastRxTime*1e-9;
            if (abs(startTimeSec - lastRxTimeSec) > 1e-9) && ...
                    ((startTimeSec - lastRxTimeSec) > (obj.SlotDurationInSec-1e-9) && ...
                    ((startTimeSec + packet.Duration) < ((obj.NextRxTime*1e-9)+1e-9)))
                flag = 0;
            end
        end
    end

    methods (Access = protected)
        function updateMACContext(obj)
            %updateMACContext Update the MAC context when new UE is connected

            obj.StatTransmittedPackets = [obj.StatTransmittedPackets; 0];
            obj.StatTransmittedBytes = [obj.StatTransmittedBytes; 0];
            obj.StatRetransmittedPackets = [obj.StatRetransmittedPackets; 0];
            obj.StatRetransmittedBytes = [obj.StatRetransmittedBytes; 0];
            obj.StatReceivedPackets = [obj.StatReceivedPackets; 0];
            obj.StatReceivedBytes = [obj.StatReceivedBytes; 0];

            obj.ElapsedTimeSinceLastLCP = [obj.ElapsedTimeSinceLastLCP; 0];
            % Configuration of logical channels for UEs
            obj.LogicalChannelConfig = [obj.LogicalChannelConfig; cell(1, obj.MaxLogicalChannels)];
            obj.LCHBjList = [obj.LCHBjList; zeros(1, obj.MaxLogicalChannels)];
            obj.LCHBufferStatus = [obj.LCHBufferStatus; zeros(1, obj.MaxLogicalChannels)];
            % Initialize LCG with lowest priority level for all the UEs. Here
            % lowest priority level is indicated by higher value
            obj.LCGPriority = [obj.LCGPriority; obj.MaxPriorityForLCH*ones(1,8)];
            % Extend the cell array to hold the RLC entity callbacks of the UE
            obj.RLCTxFcn = [obj.RLCTxFcn; cell(1, obj.MaxLogicalChannels)];
            obj.RLCRxFcn = [obj.RLCRxFcn; cell(1, obj.MaxLogicalChannels)];

            % Create Tx/Rx contexts
            obj.UplinkRxContext = [obj.UplinkRxContext; cell(1, obj.NumSymInFrame)];
            obj.DownlinkTxContext = [obj.DownlinkTxContext; cell(1, obj.NumSymInFrame)];
            obj.RxContextFeedback = [obj.RxContextFeedback; cell(1, obj.NumSymInFrame, obj.NumHARQ)];

            % Calculate unique CSI-RS transmission time and periodicity
            obj.CSIRSTxInfo = calculateCSIRSPeriodicity(obj, [obj.CSIRSMapping.CSIRSConfig obj.CSIRSConfigurationRSRP]);

            % Calculate unique SRS reception time and periodicity
            obj.SRSRxInfo = calculateSRSPeriodicity(obj, obj.SRSConfiguration);
        end

        function dlControlRequest(obj)
            %dlControlRequest Request from MAC to PHY to send non-data DL transmissions
            %   dlControlRequest(OBJ) sends a request to PHY for non-data downlink
            %   transmission scheduled for the current slot. MAC sends it at the
            %   start of a DL slot for all the scheduled DL transmissions in
            %   the slot (except PDSCH, which is sent using dataTx
            %   function of this class).

            % Check if current slot is a slot with DL symbols. For FDD (Value 0),
            % there is no need to check as every slot is a DL slot. For
            % TDD (Value 1), check if current slot has any DL symbols
            csirsConfigLen = numel(obj.CSIRSMapping);
            maxNumCSIRS = numel(obj.CSIRSConfigurationRSRP) + csirsConfigLen;
            dlControlType = zeros(1, maxNumCSIRS);
            dlControlPDUs = cell(1, maxNumCSIRS);
            numDLControlPDU = 0; % Variable to hold the number of DL control PDUs
            % Set carrier configuration object
            carrier = obj.CarrierConfigDL;
            carrier.NSlot = obj.CurrSlot;
            carrier.NFrame = obj.CurrFrame;

            % To account for consecutive symbols in CDM pattern
            additionalCSIRSSyms = [0 0 0 0 1 0 1 1 0 1 1 1 1 1 3 1 1 3];
            for csirsIdx = 1:csirsConfigLen % CSI-RS for downlink channel measurement
                csirsConfig = obj.CSIRSMapping(csirsIdx).CSIRSConfig;
                csirsSymbolRange(1) = min(csirsConfig.SymbolLocations); % First CSI-RS symbol
                csirsSymbolRange(2) = max(csirsConfig.SymbolLocations) + ... % Last CSI-RS symbol
                    additionalCSIRSSyms(csirsConfig.RowNumber);
                % Check whether the mode is FDD OR if it is TDD then all the CSI-SRS symbols must be DL symbols
                if obj.DuplexModeNumber == 0 || all(obj.DLULSlotFormat(obj.CurrDLULSlotIndex + 1, csirsSymbolRange+1) == obj.DLType)
                    % Check if the current slot is CSI-RS transmission slot based on configured CSI-RS periodicity and offset
                    if (~isnumeric(csirsConfig.CSIRSPeriod) && csirsConfig.CSIRSPeriod == "on") || ~mod(obj.NumSlotsFrame*obj.CurrFrame + obj.CurrSlot - csirsConfig.CSIRSPeriod(2), csirsConfig.CSIRSPeriod(1))
                        numDLControlPDU = numDLControlPDU + 1;
                        dlControlType(numDLControlPDU) = 0; % CSIRS PDU
                        % Passing empty CSIRS beam index
                        dlControlPDUs{numDLControlPDU} = {csirsConfig, [], obj.CSIRSMapping(csirsIdx).RNTI};
                    end
                end
            end
            obj.DlControlRequestFcn(dlControlType(1:numDLControlPDU), dlControlPDUs(1:numDLControlPDU), obj.TimestampInfo); % Send DL control request to Ph
        end

        function ulControlRequest(obj)
            %ulControlRequest Request from MAC to PHY to receive non-data UL transmissions
            %   ulControlRequest(OBJ) sends a request to PHY for non-data
            %   uplink reception scheduled for the current slot. MAC
            %   sends it at the start of a UL slot for all the scheduled UL
            %   receptions in the slot (except PUSCH, which is received
            %   using dataRx function of this class).

            if ~isempty(obj.SRSConfiguration) % Check if SRS is enabled
                % Check if current slot is a slot with UL symbols. For FDD
                % (value 0), there is no need to check as every slot is a
                % UL slot. For TDD (value 1), check if current slot has any
                % UL symbols
                if obj.DuplexModeNumber == 0 || ~isempty(find(obj.DLULSlotFormat(obj.CurrDLULSlotIndex + 1, :) == obj.ULType, 1))
                    ulControlType = zeros(1, length(obj.UEs));
                    ulControlPDUs = cell(1, length(obj.UEs));
                    numSRSUEs = 0; % Initialize number of UEs from which SRS is expected in this slot
                    % Set carrier configuration object
                    carrier = obj.CarrierConfigUL;
                    carrier.NSlot = obj.CurrSlot;
                    carrier.NFrame = obj.CurrFrame;
                    for rnti=1:length(obj.UEs) % Send SRS reception request to PHY for the UEs
                        srsConfigUE = obj.SRSConfiguration(rnti);
                        if ~isempty(srsConfigUE)
                            srsLocations = srsConfigUE.SymbolStart : (srsConfigUE.SymbolStart + srsConfigUE.NumSRSSymbols-1); % SRS symbol locations
                            % Check whether the mode is FDD OR if it is TDD then all the SRS symbols must be UL symbols
                            if obj.DuplexModeNumber == 0 || all(obj.DLULSlotFormat(obj.CurrDLULSlotIndex + 1, srsLocations+1) == obj.ULType)
                                % Check if the current slot is SRS reception slot based on configured SRS periodicity and offset
                                if (~isnumeric(srsConfigUE.SRSPeriod) && srsConfigUE.SRSPeriod == "on") || ~mod(obj.NumSlotsFrame*obj.CurrFrame + obj.CurrSlot - srsConfigUE.SRSPeriod(2), srsConfigUE.SRSPeriod(1))
                                    numSRSUEs = numSRSUEs+1;
                                    ulControlType(numSRSUEs) = 1; % SRS PDU
                                    ulControlPDUs{numSRSUEs}{1} = rnti;
                                    ulControlPDUs{numSRSUEs}{2} = srsConfigUE;
                                end
                            end
                        end
                    end
                    ulControlType = ulControlType(1:numSRSUEs);
                    ulControlPDUs = ulControlPDUs(1:numSRSUEs);
                    obj.UlControlRequestFcn(ulControlType, ulControlPDUs, obj.TimestampInfo); % Send UL control request to PHY
                    if numSRSUEs > 0
                        obj.LastRxTime = obj.TimestampInfo.Timestamp;
                    end
                end
            end
        end

        function pduLen = sendMACPDU(obj, rnti, downlinkGrant)
            %sendMACPDU Sends MAC PDU to PHY as per the parameters of the downlink grant
            % Based on the NDI in the downlink grant, either new
            % transmission or retransmission would be indicated to PHY

            macPDU = [];
            % Populate PDSCH information to be sent to PHY, along with the MAC
            % PDU. For runtime optimization, only set a field if its value is different
            % from last PDSCH for the UE
            pdschConfig = obj.PDSCHInfo(rnti).PDSCHConfig;
            obj.PDSCHInfo(rnti).PDSCHConfig.PRBSet = downlinkGrant.PRBSet;
            % Get the corresponding row from the mcs table
            mcsInfo = obj.MCSTableDL(downlinkGrant.MCSIndex + 1, :);
            modSchemeBits = mcsInfo(1); % Bits per symbol for modulation scheme(stored in column 1)
            obj.PDSCHInfo(rnti).TargetCodeRate = mcsInfo(2)/1024; % Coderate (stored in column 2)
            % Modulation scheme and corresponding bits/symbol
            modScheme = nr5g.internal.getModulationScheme(modSchemeBits); % Get modulation scheme string
            if pdschConfig.Modulation ~= modScheme(1)
                obj.PDSCHInfo(rnti).PDSCHConfig.Modulation = modScheme(1);
            end
            if downlinkGrant.StartSymbol ~= pdschConfig.SymbolAllocation(1)
                obj.PDSCHInfo(rnti).PDSCHConfig.SymbolAllocation(1) = downlinkGrant.StartSymbol;
            end
            if downlinkGrant.NumSymbols ~= pdschConfig.SymbolAllocation(2)
                obj.PDSCHInfo(rnti).PDSCHConfig.SymbolAllocation(2) = downlinkGrant.NumSymbols;
            end
            obj.PDSCHInfo(rnti).NSlot = obj.CurrSlot;
            obj.PDSCHInfo(rnti).HARQID = downlinkGrant.HARQID;
            obj.PDSCHInfo(rnti).RV = downlinkGrant.RV;
            obj.PDSCHInfo(rnti).PrecodingMatrix = downlinkGrant.W;
            obj.PDSCHInfo(rnti).BeamIndex = downlinkGrant.BeamIndex;
            if downlinkGrant.NumLayers ~= pdschConfig.NumLayers
                obj.PDSCHInfo(rnti).PDSCHConfig.NumLayers = downlinkGrant.NumLayers;
            end
            if downlinkGrant.MappingType ~= pdschConfig.MappingType
                obj.PDSCHInfo(rnti).PDSCHConfig.MappingType = mappingType;
            end
            if downlinkGrant.MappingType == 'A'
                dmrsAdditonalPos = obj.PDSCHDMRSAdditionalPosTypeA;
            else
                dmrsAdditonalPos = obj.PDSCHDMRSAdditionalPosTypeB;
            end
            if dmrsAdditonalPos ~= pdschConfig.DMRS.DMRSAdditionalPosition
                obj.PDSCHInfo(rnti).PDSCHConfig.DMRS.DMRSAdditionalPosition = dmrsAdditonalPos;
            end
            if downlinkGrant.DMRSLength ~= pdschConfig.DMRS.DMRSLength
                obj.PDSCHInfo(rnti).PDSCHConfig.DMRS.DMRSLength = downlinkGrant.DMRSLength;
            end
            if downlinkGrant.NumCDMGroupsWithoutData ~= pdschConfig.DMRS.NumCDMGroupsWithoutData
                obj.PDSCHInfo(rnti).PDSCHConfig.DMRS.NumCDMGroupsWithoutData = downlinkGrant.NumCDMGroupsWithoutData;
            end

            % Carrier configuration
            carrierConfig = obj.CarrierConfigDL;
            carrierConfig.NFrame = obj.CurrFrame;
            carrierConfig.NSlot = obj.PDSCHInfo(rnti).NSlot;

            pduLen = downlinkGrant.TBS; % In bytes
            if downlinkGrant.Type == "newTx"
                % Generate MAC PDU
                macPDU = constructMACPDU(obj, pduLen, rnti);
            end

            obj.PDSCHInfo(rnti).TBS = pduLen;
            % Set reserved REs information. Generate 0-based
            % carrier-oriented CSI-RS indices in linear indexed form
            obj.PDSCHInfo(rnti).PDSCHConfig.ReservedRE = [];
            for csirsIdx = 1:numel(obj.CSIRSMapping)
                csirsConfig = obj.CSIRSMapping(csirsIdx).CSIRSConfig;
                csirsLocations = csirsConfig.SymbolLocations; % CSI-RS symbol locations
                 % (Mode is FDD) or (Mode is TDD and CSI-RS symbols are DL symbols)
                if obj.DuplexModeNumber == 0 || all(obj.DLULSlotFormat(obj.CurrDLULSlotIndex + 1, csirsLocations+1) == obj.DLType)
                   % Check if the current slot is CSI-RS transmission slot based on configured CSI-RS periodicity and offset
                    if (~isnumeric(csirsConfig.CSIRSPeriod) && csirsConfig.CSIRSPeriod == "on") || ~mod(obj.NumSlotsFrame*obj.CurrFrame + obj.CurrSlot - csirsConfig.CSIRSPeriod(2), csirsConfig.CSIRSPeriod(1))
                        obj.PDSCHInfo(rnti).PDSCHConfig.ReservedRE = [obj.PDSCHInfo(rnti).PDSCHConfig.ReservedRE; nrCSIRSIndices(carrierConfig, csirsConfig, 'IndexBase', '0based')]; % Reserve CSI-RS REs
                    end
                end
            end
            for idx = 1:length(obj.CSIRSConfigurationRSRP)
                csirsConfig = obj.CSIRSConfigurationRSRP(idx);
                csirsLocations = csirsConfig.SymbolLocations; % CSI-RS symbol locations
                % (Mode is FDD) or (Mode is TDD and CSI-RS symbols are DL symbols)
                if obj.DuplexModeNumber == 0 || all(obj.DLULSlotFormat(obj.CurrDLULSlotIndex + 1, cell2mat(csirsLocations) + 1) == obj.DLType)
                    % Check if the current slot is CSI-RS transmission slot based on configured CSI-RS periodicity and offset
                    if (~isnumeric(csirsConfig.CSIRSPeriod) && csirsConfig.CSIRSPeriod == "on") || ~mod(obj.NumSlotsFrame*obj.CurrFrame + obj.CurrSlot - csirsConfig.CSIRSPeriod(2), csirsConfig.CSIRSPeriod(1))
                        obj.PDSCHInfo(rnti).PDSCHConfig.ReservedRE = [obj.PDSCHInfo(rnti).PDSCHConfig.ReservedRE; nrCSIRSIndices(carrierConfig, obj.CSIRSConfigurationRSRP(idx), 'IndexBase', '0based')]; % Reserve CSI-RS REs
                    end
                end
            end
            obj.TxDataRequestFcn(obj.PDSCHInfo(rnti), macPDU, obj.TimestampInfo);
        end

        function rxRequestToPHY(obj, rnti, uplinkGrant)
            %rxRequestToPHY Send Rx request to PHY

            % Populate PUSCH information to be sent to PHY for reception.
            % For runtime optimization, only set a field if its value is
            % different from last PUSCH for the UE
            puschConfig = obj.PUSCHInfo(rnti).PUSCHConfig; % Information to be passed to PHY for PUSCH reception
            obj.PUSCHInfo(rnti).PUSCHConfig.PRBSet = uplinkGrant.PRBSet;
            % Get the corresponding row from the mcs table
            mcsInfo = obj.MCSTableUL(uplinkGrant.MCSIndex + 1, :);
            modSchemeBits = mcsInfo(1); % Bits per symbol for modulation scheme (stored in column 1)
            obj.PUSCHInfo(rnti).TargetCodeRate = mcsInfo(2)/1024; % Coderate (stored in column 2)
            % Modulation scheme and corresponding bits/symbol
            modScheme = nr5g.internal.getModulationScheme(modSchemeBits); % Get modulation scheme string
            if puschConfig.Modulation ~= modScheme(1)
                obj.PUSCHInfo(rnti).PUSCHConfig.Modulation = modScheme(1);
            end
            if uplinkGrant.StartSymbol ~= puschConfig.SymbolAllocation(1)
                obj.PUSCHInfo(rnti).PUSCHConfig.SymbolAllocation(1) = uplinkGrant.StartSymbol;
            end
            if uplinkGrant.NumSymbols ~= puschConfig.SymbolAllocation(2)
                obj.PUSCHInfo(rnti).PUSCHConfig.SymbolAllocation(2) = uplinkGrant.NumSymbols;
            end
            obj.PUSCHInfo(rnti).NSlot = obj.CurrSlot;
            obj.PUSCHInfo(rnti).HARQID = uplinkGrant.HARQID;
            obj.PUSCHInfo(rnti).RV = uplinkGrant.RV;
            if uplinkGrant.NumLayers ~= puschConfig.NumLayers
                obj.PUSCHInfo(rnti).PUSCHConfig.NumLayers = uplinkGrant.NumLayers;
            end
            if uplinkGrant.NumAntennaPorts ~= puschConfig.NumAntennaPorts
                obj.PUSCHInfo(rnti).PUSCHConfig.NumAntennaPorts = uplinkGrant.NumAntennaPorts;
            end
            obj.PUSCHInfo(rnti).PUSCHConfig.TPMI = uplinkGrant.TPMI;
            if uplinkGrant.MappingType ~= puschConfig.MappingType
               obj.PUSCHInfo(rnti).PUSCHConfig.MappingType = uplinkGrant.MappingType;
            end
            if uplinkGrant.MappingType == 'A'
                dmrsAdditonalPos = obj.PUSCHDMRSAdditionalPosTypeA;
            else
                dmrsAdditonalPos = obj.PUSCHDMRSAdditionalPosTypeB;
            end
            if dmrsAdditonalPos ~= puschConfig.DMRS.DMRSAdditionalPosition
                obj.PUSCHInfo(rnti).PUSCHConfig.DMRS.DMRSAdditionalPosition = dmrsAdditonalPos;
            end
            if uplinkGrant.DMRSLength ~= puschConfig.DMRS.DMRSLength
                obj.PUSCHInfo(rnti).PUSCHConfig.DMRS.DMRSLength = uplinkGrant.DMRSLength;
            end
            if uplinkGrant.NumCDMGroupsWithoutData ~= puschConfig.DMRS.NumCDMGroupsWithoutData
                obj.PUSCHInfo(rnti).PUSCHConfig.DMRS.NumCDMGroupsWithoutData = uplinkGrant.NumCDMGroupsWithoutData;
            end
            % Carrier configuration
            carrierConfig = obj.CarrierConfigUL;
            carrierConfig.NSlot = obj.PUSCHInfo(rnti).NSlot;

            % Calculate TBS
            obj.PUSCHInfo(rnti).TBS = uplinkGrant.TBS; % TBS in bytes
            obj.PUSCHInfo(rnti).NewData = uplinkGrant.Type=="newTx";

            % Call PHY to start receiving PUSCH
            obj.RxDataRequestFcn(obj.PUSCHInfo(rnti), obj.TimestampInfo);
        end

        function nextInvokeTime = getNextInvokeTime(obj, currentTime)
            %getNextInvokeTime Return the next invoke time in nanoseconds

            % Find the duration completed in the current symbol
            durationCompletedInCurrSlot = mod(currentTime, obj.SlotDurationInNS);
            currSymDurCompleted = obj.SymbolDurationsInSlot(obj.CurrSymbol+1) - obj.SymbolEndTimesInSlot(obj.CurrSymbol+1) + durationCompletedInCurrSlot;

            totalSymbols = obj.NumSymInFrame;
            symbolNumFrame = obj.CurrSlot*obj.NumSymbols + obj.CurrSymbol;
            % Next Tx start symbol
            nextTxStartSymbol = Inf;
            if ~isempty(obj.DownlinkTxContext)
                nextTxStartSymbol = find(~cellfun('isempty',obj.DownlinkTxContext(:, symbolNumFrame+2:totalSymbols)), 1);
                nextTxStartSymbol = ceil(nextTxStartSymbol/numel(obj.UEs));
                if isempty(nextTxStartSymbol)
                    nextTxStartSymbol = find(~cellfun('isempty',obj.DownlinkTxContext(:, 1:symbolNumFrame)), 1);
                    nextTxStartSymbol = (totalSymbols-symbolNumFrame-1) + ceil(nextTxStartSymbol/numel(obj.UEs));
                end
            end

            % Next Rx start symbol
            nextRxStartSymbol = Inf;
            if ~isempty(obj.UplinkRxContext)
                nextRxStartSymbol = find(~cellfun('isempty',obj.UplinkRxContext(:, symbolNumFrame+2:totalSymbols)), 1);
                nextRxStartSymbol = ceil(nextRxStartSymbol/numel(obj.UEs));
                if isempty(nextRxStartSymbol)
                    nextRxStartSymbol = find(~cellfun('isempty',obj.UplinkRxContext(:, 1:symbolNumFrame)), 1);
                    nextRxStartSymbol = (totalSymbols-symbolNumFrame-1) + ceil(nextRxStartSymbol/numel(obj.UEs));
                end
            end

            nextInvokeSymbol = min([nextTxStartSymbol nextRxStartSymbol Inf]);
            nextInvokeTime = Inf;
            % Set Next Rx time
            if nextInvokeSymbol ~= Inf
                nextInvokeTime = currentTime + durationToSymNum(obj, nextInvokeSymbol) - currSymDurCompleted;
                if nextInvokeSymbol == nextRxStartSymbol
                    obj.NextRxTime = nextInvokeTime;
                elseif ~isempty(nextRxStartSymbol)
                    obj.NextRxTime = currentTime + durationToSymNum(obj, nextRxStartSymbol) - currSymDurCompleted;
                else
                    obj.NextRxTime = Inf;
                end
            end

            % Next control transmission time
            controlTxStartTime = min(obj.CSIRSTxInfo(:, 2));
            % Next control reception time
            controlRxStartTime = min(obj.SRSRxInfo(:, 2));
            obj.NextRxTime = min([obj.NextRxTime controlRxStartTime]);
            nextInvokeTime = min([obj.SchedulerNextInvokeTime nextInvokeTime controlTxStartTime controlRxStartTime]);
        end
    end
end
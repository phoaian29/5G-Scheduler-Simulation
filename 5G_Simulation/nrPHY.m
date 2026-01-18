classdef (Abstract) nrPHY < handle
    %nrPHY Define NR physical layer base class
    %   The class acts as a base class for all the physical layer types. It
    %   contains the code applicable for any PHY flavor and node type (UE or
    %   gNB).
    %
    %   Note: This is an internal undocumented class and its API and/or
    %   functionality may change in subsequent releases.

    %   Copyright 2022-2024 The MathWorks, Inc.

    properties(SetAccess=protected)
        %NumTransmitAntennas Number of transmit antennas
        NumTransmitAntennas

        %NumReceiveAntennas Number of receive antennas
        NumReceiveAntennas

        %TransmitPower Tx power in dBm
        TransmitPower

        %ReceiveGain Rx antenna gain in dBi
        ReceiveGain

        %NoiseFigure Noise figure at the receiver
        NoiseFigure
    end

    properties(Access=protected)
        %CarrierInformation Carrier related information
        CarrierInformation

        %WaveformInfo Waveform related information
        WaveformInfo

        %AntNoiseTemperature Antenna temperature at node in Kelvin
        % It is used for thermal noise calculation
        AntNoiseTemperature = 290

        %CQITableValues CQI table with each row containing associated modulation and code rate as the two columns
        CQITableValues

        %MCSTableValues MCS table with each row containing associated MCS index and target code rate as the two columns
        MCSTableValues

        %PacketStruct Packet representation
        PacketStruct = wirelessnetwork.internal.wirelessPacket

        %SendPacketFcn Function handle to transmit the packet
        SendPacketFcn

        %DataRxContext Data Rx context for the PHY
        % Cell array of size 'N' where N is the number of symbols in a 10 ms frame.
        % For gNB PHY, the cell elements are populated with structure of type
        % PUSCHInfo (See rxDataRequest function in nrGNBPHY.m for structure
        % details). For UE PHY, the cell elements are populated with structures of
        % type PDSCHInfo (See rxDataRequest function in nrUEPHY.m for structure
        % details). The information in the structure is used by the receiver (UE or
        % gNB) for Rx reception and processing. A node reads the complete packet at
        % the symbol in which reception ends. So, an element at index 'i' contains
        % the information for reception which ends at symbol index 'i' w.r.t the
        % start of the frame. There can be array of structures at index 'i', if
        % multiple receptions were scheduled to end at symbol index 'i'. Cell
        % element at 'i' is empty, if no reception was scheduled to end at symbol
        % index 'i'
        DataRxContext

        %LastRunTime Time (in nanoseconds) at which the PHY layer was invoked last time
        LastRunTime = 0;

        %NextRxTime Next Rx time
        % Array of size 'N' where N is the number of symbols in a 10 ms frame.
        % Value at index 'i' contains contains the absolute Rx end time (in
        % nanoseconds) if any reception ends at symbol index 'i', otherwise it
        % contains Inf
        NextRxTime

        %NextTxTime Next Tx time
        % Array of size 'N' where N is the number of symbols in a 10 ms frame.
        % Value at index 'i' contains contains the absolute Tx start time (in
        % nanoseconds) if any Tx starts at symbol index 'i', otherwise it
        % contains Inf
        NextTxTime

        %RxIndicationFcn Function handle to send data to MAC
        RxIndicationFcn

        %CarrierConfig Carrier configuration
        % CarrierConfig is an object of type nrCarrierConfig
        CarrierConfig

        %CurrTimeInfo Current timing information
        CurrTimeInfo

        %PhyTxProcessingDelay PHY Tx processing delay in terms of number of symbols
        PhyTxProcessingDelay = 0;

        %MACPDUInfo PDU information sent to MAC
        % It is a structure with the following fields:
        %   NodeID  - Node ID of the connected UE
        %   RNTI    - RNTI of the connected UE
        %   TBS     - Transport block size
        %   MACPDU  - PDU sent to the MAC
        %   CRCFlag - Packet decode success (value as 0) or failure(value as 1)
        %   HARQID  - HARQ ID of the packet
        %   Tags    - Array of structures where each structure contains these
        %             fields.
        %             Name      - Name of the tag.
        %             Value     - Data associated with the tag.
        %             ByteRange - Specific range of bytes within the packet to
        %                         which the tag applies.
        MACPDUInfo

        %HARQBuffers Buffers to store most recent HARQ transport blocks
        % N-by-NumHARQProcesses cell array to buffer transport blocks for the HARQ
        % processes, where 'N' is the number of UEs if it is gNB PHY and
        % N=1, if it is UE PHY. The physical layer stores the transport
        % blocks for retransmissions
        HARQBuffers

        %ReTxTagBuffer Buffers to store tags of the most recent HARQ transport blocks
        % N-by-NumHARQProcesses cell array to buffer packet tags for the HARQ
        % processes, where 'N' is the number of UEs if it is gNB PHY and
        % N=1, if it is UE PHY. The physical layer stores the transport
        % blocks's tags for retransmissions
        ReTxTagBuffer = {};

        %NotificationFcn A function handle to inform node about the events
        NotificationFcn = []

        %PacketTxStartedStruct Event data structure
        PacketTxStartedStruct = struct('CurrentTime',0,'DuplexMode',[], ...
            'RNTI',[],'TimingInfo',[0 0 0],'LinkType',[],'HARQID',[],'SignalType',[], ...
            'Duration',[],'PDU',[],'TransmissionType',[]);

        %PacketTransmissionStarted Contains event data for the each node
        PacketTransmissionStarted

        %PacketRxEndedStruct Event data structure
        PacketRxEndedStruct = struct('CurrentTime',0,'DuplexMode',[], ...
            'RNTI',[],'TimingInfo', [0 0 0],'LinkType',[],'HARQID',[],'SignalType',[], ...
            'Duration',[],'PDU',[],'CRCFlag',[],'SINR',-Inf,'ChannelMeasurements',[]);

        %PacketReceptionEnded Contains event data for the each node
        PacketReceptionEnded
    end

    properties (SetAccess=private, GetAccess=protected)
        %RxIndicationStruct Format of PDSCH/PUSCH Rx information sent to MAC 
        RxIndicationStruct = struct('NodeID',[],'RNTI',[],'HARQID',[],'TBS',[],'CRCFlag',[],'MACPDU',[]);
    end

    properties (Constant, Hidden)
        %FrameDurationInNS Frame duration in nanosecond
        FrameDurationInNS = 10e6;

        %PXSCHPacketType PDSCH/PUSCH packet type identifier in wireless packet
        PXSCHPacketType = 0;

        %CSIRSPacketType CSI-RS packet type identifier in wireless packet
        CSIRSPacketType = 1;

        %SRSPacketType SRS packet type identifier in wireless packet
        SRSPacketType = 2;
    end

    properties(Hidden)
        %StatTransmittedPackets Number of PDSCH/PUSCH packets transmitted
        StatTransmittedPackets

        %StatReceivedPackets Number of PDSCH/PUSCH packets received
        StatReceivedPackets

        %StatDecodeFailures Number of PDSCH/PUSCH decode failures
        StatDecodeFailures
    end

    properties(SetAccess=protected, Hidden)
        %RxBuffer Reception buffer (of type wirelessnetwork.internal.interferenceBuffer) to hold incoming packets till processing
        RxBuffer
    end

    methods(Access = public)
        function obj = nrPHY(param, notificationFcn)
            % Constructor

            obj.NotificationFcn = notificationFcn;
            inputParam = {'TransmitPower', 'NumTransmitAntennas', 'NumReceiveAntennas', 'NoiseFigure', 'ReceiveGain'};
            for idx=1:numel(inputParam)
                obj.(char(inputParam{idx})) = param.(inputParam{idx});
            end

            % 5G NR packet
            obj.PacketStruct.Type = 2; % 5G packet
            obj.PacketStruct.DirectToDestination = 0;
            obj.PacketStruct.NumTransmitAntennas = obj.NumTransmitAntennas;

            % Initialize currTimingInfo
            obj.CurrTimeInfo = struct('Time', 0, 'NFrame', 0, 'NSlot', 0, 'NSymbol', 0);
        end

        function registerTxHandle(obj, sendPacketFcn)
            %registerTxHandle Register function handle for transmission
            %
            %   SENDPACKETFCN Function handle provided by node to PHY for packet transmission

            obj.SendPacketFcn = sendPacketFcn;
        end

        function nextInvokeTime = run(obj, currentTime, packets)
            %run Run the PHY layer operations and return the next invoke time (in nanoseconds)
            %   NEXTINVOKETIME = run(OBJ, CURRENTTIME, PACKETS) runs the PHY layer
            %   operations and returns the next invoke time (in nanoseconds).
            %
            %   NEXTINVOKETIME is the next invoke time (in nanoseconds) for
            %   PHY.
            %
            %   CURRENTTIME is the current time (in nanoseconds).
            %
            %   PACKETS are the received packets from other nodes.

            nextInvokeTime = Inf;
            if ~isempty(obj.CarrierInformation) % If carrier is configured
                symEndTimes = obj.CarrierInformation.SymbolTimings;
                slotDuration = obj.CarrierInformation.SlotDuration; % In nanoseconds

                % Find the duration completed in the current slot
                durationCompletedInCurrSlot = mod(currentTime, slotDuration);

                % Calculate the current NFrame, slot and symbol
                currTimeInfo = obj.CurrTimeInfo;
                currTimeInfo.Time = currentTime;
                currTimeInfo.NFrame = floor(currentTime/obj.FrameDurationInNS);
                currTimeInfo.NSlot = mod(floor(currentTime/slotDuration), obj.CarrierInformation.SlotsPerFrame);
                currTimeInfo.NSymbol = find(durationCompletedInCurrSlot < symEndTimes, 1) - 1;

                % PHY transmission.
                phyTx(obj, currTimeInfo);

                % Store the received packets
                storeReception(obj, packets);

                % PHY reception
                phyRx(obj, currTimeInfo);

                % Get the next invoke time for PHY
                nextInvokeTime = getNextInvokeTime(obj);
            end
            % Update the last run time
            obj.LastRunTime = currentTime;
        end
    end

    methods(Access = protected)
        function setCarrierInformation(obj, carrierInfo)
            % Set the carrier configuration
           
            slotDuration = 1/(carrierInfo.SubcarrierSpacing/15); % In ms
            carrierInfo.SlotDuration = slotDuration * 1e6; % In nanoseconds
            carrierInfo.SlotsPerSubframe = 1/slotDuration; % Number of slots per 1 ms subframe
            slotsPerFrame = carrierInfo.SlotsPerSubframe*10;
            carrierInfo.SlotsPerFrame = slotsPerFrame; % Number of slots per frame
            carrierInfo.SymbolsPerFrame = slotsPerFrame*14;
            carrierInfo.SymbolTimings = round(((1:14)*slotDuration)/14, 4) * 1e6; % Symbol end times in nanoseconds
            symbolStartTimes = round(((0:13)*slotDuration)/14, 4) * 1e6; % In nanoseconds
            carrierInfo.SymbolDurations = carrierInfo.SymbolTimings - symbolStartTimes;
            obj.CarrierInformation = carrierInfo;

            % Initialize data Rx context
            obj.DataRxContext = cell(carrierInfo.SymbolsPerFrame, 1);

            % Set carrier waveform properties
            obj.WaveformInfo = nrOFDMInfo(carrierInfo.NumResourceBlocks, carrierInfo.SubcarrierSpacing);

            % Create an nrCarrierConfig object
            obj.CarrierConfig = nrCarrierConfig;
            obj.CarrierConfig.SubcarrierSpacing = carrierInfo.SubcarrierSpacing;
            obj.CarrierConfig.NSizeGrid = carrierInfo.NumResourceBlocks;
            obj.CarrierConfig.NCellID = carrierInfo.NCellID;
        end

        function T = samplesInSlot(~, carrier)
            % Get the OFDM symbol lengths of a slot in terms of samples

            ofdmInfo = nrOFDMInfo(carrier);
            L = carrier.SymbolsPerSlot;
            symbolLengths = ofdmInfo.SymbolLengths(mod(carrier.NSlot,carrier.SlotsPerSubframe)*L + (1:L));
            T = sum(symbolLengths);
        end

        function nVar = calculateThermalNoise(obj)
            % Calculate thermal noise

            noiseFigure = 10^(obj.NoiseFigure/10);
            sampleRate = obj.WaveformInfo.SampleRate;
            nFFT = obj.WaveformInfo.Nfft;

            % Thermal noise (in Watts)
            Nt = physconst('Boltzmann') * (obj.AntNoiseTemperature + 290*(noiseFigure-1)) * sampleRate;
            N0 = sqrt(Nt/2);
            nVar = 2.0 * nFFT * N0^2;
        end

        function setCQITable(obj, cqiTable)
            % Set CQI table values

            cqiTableValues = nr5g.internal.nrCQITables(cqiTable);
            obj.CQITableValues = cqiTableValues(:, 2:3); % Keep 2 columns: modulation and coderate
        end

        function setMCSTable(obj, mcsTable)
            % Set MCS table values

            obj.MCSTableValues = [mcsTable.("Modulation Order") mcsTable.("Code Rate x 1024")];
        end

        function carrierStruct = createCarrierStruct(~, param)
            % Create carrier information structure

            structFields = {'SubcarrierSpacing', 'NumResourceBlocks', 'ChannelBandwidth', ...
                'DLCarrierFrequency', 'ULCarrierFrequency', 'NCellID', 'DuplexMode'};

            for idx=1:numel(structFields)
                carrierStruct.(char(structFields{idx})) = param.(structFields{idx});
            end
        end

        function [startTime, endTime] = pktTiming(obj, nFrame, nSlot, allocationStartSym, numSym)
            % Calculate packet rx start time and end time (in seconds)

            slotDuration = obj.CarrierInformation.SlotDuration;
            startTime = (10e6*nFrame + slotDuration*nSlot + sum(obj.CarrierInformation.SymbolDurations(1:allocationStartSym)))/1e9;
            endTime = startTime + (sum(obj.CarrierInformation.SymbolDurations(...
                allocationStartSym+1:allocationStartSym+numSym))/1e9);
        end

        function [txSlot, txSlotAFN] = txSlotInfo(~, slotsPerSubframe, currTimingInfo)
            % Returns tx slot information

            if currTimingInfo.NSymbol == 0 % Current symbol is first in the slot hence transmission was done in the last slot
                if currTimingInfo.NSlot > 0
                    txSlot = currTimingInfo.NSlot-1;
                    txSlotAFN = currTimingInfo.NFrame; % Tx slot was in the current frame
                else
                    txSlot = slotsPerSubframe*10-1;
                    txSlotAFN = currTimingInfo.NFrame - 1; % Tx slot was in the previous frame
                end
            else
                % Get the Tx slot
                txSlot = currTimingInfo.NSlot;
                txSlotAFN = currTimingInfo.NFrame; % Tx slot was in the current frame
            end
        end

        function [startSampleIdx, endSampleIdx] = sampleIndices(obj, slotNum, startSym, endSym)
            %  Get sample indices corresponding to allocation in the slot

            slotNumSubFrame = mod(slotNum, obj.WaveformInfo.SlotsPerSubframe);
            startSymSubframe = slotNumSubFrame*obj.WaveformInfo.SymbolsPerSlot+1; % Start symbol of slot in the subframe
            lastSymSubframe = startSymSubframe+obj.WaveformInfo.SymbolsPerSlot-1; % Last symbol of slot in the subframe
            symbolLengths = obj.WaveformInfo.SymbolLengths(startSymSubframe:lastSymSubframe); % Length of symbols of slot
            startSampleIdx = sum(symbolLengths(1:startSym))+1;
            endSampleIdx = sum(symbolLengths(1:endSym+1));
        end
    end

    methods(Abstract)
        %registerMACHandle Register MAC interface functions at PHY for sending information to MAC
        %
        % registerMACHandle(obj, SENDMACPDUFCN, SENDCHANNELQUALITYFCN) registers
        % the callback function to send decoded MAC PDUs and measured channel
        % quality to MAC.
        %
        % SENDMACPDUFCN callback is a function handle with the signature, fcn(
        % RXINFO). PHY uses this handle to send the decoded data packet to MAC.
        % RXINFO is a structure containing the data and associated information, as
        % defined by the property 'RxIndicationStruct'.
        %
        % For gNB PHY, SENDCHANNELQUALITYFCN is a function handle with the
        % signature fcn(RANK, PMISET, CQI). PHY uses this handle to send the
        % measured DL channel measurements to MAC.
        % RANK - Rank indicator
        % PMISET - PMI set corresponding to RANK. It is a structure with fields
        % 'i11', 'i12', 'i13', 'i2'.
        % CQI - CQI corresponding to RANK and PMISET. It is a vector of size 'N',
        % where 'N' is number of RBs in bandwidth. Value at index 'i' represents
        % CQI value at RB-index 'i'.
        %
        % For UE PHY, SENDCHANNELQUALITYFCN is a function handle with the signature
        % fcn(CSIMEASUREMENT). PHY uses this handle to send the measured UL
        % channel measurements to MAC. 
        % CSIMEASUREMENT is a structure and contains these fields.
        %   RNTI           - UE corresponding to the SRS
        %   RankIndicator  - Rank indicator
        %   TPMI           - Measured transmitted precoding matrix indicator (TPMI)
        %   CQI            - CQI corresponding to RANK and TPMI. It is a vector
        %                    of size 'N', where 'N' is number of RBs in bandwidth. Value
        %                    at index 'i' represents CQI value at RB-index 'i'.
        registerMACHandle(obj, sendMACPDUFcn, sendChannelQualityFcn)

        %txDataRequest Data Tx request from MAC to PHY
        % txDataRequest(OBJ, TXINFO, MACPDU, TIMINGINFO) is the request from MAC to
        % PHY to transmit PDSCH (for gNB) or PUSCH (for UE). MAC calls it at the
        % start of Tx time.
        %
        % TXINFO is the information sent by MAC which is required for PHY
        % processing and transmission. For gNB, it's a structure of type PDSCHInfo
        % (See txDataRequest function in nrGNBPHY.m for structure details). For UE,
        % it's a structure of type PUSCHInfo (See txDataRequest function in
        % nrUEPHY.m for structure details).
        %
        % MACPDU is the MAC transport block
        %
        % TIMINGINFO is a structure that contains the these fields.
        %   NFrame      - Current frame number
        %   NSlot       - Current slot number in a 10 millisecond frame
        %   NSymbol     - Current symbol number in the current slot
        %   Timestamp   - Current time in nanoseconds.
        txDataRequest(obj, txInfo, macPDU, timingInfo)

        %rxDataRequest Data Rx request from MAC to PHY
        % rxDataRequest(OBJ, RXINFO, TIMINGINFO) is the request from MAC to PHY to
        % receive PUSCH (for gNB) or PDSCH (for UE). The PHY expects to receive
        % this request at the start of reception time.
        %
        % RXINFO is the information sent by MAC which is required by PHY to receive
        % the packet. For gNB, it's a structure of type PUSCHInfo (See
        % rxDataRequest function in nrGNBPHY.m for structure details). For UE, it's
        % a structure of type PDSCHInfo (See rxDataRequest function in nrUEPHY.m
        % for structure details).
        %
        % TIMINGINFO See "txDataRequest" description above for this.
        rxDataRequest(obj, rxInfo, timingInfo)

        %dlControlRequest Downlink control request from MAC to PHY
        % dlControlRequest(OBJ, PDUTYPES, DLCONTROLPDUS, TIMINGINFO) is an
        % indication from MAC for non-data downlink transmissions/receptions. For
        % gNB, it is sent by gNB MAC for DL transmissions. For UE, it is sent by UE
        % MAC for DL receptions. MAC sends it at the start of a DL slot for all the
        % scheduled DL transmission/receptions in the slot. This interface is used
        % for all other DL transmission/reception except for PDSCH
        % transmission/reception.
        %
        % PDUTYPES is an array of DL packet types.
        %
        % DLCONTROLPDUS is an array of DL control PDUs corresponding to PDUTYPES.
        %
        % TIMINGINFO See "txDataRequest" description above for this.
        dlControlRequest(obj, pduTypes, dlControlPDUs, timingInfo)

        %ulControlRequest Uplink control request from MAC to PHY
        % ulControlRequest(OBJ, PDUTYPES, ULCONTTROLPDUS, TIMINGINFO) is an
        % indication from MAC for non-data uplink transmissions/receptions. For gNB
        % it is sent by gNB MAC for UL receptions. For UE, it is sent by UE MAC for
        % UL transmissions. MAC sends it at the start of a UL slot for all the
        % scheduled UL transmission/receptions in the slot.  This interface is used
        % for all other UL transmission/reception except for PUSCH
        % transmission/reception.
        %
        % PDUTYPES is an array of UL packet types.
        %
        % ULCONTROLPDUS is an array of UL control PDUs corresponding to PDUTYPES.
        %
        % TIMINGINFO See "txDataRequest" description above for this.
        ulControlRequest(obj, pduTypes, ulControlPDUs, timingInfo)
    end

    methods(Abstract, Access=protected)
        %phyTx PHY transmission
        % phyTx(OBJ, TIMINGINFO) transmits the scheduled PHY packets.
        % TIMINGINFO is a structure that contains the following
        % fields.
        % TIMINGINFO is a structure that contains the these fields.
        %   NFrame      - Current frame number
        %   NSlot       - Current slot number in a 10 millisecond frame
        %   NSymbol     - Current symbol number in the current slot
        %   Timestamp   - Current time in nanoseconds
        phyTx(obj, timingInfo);

        %phyRx PHY reception
        % phyRx(OBJ, TIMINGINFO) receives the scheduled PHY packets.
        %
        % TIMINGINFO See "phyTx" description above for this.
        phyRx(obj, timingInfo)
    end

    methods(Access=protected)
        function nextInvokeTime = getNextInvokeTime(obj)
            %getNextInvokeTime Return the next invoke time in nanoseconds

            nextInvokeTime = min([min(obj.NextTxTime) min(obj.NextRxTime)]);
        end

        function storeReception(obj, packets)
            %storeReception Receive the incoming packets and add them to the reception buffer

            % Loop for all packets
            for pktIdx = 1:numel(packets)
                packetInfo = packets(pktIdx);

                % Skip the packets that are configured as DirectToDestination
                if packetInfo.DirectToDestination == 0
                    addPacket(obj.RxBuffer, packetInfo);
                end
            end
        end
    end
end
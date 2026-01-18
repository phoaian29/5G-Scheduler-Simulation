classdef (Abstract) nrGNBPHY < nr5g.internal.nrPHY
    %nrGNBPHY Define NR physical layer base class for gNB
    %   The class acts as a base class for all the physical layer types of gNB.
    %
    %   Note: This is an internal undocumented class and its API and/or
    %   functionality may change in subsequent releases.

    %   Copyright 2023-2024 The MathWorks, Inc.

    properties(SetAccess=protected)
        %UEInfo Information about the UEs connected to the gNB
        % N-by-1 array where 'N' is the number of UEs. Each element in the array is
        % a structure with three fields.
        %   ID      - Node ID of the UE
        %   Name    - Node name of the UE
        %   RNTI    - RNTI of the connected UE
        UEInfo
    end

    properties (Access = protected)
        %CSIRSInfo CSI-RS transmission information sent by MAC for the current slot
        % It is an array of objects of type nrCSIRSConfig. Each element contains a
        % CSI-RS configuration which corresponds to a CSI-RS packet. Value as empty
        % means that no CSI-RS Tx is scheduled for the current slot.
        CSIRSInfo = []

        %SRSInfo Rx context for the sounding reference signal (SRS)
        % This information is populated by MAC and is used by PHY to receive
        % scheduled SRS. It is a cell array of size 'N' where N is the number of
        % symbols in a 10 ms frame. The cell elements are populated with objects of
        % type nrSRSConfig. An element at index 'i' contains the configuration of
        % SRS which is sent between the symbol index 'i-14' to 'i' (i.e during the
        % slot). Cell element at 'i' is empty if no SRS reception was scheduled in
        % the slot.
        SRSInfo = {}

        %PDSCHInfo PDSCH information sent by MAC for the current slot
        % It is an array of structures of PDSCHInfo (See txDataRequest function for
        % structure details). An element at index 'i' contains the information
        % required by PHY to transmit a MAC PDU stored at index 'i' of property
        % 'MacPDU'.
        PDSCHInfo = []

        %MacPDU PDUs sent by MAC which are scheduled to be sent in the current slot
        % It is a cell array of downlink MAC PDUs to be sent in the current slot.
        % Each element in the array is a MAC PDU and corresponds to a structure
        % element in the property PDSCHInfo.
        MacPDU = {}

        %SRSIndicationFcn Function handle to send the measured UL channel quality to MAC
        SRSIndicationFcn

        %RankIndicator UL Rank to calculate precoding matrix and CQI
        % Vector of length 'N' where N is number of UEs. Value at index 'i'
        % contains UL rank of UE with RNTI 'i'
        RankIndicator

        %PHYStatsInfo PHY statistics for all UEs
        PHYStatsInfo

        %PUSCHConfig PUSCH configuration for SRS measurements
        PUSCHConfig = nrPUSCHConfig
    end

    methods
        function obj = nrGNBPHY(param, notificationFcn)
            % Constructor

            obj = obj@nr5g.internal.nrPHY(param, notificationFcn); % Call base class constructor
            setCarrierInformation(obj, createCarrierStruct(obj, param)); % Set carrier information
            
            if isfield(param, 'CQITable')
               setCQITable(obj, param.CQITable);
            end

            if isfield(param, 'MCSTable')
                setMCSTable(obj, param.MCSTable);
            end

            % Initialize MACPDUInfo
            obj.MACPDUInfo = struct('NodeID', 0, 'RNTI', 0, 'TBS', 0, 'MACPDU', [], 'CRCFlag', 1, 'HARQID', 0, 'Tags', []);

            % Initialize interference buffer
            obj.RxBuffer = wirelessnetwork.internal.interferenceBuffer(CenterFrequency=obj.CarrierInformation.ULCarrierFrequency, ...
                Bandwidth=obj.CarrierInformation.ChannelBandwidth, SampleRate=obj.WaveformInfo.SampleRate, ResultantWaveformDataType="single", ...
                DisableValidation=true);

            % Initialize per symbol context
            symbolsPerFrame = obj.CarrierInformation.SlotsPerSubframe*10*14;
            obj.SRSInfo = cell(symbolsPerFrame, 1);
            obj.NextTxTime = Inf*ones(symbolsPerFrame, 1);
            obj.NextRxTime = Inf*ones(symbolsPerFrame, 1);

            % Set NR packet fields
            obj.PacketStruct.CenterFrequency = obj.CarrierInformation.DLCarrierFrequency;
            obj.PacketStruct.Bandwidth = obj.CarrierInformation.ChannelBandwidth;

            obj.PUSCHConfig.PRBSet = (0:obj.CarrierInformation.NumResourceBlocks-1);
        end

        function addConnection(obj, connectionConfig)
            %addConnection Configures the gNB PHY with a UE connection information
            %   addConnection(OBJ, CONNECTIONCONFIG) adds the UE connection information
            %   to gNB PHY. CONNECTIONCONFIG is a structure including the following
            %   fields:
            %       RNTI      - RNTI of the UE.
            %       UEID      - Node ID of the UE
            %       UEName    - Node name of the UE
            %       NumHARQ   - Number of HARQ processes for the UE
            %       DuplexMode- "FDD" or "TDD"

            nodeInfo = struct('ID', connectionConfig.UEID, 'Name', connectionConfig.UEName, 'RNTI', connectionConfig.RNTI);
            obj.UEInfo = [obj.UEInfo; nodeInfo];

            phyStatsInfo = struct('UEID', connectionConfig.UEID, 'UEName', connectionConfig.UEName, ...
                'RNTI', connectionConfig.RNTI,'TransmittedPackets', 0, ...
                'ReceivedPackets', 0, 'DecodeFailures', 0);
            obj.PHYStatsInfo = [obj.PHYStatsInfo; phyStatsInfo];

            obj.StatTransmittedPackets = [obj.StatTransmittedPackets; 0];
            obj.StatReceivedPackets = [obj.StatReceivedPackets; 0];
            obj.StatDecodeFailures = [obj.StatDecodeFailures; 0];

            % UL rank indicator (fixed to 1)
            obj.RankIndicator = [obj.RankIndicator 1];

            % Set the PacketTransmissionStarted event information
            txStartStruct = obj.PacketTxStartedStruct;
            txStartStruct.DuplexMode = connectionConfig.DuplexMode;
            txStartStruct.RNTI = connectionConfig.RNTI;
            txStartStruct.LinkType = "DL";
            obj.PacketTransmissionStarted = [obj.PacketTransmissionStarted; txStartStruct];

            % Set the PacketReceptionEnded event information
            rxEndStruct = obj.PacketRxEndedStruct;
            rxEndStruct.DuplexMode = connectionConfig.DuplexMode;
            rxEndStruct.RNTI = connectionConfig.RNTI;
            rxEndStruct.LinkType = "UL";
            obj.PacketReceptionEnded = [obj.PacketReceptionEnded; rxEndStruct];

            obj.HARQBuffers = [obj.HARQBuffers; cell(1,connectionConfig.NumHARQ)]; % Add HARQ buffers for the UE

            % Add packet tag information buffer for the UE
            obj.ReTxTagBuffer = [obj.ReTxTagBuffer; cell(1,connectionConfig.NumHARQ)];
        end

        function registerMACHandle(obj, sendMACPDUFcn, sendULChannelQualityFcn)
            %registerMACHandle Register MAC interface functions at PHY, for sending information to MAC

            obj.RxIndicationFcn = sendMACPDUFcn;
            obj.SRSIndicationFcn = sendULChannelQualityFcn;
        end

        function scaledTransmitPower = scaleTransmitPower(obj)
            % scaleTransmitPower Return the scaled gNB transmit power to the node
            
            % Apply FFT Scaling to the transmit power
            powerScalingFactor = sqrt((obj.WaveformInfo.Nfft^2) / (12*obj.CarrierConfig.NSizeGrid));
            scaledTransmitPower = mag2db(db2mag(obj.TransmitPower-30)*powerScalingFactor) + 30; % dBm
        end

        function phyStats = statistics(obj)
            %statistics Return the PHY statistics for all the connected UEs
            %
            %   PHYSTATS = statistics(OBJ) returns the PHY statistics for all the connected UEs.
            %
            %   PHYSTATS - Nx1 array of structures, where N is the number of UEs
            %   connected to the gNB. Each structure contains following fields.
            %       UEID                - Node ID of the UE
            %       UEName              - Node name of the UE
            %       RNTI                - RNTI of the UE
            %       TransmittedPackets  - Number of transmitted PDSCHs
            %       ReceivedPackets     - Number of received PUSCHs
            %       DecodeFailures      - Number of PUSCH decode failures

            numUEs = numel(obj.UEInfo);
            phyStats = obj.PHYStatsInfo;

            for ueIdx=1:numUEs
                phyStats(ueIdx).TransmittedPackets = obj.StatTransmittedPackets(ueIdx);
                phyStats(ueIdx).ReceivedPackets = obj.StatReceivedPackets(ueIdx);
                phyStats(ueIdx).DecodeFailures = obj.StatDecodeFailures(ueIdx);
            end
        end

        function txDataRequest(obj, pdschInfo, packetInfo, timingInfo)
            %txDataRequest Tx request from MAC to PHY to start PDSCH transmission
            %   txDataRequest(OBJ, PDSCHINFO, PACKETINFO, TIMINGINFO) sets
            %   the Tx context for PDSCH transmission in the current
            %   symbol.
            %
            %   PDSCHInfo is a structure which is sent by MAC and contains the
            %   information required by the PHY for the PDSCH transmission. It contains
            %   these fields.
            %       NSlot           - Slot number of the PDSCH transmission
            %       HARQID          - HARQ process ID
            %       NewData         - Defines if it is a new transmission (Value 1) or re-transmission (Value 0)
            %       RV              - Redundancy version of the transmission
            %       TargetCodeRate  - Target code rate
            %       TBS             - Transport block size in bytes
            %       PrecodingMatrix - Precoding matrix
            %       BeamIndex       - Column index in the beam weights table configured at PHY
            %       PDSCHConfig     - PDSCH configuration object as described in
            %                         <a href="matlab:help('nrPDSCHConfig')">nrPDSCHConfig</a>
            %
            %   PACKETINFO is the downlink MAC PDU sent by MAC for transmission and the
            %   associated information. It is a structure with these fields.
            %      Packet       - MAC PDU
            %      PacketLength - MAC PDU length in bytes
            %      Tags         - Array of structures where each structure
            %                     contains these fields.
            %                     Name      - Name of the tag.
            %                     Value     - Data associated with the tag.
            %                     ByteRange - Specific range of bytes within the
            %                                 packet to which the tag applies.
            %
            %   TIMINGINFO is a structure that contains the these fields.
            %     NFrame      - Current frame number
            %     NSlot       - Current slot number in the 10 millisecond frame
            %     NSymbol     - Current symbol number in the current slot
            %     Timestamp   - Reception start timestamp in nanoseconds

            symbolNumFrame = pdschInfo.NSlot * 14 + pdschInfo.PDSCHConfig.SymbolAllocation(1);
            % Calculate the time to transmit and store it at the corresponding symbol
            % number
            if (obj.PhyTxProcessingDelay == 0)
                obj.NextTxTime(symbolNumFrame+1) = timingInfo.Timestamp;
            else
                symDur = obj.CarrierInformation.SymbolDurations; % In nanoseconds
                phyProcessingStart = currTimingInfo.NSymbol;
                phyProcessingEnd = currTimingInfo.NSymbol + obj.PhyTxProcessingDelay - 1;
                phyProcessingSyms = (phyProcessingStart:phyProcessingEnd) + 1;
                obj.NextTxTime(symbolNumFrame+obj.PhyTxProcessingDelay+1) = timingInfo.Timestamp + sum(symDur(phyProcessingSyms));
            end
            % Update the Tx context. There can be multiple simultaneous PDSCH
            % transmissions for different UEs
            obj.MacPDU{end+1} = packetInfo;
            obj.PDSCHInfo = [obj.PDSCHInfo pdschInfo];
        end

        function rxDataRequest(obj, puschInfo, timingInfo)
            %rxDataRequest Rx request from MAC to PHY to start PUSCH reception
            %   rxDataRequest(OBJ, PUSCHINFO, TIMINGINFO) is a request to start PUSCH
            %   reception. It starts a timer for PUSCH end time (which on firing
            %   receives the PUSCH). The PHY expects the MAC to send this request at
            %   the start of reception time.
            %
            %   PUSCHInfo is a structure which is sent by MAC and contains the
            %   information required by the PHY for the PUSCH reception. It contains
            %   these fields.
            %       NSlot           - Slot number of the PUSCH transmission
            %       HARQID          - HARQ process ID
            %       NewData         - Defines if it is a new transmission (Value 1) or
            %                         re-transmission (Value 0)
            %       RV              - Redundancy version of the transmission
            %       TargetCodeRate  - Target code rate
            %       TBS             - Transport block size in bytes
            %       PUSCHConfig     - PUSCH configuration object as described in
            %                        <a href="matlab:help('nrPUSCHConfig')">nrPUSCHConfig</a>
            %
            %   TIMINGINFO is a structure that contains the following
            %   fields.
            %     NFrame      - Current frame number
            %     NSlot       - Current slot number in the 10 millisecond frame
            %     NSymbol     - Current symbol number in the current slot
            %     Timestamp   - Reception start timestamp in nanoseconds

            puschStartSym = puschInfo.PUSCHConfig.SymbolAllocation(1);
            symbolNumFrame = puschInfo.NSlot*14 + puschStartSym; % PUSCH Rx start symbol number w.r.t start of 10 ms frame

            % PUSCH to be read at the end of last symbol in PUSCH reception
            numPUSCHSym =  puschInfo.PUSCHConfig.SymbolAllocation(2);
            puschRxSymbolFrame = symbolNumFrame + numPUSCHSym;
            symDur = obj.CarrierInformation.SymbolDurations; % In nanoseconds
            startSymbolIdx = puschStartSym + 1;
            endSymbolIdx = puschStartSym + numPUSCHSym;

            % Add the PUSCH Rx information at the index corresponding to the symbol
            % where PUSCH Rx ends
            obj.DataRxContext{puschRxSymbolFrame}{end+1} = puschInfo;
            % Store data reception time (in nanoseconds) information
            obj.NextRxTime(puschRxSymbolFrame) = timingInfo.Timestamp + ...
                sum(symDur(startSymbolIdx:endSymbolIdx));
        end

        function dlControlRequest(obj, pduType, dlControlPDU, timingInfo)
            %dlControlRequest Downlink control (non-data) transmission request from MAC to PHY
            %   dlControlRequest(OBJ, PDUTYPE, DLCONTROLPDU) is a request from MAC for
            %   downlink control transmission. MAC sends it at the start of a DL slot
            %   for all the scheduled non-data DL transmission in the slot.
            %
            %   PDUTYPE is an array of packet types. Currently, only packet
            %   type 0 (CSI-RS) is supported.
            %
            %   DLCONTROLPDU is an array of DL control information PDUs. Each PDU is
            %   stored at the index corresponding to its type in PDUTYPE. Currently
            %   supported CSI-RS information PDU is an object of type nrCSIRSConfig.

            % Update the Tx context
            if ~isempty(dlControlPDU)
                obj.CSIRSInfo = dlControlPDU;
                txSymbolFrame = (timingInfo.NSlot)*14;
                obj.NextTxTime(txSymbolFrame+1) = timingInfo.Timestamp;
            end
        end

        function ulControlRequest(obj, pduType, ulControlPDU, timingInfo)
            %ulControlRequest Uplink control (non-data) reception request from MAC to PHY
            %   ulControlRequest(OBJ, PDUTYPE, ULCONTROLPDU, TIMINGINFO) is a request
            %   from MAC for uplink control reception. MAC sends it at the start of a UL slot
            %   for all the scheduled non-data UL receptions in the slot.
            %
            %   PDUTYPE is an array of packet types. Currently, only packet type 1
            %   (SRS) is supported.
            %
            %   ULCONTROLPDU is an array of UL control information PDUs. Each PDU is
            %   stored at the index corresponding to its type in PDUTYPE. Currently
            %   supported SRS information PDU is an object of type nrSRSConfig.
            %
            %   TIMINGINFO is a structure that contains the following
            %   fields.
            %     NFrame      - Current frame number
            %     NSlot       - Current slot number in the 10 millisecond frame
            %     NSymbol     - Current symbol number in the current slot
            %     Timestamp   - Reception start timestamp in nanoseconds

            % SRS would be read at the end of the current slot
            rxSymbolFrame = (timingInfo.NSlot + 1) * obj.CarrierConfig.SymbolsPerSlot;
            obj.SRSInfo{rxSymbolFrame} = ulControlPDU;
            obj.NextRxTime(rxSymbolFrame) = timingInfo.Timestamp + (15e6/obj.CarrierInformation.SubcarrierSpacing); % In nanoseconds
        end
    end

    methods(Access=protected)
        function phyTx(obj, currTimingInfo)
            %phyTx Physical layer transmission

            symbolNumFrame = mod(currTimingInfo.NSlot*14 + currTimingInfo.NSymbol, ...
                obj.CarrierInformation.SymbolsPerFrame);
            if obj.NextTxTime(symbolNumFrame+1) ~= Inf % Check if any Tx is scheduled now
                pdschTx(obj, currTimingInfo); % Transmit PDSCH(s)
                csirsTx(obj, currTimingInfo); % Transmit CSI-RS
                obj.NextTxTime(symbolNumFrame+1) = Inf; % Reset
                eventData = obj.PacketTransmissionStarted;
                for idx=1:numel(eventData)
                    if ~isempty(eventData(idx).SignalType)
                        eventData(idx).TimingInfo = [currTimingInfo.NFrame currTimingInfo.NSlot currTimingInfo.NSymbol];
                        obj.NotificationFcn('PacketTransmissionStarted', eventData(idx));
                        % Reset the event data context
                        obj.PacketTransmissionStarted(idx).HARQID = [];
                        obj.PacketTransmissionStarted(idx).SignalType = [];
                        obj.PacketTransmissionStarted(idx).PDU =[];
                        obj.PacketTransmissionStarted(idx).TransmissionType = [];
                    end
                end

            end
        end

        function phyRx(obj, currTimingInfo)
            %phyRx Physical layer reception

            symbolNumFrame = mod(currTimingInfo.NSlot*14 + currTimingInfo.NSymbol-1, ...
                obj.CarrierInformation.SymbolsPerFrame); % Previous symbol number in the 10 ms frame
            if obj.NextRxTime(symbolNumFrame+1) ~= Inf % Check if any Rx is scheduled now
                puschRx(obj, currTimingInfo); % Receive PUSCH(s)
                srsRx(obj, currTimingInfo); % Receive SRS(s)
                obj.NextRxTime(symbolNumFrame+1) = Inf; % Reset

                eventData = obj.PacketReceptionEnded;
                for idx=1:numel(eventData)
                    if ~isempty(eventData(idx).SignalType)
                        obj.NotificationFcn('PacketReceptionEnded', eventData(idx));
                        % Reset the event data context
                        obj.PacketReceptionEnded(idx).HARQID = [];
                        obj.PacketReceptionEnded(idx).SignalType = [];
                        obj.PacketReceptionEnded(idx).Duration = [];
                        obj.PacketReceptionEnded(idx).PDU = [];
                        obj.PacketReceptionEnded(idx).CRCFlag = [];
                        obj.PacketReceptionEnded(idx).SINR = -Inf;
                        obj.PacketReceptionEnded(idx).ChannelMeasurements = [];
                    end
                end
            end
        end
    end

    methods(Abstract)
        %pdschData PHY Tx processing of PDSCH
        % DATA = pdschData(OBJ, PDSCHINFO, MACPDU) returns the data after Tx
        % processing of PDSCH. MACPDU is the PDU sent by MAC. PDSCHINFO is a
        % structure which is sent by MAC and contains the information required by
        % the PHY for the PDSCH transmission. It contains these fields.
        %       NSlot           - Slot number of the PDSCH transmission
        %       HARQID          - HARQ process ID
        %       NewData         - Defines if it is a new transmission (Value 1) or
        %                         re-transmission (Value 0)
        %       RV              - Redundancy version of the transmission
        %       TargetCodeRate  - Target code rate
        %       TBS             - Transport block size in bytes
        %       PrecodingMatrix - Precoding matrix
        %       PDSCHConfig     - PDSCH configuration object as described in
        %                         <a href="matlab:help('nrPDSCHConfig')">nrPDSCHConfig</a>
        %
        % DATA is PHY-processed PDSCH. DATA could be simply MACPDU for abstract
        % flavours of PHY or processed PDSCH waveform for full PHY flavours.
        data = pdschData(obj, pdschInfo, macPDU);

        %pdschPacket Create PDSCH packet(s) from PDSCH data
        % PACKET = pdschPacket(OBJ, PDSCHINFOLIST, PDSCHDATALIST, TXSTARTTIME)
        % returns the PDSCH packet(s) to be transmitted. PDSCHINFOLIST is the
        % information about PDSCH(s) to be transmitted as a structure array.
        % PDSCHDATALIST is a cell array of PDSCH data where each element
        % corresponds to output of pdschData method. An element at its index 'i'
        % corresponds to the element at index 'i' in the input PDSCHINFOLIST.
        % TXSTARTTIME is the transmission start time in seconds. PACKET represents
        % the PDSCH packet(s) ready to be transmitted. For abstract PHY, PACKET is
        % an array of PDSCH packets with each element representing a PDSCH
        % containing its respective MAC PDU. For full PHY, PACKET is a single PDSCH
        % packet containing the combined waveform corresponding to all PDSCH(s) to
        % be transmitted
        packet = pdschPacket(obj, pdschInfoList, pdschDataList, txStartTime);

        %csirsData PHY Tx processing of CSI-RS
        % DATA = csirsDATA(OBJ, CSIRSCONFIG) returns the data after PHY Tx
        % processing of CSI-RS based on CSI-RS configuration, CSIRSCONFIG. DATA
        % could be empty ([]) for abstract flavours of PHY or processed CSI-RS
        % waveform for full PHY flavours
        data = csirsData(obj, csirsConfig);

        %csirsPacket Create CSI-RS packet(s) from CSI-RS configuration(s)
        % PACKET = csirsPacket(OBJ, CSIRSINFOLIST, CSIRSDATALIST, TXSTARTTIME)
        % returns the CSI-RS packet(s) to be transmitted. CSIRSINFOLIST is the
        % configuration of CSI-RS(s) to be transmitted as an array of type
        % nrCSIRSConfig. CSIRSDATALIST is a cell array of CSI-RS data where each
        % element corresponds to output of csirsData method. An element at its
        % index 'i' corresponds to the element at index 'i' in the input
        % CSIRSINFOLIST. TXSTARTTIME is the transmission start time in seconds.
        % PACKET represents the CSI-RS packet(s) ready to be transmitted. For
        % abstract PHY, PACKET is an array of CSI-RS packets. For full PHY, PACKET
        % is a single CSI-RS packet containing the combined waveform corresponding
        % to all CSI-RS(s) to be transmitted
        packet = csirsPacket(obj, pdschInfoList, pdschDataList, txStartTime);

        %decodePUSCH PHY Rx processing of PUSCH
        % [MACPDULIST, CRCFLAGLIST, SINRLIST] = decodePUSCH(OBJ, PUSCHINFOLIST, STARTTIME,
        % ENDTIME, CARRIERCONFIGINFO) returns the decoded packet(s), MACPDULIST,
        % from received PUSCH(s) in the specified time window. PUSCHInfoList is a
        % structure array where each element corresponds to a PUSCH to be received
        % and contains the information required to receive and decode PUSCH. Each
        % element contains these fields.
        %   NSlot           - Slot number of the PUSCH transmission
        %   HARQID          - HARQ process ID
        %   NewData         - Defines if it is a new transmission (Value 1) or re-transmission (Value 0)
        %   RV              - Redundancy version of the transmission
        %   TargetCodeRate  - Target code rate
        %   TBS             - Transport block size in bytes
        %   PUSCHConfig     - PUSCH configuration object as described in
        %                     <a href="matlab:help('nrPUSCHConfig')">nrPUSCHConfig</a>
        %
        % STARTTIME and ENDTIME represent the time window (in seconds) where PUSCH
        % packets fall. CARRIERCONFIGINFO is the carrier related information of
        % type <a href="matlab:help('nrCarrierConfig')">nrCarrierConfig</a>.
        % MACPDULIST is a cell array where each element is a MAC PDU. CRCFLAGLIST
        % is an array of length same as MACPDULIST and contains binary values.
        % Value 0 and 1 represent CRC success and failure, respectively.
        % EFFECTIVESINRLIST is a numeric array where each element is the effective SINR of the
        % received packet.
        [macPDUList, crcFlagList, effectiveSINRList] = decodePUSCH(obj, puschInfoList, startTime, endTime, carrierConfigInfo)

        %decodeSRS PHY Rx processing of SRS
        % [srsMeasurementList, EFFECTIVESINRLIST] = decodeSRS(OBJ, STARTTIME, ENDTIME,
        % CARRIERCONFIGINFO) returns the SRS measurements for all the SRS received
        % in the time window defined by STARTTIME and ENDTIME. CARRIERCONFIGINFO is
        % the carrier related information of type <a
        % href="matlab:help('nrCarrierConfig')">nrCarrierConfig</a>.
        % SRSMEASUREMENTLIST is a structure array where each element contains these
        % fields.
        %   RNTI          - RNTI of the UE
        %   RankIndicator - Measured rank
        %   TPMI          - Measured tpmi
        %   CQI           - Measured CQI
        % EFFECTIVESINRLIST is a numeric array where each element is the effective SINR of the
        % received packet.
        [srsMeasurementList, effectiveSINRList] = decodeSRS(obj, startTime, endTime, carrierConfigInfo)
    end

    methods (Access = private)
        function pdschTx(obj, currTimingInfo)
            % Transmit the PDSCH(s) scheduled for current time

            pdschDataList = cell(size(obj.PDSCHInfo,2), 1);
            for i=1:size(obj.PDSCHInfo,2) % For each PDSCH scheduled to be sent now
                if ~isempty(obj.MacPDU{i}) % For new transmission
                    % Get the PDSCH data after PHY processing
                    pdschDataList{i} = pdschData(obj, obj.PDSCHInfo(i), obj.MacPDU{i}.Packet);
                    % Store the associated packet tags to deal with possible retransmission
                    obj.ReTxTagBuffer{obj.PDSCHInfo(i).PDSCHConfig.RNTI, ...
                        obj.PDSCHInfo(i).HARQID+1} = obj.MacPDU{i}.Tags;
                else % For retransmission
                    % Get the PDSCH data after PHY processing
                    pdschDataList{i} = pdschData(obj, obj.PDSCHInfo(i), []);
                end
                obj.StatTransmittedPackets(obj.PDSCHInfo(i).PDSCHConfig.RNTI) = obj.StatTransmittedPackets(obj.PDSCHInfo(i).PDSCHConfig.RNTI) + 1;
            end

            if ~isempty(obj.PDSCHInfo)
                % Create PDSCH packet(s) and send
                pktList = pdschPacket(obj, obj.PDSCHInfo, pdschDataList, currTimingInfo.Time/1e9);
                for i=1:numel(pktList)
                    obj.SendPacketFcn(pktList(i));
                end

                for i=1:numel(pdschDataList)
                    % Update event data
                    rnti = obj.PDSCHInfo(i).PDSCHConfig.RNTI;
                    harqID = obj.PDSCHInfo(i).HARQID;
                    if isempty(obj.MacPDU{i})
                        obj.PacketTransmissionStarted(rnti).TransmissionType = "ReTx";
                        macPDU = obj.HARQBuffers{rnti, harqID+1};
                    else
                        obj.PacketTransmissionStarted(rnti).TransmissionType = "NewTx";
                        macPDU = obj.MacPDU{i}.Packet;
                        obj.HARQBuffers{rnti, harqID+1} = macPDU;
                    end
                    obj.PacketTransmissionStarted(rnti).HARQID = harqID;
                    obj.PacketTransmissionStarted(rnti).SignalType = "PDSCH";
                    obj.PacketTransmissionStarted(rnti).PDU = macPDU;
                    obj.PacketTransmissionStarted(rnti).Duration = pktList(1).Duration; % Assume all the packets has same duration
                end
            end

            % Transmission done. Clear the Tx contexts
            obj.PDSCHInfo = [];
            obj.MacPDU = {};
        end

        function csirsTx(obj, currTimingInfo)
            % Transmit CSI-RS scheduled for current time

            numCSIRS = size(obj.CSIRSInfo,2);
            if numCSIRS == 0
                return;
            end
            csirsPacketList = cell(numCSIRS, 1);
            obj.CarrierConfig.NSlot = currTimingInfo.NSlot;
            obj.CarrierConfig.NFrame = currTimingInfo.NFrame;
            for i = 1:numCSIRS % For each CSI-RS scheduled to be sent now
                csirsConfig = obj.CSIRSInfo{i};
                csirsPacketList{i} = csirsData(obj, csirsConfig{1}); % Get the CSI-RS packet data
            end

            % Create CSI-RS packet(s) and send
            pktList = csirsPacket(obj, obj.CSIRSInfo, csirsPacketList, currTimingInfo.Time/1e9);
            for i=1:numel(pktList)
                obj.SendPacketFcn(pktList(i));
                % CSI-RS packet can be single packet transmitted to one or
                % more nodes. So, update the context for required nodes
                rntiList = pktList(i).Metadata.RNTI;
                for ueIdx=1:numel(rntiList)
                    if isempty(obj.PacketTransmissionStarted(rntiList(ueIdx)).SignalType)
                        obj.PacketTransmissionStarted(rntiList(ueIdx)).SignalType = "CSIRS";
                        obj.PacketTransmissionStarted(rntiList(ueIdx)).Duration = pktList(i).Duration;
                    else
                        obj.PacketTransmissionStarted(rntiList(ueIdx)).SignalType = obj.PacketTransmissionStarted(rntiList(ueIdx)).SignalType+"+CSIRS";
                        obj.PacketTransmissionStarted(rntiList(ueIdx)).Duration = max(obj.PacketTransmissionStarted(rntiList(ueIdx)).Duration, pktList(i).Duration);
                    end
                end
            end

            % Transmission done. Clear the Tx context
            obj.CSIRSInfo = [];
        end

        function puschRx(obj, currTimingInfo)
            %puschRx Receive the PUSCH(s) scheduled for current time

            % Read context of PUSCH(s) scheduled for current time
            symbolNumFrame = mod(currTimingInfo.NSlot*14 + currTimingInfo.NSymbol - 1, ...
                obj.CarrierInformation.SymbolsPerFrame); % Previous symbol in a 10 ms frame
            puschInfoList = obj.DataRxContext{symbolNumFrame+1};

            if isempty(puschInfoList)
                return;
            end
            % Set carrier information
            carrierConfigInfo = obj.CarrierConfig;
            slotsPerSubframe = obj.WaveformInfo.SlotsPerSubframe;
            [carrierConfigInfo.NSlot, carrierConfigInfo.NFrame] = txSlotInfo(obj, slotsPerSubframe, currTimingInfo);

            % Initializations
            minStartTime = Inf;
            maxEndTime = 0;

            numPUSCH = size(puschInfoList,2);
            % Calculate the time window of PUSCH(s) to be received now
            for i=1:numPUSCH
                puschInfo = puschInfoList{i};
                [pktStartTime, pktEndTime] = pktTiming(obj, carrierConfigInfo.NFrame, ...
                    carrierConfigInfo.NSlot, puschInfo.PUSCHConfig.SymbolAllocation(1), ...
                    puschInfo.PUSCHConfig.SymbolAllocation(2));
                minStartTime = min([minStartTime pktStartTime]);
                maxEndTime = max([maxEndTime pktEndTime]);
            end

            % Decode the PUSCH(s) to extract MAC PDU(s), and
            % corresponding effective SINR
            [macPDUList, crcFlagList, effectiveSINRList] = decodePUSCH(obj, cell2mat(puschInfoList), minStartTime, maxEndTime, carrierConfigInfo);
            [transmitterIDList, tagList] = getTagsInfo(obj, puschInfoList, minStartTime, maxEndTime, carrierConfigInfo);
            % Calculate the rx timing information
            if currTimingInfo.NSymbol == 0 % Reception ended in previous slot
                rxTimingInfo = [carrierConfigInfo.NFrame carrierConfigInfo.NSlot 13];
            else % Reception ended in current slot
                rxTimingInfo = [carrierConfigInfo.NFrame carrierConfigInfo.NSlot currTimingInfo.NSymbol-1];
            end
            % Send the PUSCH decode information to MAC and update the PHY stats
            for i=1:numPUSCH % For each PUSCH to be received
                puschInfo = puschInfoList{i};
                macPDUInfo = obj.MACPDUInfo;
                macPDUInfo.RNTI = puschInfo.PUSCHConfig.RNTI;
                macPDUInfo.HARQID = puschInfo.HARQID;
                macPDUInfo.MACPDU = macPDUList{i};
                macPDUInfo.CRCFlag = crcFlagList(i);
                macPDUInfo.TBS = puschInfo.TBS;
                macPDUInfo.NodeID = transmitterIDList(i);
                macPDUInfo.Tags = tagList{i};
                % Rx callback to MAC
                obj.RxIndicationFcn(macPDUInfo, currTimingInfo.Time);
                % Increment the number of received packets for UE
                rnti = macPDUInfo.RNTI;
                obj.StatReceivedPackets(rnti) = obj.StatReceivedPackets(rnti) + 1;
                % Increment the number of decode failures received for UE
                obj.StatDecodeFailures(rnti) = obj.StatDecodeFailures(rnti) + crcFlagList(i);

                % Update Rx event information
                obj.PacketReceptionEnded(rnti).TimingInfo = rxTimingInfo;
                obj.PacketReceptionEnded(rnti).HARQID = macPDUInfo.HARQID;
                obj.PacketReceptionEnded(rnti).SignalType = "PUSCH";
                obj.PacketReceptionEnded(rnti).Duration = pktEndTime-pktStartTime;
                obj.PacketReceptionEnded(rnti).PDU = macPDUList{i};
                obj.PacketReceptionEnded(rnti).CRCFlag = crcFlagList(i);
                obj.PacketReceptionEnded(rnti).SINR = effectiveSINRList(i);
            end
            obj.DataRxContext{symbolNumFrame+1} = {}; % Clear the context
        end

        function srsRx(obj, currTimingInfo)
            %srsRx  Receive SRS(s)

            % Read context of SRS scheduled for current time
            symbolNumFrame = mod(currTimingInfo.NSlot*14 + currTimingInfo.NSymbol - 1, ...
                obj.CarrierInformation.SymbolsPerFrame);
            srsInfoList = obj.SRSInfo{symbolNumFrame+1};
            if isempty(srsInfoList)
                return;
            end

            % Set carrier information
            carrierConfigInfo = obj.CarrierConfig;
            slotsPerSubframe = obj.WaveformInfo.SlotsPerSubframe;
            [carrierConfigInfo.NSlot, carrierConfigInfo.NFrame] = txSlotInfo(obj, slotsPerSubframe, currTimingInfo);

            [startTime, endTime] = pktTiming(obj, carrierConfigInfo.NFrame, ...
                carrierConfigInfo.NSlot, 0, carrierConfigInfo.SymbolsPerSlot);

            % Channel measurement on SRS
            [srsMeasurement, effectiveSINR] = decodeSRS(obj, startTime, endTime, carrierConfigInfo);

            % Report measurements to MAC
            duration = endTime-startTime;
            % Calculate the rx timing information
            if currTimingInfo.NSymbol == 0 % Reception ended in previous slot
                rxTimingInfo = [carrierConfigInfo.NFrame carrierConfigInfo.NSlot 13];
            else % Reception ended in current slot
                rxTimingInfo = [carrierConfigInfo.NFrame carrierConfigInfo.NSlot currTimingInfo.NSymbol-1];
            end
            for i=1:numel(srsMeasurement)
                srsInfo = srsMeasurement(i);
                obj.SRSIndicationFcn(srsInfo);
                rnti = srsInfo.RNTI;
                obj.PacketReceptionEnded(rnti).ChannelMeasurements.SINR = effectiveSINR(i);
                obj.PacketReceptionEnded(rnti).ChannelMeasurements.SRSBasedULMeasurements = struct('RI',srsInfo.RankIndicator,'TPMI',srsInfo.TPMI,'MCSIndex',srsInfo.MCSIndex);
                obj.PacketReceptionEnded(rnti).ChannelMeasurements.SRSBasedDLMeasurements = srsInfo.SRSBasedDLMeasurements;
                if isempty(obj.PacketReceptionEnded(rnti).SignalType)
                    obj.PacketReceptionEnded(rnti).SignalType = "SRS";
                    obj.PacketReceptionEnded(rnti).Duration = duration;
                    obj.PacketReceptionEnded(rnti).TimingInfo = rxTimingInfo;
                else
                    obj.PacketReceptionEnded(rnti).Duration = max(obj.PacketReceptionEnded(rnti).Duration, duration);
                    obj.PacketReceptionEnded(rnti).TimingInfo(3) = max(obj.PacketReceptionEnded(rnti).TimingInfo(3), rxTimingInfo(3));
                    obj.PacketReceptionEnded(rnti).SignalType = obj.PacketReceptionEnded(rnti).SignalType+"+SRS";
                end
            end
            obj.SRSInfo{symbolNumFrame + 1} = []; % Clear the context
        end

        function [transmitterIDList, tagList] = getTagsInfo(obj, puschInfoList, startTime, endTime, carrierConfigInfo)
            %getTagsInfo Returns the tags associated with the give PUSCH configurations

            numPUSCHs = size(puschInfoList,2);
            tagList = cell(numPUSCHs,1);
            transmitterIDList = zeros(numPUSCHs, 1);
            % Read all the relevant packets (i.e. either of interest or sent on same carrier
            % frequency)
            packetInfoList = packetList(obj.RxBuffer, startTime, endTime);
            numPackets = size(packetInfoList,1);

            for i=1:numPUSCHs % For each PUSCH to be received
                puschInfo = puschInfoList{i};
                for j=1:numPackets % Search PUSCH of interest in the list of received packets
                    packet = packetInfoList(j);
                    if (packet.Metadata.PacketType == obj.PXSCHPacketType) && ... % Check for PUSCH
                            (carrierConfigInfo.NCellID == packet.Metadata.NCellID) && ... % Check for PUSCH of interest
                            (puschInfo.PUSCHConfig.RNTI == packet.Metadata.RNTI) && ...
                            (startTime == packet.StartTime)
                        % Extract TransmitterID and Tags from the packet of interest
                        transmitterIDList(i) = packet.TransmitterID;
                        tagList{i} = packet.Tags;
                        break;
                    end
                end
            end
        end
    end
end
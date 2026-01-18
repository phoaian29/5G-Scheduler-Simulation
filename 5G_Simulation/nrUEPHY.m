classdef (Abstract) nrUEPHY < nr5g.internal.nrPHY
    %nrUEPHY Define NR physical layer base class for UE
    %   The class acts as a base class for all the physical layer types of UE.
    %
    %   Note: This is an internal undocumented class and its API and/or
    %   functionality may change in subsequent releases.

    %   Copyright 2023-2024 The MathWorks, Inc.

    properties (SetAccess=protected)
        %RNTI RNTI of the UE
        RNTI

        %L2SMIUI L2SM context for inter user interference
        L2SMIUI

        %L2SM L2SM context for CSI-RS packets
        L2SMCSI
    end

    properties (Access=protected)
        %GNBTransmitPower gNB transmit power in dBm
        GNBTransmitPower

        %GNBReceivedPower gNB received power at the UE in dBm
        GNBReceivedPower

        %PoPUSCCH Nominal transmit power of a UE per resource block in dBm
        %   Specify the nominal transmit power of UE in dBm. The default
        %   value is 0 dBm (3GPP TS 38.213 Section 7.1).
        PoPUSCH

        %AlphaPUSCH Fractional power control multiplier of UE
        %   Specify the fractional power control multiplier of UE. The default
        %   value is 1 (3GPP TS 38.213 Section 7.1).
        AlphaPUSCH

        %DeltaMCS Parameter flag for delta MCS
        %   Specify whether the UE needs to adjust its transmit power based on the
        %   transport format. The default value is false (3GPP TS 38.213 Section 7.1).
        DeltaMCS = false;

        %MacPDU PDU sent by MAC which is scheduled to be sent in the current slot
        % The MAC PDU corresponds to the information in the property PUSCHInfo.
        MacPDU = []

        %PUSCHInfo PUSCH information sent by MAC for the current slot
        % It is a structure of type PUSCHINFO (See txDataRequest function for
        % structure details). It contains the information required by PHY to
        % transmit a MAC PDU stored in object property 'MacPDU'.
        PUSCHInfo = []

        %SRSInfo SRS information PDU sent by MAC for the current slot
        % It is an object of type nrSRSConfig containing the configuration of SRS
        % to be sent in current slot.  Value as empty means that no SRS Tx is
        % scheduled for the current slot
        SRSInfo = []

        %CSIRSInfo Rx context for the channel state information reference signals (CSI-RS)
        % This information is populated by MAC and is used by PHY to receive
        % scheduled CSI-RS. It is a cell array of size 'N' where N is the number of
        % symbols in a 10 ms frame. The cell elements are populated with objects of
        % type nrCSIRSConfig. An element at index 'i' contains the configuration of
        % CSI-RS configuration which is sent between the symbol index 'i-14' to 'i'
        % (i.e during the slot). Cell element at 'i' is empty if no CSI-RS
        % reception was scheduled in the slot.
        CSIRSInfo

        %CSIRSIndicationFcn Function handle to send the measured DL channel quality to MAC
        CSIRSIndicationFcn

        %CSIReportConfig CSI report configuration
        % The detailed explanation of this structure and its fields is
        % present as ReportConfig in <a href="matlab:help('nr5g.internal.nrCQISelect')">nrCQISelect</a> function
        CSIReportConfig

        %AdjustedTransmitPower Transmit power of UE in dBm after uplink power control
        AdjustedTransmitPower

        %CSIReferenceResource CSI reference resource for CQI measurements
        CSIReferenceResource = nrPDSCHConfig
    end

    properties (Constant, Access=protected)
        %SRSPowerControlType Integer mapping for SRS power control type
        SRSPowerControlType = 0;

        %PUSCHPowerControlType Integer mapping for PUSCH power control type
        PUSCHPowerControlType = 1;
    end

    methods
        function obj = nrUEPHY(param, notificationFcn)
            % Constructor

            obj = obj@nr5g.internal.nrPHY(param, notificationFcn); % Call base class constructor
            % Initialize UE PHY statistics
            obj.StatTransmittedPackets = 0;
            obj.StatReceivedPackets = 0;
            obj.StatDecodeFailures = 0;
            obj.AdjustedTransmitPower = obj.TransmitPower;

            % Initialize MACPDUInfo
            obj.MACPDUInfo = struct('NodeID', 0, 'RNTI', 0, 'TBS', 0, 'MACPDU', [], 'CRCFlag', 1, 'HARQID', 0, 'Tags', []);
        end

        function addConnection(obj, connectionConfig)
            %addConnection Configures the UE PHY with connection information
            %   addConnection(OBJ, CONNECTIONCONFIG) adds the cell connection related
            %   information to UE PHY. CONNECTIONCONFIG is a structure including the
            %   following fields:
            %       RNTI                     - Radio network temporary identifier of the UE
            %                                  node, returned as an integer in the range [1,
            %                                  65,522]. For more information about radio
            %                                  network temporary identifier, see Table 7.1-1
            %                                  in 3GPP TS 38.321, version 18.1.0.
            %       NCellID                  - Physical cell ID. values: 0 to 1007 (TS 38.211, sec 7.4.2.1)
            %       DuplexMode               - "FDD" or "TDD"
            %       SubcarrierSpacing        - Subcarrier spacing
            %       NumResourceBlocks        - Number of RBs
            %       NumHARQ                  - Number of HARQ processes on UE
            %       ChannelBandwidth         - DL or UL channel bandwidth in Hz
            %       DLCarrierFrequency       - DL carrier frequency
            %       ULCarrierFrequency       - UL carrier frequency
            %       CSIReportConfiguration   - CSI report configuration
            %       PoPUSCH                  - Nominal UE transmit power per resource block
            %       AlphaPUSCH               - Fractional power control multiplier of a UE
            %       GNBTransmitPower         - Transmit power of the gNB

            obj.RNTI = connectionConfig.RNTI;
            setCarrierInformation(obj, createCarrierStruct(obj, connectionConfig)); % Set carrier information
            symbolsPerFrame = obj.CarrierInformation.SlotsPerSubframe*10*14;
            % Create per symbol context
            obj.CSIRSInfo = cell(symbolsPerFrame, 1);
            obj.NextTxTime = Inf*ones(symbolsPerFrame, 1);
            obj.NextRxTime = Inf*ones(symbolsPerFrame, 1);

            if ~isempty(connectionConfig.CSIReportConfiguration)
                % Set CSI report configuration
                obj.CSIReportConfig =  connectionConfig.CSIReportConfiguration;
                setCQITable(obj, obj.CSIReportConfig.CQITable);
            end

            % Set the packet properties
            obj.PacketStruct.Metadata.RNTI = connectionConfig.RNTI;
            obj.PacketStruct.Metadata.NCellID = obj.CarrierInformation.NCellID;
            obj.PacketStruct.CenterFrequency = connectionConfig.ULCarrierFrequency;
            obj.PacketStruct.Bandwidth = connectionConfig.ChannelBandwidth;

            % Set the power control parameters
            obj.PoPUSCH = connectionConfig.PoPUSCH;
            obj.AlphaPUSCH = connectionConfig.AlphaPUSCH;
            obj.GNBTransmitPower = connectionConfig.GNBTransmitPower;

            % Update RNTI
            obj.MACPDUInfo.RNTI = connectionConfig.RNTI;

            % Initialize interference buffer
            obj.RxBuffer = wirelessnetwork.internal.interferenceBuffer(CenterFrequency=obj.CarrierInformation.DLCarrierFrequency, ...
                Bandwidth=obj.CarrierInformation.ChannelBandwidth, SampleRate=obj.WaveformInfo.SampleRate, ResultantWaveformDataType="single", ...
                DisableValidation=true);

            % Set the PacketTransmissionStarted event information
            obj.PacketTransmissionStarted = obj.PacketTxStartedStruct;
            obj.PacketTransmissionStarted.DuplexMode = connectionConfig.DuplexMode;
            obj.PacketTransmissionStarted.RNTI = connectionConfig.RNTI;
            obj.PacketTransmissionStarted.LinkType = "UL";

            % Set the PacketReceptionEnded event information
            obj.PacketReceptionEnded = obj.PacketRxEndedStruct;
            obj.PacketReceptionEnded.DuplexMode = connectionConfig.DuplexMode;
            obj.PacketReceptionEnded.RNTI = connectionConfig.RNTI;
            obj.PacketReceptionEnded.LinkType = "DL";
            obj.HARQBuffers = cell(1, connectionConfig.NumHARQ); % To hold the received MAC packets

            % Add packet tag information buffer
            obj.ReTxTagBuffer = cell(1,connectionConfig.NumHARQ);

            % Initialize CSI-reference resource
            obj.CSIReferenceResource.PRBSet = (0:obj.CarrierConfig.NSizeGrid-1);
            obj.CSIReferenceResource.SymbolAllocation = [2 obj.CarrierConfig.SymbolsPerSlot-2];
        end

        function registerMACHandle(obj, sendMACPDUFcn, sendDLChannelQualityFcn)
            %registerMACHandle Register MAC interface functions at PHY, for sending
            % information to MAC

            obj.RxIndicationFcn = sendMACPDUFcn;
            obj.CSIRSIndicationFcn = sendDLChannelQualityFcn;
        end

        function phyStats = statistics(obj)
            %statistics Return the UE PHY statistics
            %   PHYSTATS = statistics(OBJ) returns the UE PHY statistics. PHYSTATS is a
            %   structure contains following fields.
            %       TransmittedPackets  - Number of transmitted PUSCHs
            %       ReceivedPackets     - Number of received PDSCHs
            %       DecodeFailures      - Number of PDSCH decode failures

            phyStats = struct('TransmittedPackets', obj.StatTransmittedPackets, ...
                'ReceivedPackets', obj.StatReceivedPackets,...
                'DecodeFailures',obj.StatDecodeFailures);
        end

        function txDataRequest(obj, puschInfo, packetInfo, timingInfo)
            %txDataRequest Tx request from MAC to PHY for starting PUSCH transmission
            %  txDataRequest(OBJ, PUSCHINFO, PACKETINFO, TIMINGINFO) sets
            %  the Tx context to indicate PUSCH transmission in the current
            %  symbol
            %
            %   PUSCHInfo is a structure which is sent by MAC and contains the
            %   information required by the PHY for the PUSCH transmission. It contains
            %   these fields.
            %       NSlot           - Slot number of the PUSCH transmission
            %       HARQID          - HARQ process ID
            %       NewData         - Defines if it is a new (value 1) or
            %                         re-transmission (value 0)
            %       RV              - Redundancy version of the transmission
            %       TargetCodeRate  - Target code rate
            %       TBS             - Transport block size in bytes
            %       PUSCHConfig     - PUSCH configuration object as described in
            %                        <a href="matlab:help('nrPUSCHConfig')">nrPUSCHConfig</a>
            %
            %   PACKETINFO is the uplink MAC PDU sent by MAC for transmission and the
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
            %       NFrame      - System frame number
            %       NSlot       - Slot number in the 10 millisecond frame
            %       NSymbol     - Symbol number in the current slot
            %       Timestamp   - Transmission start timestamp in nanoseconds

            obj.PUSCHInfo = puschInfo;
            obj.MacPDU = packetInfo;

            txStartSym = puschInfo.PUSCHConfig.SymbolAllocation(1);
            txSymbolFrame = puschInfo.NSlot*14 + txStartSym; % PUSCH Tx start symbol number w.r.t start of 10 ms frame
            obj.NextTxTime(txSymbolFrame+1) = timingInfo.Timestamp;
        end

        function dlControlRequest(obj, pduType, dlControlPDU, timingInfo)
            %dlControlRequest Downlink control (non-data) reception request from MAC to PHY
            %   dlControlRequest(OBJ, PDUTYPES, DLCONTROLPDUS, TIMINGINFO) is a request
            %   from MAC for downlink control receptions. MAC sends it at the start of
            %   a DL slot for all the scheduled non-data DL receptions in the slot
            %   (Data i.e. PDSCH reception information is sent by MAC using
            %   rxDataRequest interface of this class).
            %
            %   PDUTYPE is an array of packet types. Currently, only
            %   packet type 0 (CSI-RS) is supported.
            %
            %   DLCONTROLPDU is an array of DL control information PDUs,
            %   corresponding to packet types in PDUTYPE. Currently
            %   supported information CSI-RS PDU is an object of type
            %   nrCSIRSConfig.
            %
            %   TIMINGINFO is a structure that contains the following fields.
            %     NFrame      - Current frame number
            %     NSlot       - Current slot number in the 10 millisecond frame
            %     NSymbol     - Current symbol number in the current slot
            %     Timestamp   - Reception start timestamp in nanoseconds.

            % Update the Rx context for DL receptions
            for i = 1:size(pduType,2)
                % Channel quality would be read at the end of the current slot
                rxSymbolFrame = (timingInfo.NSlot+1) * 14;
                obj.CSIRSInfo{rxSymbolFrame}{end+1} = dlControlPDU{i};
                obj.NextRxTime(rxSymbolFrame) = timingInfo.Timestamp + (15e6/obj.CarrierInformation.SubcarrierSpacing); % In nanoseconds
            end
        end

        function ulControlRequest(obj, pduType, ulControlPDU, timingInfo)
            %ulControlRequest Uplink transmission request (non-data) from
            %MAC to PHY
            %   ulControlRequest(OBJ, PDUTYPES, ULCONTROLPDU) is a request from MAC for
            %   uplink transmissions. MAC sends it at the start of a UL slot for all
            %   the scheduled non-data UL transmissions in the slot (Data i.e. PUSCH
            %   transmissions information is sent by MAC using txDataRequest interface
            %   of this class).
            %
            %   PDUTYPE is an array of packet types. Currently, only packet type 1
            %   (SRS) is supported.
            %
            %   ULCONTROLPDU is an array of UL control information PDUs, corresponding
            %   to packet types in PDUTYPE. Currently supported information SRS PDU is
            %   an object of type nrSRSConfig.
            %
            %   TIMINGINFO is a structure that contains the following fields.
            %     NFrame      - Current frame number
            %     NSlot       - Current slot number in the 10 millisecond frame
            %     NSymbol     - Current symbol number in the current slot
            %     Timestamp   - Reception start timestamp in nanoseconds

            % Update the Tx context
            obj.SRSInfo = [ulControlPDU{:}];
            txSymbolFrame = (timingInfo.NSlot)*14;
            obj.NextTxTime(txSymbolFrame+1) = timingInfo.Timestamp;
        end

        function rxDataRequest(obj, pdschInfo, timingInfo)
            %rxDataRequest Rx request from MAC to PHY for starting PDSCH reception
            %   rxDataRequest(OBJ, PDSCHINFO, TIMINGINFO) is a request to start PDSCH
            %   reception. It starts a timer for PDSCH end time (which on triggering
            %   receives the complete PDSCH). The PHY expects the MAC to send this
            %   request at the start of reception time.
            %
            %   pdschInfo is an structure which is sent by MAC and contains the
            %   information required by the PHY for the PDSCH reception. It contains
            %   these fields.
            %       NSlot           - Slot number of the PDSCH reception
            %       HARQID          - HARQ process ID
            %       NewData         - Defines if it is a new (value 1) or
            %                         re-transmission (value 0)
            %       RV              - Redundancy version of the transmission
            %       TargetCodeRate  - Target code rate
            %       TBS             - Transport block size in bytes
            %       PrecodingMatrix - Precoding matrix
            %       BeamIndex       - Column index in the beam weights table configured at PHY
            %       PDSCHConfig     - PDSCH configuration object as described in
            %                         <a href="matlab:help('nrPDSCHConfig')">nrPDSCHConfig</a>
            %
            %   TIMINGINFO is a structure that contains the following
            %   fields.
            %     NFrame      - Current frame number
            %     NSlot       - Current slot number in the 10 millisecond frame
            %     NSymbol     - Current symbol number in the current slot
            %     Timestamp   - Reception start timestamp in nanoseconds

            pdschStartSym = pdschInfo.PDSCHConfig.SymbolAllocation(1);
            symbolNumFrame = pdschInfo.NSlot*14 + pdschStartSym; % PDSCH Rx start symbol number w.r.t start of 10 ms frame

            % PDSCH to be read at the end of last symbol in PDSCH reception
            numPDSCHSym =  pdschInfo.PDSCHConfig.SymbolAllocation(2);
            pdschRxSymbolFrame = symbolNumFrame + numPDSCHSym;

            symDur = obj.CarrierInformation.SymbolDurations; % In nanoseconds
            startSymbolIdx = pdschStartSym + 1;
            endSymbolIdx = pdschStartSym + numPDSCHSym;

            % Add the PDSCH Rx information at the index corresponding to
            % the symbol where PDSCH Rx ends
            obj.DataRxContext{pdschRxSymbolFrame} = pdschInfo;
            % Update next reception time (in nanoseconds)
            obj.NextRxTime(pdschRxSymbolFrame) = timingInfo.Timestamp + ...
                sum(symDur(startSymbolIdx:endSymbolIdx));
        end
    end

    methods(Access=protected)
        function phyTx(obj, currTimingInfo)
            %phyTx Physical layer transmission

            symbolNumFrame = mod(currTimingInfo.NSlot*14 + currTimingInfo.NSymbol, ...
                obj.CarrierInformation.SymbolsPerFrame);

            if obj.NextTxTime(symbolNumFrame+1) ~= Inf % Check if any Tx is scheduled now
                puschTx(obj, currTimingInfo); % Transmit PUSCH
                srsTx(obj, currTimingInfo); % Transmit CSI-RS
                obj.NextTxTime(symbolNumFrame+1) = Inf; % Reset

                % Notify the valid Tx event
                obj.PacketTransmissionStarted.TimingInfo = [currTimingInfo.NFrame currTimingInfo.NSlot currTimingInfo.NSymbol];
                obj.NotificationFcn('PacketTransmissionStarted', obj.PacketTransmissionStarted);
                % Reset the event data context
                obj.PacketTransmissionStarted.SignalType = [];
                obj.PacketTransmissionStarted.HARQID = [];
                obj.PacketTransmissionStarted.PDU = [];
                obj.PacketTransmissionStarted.TransmissionType = [];
                obj.PacketTransmissionStarted.Duration = [];
            end
        end

        function phyRx(obj, currTimingInfo)
            %phyRx Physical layer reception

            symbolNumFrame = mod(currTimingInfo.NSlot*14 + currTimingInfo.NSymbol-1, ...
                obj.CarrierInformation.SymbolsPerFrame); % Previous symbol number in the 10 ms frame

            if obj.NextRxTime(symbolNumFrame+1) ~= Inf % Check if any Rx is scheduled now
                pdschRx(obj, currTimingInfo); % Receive PDSCH
                csirsRx(obj, currTimingInfo); % Receive CSI-RS
                obj.NextRxTime(symbolNumFrame+1) = Inf; % Reset

                % Notify the valid Rx event
                obj.NotificationFcn('PacketReceptionEnded', obj.PacketReceptionEnded);
                % Reset the event data context
                obj.PacketReceptionEnded.HARQID = [];
                obj.PacketReceptionEnded.SignalType = [];
                obj.PacketReceptionEnded.Duration = [];
                obj.PacketReceptionEnded.PDU = [];
                obj.PacketReceptionEnded.CRCFlag = [];
                obj.PacketReceptionEnded.SINR = [];
                obj.PacketReceptionEnded.ChannelMeasurements = [];
            end
        end

        function [dlRank, pmiSet, cqi, precodingMatrix, sinr] = csirsRxProcessing(obj, csirsConfig, pktStartTime, pktEndTime, carrierConfigInfo)
            % Return CSI-RS measurment

            % Get CSI-RS packet of interest and the interfering packets
            [csirsPacket, interferingPackets] = packetListIntfBuffer(obj, obj.CSIRSPacketType, ...
                pktStartTime, pktEndTime);

            nVar = calculateThermalNoise(obj);
            % Received power of gNB at UE for pathloss calculation
            obj.GNBReceivedPower = csirsPacket.Power;

            rnti = obj.RNTI; % Get the RNTI of the current UE

            % Initialize variable to store the matched CSI-RS packet
            packetOfInterest = [];

            % Loop over CSI-RS packet to find the one that matches the current UE's RNTI
            for pktIdx = 1:numel(csirsPacket)
                metadata = csirsPacket(pktIdx).Metadata;
                rntiList = metadata.RNTI;
                % Check if the current UE's RNTI matches any RNTI in the sublist
                if any(rnti == rntiList)
                    packetOfInterest = csirsPacket(pktIdx);
                    break;
                end
            end

            % Estimate channel for packet of interest and interferers
            [estChannelGrid, estChannelGridIntf] = estimateChannelGrid(obj, packetOfInterest, ...
                interferingPackets, carrierConfigInfo);

            % Prepare LQM input for interferers
            intf = prepareLQMInputIntf(obj, obj.L2SMIUI, interferingPackets, estChannelGridIntf, ...
                carrierConfigInfo, nVar);

            % Compute downlink rank and precoder based on the channel
            [dlRank,pmiSet,pmiInfo] = nr5g.internal.nrRISelect(carrierConfigInfo, csirsConfig, ...
                obj.CSIReportConfig, estChannelGrid, nVar, 'MaxSE');
            blerThreshold = 0.1;
            overhead = 0;
            if obj.CSIReferenceResource.NumLayers ~= dlRank
                obj.CSIReferenceResource.NumLayers = dlRank;
            end
            precodingMatrix = pmiInfo.W;
            % For the given precoder prepare the LQM input

            [obj.L2SMCSI, sig] = nr5g.internal.L2SM.prepareLQMInput(obj.L2SMCSI, ...
                carrierConfigInfo,csirsConfig,estChannelGrid,nVar,pmiInfo.W.');
            % Determine SINRs from Link Quality Model (LQM)
            [obj.L2SMCSI, sinr] = nr5g.internal.L2SM.linkQualityModel(obj.L2SMCSI,sig,intf);
            % CQI Selection
            [obj.L2SMCSI, cqi, cqiInfo] = nr5g.internal.L2SM.cqiSelect(obj.L2SMCSI, ...
                carrierConfigInfo,obj.CSIReferenceResource,overhead,sinr,obj.CQITableValues,blerThreshold);
            cqi = max([cqi, 1]); % Ensure minimum CQI as 1
            sinr = cqiInfo.EffectiveSINR;
        end

        function [packetsofInterest, interferingPackets] = packetListIntfBuffer(obj, packetType, rxStartTime, rxEndTime)
            %packetListIntfBuffer Return packet of interest and interfering packets for reference signal

            % Get all the relevant packets from the interference buffer
            relevantPackets = packetList(obj.RxBuffer, rxStartTime, rxEndTime);

            % Initialization
            packetsofInterest = relevantPackets;
            interferingPackets = relevantPackets;
            numPkts = numel(relevantPackets);
            pktOfInterestCount = 0;
            intfPktCount = 0;

            % Divide relevant packets as packets of interest and interfering packets
            for pktIdx = 1:numPkts
                if relevantPackets(pktIdx).Metadata.PacketType==packetType && ...
                        obj.CarrierInformation.NCellID ~= relevantPackets(pktIdx).Metadata.NCellID
                    % For CSI-RS packet, only consider CSI-RS from other cells as
                    % interference. Likewise for SRS
                    intfPktCount = intfPktCount + 1;
                    interferingPackets(intfPktCount) = relevantPackets(pktIdx);
                elseif (relevantPackets(pktIdx).Metadata.PacketType==packetType && ...
                        obj.CarrierInformation.NCellID == relevantPackets(pktIdx).Metadata.NCellID)
                    pktOfInterestCount = pktOfInterestCount + 1;
                    packetsofInterest(pktOfInterestCount) = relevantPackets(pktIdx);
                end
            end
            interferingPackets = interferingPackets(1:intfPktCount);
            packetsofInterest = packetsofInterest(1:pktOfInterestCount);
        end

        function [estChannelGrid, estChannelGridIntf] = estimateChannelGrid(obj, packetOfInterest, ...
                interferingPackets, carrierConfigInfo)
            %estPerfectChannelGrid Returns estimated channel grid w.r.t transmitter of interest and interferers

            % Estimate channel for transmitter of interest. Scale path gains to
            % accommodate for receiver gains and pathloss
            pathGains = packetOfInterest.Metadata.Channel.PathGains * db2mag(packetOfInterest.Power-30) * db2mag(obj.ReceiveGain);
            % Timing and channel estimation
            offset = nrPerfectTimingEstimate(pathGains, packetOfInterest.Metadata.Channel.PathFilters.');
            estChannelGrid = nrPerfectChannelEstimate(carrierConfigInfo, pathGains, packetOfInterest.Metadata.Channel.PathFilters.', ...
                offset, packetOfInterest.Metadata.Channel.SampleTimes);

            % Estimate channel for interferers
            numIntf = numel(interferingPackets);
            estChannelGridIntf = cell(1,numIntf);
            for pktIdx = 1:numIntf
                currPacketInfo = interferingPackets(pktIdx);
                if obj.CarrierInformation.NCellID == currPacketInfo.Metadata.NCellID
                    % Packet belongs to the cell of interest. These packets contribute to inter
                    % user interference. These packets will have same channel as the packet of
                    % interest. Only precoding matrix will be unique
                    estChannelGridIntf{pktIdx} = estChannelGrid;
                else
                    pathGains = currPacketInfo.Metadata.Channel.PathGains * db2mag(currPacketInfo.Power-30) * db2mag(obj.ReceiveGain);

                    if (currPacketInfo.Metadata.Channel.PathFilters==1)
                        % Fast fading is not configured for this link. Channel estimation
                        % considering only transmit power, pathloss and receiver gain
                        estChannelGridIntf{pktIdx} = permute(pathGains,[1 2 4 3]);
                    else
                        % Fast fading is configured for neighboring gNBs interfering links
                        % Compute timing and channel estimation
                        offset = nrPerfectTimingEstimate(pathGains, currPacketInfo.Metadata.Channel.PathFilters.');
                        estChannelGridIntf{pktIdx} = nrPerfectChannelEstimate(carrierConfigInfo, pathGains, currPacketInfo.Metadata.Channel.PathFilters.', ...
                            offset, currPacketInfo.Metadata.Channel.SampleTimes);
                    end
                end
            end
        end

        function intf = prepareLQMInputIntf(~, l2sm, packetInfoListIntf, estChannelGridsIntf, carrierConfigInfo, nVar)
            % Prepare LQM input for interfering packets

            intf=[];
            numPkts = numel(packetInfoListIntf);

            % Loop for valid packets and prepare Link Quality Model (LQM)
            % for interfering packets
            for pktIdx = 1:numPkts
                % Packet to be processed by LQM
                pkt = packetInfoListIntf(pktIdx);
                pktConfiguration = pkt.Metadata.PacketConfig;

                % Set precoder for interfering packet
                if ~isempty(pkt.Metadata.PrecodingMatrix)
                    precoder = pkt.Metadata.PrecodingMatrix;
                else
                    precoder = [1/sqrt(pkt.NumTransmitAntennas) zeros(1,pkt.NumTransmitAntennas-1)];
                end

                [~, lqmiInfo] = nr5g.internal.L2SM.prepareLQMInput(l2sm,carrierConfigInfo,pktConfiguration,estChannelGridsIntf{pktIdx},nVar,precoder);

                if isempty(intf)
                    intf = repmat(lqmiInfo,numPkts,1);
                else
                    intf(pktIdx) = lqmiInfo;
                end
            end
        end

    end

    methods(Abstract)
        %puschData PHY Tx processing of PUSCH
        % DATA = puschData(OBJ, PUSCHINFO, MACPDU) returns the data after Tx
        % processing of PUSCH. MACPDU is the PDU sent by MAC. PUSCHINFO is a
        % structure which is sent by MAC and contains the information required by
        % the PHY for the PUSCH transmission. It contains these fields.
        %       NSlot           - Slot number of the PUSCH transmission
        %       HARQID          - HARQ process ID
        %       NewData         - Defines if it is a new (value 1) or
        %                         re-transmission (value 0)
        %       RV              - Redundancy version of the transmission
        %       TargetCodeRate  - Target code rate
        %       TBS             - Transport block size in bytes
        %       PUSCHConfig     - PUSCH configuration object as described in
        %                         <a href="matlab:help('nrPUSCHConfig')">nrPUSCHConfig</a>
        %
        % DATA is PHY-processed PUSCH. DATA could be simply MACPDU for abstract
        % flavours of PHY or processed PUSCH waveform for full PHY flavours.
        data = puschData(obj, pdschInfo, macPDU);

        %srsData PHY Tx processing of SRS
        % DATA = srsDATA(OBJ, SRSCONFIG) returns the data after PHY Tx processing
        % of SRS based on SRS configuration, SRSCONFIG. DATA could be empty ([])
        % for abstract flavours of PHY or processed SRS waveform for full PHY
        % flavours
        data = srsData(obj, srsConfig);

        %decodePDSCH PHY Rx processing of PDSCH
        % [MACPDU, CRCFLAG, EFFECTIVESINR] = decodePDSCH(OBJ, PDSCHINFO, PKTSTARTTIME, PKTENDTIME,
        % CARRIERCONFIGINFO) returns the decoded packet, MACPDU from received
        % PDSCH. PDSCHInfo is a structure which is sent by MAC and contains the
        % information required by the PHY to receive PDSCH. It contains these
        % fields.
        %   NSlot           - Slot number of the PDSCH transmission
        %   HARQID          - HARQ process ID
        %   NewData         - Defines if it is a new (Value 1) or re-transmission (Value 0)
        %   RV              - Redundancy version of the transmission
        %   TargetCodeRate  - Target code rate
        %   TBS             - Transport block size in bytes
        %   PDSCHConfig     - PUSCH configuration object as described in
        %                     <a href="matlab:help('nrPDSCHConfig')">nrPDSCHConfig</a>
        %
        % PKTSTARTTIME, PKTENDTIME are the start and end time of PDSCH in seconds,
        % respectively. CARRIERCONFIGINFO is the carrier related information of
        % type <a href="matlab:help('nrCarrierConfig')">nrCarrierConfig</a>.
        % MACPDU is a numeric array where each element is in the range [0 255]. CRCFLAG
        % is a numeric scalar and contains binary values. Value 0 and 1
        % represent CRC success and failure, respectively.
        % SINR is a numeric scalar value represents the effective SINR of the received
        % packet.
        [macPDU, crcFlag, effectiveSINR] = decodePDSCH(obj, puschInfo, pktStartTime, pktEndTime, carrierConfigInfo)

        %decodeCSIRS PHY Rx processing of CSI-RS
        % [RANK, PMISET, CQI, PRECODINGMATRIX, EFFECTIVESINR] = decodeCSIRS(OBJ, CSIRSCONFIG,
        % PKTSTARTTIME, PKTENDTIME, CARRIERCONFIGINFO) returns the CSI-RS
        % measurements based on received CSI-RS. CSIRSCONFIG is the CSI-RS
        % configuration and is an object of type <a
        % href="matlab:help('nrCSIRSConfig')">nrCSIRSConfig</a>. PKTSTARTTIME,
        % PKTENDTIME are the start and end time of CSI-RS in seconds, respectively.
        % CARRIERCONFIGINFO is the carrier related information of type <a
        % href="matlab:help('nrCarrierConfig')">nrCarrierConfig</a>. RANK is the
        % measured DL rank. PMISET is the measured PMI values and PRECODINGMATRIX
        % is the corresponding precoding matrix. CQI is the measured CQI over the
        % carrier bandwidth. EFFECTIVESINR is a numeric value represents the effective SINR of the
        % received packet.
        [rank, pmiSet, cqi, precodingMatrix, effectiveSINR] = decodeCSIRS(obj, rnti, csirsConfig, pktStartTime, pktEndTime, carrierConfigInfo)
    end

    methods (Access=private)
        function puschTx(obj, currTimingInfo)
            % Transmit the PUSCH scheduled for current time

            if ~isempty(obj.PUSCHInfo)
                % Fill PUSCH packet details
                packet = obj.PacketStruct;
                puschInfo = obj.PUSCHInfo;
                txPowerControl(obj, obj.PUSCHPowerControlType);
                if isempty(obj.MacPDU)
                    obj.PacketTransmissionStarted.TransmissionType = "ReTx";
                    macPDU = obj.HARQBuffers{puschInfo.HARQID+1};
                    packet.Data = puschData(obj, puschInfo, []);
                else
                    obj.PacketTransmissionStarted.TransmissionType = "NewTx";
                    macPDU = obj.MacPDU.Packet;
                    obj.HARQBuffers{puschInfo.HARQID+1} = macPDU;
                    % Get the PUSCH data after PHY processing
                    packet.Data = puschData(obj, puschInfo, obj.MacPDU.Packet);
                    % Store the associated packet tags to deal with possible retransmission
                    obj.ReTxTagBuffer{puschInfo.HARQID+1} = obj.MacPDU.Tags;
                end
                % Apply FFT Scaling to the transmit power
                powerScalingFactor = sqrt((obj.WaveformInfo.Nfft^2) / (12*numel(obj.PUSCHInfo.PUSCHConfig.PRBSet)));
                packet.Power = mag2db(db2mag(obj.AdjustedTransmitPower-30)*powerScalingFactor) + 30; %dBm
                packet.StartTime = currTimingInfo.Time/1e9;
                % MIMO precoding, TS 38.211 Section 6.3.1.5
                if (strcmpi(puschInfo.PUSCHConfig.TransmissionScheme,'codebook'))
                    wtx = nrPUSCHCodebook(puschInfo.PUSCHConfig.NumLayers,puschInfo.PUSCHConfig.NumAntennaPorts, ...
                        puschInfo.PUSCHConfig.TPMI,puschInfo.PUSCHConfig.TransformPrecoding);
                else % 'nonCodebook'
                    wtx = eye(puschInfo.PUSCHConfig.NumLayers);
                end
                packet.Metadata.PrecodingMatrix = wtx;
                packet.Metadata.PacketConfig = puschInfo.PUSCHConfig; % PUSCH Configuration
                packet.Metadata.TargetCodeRate = puschInfo.TargetCodeRate;
                packet.Metadata.RNTI = puschInfo.PUSCHConfig.RNTI;
                packet.Metadata.PacketType = obj.PXSCHPacketType;
                startSym = puschInfo.PUSCHConfig.SymbolAllocation(1);
                endSym = puschInfo.PUSCHConfig.SymbolAllocation(1)+puschInfo.PUSCHConfig.SymbolAllocation(2)-1;
                [startSampleIdx, endSampleIdx] = sampleIndices(obj, puschInfo.NSlot, ...
                    startSym, endSym);
                packet.Duration = round(sum(obj.CarrierInformation.SymbolDurations(startSym+1:endSym+1))/1e9,9);
                packet.Metadata.NumSamples = endSampleIdx-startSampleIdx+1;
                packet.SampleRate = obj.WaveformInfo.SampleRate;
                packet.Tags = obj.ReTxTagBuffer{puschInfo.HARQID+1};

                % Update stats
                obj.StatTransmittedPackets = obj.StatTransmittedPackets + 1;

                %Send the packet
                obj.SendPacketFcn(packet);

                % Update event data
                obj.PacketTransmissionStarted.HARQID = puschInfo.HARQID;
                obj.PacketTransmissionStarted.SignalType = "PUSCH";
                obj.PacketTransmissionStarted.PDU = macPDU;
                obj.PacketTransmissionStarted.Duration = packet.Duration;
            end
            % Transmission done. Clear the Tx contexts
            obj.PUSCHInfo = [];
            obj.MacPDU = [];
        end

        function srsTx(obj, currTimingInfo)
            % Transmit SRS scheduled for current time

            if ~isempty(obj.SRSInfo)
                % Fill SRS packet details
                packet = obj.PacketStruct;
                txPowerControl(obj, obj.SRSPowerControlType);
                obj.CarrierConfig.NSlot = currTimingInfo.NSlot;
                packet.Data = srsData(obj, obj.SRSInfo);  % Get the SRS data after PHY processing
                % Apply FFT Scaling to the transmit power
                powerScalingFactor = sqrt((obj.WaveformInfo.Nfft^2) / (12*obj.SRSInfo.NRB));
                packet.Power = mag2db(db2mag(obj.AdjustedTransmitPower-30)*powerScalingFactor) + 30; %dBm
                % Other packet information
                packet.Metadata.NumSamples = samplesInSlot(obj, obj.CarrierConfig);
                packet.StartTime = currTimingInfo.Time/1e9;
                packet.Duration = round(obj.CarrierInformation.SlotDuration/1e9,9);
                packet.Metadata.PacketType = obj.SRSPacketType;
                packet.SampleRate = obj.WaveformInfo.SampleRate;
                packet.Metadata.PacketConfig = obj.SRSInfo; % SRS Configuration

                %Send the packet
                obj.SendPacketFcn(packet);

                % Update event data to include SRS information
                if isempty(obj.PacketTransmissionStarted.SignalType)
                    obj.PacketTransmissionStarted.SignalType = "SRS";
                    obj.PacketTransmissionStarted.Duration = packet.Duration;
                else
                    obj.PacketTransmissionStarted.SignalType = obj.PacketTransmissionStarted.SignalType+"+SRS";
                    obj.PacketTransmissionStarted.Duration = max(obj.PacketTransmissionStarted.Duration,packet.Duration);
                end

            end
            % Transmission done. Clear the Tx context
            obj.SRSInfo = [];
        end

        function pdschRx(obj, currTimingInfo)
            %pdschRx Receive the PDSCH scheduled for current time

            % Read context of PUSCH(s) scheduled for current time
            symbolNumFrame = mod(currTimingInfo.NSlot*14 + currTimingInfo.NSymbol - 1, ...
                obj.CarrierInformation.SymbolsPerFrame); % Previous symbol in a 10 ms frame
            pdschInfo = obj.DataRxContext{symbolNumFrame + 1};
            if ~isempty(pdschInfo)
                % Set carrier information
                carrierConfigInfo = obj.CarrierConfig;
                slotsPerSubframe = obj.WaveformInfo.SlotsPerSubframe;
                [carrierConfigInfo.NSlot, carrierConfigInfo.NFrame] = txSlotInfo(obj, slotsPerSubframe, currTimingInfo);

                [pktStartTime, pktEndTime] = pktTiming(obj, carrierConfigInfo.NFrame, ...
                    carrierConfigInfo.NSlot, pdschInfo.PDSCHConfig.SymbolAllocation(1), ...
                    pdschInfo.PDSCHConfig.SymbolAllocation(2));
                [macPDU, crcFlag, effectiveSINR] = decodePDSCH(obj, pdschInfo, pktStartTime, pktEndTime, carrierConfigInfo);

                macPDUInfo = obj.MACPDUInfo;
                macPDUInfo.HARQID = pdschInfo.HARQID;
                macPDUInfo.MACPDU = macPDU;
                macPDUInfo.CRCFlag = crcFlag;
                macPDUInfo.TBS = pdschInfo.TBS;
                [macPDUInfo.NodeID, macPDUInfo.Tags] = getTagsInfo(obj, pdschInfo, pktStartTime, pktEndTime, carrierConfigInfo);
                % Increment the number of received packets for UE
                obj.StatReceivedPackets = obj.StatReceivedPackets + 1;
                % Increment the number of decode failures for UE
                obj.StatDecodeFailures = obj.StatDecodeFailures +  crcFlag;
                % Rx callback to MAC
                obj.RxIndicationFcn(macPDUInfo, currTimingInfo.Time);
                obj.DataRxContext{symbolNumFrame + 1} = {}; % Clear the context
                % Calculate the rx timing information
                if currTimingInfo.NSymbol == 0 % Reception ended in previous slot
                    obj.PacketReceptionEnded.TimingInfo = [carrierConfigInfo.NFrame carrierConfigInfo.NSlot 13];
                else % Reception ended in current slot
                    obj.PacketReceptionEnded.TimingInfo = [carrierConfigInfo.NFrame carrierConfigInfo.NSlot currTimingInfo.NSymbol-1];
                end
                obj.PacketReceptionEnded.HARQID = macPDUInfo.HARQID;
                obj.PacketReceptionEnded.SignalType = "PDSCH";
                obj.PacketReceptionEnded.Duration = pktEndTime-pktStartTime;
                obj.PacketReceptionEnded.PDU = macPDU;
                obj.PacketReceptionEnded.CRCFlag = crcFlag;
                obj.PacketReceptionEnded.SINR = effectiveSINR;
            end
        end

        function csirsRx(obj, currTimingInfo)
            %csirsRx Receive CSI-RS

            symbolNumFrame = mod(currTimingInfo.NSlot*14 + currTimingInfo.NSymbol-1, ...
                obj.CarrierInformation.SymbolsPerFrame); % Previous symbol number in a 10 ms frame
            csirsInfo = obj.CSIRSInfo{symbolNumFrame + 1};

            for idx=1:size(csirsInfo,1)
                csirsConfig = csirsInfo{idx};
                % Set carrier information
                carrierConfigInfo = obj.CarrierConfig;
                slotsPerSubframe = obj.WaveformInfo.SlotsPerSubframe;
                [carrierConfigInfo.NSlot, carrierConfigInfo.NFrame] = txSlotInfo(obj, slotsPerSubframe, currTimingInfo);

                startSym = 0;
                numSym = carrierConfigInfo.SymbolsPerSlot;
                [pktStartTime, pktEndTime] = pktTiming(obj, carrierConfigInfo.NFrame, ...
                    carrierConfigInfo.NSlot, startSym, numSym);
                % Channel measurement on CSI-RS
                [dlRank, pmiSet, cqiRBs, precodingMatrix, effectiveSINR] = decodeCSIRS(obj, csirsConfig, pktStartTime, pktEndTime, carrierConfigInfo);
                obj.CSIRSIndicationFcn(dlRank, pmiSet, cqiRBs, precodingMatrix);

                %Assuming there is only one CSI-RS reception
                % Calculate the rx timing information
                if currTimingInfo.NSymbol == 0 % Reception ended in previous slot
                    rxTimingInfo = [carrierConfigInfo.NFrame carrierConfigInfo.NSlot 13];
                else % Reception ended in current slot
                    rxTimingInfo = [carrierConfigInfo.NFrame carrierConfigInfo.NSlot currTimingInfo.NSymbol-1];
                end
                if isempty(obj.PacketReceptionEnded.SignalType)
                    obj.PacketReceptionEnded.Duration = pktEndTime-pktStartTime;
                    obj.PacketReceptionEnded.TimingInfo = rxTimingInfo;
                    obj.PacketReceptionEnded.SignalType = "CSIRS";
                else
                    obj.PacketReceptionEnded.Duration = max(obj.PacketReceptionEnded.Duration,pktEndTime-pktStartTime);
                    obj.PacketReceptionEnded.TimingInfo(3) = max(obj.PacketReceptionEnded.TimingInfo(3), rxTimingInfo(3));
                    obj.PacketReceptionEnded.SignalType = obj.PacketReceptionEnded.SignalType+"+CSIRS";
                end
                obj.PacketReceptionEnded.ChannelMeasurements.W = precodingMatrix;
                obj.PacketReceptionEnded.ChannelMeasurements.CQI = cqiRBs;
                obj.PacketReceptionEnded.ChannelMeasurements.PMI = pmiSet;
                obj.PacketReceptionEnded.ChannelMeasurements.RI = dlRank;
                obj.PacketReceptionEnded.ChannelMeasurements.SINR = effectiveSINR;
            end
            obj.CSIRSInfo{symbolNumFrame+1} = {}; % Clear the context
        end

        function txPowerControl(obj, type)
            % Uplink transmit power calculation

            if ~isempty(obj.GNBReceivedPower)
                pcMax = obj.TransmitPower; % Peak UE transmit power
                scs = obj.CarrierInformation.SubcarrierSpacing;
                mu = log2(scs/15); % numerology (0, 1, 2 or 3)

                % Computing the pathloss of the channel
                pathloss = obj.GNBTransmitPower - obj.GNBReceivedPower;

                if type == obj.PUSCHPowerControlType
                    numRB = numel(obj.PUSCHInfo.PUSCHConfig.PRBSet);
                    numPUSCHSymbol = obj.PUSCHInfo.PUSCHConfig.SymbolAllocation(2);

                    % Converting TBS to bits and calculating bits per resource element
                    bpre = obj.PUSCHInfo.TBS * 8/(numRB * numPUSCHSymbol * 12);

                    if obj.PUSCHInfo.PUSCHConfig.NumLayers == 1 && obj.DeltaMCS
                        delta = 10 * log10(2.^(bpre * 1.25) - 1);
                    else
                        delta = 0;
                    end
                    % Adjusted transmit power for PUSCH based power control calculation
                    obj.AdjustedTransmitPower = min(pcMax, obj.PoPUSCH + 10*log10(2^mu * numRB) + obj.AlphaPUSCH * pathloss + delta);
                elseif type == obj.SRSPowerControlType
                    numRB = obj.SRSInfo.NRB;
                    % Adjusted transmit power for SRS based power control calculation
                    obj.AdjustedTransmitPower = min(pcMax, obj.PoPUSCH + 10*log10(2^mu * numRB) + obj.AlphaPUSCH * pathloss);
                end
            else
                obj.AdjustedTransmitPower = obj.TransmitPower;
            end
        end

        function [transmitterID, tags] = getTagsInfo(obj, pdschInfo, pktStartTime, pktEndTime, carrierConfigInfo)
            %getTagsInfo Get the tags associated with the given PDSCH configuration

            % Read all the relevant packets (i.e. either of interest or sent on same carrier
            % frequency) received in the time window
            packetInfoList = packetList(obj.RxBuffer, pktStartTime, pktEndTime);
            packetOfInterest = [];
            foundPacketOfInterest = false;
            for j=1:size(packetInfoList,1) % Search PDSCH of interest in the list of received packets
                packet = packetInfoList(j);
                if (packet.Metadata.PacketType == obj.PXSCHPacketType) && ... % Check for PDSCH
                        (carrierConfigInfo.NCellID == packet.Metadata.NCellID) && ... % Check for PDSCH of interest
                        any(pdschInfo.PDSCHConfig.RNTI == packet.Metadata.RNTI) && ...
                        (pktStartTime == packet.StartTime)
                    packetOfInterest = packet;
                    foundPacketOfInterest = true;
                    break;
                end
            end

            % If no packet of interest is found, return empty values
            if ~foundPacketOfInterest
                transmitterID = [];
                tags = [];
                return;
            end

            transmitterID = packetOfInterest.TransmitterID;
            % Remove the "UETagInfo" tag from the tag list, which includes the relevant
            % information to identify the tags of a UE
            [~, phyTag] = ...
                wirelessnetwork.internal.packetTags.remove(packetOfInterest.Tags, ...
                "UETagInfo");
            if isempty(phyTag)
                tags = packetOfInterest.Tags;
            else
                % Identify the index of the UE based on the RNTI match between the packet
                % metadata and the PDSCH configuration
                numUEsScheduled = 1:numel(packetOfInterest.Metadata.RNTI);
                ueRNTIIdx = numUEsScheduled(pdschInfo.PDSCHConfig.RNTI == ...
                    packetOfInterest.Metadata.RNTI);
                % Use the retrieved tag indexing information to find the specific tags related
                % to the UE of interest within the packet
                ueTagIndices = phyTag.Value(2*ueRNTIIdx-1:2*ueRNTIIdx);
                % Extract the relevant tags for the UE from the packet based on the identified
                % indices
                tags = packetOfInterest.Tags(ueTagIndices(1):ueTagIndices(2));
            end
        end
    end
end
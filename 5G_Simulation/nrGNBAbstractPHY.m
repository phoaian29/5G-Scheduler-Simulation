classdef nrGNBAbstractPHY < nr5g.internal.nrGNBPHY
    %nrGNBAbstractPHY Implements abstract physical (PHY) layer for gNB.
    %   The class implements the abstraction specific aspects of gNB PHY.
    %
    %   Note: This is an internal undocumented class and its API and/or
    %   functionality may change in subsequent releases.

    %   Copyright 2022-2024 The MathWorks, Inc.

    properties (Access = private)
        %L2SMs Link-to-system-mapping (L2SM) context for UEs
        % It is an array of objects of length 'N' where N is the number of UEs in
        % the cell
        L2SMs

        %L2SMsSRS L2SM context for SRS
        % It is an array of objects of length 'N' where N is the number of UEs in
        % the cell.
        L2SMsSRS

        %L2SMIntf L2SM context for interference
        L2SMIntf

        %CSIMeasurementSignalDLType The value of "CSIMeasurementSignalDLType" is 1 if the
        % specified value of CSIMeasurementSignalDL is 'SRS'.  It is 0 if the specified
        % value of "CSIMeasurementSignalDL" is 'CSI-RS'.
        CSIMeasurementSignalDLType (1, 1) = 0;
    end

    methods
        function obj = nrGNBAbstractPHY(param, notificationFcn)
            %nrGNBAbstractPHY Construct an abstract gNB PHY object
            %   OBJ = nrGNBAbstractPHY(PARAM,NOTIFICATIONFCN) constructs a gNB abstract PHY object.
            %
            %   PARAM is a structure with the fields:
            %       NCellID             - Cell ID
            %       DuplexMode          - "FDD" or "TDD"
            %       ChannelBandwidth    - DL or UL channel bandwidth in Hz. In FDD mode,
            %                             each of the DL and UL operations happen in
            %                             separate bands of this size. In TDD mode,
            %                             both DL and UL share single band of this
            %                             size.
            %       DLCarrierFrequency  - DL Carrier frequency in Hz
            %       ULCarrierFrequency  - UL Carrier frequency in Hz
            %       NumResourceBlocks   - Number of resource blocks
            %       SubCarrierSpacing   - Subcarrier spacing
            %       TransmitPower       - Tx power in dBm
            %       NumTransmitAntennas - Number of GNB Tx antennas
            %       NumReceiveAntennas  - Number of GNB Rx antennas
            %       NoiseFigure         - Noise figure
            %       ReceiveGain         - Receiver gain at gNB in dBi
            %       CQITable            - Name of the CQI table to be used
            %       MCSTable            - Name of the MCS table to be used
            %
            %   NOTIFICATIONFCN - It is a handle of the node's processEvents
            %   method

            obj = obj@nr5g.internal.nrGNBPHY(param, notificationFcn); % Call base class constructor

            % Set PHY abstraction specific fields
            obj.PacketStruct.Abstraction = true;
            obj.PacketStruct.Metadata = struct('NCellID', obj.CarrierInformation.NCellID, 'RNTI', [], ...
                'PrecodingMatrix', [], 'NumSamples', [], 'Channel', obj.PacketStruct.Metadata.Channel);

            % Initialize L2SM for holding interference context
            obj.L2SMIntf = nr5g.internal.L2SM.initialize(obj.CarrierConfig);
        end

        function addConnection(obj, connectionConfig)
            %addConnection Configures the gNB PHY with a UE connection information
            %
            %   connectionConfig is a structure including the following fields:
            %       RNTI      - RNTI of the UE.
            %       UEID      - Node ID of the UE
            %       UEName    - Node name of the UE
            %       NumHARQ   - Number of HARQ processes for the UE

            addConnection@nr5g.internal.nrGNBPHY(obj, connectionConfig);

            % Set PHY abstraction specific context
            % Initialize L2SMs
            obj.L2SMs = [obj.L2SMs; nr5g.internal.L2SM.initialize(obj.CarrierConfig, connectionConfig.NumHARQ, 1)];
            obj.L2SMsSRS = [obj.L2SMsSRS; nr5g.internal.L2SM.initialize(obj.CarrierConfig)];
            obj.CSIMeasurementSignalDLType = connectionConfig.CSIMeasurementSignalDLType;
        end

        function data = pdschData(obj, pdschInfo, macPDU)
            % Return the MAC packet without any PHY processing

            if isempty(macPDU)
                % MAC PDU not sent by MAC which indicates retransmission. Get the MAC PDU
                % from the HARQ buffers
                data = obj.HARQBuffers{pdschInfo.PDSCHConfig.RNTI, pdschInfo.HARQID+1};
            else
                % New transmission. Buffer the transport block
                data = macPDU;
                obj.HARQBuffers{pdschInfo.PDSCHConfig.RNTI, pdschInfo.HARQID+1} = macPDU;
            end
        end

        function data = csirsData(~, ~)
            % Return empty as abstract PHY does not send any CSI-RS waveform

            data = [];
        end

        function packetList = pdschPacket(obj, pdschInfoList, pdschDataList, txStartTime)
            % Populate and return PDSCH packets (one packet per PDSCH)

            packetStruct = obj.PacketStruct;
            % Apply FFT Scaling to the transmit power
            packetStruct.Power = scaleTransmitPower(obj);
            packetStruct.StartTime = txStartTime;
            packetStruct.SampleRate = obj.WaveformInfo.SampleRate;
            packetStruct.Metadata.PacketType = obj.PXSCHPacketType;

            % Abstract PHY creates a packet per PDSCH
            packetList = repmat(packetStruct, numel(pdschInfoList), 1);

            for i=1:numel(pdschInfoList) % For each PDSCH
                pdschInfo = pdschInfoList(i);
                packetList(i).Metadata.PrecodingMatrix = pdschInfo.PrecodingMatrix;
                packetList(i).Metadata.PacketConfig = pdschInfo.PDSCHConfig; % PDSCH Configuration
                packetList(i).Metadata.RNTI = pdschInfo.PDSCHConfig.RNTI;

                % Calculate packet duration
                startSymIdx = pdschInfo.PDSCHConfig.SymbolAllocation(1)+1;
                pdschNumSym = pdschInfo.PDSCHConfig.SymbolAllocation(2);
                endSymIdx = startSymIdx+pdschNumSym-1;
                slotNumSubFrame = mod(pdschInfo.NSlot, obj.WaveformInfo.SlotsPerSubframe);
                startSymSubframe = slotNumSubFrame*obj.WaveformInfo.SymbolsPerSlot + 1; % Start symbol of Tx slot in the subframe
                lastSymSubframe = startSymSubframe + obj.WaveformInfo.SymbolsPerSlot - 1; % Last symbol of Tx slot in the subframe
                symbolLengths = obj.WaveformInfo.SymbolLengths(startSymSubframe:lastSymSubframe); % Length of symbols of Tx slot
                startSampleIdx = sum(symbolLengths(1:startSymIdx-1))+1;
                endSampleIdx = sum(symbolLengths(1:endSymIdx));
                packetList(i).Duration = round(sum(obj.CarrierInformation.SymbolDurations(startSymIdx:endSymIdx))/1e9,9);
                packetList(i).Data = pdschDataList{i};
                packetList(i).Metadata.NumSamples = endSampleIdx-startSampleIdx+1;
                packetList(i).Tags = obj.ReTxTagBuffer{pdschInfo.PDSCHConfig.RNTI, pdschInfo.HARQID+1};
            end
        end

        function packetList = csirsPacket(obj, csirsInfoList, csirsDataList, txStartTime)
            % Populate and return CSI-RS packet

            packetStruct = obj.PacketStruct;
            % Apply FFT Scaling to the transmit power
            packetStruct.Power = scaleTransmitPower(obj);
            packetStruct.Metadata.PacketType = obj.CSIRSPacketType;
            packetStruct.StartTime = txStartTime;
            packetStruct.Duration = round(obj.CarrierInformation.SlotDuration/1e9,9);
            packetStruct.Metadata.NumSamples = samplesInSlot(obj, obj.CarrierConfig);
            packetStruct.SampleRate = obj.WaveformInfo.SampleRate;

            % Abstract PHY creates a packet per CSI-RS
            packetList = repmat(packetStruct, numel(csirsInfoList), 1);

            for i=1:size(csirsInfoList,2)
                % csirsInfo is a cell array containing three elements:
                % CSI-RS configuration, Beam index, RNTI
                csirsInfo = csirsInfoList{i};
                packetList(i).Metadata.RNTI = csirsInfo{3}; % RNTI
                packetList(i).Data = csirsDataList{i};
                packetList(i).Metadata.PacketConfig = csirsInfo{1}; % CSI-RS configuration
            end
        end

        function [macPDU, crcFlag, sinr] = decodePUSCH(obj, puschInfoList, startTime, endTime, carrierConfigInfo)
            % Return the decoded MAC PDU along with the crc result

            numPUSCHs = size(puschInfoList,2);
            macPDU = cell(numPUSCHs,1);
            crcFlag = ones(numPUSCHs, 1);
            sinr = -Inf(numPUSCHs, 1);
            % Read all the relevant packets (i.e either of interest or sent on same carrier
            % frequency) received during the PUSCH reception
            packetInfoList = packetList(obj.RxBuffer, startTime, endTime);

            % Eliminate any PUSCH packets which are not sent on the overlapping
            % resource blocks to PUSCH of interest. Also, separate out PUSCH of
            % interest
            numPkts = numel(packetInfoList);
            for i=1:numPUSCHs % For each PUSCH to be received
                puschInfo = puschInfoList(i);
                interferingPackets = packetInfoList;
                prbSetPacket = puschInfo.PUSCHConfig.PRBSet;
                numRBPacket = numel(prbSetPacket);
                packetOfInterest = [];
                intfPktCount=0;
                for pktIdx = 1:numPkts
                    metadata = packetInfoList(pktIdx).Metadata;
                    if (metadata.PacketType == obj.PXSCHPacketType) % Check for PUSCH
                        if (carrierConfigInfo.NCellID == metadata.NCellID) && ... % Check for PUSCH of interest
                                (puschInfo.PUSCHConfig.RNTI == metadata.RNTI)
                            packetOfInterest = packetInfoList(pktIdx);
                        else
                            prbSetInterferer = metadata.PacketConfig.PRBSet;
                            isMatched = false;
                            numRBInterferer = numel(prbSetInterferer);
                            % Check for interfering RBs
                            for j=1:numRBPacket
                                rbOfInterest = prbSetPacket(j);
                                for k=1:numRBInterferer
                                    interferingRB = prbSetInterferer(k);
                                    if interferingRB == rbOfInterest
                                        isMatched = true; % Packet is an interfering one
                                        intfPktCount = intfPktCount+1;
                                        interferingPackets(intfPktCount) = packetInfoList(pktIdx);
                                        break;
                                    elseif interferingRB>rbOfInterest
                                        break;
                                    end
                                end
                                if isMatched
                                    break;
                                end
                            end
                        end
                    end
                end
                interferingPackets = interferingPackets(1:intfPktCount);

                if ~isempty(packetOfInterest)
                    % Estimate channel for all for relevant packets i.e. packet of interest and
                    % interferers (inter-cell interferers and inter-user interferers)
                    [estChannelGrid, estChannelGridIntf] = estimateChannelGrid(obj, packetOfInterest, ...
                        interferingPackets, carrierConfigInfo);

                    % Read MAC PDU
                    macPDU{i} = packetOfInterest.Data;

                    % Calculate crc result using l2sm
                    [crcFlag(i), sinr(i)] = l2smCRC(obj, packetOfInterest, interferingPackets, estChannelGrid, ...
                        estChannelGridIntf, puschInfo, carrierConfigInfo);
                end
            end
        end

        function [srsMeasurement, sinr] = decodeSRS(obj, startTime, endTime, carrierConfigInfo)
            % Return SRS measurement for the UEs

            % Get SRS packets of interest and the interfering packets
            [srsPackets, interferingPackets] = packetListIntfBuffer(obj, obj.SRSPacketType, ...
                startTime, endTime);
            nVar = calculateThermalNoise(obj);

            % Set PxSCH MCS table
            mcsTable = "qam256";

            estChannelGridList = cell(numel(srsPackets), 1);
            % Estimate channel for SRS of interest and interferers
            [estChannelGridList{1}, estChannelGridIntf] = estimateChannelGrid(obj, srsPackets(1), ...
                interferingPackets, carrierConfigInfo);

            % Estimate channel for other SRS of interest
            for i=2:numel(srsPackets)
                [estChannelGridList{i}, ~] = estimateChannelGrid(obj, srsPackets(i), ...
                    [], carrierConfigInfo);
            end

            % Prepare LQM input for interferers
            intf = prepareLQMInputIntf(obj, obj.L2SMIntf, interferingPackets, estChannelGridIntf, ...
                carrierConfigInfo, nVar);

            numSRSPkts = numel(srsPackets);
            srsMeasurement = repmat(struct('RNTI',0,'RankIndicator',0,'TPMI',0,'MCSIndex',0,'SRSBasedDLMeasurements',[]),numSRSPkts,1);
            sinr = inf([numSRSPkts 1]);
            numValidReports = 0;
            % Measure channel quality for each SRS
            for i=1:numSRSPkts
                srsPacket = srsPackets(i);
                rnti = srsPacket.Metadata.RNTI;

                % Channel model of the SRS packet
                estChannelGrid = estChannelGridList{i};

                % Compute uplink rank selection and PMI for the UEs SRS transmission
                if srsPacket.Metadata.PacketConfig.NumSRSPorts > 1
                    [rank,pmi,~,~]= nr5g.internal.nrULCSIMeasurements(carrierConfigInfo,srsPacket.Metadata.PacketConfig,obj.PUSCHConfig,estChannelGrid,nVar,mcsTable,obj.CarrierConfig.NSizeGrid);
                else
                    rank = 1;
                    pmi = 0;
                end

                if ~any(isnan(pmi))
                    blerThreshold = 0.1;
                    overhead = 0;
                    % Update number of layers with the calculated uplink rank
                    obj.PUSCHConfig.NumLayers = rank;
                    wtx = nrPUSCHCodebook(rank,size(estChannelGrid,4),pmi);

                    % For the given precoder prepare the LQM input
                    [obj.L2SMsSRS(rnti), sig] = nr5g.internal.L2SM.prepareLQMInput(obj.L2SMsSRS(rnti), ...
                        carrierConfigInfo,srsPacket.Metadata.PacketConfig,estChannelGrid,nVar,wtx);
                    % Determine SINRs from Link Quality Model (LQM)
                    [obj.L2SMsSRS(rnti),SINRs] = nr5g.internal.L2SM.linkQualityModel(obj.L2SMsSRS(rnti),sig,intf);
                    % MCS Selection
                    [obj.L2SMsSRS(rnti),mcsIndex,mcsInfo] = nr5g.internal.L2SM.cqiSelect(obj.L2SMsSRS(rnti), ...
                        carrierConfigInfo,obj.PUSCHConfig,overhead,SINRs,obj.MCSTableValues,blerThreshold);

                    numValidReports = numValidReports + 1;
                    % Set SRS measurement structure
                    srsMeasurement(numValidReports).RNTI = rnti;
                    srsMeasurement(numValidReports).RankIndicator = rank;
                    srsMeasurement(numValidReports).TPMI = pmi;
                    srsMeasurement(numValidReports).MCSIndex = mcsIndex;
                    sinr(numValidReports) = mcsInfo.EffectiveSINR;

                    % Estimate SRS based DL channel measurements
                    if obj.CSIMeasurementSignalDLType
                        prgBundleSize = [];
                        carrier = carrierConfigInfo;
                        estChannelGrid = permute(estChannelGrid,[1 2 4 3]);
                        srsConfig = srsPacket.Metadata.PacketConfig;
                        pdschConfig = nrPDSCHConfig;
                        pdschConfig.PRBSet = (0:carrierConfigInfo.NSizeGrid-1);
                        enablePRGLevelMCS = 0;

                        % Compute rank, precoder, MCS, rank-reduced wideband channel matrix, and effective DL SINR
                        % based on the SRS measurements
                        [rank,w,mcsIndex,dlChannelMatrix,effectivesinrDL] = nr5g.internal.nrSRSDLChannelMeasurements(carrier,srsConfig,pdschConfig,...
                            estChannelGrid,nVar,mcsTable,prgBundleSize,enablePRGLevelMCS);

                        % Set SRS measurement structure to include DL CSI
                        srsBasedDLMeasurements.MCSIndex = mcsIndex;
                        srsBasedDLMeasurements.W = w.';
                        srsBasedDLMeasurements.RI = rank;
                        srsBasedDLMeasurements.H = dlChannelMatrix;
                        srsBasedDLMeasurements.nVar = nVar;
                        srsBasedDLMeasurements.sinr = effectivesinrDL;
                        srsMeasurement(numValidReports).SRSBasedDLMeasurements = srsBasedDLMeasurements;
                    end
                end
            end
            srsMeasurement = srsMeasurement(1:numValidReports);
            sinr = sinr(1:numValidReports);
        end
    end

    methods (Access = private)
        function [crcFlag, sinr] = l2smCRC(obj, packetOfInterest, interferingPackets, ...
                estChannelGrid, estChannelGridsIntf, puschInfo, carrierConfigInfo)
            % Return the CRC flag (0 - success, 1 - failure)

            % Noise Variance
            nVar = calculateThermalNoise(obj);

            % Prepare LQM input for interferers
            intf = prepareLQMInputIntf(obj, obj.L2SMIntf, interferingPackets, estChannelGridsIntf, carrierConfigInfo, nVar);

            rnti = puschInfo.PUSCHConfig.RNTI;
            % Extract PUSCH Indices for the packet of interest
            [~, info] = nrPUSCHIndices(carrierConfigInfo, puschInfo.PUSCHConfig);

            % Prepare HARQ context for the packet of interest
            harqInfo = struct('HARQProcessID', puschInfo.HARQID, 'RedundancyVersion', puschInfo.RV, ...
                'TransportBlockSize', puschInfo.TBS*8, 'NewData', puschInfo.NewData);
            obj.L2SMs(rnti) = nr5g.internal.L2SM.txHARQ(obj.L2SMs(rnti), harqInfo, puschInfo.TargetCodeRate, info.G);
            % Prepare Link Quality Model inputs for the packet of interest
            [obj.L2SMs(rnti), sig] = nr5g.internal.L2SM.prepareLQMInput(obj.L2SMs(rnti),carrierConfigInfo, ...
                packetOfInterest.Metadata.PacketConfig, estChannelGrid,nVar, packetOfInterest.Metadata.PrecodingMatrix);

            % Link Quality Model (LQM) with interference
            [obj.L2SMs(rnti),sinr] = nr5g.internal.L2SM.linkQualityModel(obj.L2SMs(rnti),sig,intf);

            % Link Performance Model
            [obj.L2SMs(rnti), crcFlag, cqiInfo] = nr5g.internal.L2SM.linkPerformanceModel(obj.L2SMs(rnti),harqInfo,puschInfo.PUSCHConfig,sinr);
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
end

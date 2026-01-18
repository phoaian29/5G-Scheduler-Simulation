classdef nrGNBFullPHY < nr5g.internal.nrGNBPHY
    %nrGNBFullPHY Implements full PHY processing for gNB.
    %   The class implements the full-PHY specific aspects of gNB PHY.
    %
    %   Note: This is an internal undocumented class and its API and/or
    %   functionality may change in subsequent releases.

    %   Copyright 2023-2024 The MathWorks, Inc.

    properties (Access = protected)
        %DLSCHEncoders Downlink shared channel (DL-SCH) encoder system objects for the UEs
        % Vector of length equal to the number of UEs in the cell. Each element is
        % an object of type nrDLSCH
        DLSCHEncoders = {}

        %ULSCHDecoders Uplink shared channel (UL-SCH) decoder system objects for the UEs
        % Vector of length equal to the number of UEs in the cell. Each element is
        % an object of type nrULSCHDecoder
        ULSCHDecoders = {}

        %TimingOffset Receiver timing offset
        % Receiver timing offset used for practical synchronization. It is an array
        % of length equal to the number of UEs. Each element at index 'i'
        % corresponds to the timing offset experienced during reception of waveform
        % from UE with RNTI 'i'
        TimingOffset

        %L2SMsSRS L2SM context for SRS
        % It is an array of objects of length 'N' where N is the number of UEs in
        % the cell.
        L2SMsSRS
    end

    properties
        %RVSequence Redundancy version sequence
        RVSequence
    end

    methods
        function obj = nrGNBFullPHY(param, notificationFcn)
            %nrGNBFullPHY Construct a gNB full PHY object
            %   OBJ = nrGNBFullPHY(PARAM,NOTIFICATIONFCN) constructs a gNB full PHY object.
            %
            %   PARAM is a structure with the fields:
            %       NCellID             - Cell ID
            %       DuplexMode          - "FDD" or "TDD"
            %       ChannelBandwidth    - DL or UL channel bandwidth in Hz. In FDD mode,
            %                             each of the DL and UL operations happen in
            %                             separate bands of this size. In TDD mode,
            %                             both DL and UL share single band of this size
            %       DLCarrierFrequency  - DL Carrier frequency in Hz
            %       ULCarrierFrequency  - UL Carrier frequency in Hz
            %       NumResourceBlocks   - Number of resource blocks
            %       SubCarrierSpacing   - Subcarrier spacing
            %       TransmitPower       - Tx power in dBm
            %       NumTransmitAntennas - Number of GNB Tx antennas
            %       NumReceiveAntennas  - Number of GNB Rx antennas
            %       NoiseFigure         - Noise figure
            %       ReceiveGain         - Receiver  gain at gNB in dBi
            %       CQITable            - Name of the CQI table to be used
            %
            %
            %   NOTIFICATIONFCN - It is a handle of the node's processEvents
            %   method

            obj = obj@nr5g.internal.nrGNBPHY(param, notificationFcn); % Call base class constructor

            % NR Packet param
            obj.PacketStruct.Abstraction = false; % Full PHY
            obj.PacketStruct.Metadata = struct('NCellID', obj.CarrierInformation.NCellID, 'RNTI', [], ...
                'PrecodingMatrix', [], 'NumSamples', [], 'Channel', obj.PacketStruct.Metadata.Channel);
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

            % Create DL-SCH encoder system objects for the UEs
            obj.DLSCHEncoders{connectionConfig.RNTI} = nrDLSCH;
            obj.DLSCHEncoders{connectionConfig.RNTI}.MultipleHARQProcesses = true;

            % Create UL-SCH decoder system objects for the UEs
            obj.ULSCHDecoders{connectionConfig.RNTI} = nrULSCHDecoder;
            obj.ULSCHDecoders{connectionConfig.RNTI}.MultipleHARQProcesses = true;
            obj.ULSCHDecoders{connectionConfig.RNTI}.LDPCDecodingAlgorithm = 'Normalized min-sum';
            obj.ULSCHDecoders{connectionConfig.RNTI}.MaximumLDPCIterationCount = 6;

            obj.TimingOffset = [obj.TimingOffset 0];
            obj.L2SMsSRS = [obj.L2SMsSRS; nr5g.internal.L2SM.initialize(obj.CarrierConfig)];
        end

        function data = pdschData(obj, pdschInfo, macPDU)
            % Return the PDSCH waveform

            % Fill the slot grid with PDSCH symbols
            pdschGrid = populatePDSCH(obj, pdschInfo, macPDU);
            % OFDM modulation
            txWaveform = nrOFDMModulate(obj.CarrierConfig, pdschGrid);
            % Signal amplitude. Account for FFT occupancy factor
            signalAmp = db2mag(scaleTransmitPower(obj)-30);
            % Apply the Tx power
            data = signalAmp*txWaveform;
        end

        function data = csirsData(obj, csirsConfig)
            % Return the CSI-RS waveform

            % Fill the slot grid with CSI-RS symbols
            csirsGrid = populateCSIRS(obj, csirsConfig);
            % OFDM modulation
            txWaveform = nrOFDMModulate(obj.CarrierConfig, csirsGrid);
            % Signal amplitude. Account for FFT occupancy factor
            signalAmp = db2mag(scaleTransmitPower(obj)-30);
            % Apply the Tx power
            data = signalAmp*txWaveform;
        end

        function [macPDU, crcFlag, sinr] = decodePUSCH(obj, puschInfoList, startTime, endTime, carrierConfigInfo)
            % Return the decoded MAC PDUs along with the respective crc result

            numPUSCHs = size(puschInfoList,2);
            macPDU = cell(numPUSCHs, 1);
            crcFlag = ones(numPUSCHs, 1);
            sinr = -Inf(numPUSCHs, 1);
            % Read all the relevant packets (i.e. either of interest or sent on same carrier
            % frequency)
            packetInfoList = packetList(obj.RxBuffer, startTime, endTime);

            for i=1:numPUSCHs % For each PUSCH to be received
                packetOfInterest = [];
                puschInfo = puschInfoList(i);
                [puschStartTime, puschEndTime] = pktTiming(obj, carrierConfigInfo.NFrame, ...
                    carrierConfigInfo.NSlot, puschInfo.PUSCHConfig.SymbolAllocation(1), ...
                    puschInfo.PUSCHConfig.SymbolAllocation(2));
                for j=1:size(packetInfoList,1) % Search PUSCH of interest in the list of received packets
                    packet = packetInfoList(j);
                    if (packet.Metadata.PacketType == obj.PXSCHPacketType) && ... % Check for PUSCH
                            (carrierConfigInfo.NCellID == packet.Metadata.NCellID) && ... % Check for PUSCH of interest
                            (puschInfo.PUSCHConfig.RNTI == packet.Metadata.RNTI) && ...
                            (puschStartTime == packet.StartTime)
                        packetOfInterest = packet;
                        % Read the combined waveform received during packet's duration
                        rxWaveform = resultantWaveform(obj.RxBuffer, puschStartTime, puschStartTime+packet.Duration);
                        channelDelay = packet.Duration -(puschEndTime-puschStartTime);
                        numSampleChannelDelay = ceil(channelDelay*packetOfInterest.SampleRate);
                        break;
                    end
                end
                if ~isempty(packetOfInterest)
                    % PUSCH Rx processing
                    [macPDU{i}, crcFlag(i)] = puschRxProcessing(obj, rxWaveform, puschInfo, ...
                        packetOfInterest, carrierConfigInfo, numSampleChannelDelay);
                end
            end
        end

        function [srsMeasurement, sinrList] = decodeSRS(obj, startTime, endTime, carrierConfigInfo)
            % Return SRS measurement for the UEs

            % Read all the relevant packets (i.e. either of interest or sent on same
            % carrier frequency) received in the time window
            rxWaveform = resultantWaveform(obj.RxBuffer, startTime, endTime);
            packetInfoList = packetList(obj.RxBuffer, startTime, endTime);

            srsIdx = 0;
            for j=1:size(packetInfoList,1)
                packet = packetInfoList(j);
                if (packet.Metadata.PacketType == obj.SRSPacketType && ...
                        carrierConfigInfo.NCellID == packet.Metadata.NCellID)
                    [srsReport, sinr] = srsRxProcessing(obj, rxWaveform, packet, carrierConfigInfo);
                    if ~isempty(srsReport)
                        srsIdx = srsIdx+1;
                        srsMeasurement(srsIdx) = srsReport;
                        sinrList(srsIdx) = sinr;
                    end
                end
            end

            if srsIdx==0
                % No valid SRS report found
                srsMeasurement = [];
                sinrList = -Inf;
            end
        end

        function packet = pdschPacket(obj, pdschInfoList, pdschDataList, txStartTime)
            % Populate and return PDSCH packet

            % Full PHY sends a single PDSCH packet containing the combined PDSCH
            % waveform corresponding to the PDSCH(s) to be sent
            packet = obj.PacketStruct;
            % Apply FFT Scaling to the transmit power
            packet.Power = scaleTransmitPower(obj); %dBm
            packet.Metadata.PacketType = obj.PXSCHPacketType;
            packet.StartTime = txStartTime;
            minStartSym = Inf;
            maxEndSym = 0;
            numPDSCH = numel(pdschInfoList);
            % Initialize an array, ueTagInfo, with dimensions 1-by-2N, where N represents
            % the number of PDSCH packets to be consolidated into a single packet. This
            % consolidated packet will incorporate tags from all individual PDSCH packets.
            % The array is designated to record the start and end indices of tags associated
            % with each PDSCH packet in the merged list. Specifically, for a given PDSCH
            % packet i, the start index of its tags is stored at position 2*i-1, and the end
            % index is stored at position 2*i within the ueTagInfo array
            ueTagInfo = zeros(1,2*numPDSCH);
            currentTagIdx = 1;

            % Fill packet fields. A single packet is sent representing the combined
            % PDSCH waveform
            combinedWaveform = zeros(size(pdschDataList(1)), "single");
            for i=1:numPDSCH % For each PDSCH
                pdschInfo = pdschInfoList(i);
                packet.Metadata.PrecodingMatrix{i} = pdschInfo.PrecodingMatrix;
                packet.Metadata.PacketConfig(i) = pdschInfo.PDSCHConfig; % PDSCH Configuration
                packet.Metadata.RNTI(i) = pdschInfo.PDSCHConfig.RNTI;
                startSymIdx = pdschInfo.PDSCHConfig.SymbolAllocation(1)+1;
                pdschNumSym = pdschInfo.PDSCHConfig.SymbolAllocation(2);
                endSymIdx = startSymIdx+pdschNumSym-1;
                if (startSymIdx < minStartSym) % Update min start symbol
                    minStartSym = startSymIdx;
                end
                if (endSymIdx > maxEndSym)  % Update max end symbol
                    maxEndSym = endSymIdx;
                end
                combinedWaveform = combinedWaveform + pdschDataList{i};
                % Append the tags of current PDSCH packet with the existing list of tags
                packet.Tags = wirelessnetwork.internal.packetTags.append(packet.Tags, ...
                    obj.ReTxTagBuffer{pdschInfo.PDSCHConfig.RNTI, pdschInfo.HARQID+1});
                % Record the start and end indices of tags associated with the current PDSCH
                % packet in the merged list
                ueTagInfo(2*i-1:2*i) = [currentTagIdx numel(packet.Tags)];
                currentTagIdx = ueTagInfo(2*i) + 1;
            end
            % Add a tag to the existing list of tags for the packet. This tag contains the
            % ueTagInfo array. Including this tag is required for receivers because it
            % enables the precise extraction and interpretation of tags pertinent to
            % specific PDSCH packets
            packet.Tags = wirelessnetwork.internal.packetTags.add(packet.Tags, ...
                "UETagInfo", ueTagInfo, [1 numel(combinedWaveform)]);
            % Trim txWaveform to span only the transmission symbols
            [startSampleIdx, endSampleIdx] = sampleIndices(obj, pdschInfoList(1).NSlot, minStartSym-1, maxEndSym-1);
            packet.Data = combinedWaveform(startSampleIdx:endSampleIdx, :);
            packet.Duration = round(sum(obj.CarrierInformation.SymbolDurations(minStartSym:maxEndSym))/1e9,9);
            packet.Metadata.NumSamples = endSampleIdx-startSampleIdx+1;
            packet.SampleRate = obj.WaveformInfo.SampleRate;
        end

        function packet = csirsPacket(obj, csirsInfoList, csirsDataList, txStartTime)
            % Populate and return CSI-RS packet

            packet = obj.PacketStruct;
            % Apply FFT Scaling to the transmit power
            packet.Power = scaleTransmitPower(obj); %dBm
            packet.Metadata.PacketType = obj.CSIRSPacketType;
            packet.StartTime = txStartTime;
            % Fill packet fields
            combinedWaveform = zeros(size(csirsDataList(1)));
            for i=1:numel(csirsInfoList) % For each CSI-RS to be sent
                combinedWaveform = combinedWaveform + csirsDataList{1};
            end
            % csirsInfo is a cell array containing three elements:
            % CSI-RS configuration, Beam index, RNTI
            csirsInfo = csirsInfoList{1};
            packet.Metadata.PacketConfig = csirsInfo{1}; % CSI-RS configuration
            packet.Metadata.RNTI = csirsInfo{3}; % RNTI
            packet.Data = combinedWaveform;
            packet.Duration = round(obj.CarrierInformation.SlotDuration/1e9,9);
            packet.Metadata.NumSamples = samplesInSlot(obj, obj.CarrierConfig);
            packet.SampleRate = obj.WaveformInfo.SampleRate;
        end
    end

    methods (Access = protected)
        function txGrid = populatePDSCH(obj, pdschInfo, macPDU)
            %populatePDSCH Populate PDSCH symbols in the Tx grid

            % Initialize Tx grid
            txGrid = zeros(obj.CarrierInformation.NumResourceBlocks*12, obj.WaveformInfo.SymbolsPerSlot, obj.NumTransmitAntennas, "single");
            obj.CarrierConfig.NSlot = pdschInfo.NSlot;

            % Set transport block in the encoder. In case of empty MAC
            % PDU sent from MAC (indicating retransmission), no need to set transport
            % block as it is already buffered in DL-SCH encoder object
            if ~isempty(macPDU)
                % A non-empty MAC PDU is sent by MAC which indicates new
                % transmission
                macPDUBitmap = int2bit(macPDU, 8);
                macPDUBitmap = reshape(macPDUBitmap', [], 1); % Convert to column vector
                setTransportBlock(obj.DLSCHEncoders{pdschInfo.PDSCHConfig.RNTI}, macPDUBitmap, 0, pdschInfo.HARQID);
            end

            W = pdschInfo.PrecodingMatrix;

            % Calculate PDSCH and DM-RS information
            [pdschIndices, pdschIndicesInfo] = nrPDSCHIndices(obj.CarrierConfig, pdschInfo.PDSCHConfig);
            dmrsSymbols = nrPDSCHDMRS(obj.CarrierConfig, pdschInfo.PDSCHConfig);
            dmrsIndices = nrPDSCHDMRSIndices(obj.CarrierConfig, pdschInfo.PDSCHConfig);

            % Encode the DL-SCH transport blocks
            obj.DLSCHEncoders{pdschInfo.PDSCHConfig.RNTI}.TargetCodeRate = pdschInfo.TargetCodeRate;
            codedTrBlock = obj.DLSCHEncoders{pdschInfo.PDSCHConfig.RNTI}.step(pdschInfo.PDSCHConfig.Modulation, ...
                pdschInfo.PDSCHConfig.NumLayers, pdschIndicesInfo.G, pdschInfo.RV, pdschInfo.HARQID);

            % PDSCH modulation and precoding
            pdschSymbols = nrPDSCH(obj.CarrierConfig, pdschInfo.PDSCHConfig, codedTrBlock);
            [pdschAntSymbols, pdschAntIndices] = nrPDSCHPrecode(obj.CarrierConfig, pdschSymbols, pdschIndices, W);
            txGrid(pdschAntIndices) = pdschAntSymbols;

            % PDSCH DM-RS precoding and mapping
            [dmrsAntSymbols, dmrsAntIndices] = nrPDSCHPrecode(obj.CarrierConfig, dmrsSymbols, dmrsIndices, W);
            txGrid(dmrsAntIndices) = dmrsAntSymbols;

            % PDSCH beamforming
            if ~isempty(pdschInfo.BeamIndex)
                numPorts = size(txGrid, 3);
                bfGrid = reshape(txGrid, [], numPorts)*repmat(obj.BeamWeightTable(:, pdschInfo.BeamIndex)', numPorts, 1);
                txGrid = reshape(bfGrid, size(txGrid));
            end
        end

        function txGrid = populateCSIRS(obj, csirsInfo)
            %populateCSIRS Populate CSI-RS symbols in the Tx grid

            % Populate Tx grid
            txGrid = zeros(obj.CarrierInformation.NumResourceBlocks*12, obj.WaveformInfo.SymbolsPerSlot, obj.NumTransmitAntennas, "single");
            csirsInd = nrCSIRSIndices(obj.CarrierConfig, csirsInfo);
            csirsSym = nrCSIRS(obj.CarrierConfig, csirsInfo);
            txGrid(csirsInd) = csirsSym;
        end

        function [macPDU, crcFlag] = puschRxProcessing(obj, rxWaveform, puschInfo, packetInfo, carrierConfigInfo, numSampleChannelDelay)
            % Decode PUSCH out of Rx waveform

            rxWaveform = applyRxGain(obj, rxWaveform);
            rxWaveform = applyThermalNoise(obj, rxWaveform);
            pathGains = packetInfo.Metadata.Channel.PathGains * db2mag(packetInfo.Power-30) * db2mag(obj.ReceiveGain);

            % Initialize slot-length waveform
            [startSampleIdx, endSampleIdx] = sampleIndices(obj, puschInfo.NSlot, 0, carrierConfigInfo.SymbolsPerSlot-1);
            slotWaveform = zeros((endSampleIdx-startSampleIdx+1)+numSampleChannelDelay, obj.NumReceiveAntennas);

            % Populate the received waveform at appropriate indices in the slot-length waveform
            startSym = puschInfo.PUSCHConfig.SymbolAllocation(1);
            endSym = startSym+puschInfo.PUSCHConfig.SymbolAllocation(2)-1;
            [startSampleIdx, ~] = sampleIndices(obj, puschInfo.NSlot, startSym, endSym);
            slotWaveform(startSampleIdx : startSampleIdx+size(rxWaveform,1)-1, :) = rxWaveform;

            % Perfect channel estimation
            offset = nrPerfectTimingEstimate(pathGains, packetInfo.Metadata.Channel.PathFilters.');
            estChannelGrid = nrPerfectChannelEstimate(pathGains,packetInfo.Metadata.Channel.PathFilters.', ...
                carrierConfigInfo.NSizeGrid,carrierConfigInfo.SubcarrierSpacing,carrierConfigInfo.NSlot,offset, ...
                packetInfo.Metadata.Channel.SampleTimes);

            % Apply MIMO deprecoding to estChannelGrid to give an estimate
            % per transmission layer
            F = eye(puschInfo.PUSCHConfig.NumAntennaPorts, packetInfo.NumTransmitAntennas);
            K = size(estChannelGrid,1);
            estChannelGrid = reshape(estChannelGrid,K*carrierConfigInfo.SymbolsPerSlot*obj.NumReceiveAntennas,packetInfo.NumTransmitAntennas);
            estChannelGrid = estChannelGrid * F.';
            estChannelGrid = estChannelGrid * packetInfo.Metadata.PrecodingMatrix.';
            estChannelGrid = reshape(estChannelGrid,K,carrierConfigInfo.SymbolsPerSlot,obj.NumReceiveAntennas,[]);

            % Perform OFDM demodulation on the received data to recreate the
            % resource grid, including padding in the event that practical
            % synchronization results in an incomplete slot being demodulated
            slotWaveform = slotWaveform(1+offset:end, :);
            rxGrid = nrOFDMDemodulate(carrierConfigInfo, slotWaveform);

            % Noise variance
            noiseEst = calculateThermalNoise(obj);

            % Get PUSCH resource elements from the received grid
            [puschIndices, ~] = nrPUSCHIndices(carrierConfigInfo, puschInfo.PUSCHConfig);
            [puschRx, puschHest] = nrExtractResources(puschIndices, rxGrid, estChannelGrid);

            % Equalization
            [puschEq, csi] = nrEqualizeMMSE(puschRx, puschHest, noiseEst);

            % Decode PUSCH physical channel
            puschInfo.PUSCHConfig.TransmissionScheme = 'nonCodebook';
            [ulschLLRs, rxSymbols] = nrPUSCHDecode(carrierConfigInfo, puschInfo.PUSCHConfig, puschEq, noiseEst);

            csi = nrLayerDemap(csi);
            Qm = size(ulschLLRs,1) / size(rxSymbols,1);
            csi = reshape(repmat(csi{1}.',Qm,1),[],1);
            ulschLLRs = ulschLLRs .* csi;

            % Decode the UL-SCH transport channel
            obj.ULSCHDecoders{puschInfo.PUSCHConfig.RNTI}.TransportBlockLength = puschInfo.TBS*8;
            obj.ULSCHDecoders{puschInfo.PUSCHConfig.RNTI}.TargetCodeRate = puschInfo.TargetCodeRate;
            [decbits, crcFlag] = obj.ULSCHDecoders{puschInfo.PUSCHConfig.RNTI}.step(ulschLLRs, ...
                puschInfo.PUSCHConfig.Modulation, puschInfo.PUSCHConfig.NumLayers, puschInfo.RV, puschInfo.HARQID);

            if puschInfo.RV == obj.RVSequence(end)
                % The last redundancy version failed. Reset the soft buffer
                resetSoftBuffer(obj.ULSCHDecoders{puschInfo.PUSCHConfig.RNTI}, puschInfo.HARQID);
            end

            % Convert bit stream to byte stream
            macPDU = bit2int(decbits, 8);
        end

        function [srsMeasurement, sinrList] = srsRxProcessing(obj, ~, packet, carrierConfigInfo)
            % SRS measurement on Rx waveform

            % Set PxSCH MCS table
            mcsTable = 'qam256';

            % Apply receive gain
            pathGains = packet.Metadata.Channel.PathGains * db2mag(packet.Power-30) * db2mag(obj.ReceiveGain);
            % Perfect channel estimation
            offset = nrPerfectTimingEstimate(pathGains, packet.Metadata.Channel.PathFilters.');
            estChannelGrid = nrPerfectChannelEstimate(pathGains,packet.Metadata.Channel.PathFilters.', ...
                carrierConfigInfo.NSizeGrid,carrierConfigInfo.SubcarrierSpacing,carrierConfigInfo.NSlot,offset, ...
                packet.Metadata.Channel.SampleTimes);

            rnti = packet.Metadata.RNTI;
            nVar = calculateThermalNoise(obj);

             % Compute uplink rank selecton and PMI for the UEs SRS transmission
            [ulRank,pmi,~,~]= nr5g.internal.nrULCSIMeasurements(carrierConfigInfo,packet.Metadata.PacketConfig,obj.PUSCHConfig,estChannelGrid,nVar,mcsTable,carrierConfigInfo.NSizeGrid);

            if ~any(isnan(pmi))
                % CQI Selection
                blerThreshold = 0.1;
                overhead = 0;
                % Update number of layers with the calculated uplink rank
                obj.PUSCHConfig.NumLayers = ulRank;

                wtx = nrPUSCHCodebook(ulRank,size(estChannelGrid,4),pmi);
                noiseEst = calculateThermalNoise(obj);
                [obj.L2SMsSRS(rnti),SINRs] = nr5g.internal.L2SM.linkQualityModel(obj.L2SMsSRS(rnti),carrierConfigInfo,obj.PUSCHConfig,estChannelGrid,noiseEst,wtx);
                [obj.L2SMsSRS(rnti),mcsIndex,mcsInfo] = nr5g.internal.L2SM.cqiSelect(obj.L2SMsSRS(rnti), ...
                    obj.CarrierConfig,obj.PUSCHConfig,overhead,SINRs,obj.MCSTableValues,blerThreshold);
                sinrList = mcsInfo.EffectiveSINR;
                % Fill SRS measurements
                srsMeasurement.RNTI = rnti;
                srsMeasurement.RankIndicator = ulRank;
                srsMeasurement.TPMI = pmi;
                srsMeasurement.MCSIndex = mcsIndex;
                srsMeasurement.SRSBasedDLMeasurements = [];
            else
                % Ignore SRS measurement report
                srsMeasurement = [];
                sinrList = -Inf;
            end
        end

        function waveformOut = applyRxGain(obj, waveformIn)
            %applyRxGain Apply receiver antenna gain

            scale = 10.^(obj.ReceiveGain/20);
            waveformOut = waveformIn.* scale;
        end

        function waveformOut = applyThermalNoise(obj, waveformIn)
            %applyThermalNoise Apply thermal noise

            noiseFigure = 10^(obj.NoiseFigure/10);
            % Thermal noise (in Watts)
            Nt = physconst('Boltzmann') * (obj.AntNoiseTemperature + 290*(noiseFigure-1)) * obj.WaveformInfo.SampleRate;
            noise = sqrt(Nt/2)*complex(randn(size(waveformIn)),randn(size(waveformIn)));
            waveformOut = waveformIn + noise;
        end
    end
end
classdef nrUEFullPHY < nr5g.internal.nrUEPHY
 %nrUEFullPHY Implements full PHY processing for UE.
    %   The class implements the full-PHY specific aspects of UE PHY.
    %
    %   Note: This is an internal undocumented class and its API and/or
    %   functionality may change in subsequent releases.

    %   Copyright 2023-2024 The MathWorks, Inc.

    properties (Access = protected)
        %ULSCHEncoder Uplink shared channel (UL-SCH) encoder system object
        % It is an object of type nrULSCH
        ULSCHEncoder

        %DLSCHDecoder Downlink shared channel (DL-SCH) decoder system object
        % It is an object of type nrDLSCHDecoder
        DLSCHDecoder

        %TimingOffset Receiver timing offset
        TimingOffset = 0;
    end

    properties
        %RVSequence Redundancy version sequence
        RVSequence
    end

    methods
        function obj = nrUEFullPHY(param, notificationFcn)
            %nrUEFullPHY Construct a UE full PHY object
            %   OBJ = nrUEFullPHY(PARAM,NOTIFICATIONFCN) constructs a UE full PHY object.
            %
            %   PARAM is a structure with the fields:
            %     TransmitPower       - UE Tx power in dBm
            %     NumTransmitAntennas - Number of Tx antennas on the UE
            %     NumReceiveAntennas  - Number of Rx antennas on the UE
            %     NoiseFigure         - Noise figure
            %     ReceiveGain         - UE Rx gain in dBi
            %
            %   NOTIFICATIONFCN - It is a handle of the node's processEvents
            %   method

            obj = obj@nr5g.internal.nrUEPHY(param, notificationFcn); % Call base class constructor

            % NR Packet param
            obj.PacketStruct.Abstraction = false; % Full PHY
            obj.PacketStruct.Metadata = struct('NCellID', [], 'RNTI', [], ...
                'PrecodingMatrix', [], 'NumSamples', [], 'Channel', obj.PacketStruct.Metadata.Channel);
        end

        function addConnection(obj, connectionConfig)
            %addConnection Configures the UE PHY with connection information
            %   connectionConfig is a structure including the following
            %   fields:
            %       RNTI                     - Radio network temporary identifier
            %                                  specified within [1, 65522]. Refer
            %                                  table 7.1-1 in 3GPP TS 38.321 version 18.1.0
            %       NCellID                  - Physical cell ID. values: 0 to 1007 (TS 38.211, sec 7.4.2.1)
            %       DuplexMode               - "FDD" or "TDD"
            %       SubcarrierSpacing        - Subcarrier spacing
            %       NumResourceBlocks        - Number of RBs
            %       NumHARQ                  - Number of HARQ processes on UE
            %       ChannelBandwidth         - DL or UL channel bandwidth in Hz
            %       DLCarrierFrequency       - DL carrier frequency
            %       ULCarrierFrequency       - UL carrier frequency
            %       CSIReportConfiguration   - CSI report configuration

            addConnection@nr5g.internal.nrUEPHY(obj, connectionConfig);
            % Create UL-SCH encoder system object
            ulschEncoder = nrULSCH;
            ulschEncoder.MultipleHARQProcesses = true;
            obj.ULSCHEncoder = ulschEncoder;

            % Create DL-SCH decoder system object
            dlschDecoder = nrDLSCHDecoder;
            dlschDecoder.MultipleHARQProcesses = true;
            dlschDecoder.LDPCDecodingAlgorithm = 'Normalized min-sum';
            dlschDecoder.MaximumLDPCIterationCount = 6;
            obj.DLSCHDecoder = dlschDecoder;

            % Initialize L2SM to hold CSI-RS and inter-user interferer context
            obj.L2SMCSI = nr5g.internal.L2SM.initialize(obj.CarrierConfig);
            obj.L2SMIUI = nr5g.internal.L2SM.initialize(obj.CarrierConfig);
        end

        function data = puschData(obj, puschInfo, macPDU)
            % Return the PUSCH waveform

            % Fill the slot grid with PUSCH symbols
            puschGrid = populatePUSCH(obj, puschInfo, macPDU);
            % OFDM modulation
            txWaveform = nrOFDMModulate(obj.CarrierConfig, puschGrid);
            % Trim txWaveform to span only the transmission symbols
            startSym = puschInfo.PUSCHConfig.SymbolAllocation(1);
            endSym = startSym+puschInfo.PUSCHConfig.SymbolAllocation(2)-1;
            [startSampleIdx, endSampleIdx] = sampleIndices(obj, puschInfo.NSlot, startSym, endSym);
            txWaveform = txWaveform(startSampleIdx:endSampleIdx, :);
            % Signal amplitude. Account for FFT occupancy factor
            signalAmp = db2mag(obj.AdjustedTransmitPower-30) * sqrt((obj.WaveformInfo.Nfft^2) / (12*numel(puschInfo.PUSCHConfig.PRBSet)));
            % Apply the Tx power
            data = signalAmp*txWaveform;
        end

        function data = srsData(obj, srsConfig)
            % Return SRS waveform

            % Fill the slot grid with SRS symbols
            srsGrid = populateSRS(obj, srsConfig);
            % OFDM modulation
            txWaveform = nrOFDMModulate(obj.CarrierConfig, srsGrid);
            % Signal amplitude. Account for FFT occupancy factor
            signalAmp = db2mag(obj.AdjustedTransmitPower-30) * sqrt((obj.WaveformInfo.Nfft^2) / (12*srsConfig.NRB));
            % Apply the Tx power
            data = signalAmp*txWaveform;
        end

        function [macPDU, crcFlag, sinr] = decodePDSCH(obj, pdschInfo, pktStartTime, pktEndTime, carrierConfigInfo)
            % Return the decoded MAC PDU along with the crc result

            % Initialization
            sinr = -Inf;
            crcFlag = 1;
            macPDU = [];

            % Read all the relevant packets (i.e. either of interest or sent on same
            % carrier frequency) received in the time window
            packetInfoList = packetList(obj.RxBuffer, pktStartTime, pktEndTime);
            packetOfInterest = [];
            for j=1:size(packetInfoList,1) % Search PDSCH of interest in the list of received packets
                packet = packetInfoList(j);
                if (packet.Metadata.PacketType == obj.PXSCHPacketType) && ... % Check for PDSCH
                        (carrierConfigInfo.NCellID == packet.Metadata.NCellID) && ... % Check for PDSCH of interest
                        any(pdschInfo.PDSCHConfig.RNTI == packet.Metadata.RNTI) && ...
                        (pktStartTime == packet.StartTime)
                    packetOfInterest = packet;
                    % Read the combined waveform received during packet's duration
                    rxWaveform = resultantWaveform(obj.RxBuffer, pktStartTime, pktStartTime+packet.Duration);
                    channelDelay = packet.Duration -(pktEndTime-pktStartTime);
                    numSampleChannelDelay = ceil(channelDelay*packetOfInterest.SampleRate);
                    break;
                end
            end

            if ~isempty(packetOfInterest)
                % PDSCH Rx processing
                [macPDU, crcFlag] = pdschRxProcessing(obj, rxWaveform, pdschInfo, packetOfInterest, carrierConfigInfo, numSampleChannelDelay);
            end
        end

        function [dlRank, pmiSet, cqiRBs, precodingMatrix, sinr] = decodeCSIRS(obj, csirsConfig, pktStartTime, pktEndTime, carrierConfigInfo)
            % Return CSI-RS measurement

            [dlRank, pmiSet, cqiRBs, precodingMatrix, sinr] = csirsRxProcessing(obj, csirsConfig, pktStartTime, pktEndTime, carrierConfigInfo);
        end
    end

    methods (Access = protected)
         function txGrid = populatePUSCH(obj, puschInfo, macPDU)
            %populatePUSCH Populate PUSCH symbols in the Tx grid

            % Initialize Tx grid
            txGrid = zeros(obj.CarrierInformation.NumResourceBlocks*12, obj.WaveformInfo.SymbolsPerSlot, obj.NumTransmitAntennas, "single");
            if ~isempty(macPDU)
                % A non-empty MAC PDU is sent by MAC which indicates new
                % transmission
                macPDUBitmap = int2bit(macPDU, 8);
                macPDUBitmap = reshape(macPDUBitmap', [], 1); % Convert to column vector
                setTransportBlock(obj.ULSCHEncoder, macPDUBitmap, puschInfo.HARQID);
            end

            % Calculate PUSCH and DM-RS information
            obj.CarrierConfig.NSlot = puschInfo.NSlot;
            [puschIndices, puschIndicesInfo] = nrPUSCHIndices(obj.CarrierConfig, puschInfo.PUSCHConfig);
            dmrsSymbols = nrPUSCHDMRS(obj.CarrierConfig, puschInfo.PUSCHConfig);
            dmrsIndices = nrPUSCHDMRSIndices(obj.CarrierConfig, puschInfo.PUSCHConfig);

            % UL-SCH encoding
            obj.ULSCHEncoder.TargetCodeRate = puschInfo.TargetCodeRate;
            codedTrBlock = obj.ULSCHEncoder(puschInfo.PUSCHConfig.Modulation, puschInfo.PUSCHConfig.NumLayers, ...
                puschIndicesInfo.G, puschInfo.RV, puschInfo.HARQID);

            % PUSCH modulation
            puschSymbols = nrPUSCH(obj.CarrierConfig, puschInfo.PUSCHConfig, codedTrBlock);

            % PUSCH mapping in the grid
            txGrid(puschIndices) = puschSymbols;

            % PUSCH DM-RS mapping
            txGrid(dmrsIndices) = dmrsSymbols;
         end

         function txGrid = populateSRS(obj, srsConfig)
             %populateSRS Populate SRS symbols in the Tx grid

             % Populate Tx grid
             txGrid = zeros(obj.CarrierInformation.NumResourceBlocks*12, obj.WaveformInfo.SymbolsPerSlot, obj.NumTransmitAntennas, "single");
             srsInd = nrSRSIndices(obj.CarrierConfig, srsConfig);
             srsSym = nrSRS(obj.CarrierConfig, srsConfig);
             txGrid(srsInd) = srsSym;
         end

         function [macPDU, crcFlag] = pdschRxProcessing(obj, rxWaveform, pdschInfo, packetInfo, carrierConfigInfo, numSampleChannelDelay)
             % Decode PDSCH out of Rx waveform

             rxWaveform = applyRxGain(obj, rxWaveform);
             rxWaveform = applyThermalNoise(obj, rxWaveform);
             pathGains = packetInfo.Metadata.Channel.PathGains * db2mag(packetInfo.Power-30) * db2mag(obj.ReceiveGain);

             % Initialize slot-length waveform
             [startSampleIdx, endSampleIdx] = sampleIndices(obj, pdschInfo.NSlot, 0, carrierConfigInfo.SymbolsPerSlot-1);
             slotWaveform = zeros((endSampleIdx-startSampleIdx+1)+numSampleChannelDelay, obj.NumReceiveAntennas);

             % Populate the received waveform at appropriate indices in the slot-length waveform
             startSym = pdschInfo.PDSCHConfig.SymbolAllocation(1);
             endSym = startSym+pdschInfo.PDSCHConfig.SymbolAllocation(2)-1;
             [startSampleIdx, ~] = sampleIndices(obj, pdschInfo.NSlot, startSym, endSym);
             slotWaveform(startSampleIdx : startSampleIdx+size(rxWaveform,1)-1, :) = rxWaveform;

             % Perfect timing estimation
             offset = nrPerfectTimingEstimate(pathGains, packetInfo.Metadata.Channel.PathFilters.');
             slotWaveform = slotWaveform(1+offset:end, :);

             % Perform OFDM demodulation on the received data to recreate the
             % resource grid, including padding in the event that practical
             % synchronization results in an incomplete slot being demodulated
             rxGrid = nrOFDMDemodulate(carrierConfigInfo, slotWaveform);

             % Perfect channel estimation
             estChannelGrid = nrPerfectChannelEstimate(pathGains,packetInfo.Metadata.Channel.PathFilters.', ...
                 carrierConfigInfo.NSizeGrid,carrierConfigInfo.SubcarrierSpacing,carrierConfigInfo.NSlot,offset, ...
                 packetInfo.Metadata.Channel.SampleTimes);

             % Extract PDSCH resources
             [pdschIndices, ~] = nrPDSCHIndices(carrierConfigInfo, pdschInfo.PDSCHConfig);
             [pdschRx, pdschHest, ~, pdschHestIndices] = nrExtractResources(pdschIndices, rxGrid, estChannelGrid);

             % Noise variance
             noiseEst = calculateThermalNoise(obj);

             % Apply precoding to channel estimate
             ueIdx = find(packetInfo.Metadata.RNTI == obj.RNTI, 1);
             precodingMatrix = packetInfo.Metadata.PrecodingMatrix{ueIdx};
             pdschHest = nrPDSCHPrecode(carrierConfigInfo,pdschHest,pdschHestIndices,permute(precodingMatrix,[2 1 3]));

             % Equalization
             [pdschEq, csi] = nrEqualizeMMSE(pdschRx,pdschHest, noiseEst);

             % PDSCH decoding
             [dlschLLRs, rxSymbols] = nrPDSCHDecode(pdschEq, pdschInfo.PDSCHConfig.Modulation, pdschInfo.PDSCHConfig.NID, ...
                 pdschInfo.PDSCHConfig.RNTI, noiseEst);

             % Scale LLRs by CSI
             csi = nrLayerDemap(csi); % CSI layer demapping

             cwIdx = 1;
             Qm = size(dlschLLRs{1},1)/size(rxSymbols{cwIdx},1); % bits per symbol
             csi{cwIdx} = repmat(csi{cwIdx}.',Qm,1);   % expand by each bit per symbol
             dlschLLRs{cwIdx} = dlschLLRs{cwIdx} .* csi{cwIdx}(:);   % scale

             obj.DLSCHDecoder.TransportBlockLength = pdschInfo.TBS*8;
             obj.DLSCHDecoder.TargetCodeRate = pdschInfo.TargetCodeRate;

             [decbits, crcFlag] = obj.DLSCHDecoder(dlschLLRs, pdschInfo.PDSCHConfig.Modulation, ...
                 pdschInfo.PDSCHConfig.NumLayers, pdschInfo.RV, pdschInfo.HARQID);

             if pdschInfo.RV == obj.RVSequence(end)
                 % The last redundancy version failed. Reset the soft
                 % buffer
                 resetSoftBuffer(obj.DLSCHDecoder, 0, pdschInfo.HARQID);
             end

             % Convert bit stream to byte stream
             macPDU = bit2int(decbits, 8);
         end

         function waveformOut = applyThermalNoise(obj, waveformIn)
             %applyThermalNoise Apply thermal noise

             noiseFigure = 10^(obj.NoiseFigure/10);
             % Thermal noise (in Watts)
             Nt = physconst('Boltzmann') * (obj.AntNoiseTemperature + 290*(noiseFigure-1)) * obj.WaveformInfo.SampleRate;
             noise = sqrt(Nt/2)*complex(randn(size(waveformIn)),randn(size(waveformIn)));
             waveformOut = waveformIn + noise;
         end

         function waveformOut = applyRxGain(obj, waveformIn)
             %applyRxGain Apply receiver antenna gain

             scale = 10.^(obj.ReceiveGain/20);
             waveformOut = waveformIn.* scale;
         end
    end
end
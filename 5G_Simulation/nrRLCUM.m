classdef nrRLCUM < nr5g.internal.nrRLC
    %nrRLCUM Implement RLC UM functionality
    %
    %   Note: This is an internal undocumented class and its API and/or
    %   functionality may change in subsequent releases.
    %
    %   RLCOBJ = nrRLCUM(RNTI, RLCBEARERCONFIG, MAXREASSEMBLYSDU,
    %   TXBUFFERSTATUSFCN, RXFORWARDFCN) implements RLC UM functionality.
    %
    %   RNTI is radio network temporary identifier.
    %
    %   RLCBEARERCONFIG is rlc bearer configuration object. For more
    %   information, please refer 'nrRLCBearerConfig' documentation.
    %
    %   MAXREASSEMBLYSDU is the maximum capacity of reassembly buffer in terms
    %   of number of SDUs.
    %
    %   TXBUFFERSTATUSFCN specifies the function callback that will be used
    %   to send the buffer status report to MAC.
    %
    %   RXFORWARDFCN specifies the function callback that will be used
    %   to forward the packets to higher layer.
    %
    %   nrRLCUM properties (configurable through constructor):
    %
    %   SNFieldLength    - Number of bits in sequence number field of RLC
    %                      entity
    %   BufferSize       - RLC Transmitter buffer size in terms of number of
    %                      packets
    %   ReassemblyTimer  - Waiting time (in milliseconds) before declaring
    %                      the reassembly failure of SDUs in the reception
    %                      buffer
    %   MaxReassemblySDU - Maximum capacity of the reassembly buffer in
    %                      terms of number of SDUs. This is also equal to
    %                      the maximum segments that can be missing per SDU
    %                      at any point of time. This value is exactly
    %                      equals to the number of HARQ processes at MAC
    %                      layer
    %
    %   nrRLCUM methods:
    %
    %   run - Run the RLC entity and return the next invoke time of RLC
    %         entity

    %   Copyright 2022-2024 The MathWorks, Inc.

    properties
        %SNFieldLength Number of bits in sequence number field of
        %transmitter and receiver entities
        %   Specify the sequence number field length as an integer scalar.
        %   The sequence number field length is one of 6 | 12. For more
        %   details, refer 3GPP TS 38.322 Section 6.2.3.3.
        SNFieldLength
    end

    % Tx configuration
    properties (Access = private)
        %TxNext Sequence number to be assigned for the next newly generated
        %UMD PDU with an SDU segment
        TxNext = 0

        %TxSegmentOffset Position of the segmented SDU in bytes within the
        %original SDU
        TxSegmentOffset = 0

        %TxBuffer Transmit queue object that stores the SDUs received from higher
        %layers
        TxBuffer

        %RequiredGrantLength Length of the required grant to transmit the
        %data in the Tx buffer
        RequiredGrantLength = 0
    end

    % Rx configuration
    properties (Access = private)
        %RxNextHighest The sequence number (SN) following the SN of the
        %unacknowledged mode data (UMD) PDU with the highest SN among
        %received UMD PDUs
        RxNextHighest = 0

        %RxNextHighestModulus Modulus value of RxNextHighest
        RxNextHighestModulus = 0

        %RxNextReassembly The earliest SN that is still considered for
        %reassembly
        RxNextReassembly = 0

        %RxNextReassemblyModulus Modulus value of RxNextReassembly
        RxNextReassemblyModulus = 0

        %RxTimerTrigger The SN following the SN which triggered reassembly
        %timer
        RxTimerTrigger = 0

        %ReassemblyTimeLeft Time (in milliseconds) that is left for
        %reassembly timer expiry
        ReassemblyTimeLeft = 0

        %RxBuffer Receiver buffer for reassembly procedure. This is a
        %N-by-1 array of nrRLCDataReassembly objects where 'N' equals to
        %the value of 'MaxReassemblySDU' property
        RxBuffer

        %ReassemblySNMap Map that shows where the segmented SDUs are stored
        %in the reassembly buffer. This is a N-by-1 column vector where 'N'
        %is the maximum reassembly buffer length. Each element contains the
        %SN of the SDUs which are under reassembly procedure. Each element
        %in the vector can take value in the range between -1 and
        %2^SNFieldLength-1. if an element is set to -1, it indicates
        %that it is not occupied by any SDUs SN
        ReassemblySNMap

        %RcvdSNList List of contiguously received full SDU SNs inside the
        %reassembly window. This is a N-by-2 matrix where 'N' is the
        %maximum reassembly buffer length. Each row has a starting SN and
        %ending SN that indicates a contiguous reception of SNs in the
        %receiving window. Value [-1, -1] in a row indicates not occupied
        RcvdSNList
    end

    % Properties that won't get modified after their initialization in the constructor
    properties (Access = private)
        %TxSeqNumFieldLength Sequence number field length of the Tx side
        TxSeqNumFieldLength

        %TotalTxSeqNum The number of SNs configured on the RLC UM
        %transmitter entity
        TotalTxSeqNum

        %PDUHeaderLength PDU header length for the first SDU awaiting in the
        %transmit buffer
        PDUHeaderLength = 1

        %PDUHeaderLengthForSegmentedSDU Estimated PDU header length for the
        %segmented SDU
        PDUHeaderLengthForSegmentedSDU

        %RxSeqNumFieldLength Sequence number field length for the Rx side
        RxSeqNumFieldLength

        %TotalRxSeqNum The number of SNs configured on the RLC UM receiver
        %entity
        TotalRxSeqNum

        %UMWindowSize Indicates the size of the reassembly window. It is
        %used to define SNs of those UMD SDUs that can be received without
        %causing an advancement of the receiving window
        UMWindowSize = 0

        %ReassemblyTimerNS Reassembly timer in nanoseconds
        ReassemblyTimerNS
    end

    methods
        %Constructor
        function obj = nrRLCUM(rnti, rlcBearerConfig, maxReassemblySDU, txBufferStatusFcn, rxForwardFcn)

            obj@nr5g.internal.nrRLC(rnti, rlcBearerConfig.LogicalChannelID);
            obj.SNFieldLength = rlcBearerConfig.SNFieldLength;
            % Initialize Tx side configuration
            if ~isempty(txBufferStatusFcn)
                obj.BufferSize = rlcBearerConfig.BufferSize;
                obj.TxBuffer = wirelessnetwork.internal.queue(obj.BufferSize);
                obj.TxSeqNumFieldLength = obj.SNFieldLength;
                obj.TotalTxSeqNum = 2^obj.TxSeqNumFieldLength;
                obj.TxBufferStatusFcn = txBufferStatusFcn;
                % Initialize RLC buffer status structure
                obj.RLCBufferStatus.RNTI = obj.RNTI;
                obj.RLCBufferStatus.LogicalChannelID = obj.LogicalChannelID;
                % Set the PDU header length of the segmented SDU based on the sequence
                % number field length
                if obj.SNFieldLength == 6
                    obj.PDUHeaderLengthForSegmentedSDU = 3;
                else
                    obj.PDUHeaderLengthForSegmentedSDU = 4;
                end
            end

            % Initialize Rx side configuration
            if ~isempty(rxForwardFcn)
                obj.ReassemblyTimer = rlcBearerConfig.ReassemblyTimer;
                obj.ReassemblyTimerNS = obj.ReassemblyTimer * 1e6;
                obj.MaxReassemblySDU = maxReassemblySDU;
                obj.RxSeqNumFieldLength = obj.SNFieldLength;
                obj.UMWindowSize = 2^(obj.RxSeqNumFieldLength - 1);
                obj.TotalRxSeqNum = 2^obj.RxSeqNumFieldLength;
                % Define reassembly buffer and SN map array
                obj.RxBuffer = repmat(nr5g.internal.nrRLCDataReassembly(maxReassemblySDU,obj.MaxPacketSize), maxReassemblySDU, 1);
                for pktIdx = 2:maxReassemblySDU
                    obj.RxBuffer(pktIdx) = nr5g.internal.nrRLCDataReassembly(maxReassemblySDU,obj.MaxPacketSize);
                end
                obj.ReassemblySNMap = -1 * ones(maxReassemblySDU, 1);
                obj.RcvdSNList = -1 * ones(maxReassemblySDU, 2);
                obj.RxForwardFcn = rxForwardFcn;
            end
        end

        function nextInvokeTime = run(obj, currentTime)
            %run Run the RLC entity and return the next invoke time of RLC
            %entity
            %
            %   NEXTINVOKETIME = run(OBJ, CURRENTTIME) runs the RLC entity
            %   and returns the next invoke time of RLC entity.
            %
            %   NEXTINVOKETIME indicates the time (in nanoseconds) at which
            %   the run function should be invoked again.
            %
            %   OBJ is an object of type nrRLCUM.
            %
            %   CURRENTTIME is an integer indicating the current time (in
            %   nanoseconds).

            if obj.ReassemblyTimeLeft > 0
                % Update the reassembly timer
                elapsedTime = currentTime - obj.LastRunTime; % In nanoseconds
                obj.ReassemblyTimeLeft = obj.ReassemblyTimeLeft - elapsedTime;
                if obj.ReassemblyTimeLeft <= 0
                    obj.ReassemblyTimeLeft = 0;
                    nextInvokeTime = Inf;
                    obj.StatReassemblyTimerExpiry = obj.StatReassemblyTimerExpiry + 1;
                    % Handle the reassembly timer expiry
                    reassemblyTimerExpiry(obj);
                else
                    nextInvokeTime = currentTime + obj.ReassemblyTimeLeft;
                end
            else
                nextInvokeTime = Inf;
            end
            obj.LastRunTime = currentTime;
        end
    end

    methods (Hidden)
        function isPacketQueued = enqueueSDU(obj, rlcSDU)
            %enqueueSDU Queue the received SDU from higher layers in the Tx
            %buffer
            %
            %   enqueueSDU(OBJ, RLCSDU) queues the received SDU in the Tx
            %   buffer. Also, it generates and stores the corresponding RLC
            %   UM header in a Tx header storage buffer.
            %
            %   RLCSDU is a structure with these fields.
            %       Packet       - Array of octets in decimal format.
            %       PacketLength - Length of packet.
            %       Tags         - Array of structures where each structure
            %                      contains these fields.
            %                      Name      - Name of the tag.
            %                      Value     - Data associated with the tag.
            %                      ByteRange - Specific range of bytes within
            %                                  the packet to which the tag
            %                                  applies.

            coder.internal.errorIf(rlcSDU.PacketLength>obj.MaxPacketSize, "nr5g:nrRLC:InvalidRLCSDUSize")
            % Store the SDU in the transmit queue
            isPacketQueued = enqueue(obj.TxBuffer, rlcSDU);
            if ~isPacketQueued
                % Update statistics when the packet is not queued
                obj.StatTransmitterQueueOverflow = obj.StatTransmitterQueueOverflow + 1;
                return;
            end

            % Increment the required grant size by the sum of expected MAC & RLC
            % headers overhead and complete RLC SDU length
            obj.RequiredGrantLength = obj.RequiredGrantLength + rlcSDU.PacketLength + ...
                obj.MaxRLCMACHeadersOH;

            % Send the updated RLC buffer status report to MAC layer
            obj.RLCBufferStatus.BufferStatus = obj.RequiredGrantLength;
            obj.TxBufferStatusFcn(obj.RLCBufferStatus);
        end

        function rlcPacketInfo = sendPDUs(obj, bytesGranted, remainingTBS, currentTime)
            %sendPDUs Send the RLC protocol data units (PDUs) that fit in
            %the grant notified by MAC layer
            %
            %   RLCPACKETINFO = sendPDUs(OBJ, BYTESGRANTED, REMAININGTBS,
            %   CURRENTTIME) sends the RLC PDUs that fit in the grant notified
            %   by MAC.
            %
            %   RLCPACKETINFO is a struct array of RLC UMD packets to be transmitted
            %   by MAC. Each element represents one UMD packet and its associated
            %   information.
            %
            %   BYTESGRANTED is a positive integer scalar, which represents
            %   the number of granted transmission bytes.
            %
            %   REMAININGTBS is a nonnegative integer scalar, which
            %   represents the remaining number of bytes in the transport
            %   block size (TBS). This helps to avoid the segmentation of
            %   RLC SDUs in round-1 of MAC logical channel prioritization
            %   (LCP) procedure.
            %
            % CURRENTTIME is the current simulation time in nanoseconds.

            txBuffer = obj.TxBuffer;
            numBytesFilled = 0;
            rlcPacketInfo = obj.RLCPacketInfo;
            numRLCPackets = 0;
            % Iterate through the RLC Tx buffer and send RLC PDUs to MAC
            % until it fulfills the granted amount of data for the
            % associated logical channel or RLC Tx buffer becomes empty
            while (numBytesFilled < bytesGranted) && (txBuffer.CurrentSize > 0)
                % Calculate the current RLC PDU length and its MAC header
                % length
                rlcSDU = peek(txBuffer);
                sdu = rlcSDU.Packet;
                pduLength = obj.PDUHeaderLength + rlcSDU.PacketLength - obj.TxSegmentOffset;
                macHeaderLength = (pduLength > 255) + 2;
                % Check if the RLC PDU along with its MAC header does not
                % fit in the assigned grant alone or in combination with
                % the remaining TBS of the associated MAC entity
                if ((numBytesFilled + pduLength + macHeaderLength) > bytesGranted) && ...
                        ((pduLength + macHeaderLength) > remainingTBS)
                    remainingGrant = bytesGranted - numBytesFilled;
                    % Check whether the remaining grant is not sufficient to send a MAC subPDU
                    % of minimum length
                    if remainingGrant < obj.MinMACSubPDULength
                        % Send the updated RLC buffer status report to MAC
                        % layer
                        obj.RLCBufferStatus.BufferStatus = obj.RequiredGrantLength;
                        obj.TxBufferStatusFcn(obj.RLCBufferStatus);
                        return;
                    end
                    % Segment the SDU, and modify the header
                    if obj.TxSegmentOffset
                        % Create the RLC header for middle SDU segment
                        header = generateDataHeader(obj, 3, ...
                            obj.TxSeqNumFieldLength, obj.TxNext, obj.TxSegmentOffset);
                    else
                        % Create the RLC header for first SDU segment
                        header = generateDataHeader(obj, 1, ...
                            obj.TxSeqNumFieldLength, obj.TxNext, obj.TxSegmentOffset);
                    end
                    obj.PDUHeaderLength = obj.PDUHeaderLengthForSegmentedSDU;
                    % Create the RLC UMD PDU by prepending the updated
                    % header to the segmented SDU and put the RLC UMD PDU
                    % in the RLC PDU set
                    segmentedSDULength = remainingGrant - numel(header) - macHeaderLength;
                    segmentedPDU = [header; ...
                        sdu(obj.TxSegmentOffset + 1:obj.TxSegmentOffset + segmentedSDULength)];
                    
                    % Increment the number of RLC packets in the list
                    numRLCPackets = numRLCPackets + 1;
                    rlcPacketInfo(numRLCPackets).Packet = segmentedPDU;
                    rlcPacketInfo(numRLCPackets).PacketLength = numel(segmentedPDU);
                    % Update existing tags to accommodate for segmentation and header addition
                    rlcPacketInfo(numRLCPackets).Tags = wirelessnetwork.internal.packetTags.segment(rlcSDU.Tags, ...
                        [obj.TxSegmentOffset+1 obj.TxSegmentOffset+segmentedSDULength]);
                    rlcPacketInfo(numRLCPackets).Tags = wirelessnetwork.internal.packetTags.adjust(rlcPacketInfo(numRLCPackets).Tags, numel(header));
                    % Update the statistics
                    obj.StatTransmittedDataPackets = obj.StatTransmittedDataPackets + 1;
                    obj.StatTransmittedDataBytes = obj.StatTransmittedDataBytes + numel(segmentedPDU);
                    % Increment the segment offset by segmented SDU length
                    obj.TxSegmentOffset = obj.TxSegmentOffset + segmentedSDULength;
                    % Increment the utilized grant by remaining grant size and update the
                    % required grant size as sum of the updated MAC header length, RLC PDU
                    % length
                    numBytesFilled = numBytesFilled + remainingGrant;
                    obj.RequiredGrantLength = obj.RequiredGrantLength - segmentedSDULength;
                else
                    if obj.TxSegmentOffset
                        % Generate the header for the last segment of the SDU transmission
                        header = generateDataHeader(obj, 2, obj.TxSeqNumFieldLength, obj.TxNext, ...
                            obj.TxSegmentOffset);
                    else
                        % Generate the header for the complete SDU transmission
                        header = generateDataHeader(obj, 0, obj.TxSeqNumFieldLength, 0, 0);
                    end
                    dequeue(txBuffer);
                    % Create the RLC UMD PDU by prepending the header to
                    % the SDU and put the RLC UMD PDU in the RLC PDUs set
                    normalPDU = [header; sdu(obj.TxSegmentOffset + 1:end)];
                    % Increment the number of RLC packets in the list
                    numRLCPackets = numRLCPackets + 1;
                    rlcPacketInfo(numRLCPackets).Packet = normalPDU;
                    rlcPacketInfo(numRLCPackets).PacketLength = numel(normalPDU);
                    % Update existing tags to accommodate for segmentation and header addition
                    rlcPacketInfo(numRLCPackets).Tags = wirelessnetwork.internal.packetTags.segment(rlcSDU.Tags, ...
                        [obj.TxSegmentOffset+1 rlcSDU.PacketLength]);
                    rlcPacketInfo(numRLCPackets).Tags = wirelessnetwork.internal.packetTags.adjust(rlcPacketInfo(numRLCPackets).Tags, numel(header));
                    % Update the statistics
                    obj.StatTransmittedDataPackets = obj.StatTransmittedDataPackets + 1;
                    obj.StatTransmittedDataBytes = obj.StatTransmittedDataBytes + numel(normalPDU);
                    sduLength = rlcSDU.PacketLength - obj.TxSegmentOffset;
                    % Increment the sequence number after the submission of
                    % last segment of the segmented SDU
                    if obj.TxSegmentOffset
                        obj.TxNext = mod(obj.TxNext + 1, obj.TotalTxSeqNum);
                        obj.TxSegmentOffset = 0;
                        obj.PDUHeaderLength = 1;
                    end
                    % Increment the utilized grant by the sum of RLC PDU
                    % length and MAC header length. Decrement the required
                    % grant length also by the sum of RLC PDU length and
                    % MAC header length
                    numBytesFilled = numBytesFilled + pduLength + macHeaderLength;
                    obj.RequiredGrantLength = obj.RequiredGrantLength - ...
                        sduLength - obj.MaxRLCMACHeadersOH;
                    % If Tx buffer is empty, no grant is required
                    if txBuffer.CurrentSize == 0
                        obj.RequiredGrantLength = 0;
                    end
                end
            end
            % Send the updated RLC buffer status report to MAC layer
            obj.RLCBufferStatus.BufferStatus = obj.RequiredGrantLength;
            obj.TxBufferStatusFcn(obj.RLCBufferStatus);
        end

        function receivePDUs(obj, rlcPacketInfo, currentTime)
            %receivePDUs Receive and process RLC PDU from the MAC layer
            %
            %   receivePDUs(OBJ, RLCPACKETINFO, CURRENTTIME) Receives and
            %   processes RLC PDU from the MAC layer. It reassembles the
            %   SDUs received from RLC PDU and delivers the SDU to higher
            %   layer.
            %
            %   RLCPACKETINFO is a structure containing these fields.
            %       NodeID       - Node identifier.
            %       Packet       - Array of bytes represented in decimals.
            %       PacketLength - Number of bytes in the packet.
            %       Tags         - Array of structures where each structure
            %                      contains these fields.
            %                      Name      - Name of the tag.
            %                      Value     - Data associated with the tag.
            %                      ByteRange - Specific range of bytes within
            %                                  the packet to which the tag
            %                                  applies.
            %
            %   CURRENTTIME - Current time in nanoseconds.

            sdu = [];
            % Decode the RLC UMD PDU received from the MAC layer
            [decodedPDU, decodingStatus] = decodeDataPDU(obj, rlcPacketInfo.Packet);
            if decodingStatus
                % Update the statistics
                obj.StatReceivedDataPackets = obj.StatReceivedDataPackets + 1;
                obj.StatReceivedDataBytes = obj.StatReceivedDataBytes + decodedPDU.PDULength;
            else
                % Update the statistics and avoid any further processing
                obj.StatDecodeFailures = obj.StatDecodeFailures + 1;
                return;
            end
            % Adjust remaining tags on removal of RLC header
            higherLayerTags = wirelessnetwork.internal.packetTags.adjust(rlcPacketInfo.Tags, ...
                -(decodedPDU.PDULength-numel(decodedPDU.Data)));
            % On the reception of complete SDU, forward it to upper layer
            if decodedPDU.SegmentationInfo == 0
                sdu = uint8(decodedPDU.Data);
            else
                snModuloValue = getModulusValue(obj, decodedPDU.SequenceNumber);
                % If the received SDU segment is an old segment, discard
                % it. Otherwise, store it in the reception buffer for
                % reassembly
                if snModuloValue < obj.RxNextReassemblyModulus
                    obj.StatDroppedDataPackets = obj.StatDroppedDataPackets + 1;
                    obj.StatDroppedDataBytes = obj.StatDroppedDataBytes + rlcPacketInfo.PacketLength;
                else
                    % Put the received SDU segment in the reassembly
                    % buffer. If the reassembly buffer is full, it will
                    % discard the oldest SDU from the buffer
                    snBufIdx = assignReassemblyBufIdx(obj, decodedPDU.SequenceNumber);
                    if snBufIdx < 0
                        [~, snBufIdx] = min(getModulusValue(obj, obj.ReassemblySNMap));
                        [numSegments, numBytes] = removeSNSegments(obj.RxBuffer(snBufIdx));
                        obj.StatDroppedDataPackets = obj.StatDroppedDataPackets + numSegments;
                        obj.StatDroppedDataBytes = obj.StatDroppedDataBytes + numBytes;
                        obj.ReassemblySNMap(snBufIdx) = decodedPDU.SequenceNumber;
                    end
                    isLastSegment = decodedPDU.SegmentationInfo == 2;
                    [numDupBytes, sdu, reassembledSDUTags] = reassembleSegment(obj.RxBuffer(snBufIdx), ...
                        decodedPDU.Data, decodedPDU.PDULength, decodedPDU.SegmentOffset, isLastSegment, higherLayerTags);
                    if numDupBytes
                        % Return if the data is duplicate
                        return;
                    end

                    isReassembled = ~isempty(sdu);
                    % Update the RLC UM receiver state based on the
                    % received segmented SDU sequence number
                    updateRxState(obj, decodedPDU.SequenceNumber, isReassembled);
                    if isReassembled
                        % Merge the tags from the segments of an SDU
                        higherLayerTags = wirelessnetwork.internal.packetTags.merge(reassembledSDUTags);
                    end
                end
            end

            % Forward the received SDU to application layer if any callback
            % is registered
            if ~isempty(obj.RxForwardFcn) && ~isempty(sdu)
                appPacket = obj.HigherLayerPacketFormat;
                appPacket.NodeID = rlcPacketInfo.NodeID;
                appPacket.Packet = sdu;
                appPacket.PacketLength = numel(sdu);
                appPacket.Tags = higherLayerTags;
                obj.RxForwardFcn(appPacket, currentTime);
            end
        end
    end

    methods (Access = private)
        function rlcUMHeader = generateDataHeader(~, segmentationInfo, seqNumFieldLength, segmentSeqNum, segmentOffset)
            %generateDataHeader Generate header for RLC UMD PDU

            % Create an UMD PDU header column vector with the maximum RLC
            % UM header length and initialize it with segmentation
            % information in the first byte
            umdPDUHeader = [segmentationInfo; zeros(3, 1)];
            % Initialize the UMD PDU header length as 1 by considering the
            % minimum RLC UM header length
            umdPDUHeaderLen = 1;

            if segmentationInfo == 0
                % Create an RLC UM header for the complete SDU. The header
                % format is segmentation information (2 bits) | reserved (6
                % bits)
                rlcUMHeader = umdPDUHeader(umdPDUHeaderLen);
            else
                % Update sequence number field in the RLC UM header based
                % on the configured size for sequence number
                if seqNumFieldLength == 6
                    % Header format is segmentation information (2 bits) |
                    % sequence number (6 bits)
                    umdPDUHeader(1:umdPDUHeaderLen) = ...
                        bitor(bitshift(segmentationInfo, 6), segmentSeqNum);
                else
                    % Set the header length to 2 bytes when the number of
                    % bits for a sequence number is 12. The header format
                    % is segmentation information (2 bits) | reserved (2
                    % bits) | sequence number (12 bits)
                    umdPDUHeaderLen = 2;
                    % Update the sequence number value in the RLC UM header
                    % by spanning it over the last 4 bits of the first byte
                    % and 2nd byte of the header
                    umdPDUHeader(1:umdPDUHeaderLen) = ...
                        [bitor(bitshift(segmentationInfo, 6), bitand(bitshift(segmentSeqNum, -8), 15));...
                        bitand(segmentSeqNum, 255)];
                end

                % Append the segment offset to the RLC UM header for middle
                % and last segments
                if (segmentationInfo == 2) || (segmentationInfo == 3)
                    umdPDUHeaderLen = umdPDUHeaderLen + 2;
                    umdPDUHeader(umdPDUHeaderLen-1: umdPDUHeaderLen) = ...
                        [bitshift(segmentOffset, -8); ...
                        bitand(segmentOffset, 255)];
                end
                rlcUMHeader = umdPDUHeader(1:umdPDUHeaderLen);
            end
        end

        function [decodedPDU, pduDecodeStatus] = decodeDataPDU(obj, rlcPDU)
            %decodeDataPDU Decode the RLC UMD PDU

            seqNumFieldLength = obj.RxSeqNumFieldLength;
            decodedPDU = obj.DataPDUInfo;
            decodedPDU.PDULength = numel(rlcPDU);
            % Extract the segmentation information present in the first 2
            % bits of the RLC UMD PDU
            decodedPDU.SegmentationInfo = bitshift(rlcPDU(1), -6);
            % Extract RLC SDU or RLC SDU segment based on the segmentation
            % information
            if decodedPDU.SegmentationInfo == 0
                if rlcPDU(1) == 0 % Check if the packet is not erroneous
                    pduDecodeStatus = true;
                    % Extract the whole RLC SDU from the RLC UMD PDU
                    decodedPDU.Data = rlcPDU(2:end);
                else
                    pduDecodeStatus = false;
                    return;
                end
            else
                if seqNumFieldLength == 6
                    pduDecodeStatus = true;
                    % Get the sequence number from the last 6 bits of the
                    % first byte from the received PDU
                    decodedPDU.SequenceNumber = bitand(rlcPDU(1), 63);
                    % Set segment offset index to 2 such that it points to
                    % segment offset field, except in case of first segment
                    % of an SDU
                    segmentOffsetIndex = 2;
                else
                    if bitand(rlcPDU(1), 48) == 0 % Check if the packet is not erroneous
                        pduDecodeStatus = true;
                        % Get the sequence number using the last 4 bits of
                        % the first byte and the complete second byte from
                        % the received PDU
                        decodedPDU.SequenceNumber = bitshift(bitand(rlcPDU(1), 15), 8) + rlcPDU(2);
                        % Set segment offset index to 3 such that it points
                        % to segment offset field, except in case of first
                        % segment of an SDU
                        segmentOffsetIndex = 3;
                    else
                        pduDecodeStatus = false;
                        return;
                    end
                end

                % Check whether the first segment of the SDU is received
                if decodedPDU.SegmentationInfo == 1
                    % Extract the RLC SDU segment from the first segmented
                    % RLC UMD PDU
                    decodedPDU.SegmentOffset = 0;
                    decodedPDU.Data = rlcPDU(segmentOffsetIndex:end);
                else
                    % Extract the RLC SDU segment offset information from
                    % the middle or last RLC UMD PDUs
                    decodedPDU.SegmentOffset = bitshift(rlcPDU(segmentOffsetIndex), 8) +  rlcPDU(segmentOffsetIndex+1);
                    % Extract the RLC SDU segment from the RLC UMD PDU
                    decodedPDU.Data = rlcPDU(segmentOffsetIndex+2:end);
                end
            end
        end

        function updateRxState(obj, currPktSeqNum, isReassembled)
            %updateRxState Process the received UMD PDU that contain RLC SDU
            % segment

            % Check whether all the SDU segments are received for the given
            % sequence number
            if isReassembled
                % When received PDU sequence number and the earliest SN
                % considered for reassembly are equal, update the SN
                % considered for reassembly to the least SN of received PDU
                % sequence numbers that has not been reassembled and
                % delivered to upper layer
                if currPktSeqNum == obj.RxNextReassembly
                    % Select the next lower edge of the reassembly window
                    minSN = mod(currPktSeqNum + 1, obj.TotalRxSeqNum);
                    % If the selected next lower edge of the reassembly
                    % window is already received as a part of some
                    % contiguous reception, choose the SN after the upper
                    % edge of the contiguous reception as a new next lower
                    % edge of the reassembly window
                    receptionIndex = (obj.RcvdSNList(:, 1) <= minSN) & ...
                        (obj.RcvdSNList(:, 2) >= minSN);
                    if any(receptionIndex)
                        minSN = obj.RcvdSNList(receptionIndex, 2) + 1;
                    end
                    obj.RxNextReassembly = minSN;
                    obj.RxNextReassemblyModulus = getModulusValue(obj, minSN);
                end
                % Update the completely received SNs context between the
                % lower edge and upper edge of the reassembly window
                updateRxGaps(obj, currPktSeqNum);
                % Remove reassembly SN map context
                bufIdx = getSDUReassemblyBufIdx(obj, currPktSeqNum);
                obj.ReassemblySNMap(bufIdx) = -1;
            elseif ~isInsideReassemblyWindow(obj, currPktSeqNum)
                handleOutOfWindowSN(obj, currPktSeqNum);
            end

            % Update the reassembly timer state as per Section 5.2.2.2.3 of
            % 3GPP TS 38.322
            updateRxStateOnRTState(obj);
        end

        function isInside = isInsideReassemblyWindow(obj, seqNum)
            %isInsideReassemblyWindow Check if the given sequence number
            %falls within the reassembly window

            isInside = false;
            % If the given sequence number falls inside the reassembly
            % window as per Section 5.2.2.2.1 of 3GPP TS 38.322, then set
            % the flag to true
            if getModulusValue(obj, seqNum) < obj.RxNextHighestModulus
                isInside = true;
            end
        end

        function modValue = getModulusValue(obj, value)
            %getModulusValue Calculate the modulus of the given value

            % Calculate modulus for the given value as per Section 7.1 of
            % 3GPP TS 38.322
            modValue = mod(value - (obj.RxNextHighest - obj.UMWindowSize), obj.TotalRxSeqNum);
        end

        function discardPDUsOutsideReassemblyWindow(obj)
            %discardPDUsOutsideReassemblyWindow Discard the received RLC
            %PDUs which fall outside of the reassembly window

            % Iterate through each SN and remove the segmented PDUs
            % associated with that SN if the SN falls outside of the
            % reassembly window
            for i = 1:obj.MaxReassemblySDU
                % Check whether the current sequence number is active
                if obj.ReassemblySNMap(i) ~= -1
                    if ~isInsideReassemblyWindow(obj, obj.ReassemblySNMap(i))
                        % Discard the SDU segments which are received for
                        % the current SN
                        [numSegments, numBytes] = removeSNSegments(obj.RxBuffer(i));
                        obj.StatDroppedDataPackets = obj.StatDroppedDataPackets + numSegments;
                        obj.StatDroppedDataBytes = obj.StatDroppedDataBytes + numBytes;
                    end
                end
            end
        end

        function reassemblyTimerExpiry(obj)
            %reassemblyTimerExpiry Perform the actions required after the
            %expiry of reassembly timer

            % Update RxNextReassembly to the SN of the first SN >=
            % RxTimerTrigger that has not been reassembled
            minSN = obj.RxTimerTrigger; % Select the next lower edge of the reassembly window
            % If the selected next lower edge of the reassembly
            % window is already received as a part of some
            % contiguous reception, choose the SN after the upper
            % edge of the contiguous reception as a new next lower
            % edge of the reassembly window
            receptionIndex = (obj.RcvdSNList(:, 1) <= minSN) & ...
                        (obj.RcvdSNList(:, 2) >= minSN);
            if any(receptionIndex)
                minSN = obj.RcvdSNList(receptionIndex, 2) + 1;
            end
            % Set RxNextReassembly to a new sequence number
            obj.RxNextReassembly = minSN;
            obj.RxNextReassemblyModulus = getModulusValue(obj, minSN);

            % Discard all the segments with SN < updated RxNextReassembly
            for seqNumIdx = 1:obj.MaxReassemblySDU
                seqNum = obj.ReassemblySNMap(seqNumIdx);
                if getModulusValue(obj, seqNum) < obj.RxNextReassemblyModulus
                    % Discard the segmented PDUs which are received for the
                    % current SN and update the statistics
                    [numSegments, numBytes] = removeSNSegments(obj.RxBuffer(seqNumIdx));
                    obj.StatDroppedDataPackets = obj.StatDroppedDataPackets + numSegments;
                    obj.StatDroppedDataBytes = obj.StatDroppedDataBytes + numBytes;
                end
            end

            % If RX_Next_Highest > RX_Next_Reassembly + 1; or
            % If RX_Next_Highest = RX_Next_Reassembly + 1 and there is
            % at least one missing byte segment of the RLC SDU associated
            % with SN = RX_Next_Reassembly before the last byte of all
            % received segments of this RLC SDU:
            %   - start t-Reassembly;
            %   - set RX_Timer_Trigger to RX_Next_Highest.
            rxNextReassembly = mod(obj.RxNextReassembly + 1, obj.TotalRxSeqNum);
            rxNextReassemblyModulus = getModulusValue(obj, rxNextReassembly);
            rxNextHighest = obj.RxNextHighestModulus;
            rnrBufIdx = getSDUReassemblyBufIdx(obj, obj.RxNextReassembly);
            if rxNextHighest > rxNextReassemblyModulus || ...
                    rxNextHighest == rxNextReassemblyModulus && ...
                    rnrBufIdx ~= -1 && anyLostSegment(obj.RxBuffer(rnrBufIdx))
                % Start the reassembly timer
                obj.ReassemblyTimeLeft = obj.ReassemblyTimerNS;
                obj.RxTimerTrigger = obj.RxNextHighest;
            end
        end

        function snBufIdx = getSDUReassemblyBufIdx(obj, sn)
            %getSDUReassemblyIdx Return the reassembly buffer index in
            %which SDU is stored

            snBufIdx = -1;
            for bufIdx = 1:obj.MaxReassemblySDU
                if obj.ReassemblySNMap(bufIdx) == sn
                    snBufIdx = bufIdx;
                    break;
                end
            end
        end

        function snBufIdx = assignReassemblyBufIdx(obj, sn)
            %assignReassemblyBufIdx Find a place to store the specified
            %SN's SDU in the reassembly buffer

            % Find out an empty RLC reassembly buffer for segmented SDU
            snBufIdx = getSDUReassemblyBufIdx(obj, sn);
            if snBufIdx ~= -1
                % Return if SDU has been allotted a buffer for reassembly
                return;
            end
            % Find an empty buffer to store the SDU
            for bufIdx = 1:obj.MaxReassemblySDU
                if obj.ReassemblySNMap(bufIdx) == -1
                    snBufIdx = bufIdx;
                    obj.ReassemblySNMap(snBufIdx) = sn;
                    break;
                end
            end
        end

        function updateRxStateOnRTState(obj)
            %updateRxStateOnRTState Update Rx state variables based on the
            %reassembly timer

            % Stop the reassembly timer if any of the following conditions
            % is met:
            %   - if RX_Timer_Trigger <= RX_Next_Reassembly; or
            %
            %   - if RX_Timer_Trigger falls outside of the reassembly
            %   window and RX_Timer_Trigger is not equal to
            %   RX_Next_Highest; or
            %
            %   - if RX_Next_Highest = RX_Next_Reassembly + 1 and there is
            %   no missing byte segment of the RLC SDU associated with SN =
            %   RX_Next_Reassembly before the last byte of all received
            %   segments of this RLC SDU
            if obj.ReassemblyTimeLeft > 0
                rxNextReassemlby = mod(obj.RxNextReassembly + 1, obj.TotalRxSeqNum);
                rnrBufIdx = getSDUReassemblyBufIdx(obj, obj.RxNextReassembly);
                if getModulusValue(obj, obj.RxTimerTrigger) <= obj.RxNextReassemblyModulus
                    obj.ReassemblyTimeLeft = 0;
                elseif ~isInsideReassemblyWindow(obj, obj.RxTimerTrigger) && (obj.RxTimerTrigger ~= obj.RxNextHighest)
                    obj.ReassemblyTimeLeft = 0;
                elseif obj.RxNextHighest == rxNextReassemlby &&  rnrBufIdx ~= -1 && ...
                        ~anyLostSegment(obj.RxBuffer(rnrBufIdx))
                    obj.ReassemblyTimeLeft = 0;
                end
            end

            if obj.ReassemblyTimeLeft == 0
                rxNextReassembly = mod(obj.RxNextReassembly + 1, obj.TotalRxSeqNum);
                rxNextReassemblyModulus = getModulusValue(obj, rxNextReassembly);
                rnrBufIdx = getSDUReassemblyBufIdx(obj, obj.RxNextReassembly);
                startRT = false;
                % Start the reassembly timer if any of the following
                % conditions is met:
                %   - At least one missing SN between lower and upper ends
                %   of the receiving window
                %   - At least one missing segment between lower and upper
                %   ends of the receiving window when upper end = lower
                %   end + 1
                if obj.RxNextHighestModulus > rxNextReassemblyModulus
                    startRT = true;
                elseif obj.RxNextHighestModulus == rxNextReassemblyModulus && ...
                        rnrBufIdx ~= -1 && anyLostSegment(obj.RxBuffer(rnrBufIdx))
                    startRT = true;
                end

                if startRT
                    obj.ReassemblyTimeLeft = obj.ReassemblyTimerNS;
                    obj.RxTimerTrigger = obj.RxNextHighest;
                end
            end
        end

        function handleOutOfWindowSN(obj, currPktSeqNum)
            %handleOutOfWindowSN Update the receiver state on reception of
            %SDU that falls outside of the reassembly window

            % Update the upper end of reassembly window to the received
            % PDU sequence number + 1. This update pulls the reassembly
            % window as the window size is fixed
            obj.RxNextHighest = mod(currPktSeqNum + 1, obj.TotalRxSeqNum);
            obj.RxNextHighestModulus = getModulusValue(obj, obj.RxNextHighest);
            obj.RxNextReassemblyModulus = getModulusValue(obj, obj.RxNextReassembly);
            % Discard the PDUs that fall outside of the reassembly window
            % due to window movement
            discardPDUsOutsideReassemblyWindow(obj);
            % Check whether the earliest SN considered for reassembly falls
            % outside of the reassembly window
            if ~isInsideReassemblyWindow(obj, obj.RxNextReassembly)
                % Select the next lower edge of the window
                minSN = mod(obj.RxNextHighest - obj.UMWindowSize, obj.TotalRxSeqNum);
                % If the selected next lower edge of the reassembly
                % window is already received as a part of some
                % contiguous reception, choose the SN after the upper
                % edge of the contiguous reception as a new next lower
                % edge of the reassembly window
                receptionIndex = (obj.RcvdSNList(:, 1) <= minSN) & (obj.RcvdSNList(:, 2) >= minSN);
                if any(receptionIndex)
                    minSN = obj.RcvdSNList(receptionIndex, 2) + 1;
                end
                obj.RxNextReassembly = minSN;
                obj.RxNextReassemblyModulus = getModulusValue(obj, minSN);
            end

            % Update the completely received SN's list in the reassembly
            % window when the upper edge of the window is changed
            validIndex = getModulusValue(obj, obj.RcvdSNList(:, 1)) > obj.UMWindowSize;
            if any(validIndex)
                obj.RcvdSNList(validIndex, :) = -1;
            end
        end

        function updateRxGaps(obj, sn)
            %updateRxGaps Update the completely received SDUs context

            % Identify whether this complete SDU reception is an extension
            % of an existing contiguous SDU reception. This can be checked
            % by finding its previous and following SDUs reception status
            prevSNRxStatus = (obj.RcvdSNList(:, 2) == mod(sn - 1, obj.TotalRxSeqNum));
            nextSNRxStatus = (obj.RcvdSNList(:, 1) == mod(sn + 1, obj.TotalRxSeqNum));
            isPrevSNContigious = any(prevSNRxStatus);
            isNextSNContigious = any(nextSNRxStatus);
            if ~isPrevSNContigious && ~isNextSNContigious
                % Create a new contiguous reception since it is not
                % extending any other existing contiguous reception
                indices = find(obj.RcvdSNList == [-1, -1], 1);
                obj.RcvdSNList(indices, 1) = sn;
                obj.RcvdSNList(indices, 2) = sn;
            elseif isPrevSNContigious && ~isNextSNContigious
                obj.RcvdSNList(prevSNRxStatus, 2) = sn;
            elseif ~isPrevSNContigious && isNextSNContigious
                obj.RcvdSNList(nextSNRxStatus, 1) = sn;
            else
                % Merge the two contiguous receptions since the new SDU
                % makes them one contiguous reception
                obj.RcvdSNList(prevSNRxStatus, 2) = obj.RcvdSNList(nextSNRxStatus, 2);
                obj.RcvdSNList(nextSNRxStatus, 1:2) = -1;
            end
        end
    end
end
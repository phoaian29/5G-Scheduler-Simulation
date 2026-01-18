classdef nrRLCAM < nr5g.internal.nrRLC
    %nrRLCAM Implement RLC AM functionality
    %
    %   Note: This is an internal undocumented class and its API and/or
    %   functionality may change in subsequent releases.
    %
    %   RLCOBJ = nrRLCAM(RNTI, RLCBEARERCONFIG, MAXREASSEMBLYSDU,
    %   TXBUFFERSTATUSFCN, RXFORWARDFCN) implements RLC AM functionality.
    %
    %   RNTI is radio network temporary identifier.
    %
    %   RLCBEARERCONFIG is rlc bearer configuration object. For more
    %   information, please refer 'nrRLCBearerConfig' documentation.
    %
    %   MAXREASSEMBLYSDU is the maximum capacity of reassembly buffer in terms
    %   of number of SDUs.
    %
    %   TXBUFFERSTATUSFCN specifies the function callback that will be used to
    %   send the buffer status report to MAC.
    %
    %   RXFORWARDFCN specifies the function callback that will be used to
    %   forward the packets to higher layer.
    %
    %
    %   nrRLCAM properties (configurable through constructor):
    %
    %   SNFieldLength       - Number of bits in sequence number field of RLC
    %                         entity
    %   BufferSize          - RLC Transmitter buffer size in terms of number of
    %                         packets
    %   PollPDU             - Allowable number of acknowledged mode data (AMD)
    %                         protocol data unit (PDU) transmissions before
    %                         requesting the status PDU
    %   PollByte            - Allowable number of service data unit (SDU) byte
    %                         transmissions before requesting the status PDU
    %   PollRetransmitTimer - Waiting time (in milliseconds) before
    %                         retransmitting the status PDU request
    %   MaxRetxThreshold    - Maximum number of retransmissions of an AMD PDU
    %   ReassemblyTimer     - Waiting time (in milliseconds) before declaring
    %                         the reassembly failure of SDUs in the reception
    %                         buffer
    %   StatusProhibitTimer - Waiting time (in milliseconds) before
    %                         transmitting the status PDU following the
    %                         previous status PDU transmission
    %   MaxReassemblySDU    - Maximum capacity of the reassembly buffer in
    %                         terms of number of SDUs. This is also equal to
    %                         the maximum segments that can be missing per SDU
    %                         at any point of time. This value is exactly
    %                         equals to the number of HARQ processes at MAC
    %                         layer
    %
    %   nrRLCAM methods:
    %
    %   run - Run the RLC entity and return the next invoke time of RLC
    %         entity

    %   Copyright 2023-2024 The MathWorks, Inc.

    properties
        %SNFieldLength Number of bits in sequence number field of transmitter and
        %receiver entities
        %   Specify the sequence number field length as an integer. The sequence
        %   number field length is one of 12 | 18. For more details, refer 3GPP TS
        %   38.322 Section 6.2.3.3.
        SNFieldLength

        %PollPDU Parameter used by the transmitting side of an AM RLC entity to
        %trigger a poll based on number of PDUs
        %   Specify the number of poll PDUs as one of 4, 8, 16, 32, 64, 128, 256,
        %   512, 1024, 2048, 4096, 6144, 8192, 12288, 16384, 20480, 24576, 28672,
        %   32768, 40960, 49152, 57344, 65536, or inf. The value inf indicates poll
        %   PDU based polling is disabled. For more details, refer 3GPP TS 38.331
        %   information element RLC-Config.
        PollPDU

        %PollByte Parameter used by the transmitting side of an AM RLC entity to
        %trigger a poll based on number of SDU bytes
        %   Specify the number of poll bytes as one of 1, 2, 5, 8, 10, 15, 25, 50,
        %   75, 100, 125, 250, 375, 500, 750, 1000, 1250, 1500, 2000, 3000, 4000,
        %   4500, 5000, 5500, 6000, 6500, 7000, 7500, 8192, 9216, 10240, 11264,
        %   12288, 13312, 14336, 15360, 16384, 17408, 18432, 20480, 25600, 30720,
        %   40960, or inf kilo bytes. The value inf indicates poll byte based
        %   polling is disabled. For more details, refer 3GPP TS 38.331 information
        %   element RLC-Config.
        PollByte

        %PollRetransmitTimer Timer used by the transmitting side of an AM RLC
        %entity in order to retransmit a poll
        %   Specify the poll retransmit timer value as one of 5, 10, 15, 20, 25,
        %   30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100, 105, 110,
        %   115, 120, 125, 130, 135, 140, 145, 150, 155, 160, 165, 170, 175, 180,
        %   190, 195, 200, 205, 210, 215, 220, 225, 230, 235, 240, 245, 250, 300,
        %   350, 400, 450, 500, 800, 1000, 2000, or 4000 ms. For more details,
        %   refer 3GPP TS 38.331 information element RLC-Config.
        PollRetransmitTimer

        %MaxRetxThreshold Maximum number of retransmissions corresponding to an RLC
        %SDU, including its segments
        %   Specify the maximum retransmission threshold as one of 1, 2, 3, 4, 6,
        %   8, 16, or 12. For more details, refer 3GPP TS 38.331 information
        %   element RLC-Config.
        MaxRetxThreshold

        %StatusProhibitTimer Timer used by the receiving side of an RLC entity
        %inorder to prohibit frequent transmission of status PDU
        %   Specify the status prohibit timer values as one of 0, 5, 10, 15, 20,
        %   25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100, 105,
        %   110, 115, 120, 125, 130, 135, 140, 145, 150, 155, 160, 165, 170, 175,
        %   180, 185, 190, 195, 200, 205, 210, 215, 220, 225, 230, 235, 240, 245,
        %   250, 300, 350, 400, 450, 500, 800, 1000, 1200, 1600, 2000, or 2400 ms.
        %   For more details, refer 3GPP TS 38.331 information element RLC-Config.
        StatusProhibitTimer
    end

    % Tx configuration
    properties (Access = private)
        %TxNext Sequence number to be assigned for the next newly generated AMD PDU
        TxNext = 0

        %TxNextAck Earliest SN that is yet to receive a positive acknowledgment
        TxNextAck = 0

        %TxSegmentOffset Position of the segmented SDU in bytes within the original
        %SDU
        TxSegmentOffset = 0

        %TxSubmitted Sequence number of the last SDU that has been submitted to
        %lower layer
        TxSubmitted = -1

        %PollSN The highest SN among the AMD PDUs submitted to MAC
        PollSN = 0

        %TxBuffer Transmit queue object that stores the SDUs received from higher
        %layers
        TxBuffer

        %PDUsWithoutPoll Number of PDUs sent after the transmission of latest
        %status report
        PDUsWithoutPoll = 0

        %BytesWithoutPollbit Number of SDU bytes sent after the transmission of
        %latest status report
        BytesWithoutPoll = 0

        %PollRetransmitTimeLeft Time left for the retransmission of status report
        %request
        PollRetransmitTimeLeft = 0

        %RetxBufferFront Index that points to the earliest SDU in the
        %retransmission buffer
        RetxBufferFront = 0

        %NumRetxBufferSDUs Number of SDUs in the retransmission buffer
        NumRetxBufferSDUs = 0

        %RetxBuffer Buffer to store the SDUs which has received the negative
        %acknowledgment. This is a N-by-1 cell array where 'N' is the maximum
        %number of SDUs which can be buffered
        RetxBuffer

        %RetxBufferContext Buffer to store the context of the SDUs in the
        %retransmission buffer. This is a N-by-(2 + 2*P) matrix where 'N' is the
        %maximum Tx buffer SDUs and 'P' is the maximum number of segment gaps
        %possible at any point of time. Values at indexes (i, 1), (i, 2), (i,
        %3:end) indicate sequence number, retransmission count, and lost segments
        %information, respectively
        RetxBufferContext

        %RequiredGrantLength Length of the required grant to transmit the data in
        %the Tx buffer
        RequiredGrantLength = 0

        %PDUHeaderLengthForSDU RLC PDU header length for complete SDU
        PDUHeaderLengthForSDU

        %PDUHeaderLengthForSegmentedSDU RLC PDU header length for segmented SDU
        PDUHeaderLengthForSegmentedSDU
    end

    % Rx configuration
    properties (Access = private)
        %RxNext SN of the last in-sequence completely received RLC SDU. It serves
        %as the lower end of the receiving window
        RxNext = 0

        % RxNextHighest SN following the SN of the RLC SDU with the highest SN
        % among received RLC SDUs
        RxNextHighest = 0

        %RxNextStatusTrigger SN following the SN of the RLC SDU which triggered
        %reassembly timer
        RxNextStatusTrigger = 0

        %RxHighestStatus The highest possible SN which can be indicated by ACK SN
        %when a status PDU needs to be constructed
        RxHighestStatus = 0

        %RxBuffer Buffer to store the segmented SDUs for reassembly. This is a
        %N-by-1 cell array where 'N' is the maximum reassembly buffer length
        RxBuffer

        %ReassemblySNMap Map that shows where the segmented SDUs are stored in the
        %reassembly buffer. This is a N-by-1 column vector where 'N' is the maximum
        %reassembly buffer length. Each element contains the SN of the SDUs which
        %are under reassembly procedure. Each element in the vector can take value
        %in the range between -1 and 2^SeqNumFieldLength-1. If an element is set to
        %-1, it indicates that is not occupied by any SDUs SN
        ReassemblySNMap

        %RcvdSNList List of contiguously received full SDU SNs that help in the
        %status PDU construction. This is a N-by-2 matrix where 'N' is the maximum
        %reassembly buffer length. Each row has a starting SN and ending SN that
        %indicates a contiguous reception of SNs in the receiving window. Value
        %[-1, -1] in a row indicates unoccupancy
        RcvdSNList

        %ReassemblyTimeLeft Time left for the reassembly procedure
        ReassemblyTimeLeft = 0

        %StatusProhibitTimeLeft Time left to avoid the transmission of status PDU
        StatusProhibitTimeLeft = 0

        %IsStatusPDUTriggered Flag that indicates whether a status report is
        %triggered. The values true and false indicate triggered and not triggered,
        %respectively
        IsStatusPDUTriggered = false
    end

    % Properties that won't get modified after their initialization in the
    % constructor
    properties (Access = private)
        %TxSeqNumFieldLength Sequence number field length of the Tx side
        TxSeqNumFieldLength

        %TotalTxSeqNum The number of SNs configured on the RLC AM transmitter
        %entity
        TotalTxSeqNum

        %AMTxWindowSize SN window size used by the transmitting side of an RLC AM
        %entity for the retransmission procedure. The window size is 2048 and
        %131072 for 12 bit and 18 bit SN, respectively
        AMTxWindowSize

        %RxSeqNumFieldLength Sequence number field length for the Rx side
        RxSeqNumFieldLength

        %TotalRxSeqNum The number of SNs configured on the RLC AM receiver entity
        TotalRxSeqNum

        %AMRxWindowSize SN window size used by the receiving side of an RLC AM
        %entity for the reassembly procedure. The window size is 2048 and 131072
        %for 12 bit and 18 bit SN, respectively
        AMRxWindowSize = 0

        %PollRetransmitTimerNS Poll retransmit timer in nanoseconds
        PollRetransmitTimerNS

        %ReassemblyTimerNS Reassembly timer in nanoseconds
        ReassemblyTimerNS

        %StatusProhibitTimerNS Status prohibit timer in nanoseconds
        StatusProhibitTimerNS
    end

    properties(Access = private)
        %WaitingForACKBuffer Buffer to store the transmitted/retransmitted SDUs
        %which are waiting for the acknowledgment. This is a N-by-1 cell array
        %where 'N' is the maximum number of SDUs which can be buffered
        WaitingForACKBuffer

        %WaitingForACKBufferContext Buffer to store the context of SDUs which are
        %waiting for the acknowledgment. This is a N-by-2 matrix where 'N' is the
        %maximum number of SDUs which can be buffered. Each row has the following
        %information: sequence number and retransmission count
        WaitingForACKBufferContext

        %NumSDUsWaitingForACK Number of SDUs that are waiting for the
        %acknowledgment
        NumSDUsWaitingForACK = 0

        %RetransmitPollFlag Flag that indicates the retransmission of poll is
        %triggered
        RetransmitPollFlag = false

        %% Rx properties

        %IsStatusPDUDelayed Flag that indicates whether the status report is
        %delayed. The values true and false indicate delayed and not delayed,
        %respectively
        IsStatusPDUDelayed = false

        %IsStatusPDUTriggeredOverSPT Flag that indicates whether the status report
        %is requested when the status prohibit timer is running. A value of 'true'
        %indicates that the request is made when the status prohibit timer is
        %running
        IsStatusPDUTriggeredOverSPT = false

        %GrantRequiredForStatusReport Grant size required for the status report
        %triggered
        GrantRequiredForStatusReport = 0
    end

    methods
        %Constructor
        function obj = nrRLCAM(rnti, rlcBearerConfig, maxReassemblySDU, txBufferStatusFcn, rxForwardFcn)

            obj@nr5g.internal.nrRLC(rnti, rlcBearerConfig.LogicalChannelID);
            obj.SNFieldLength = rlcBearerConfig.SNFieldLength;

            % Initialize Tx side configuration
            obj.PollPDU = rlcBearerConfig.PollPDU;
            obj.PollByte = rlcBearerConfig.PollByte;
            obj.PollRetransmitTimer = rlcBearerConfig.PollRetransmitTimer;
            obj.MaxRetxThreshold = rlcBearerConfig.MaxRetxThreshold;
            obj.BufferSize = rlcBearerConfig.BufferSize;
            obj.TxBuffer = wirelessnetwork.internal.queue(obj.BufferSize);
            obj.TxSeqNumFieldLength = obj.SNFieldLength;
            obj.TotalTxSeqNum = 2^obj.TxSeqNumFieldLength;
            obj.AMTxWindowSize = 2^(obj.TxSeqNumFieldLength - 1);
            obj.TxBufferStatusFcn = txBufferStatusFcn;
            obj.MaxReassemblySDU = maxReassemblySDU;
            obj.RetxBufferContext = -1 * ones(obj.BufferSize, 2 + obj.MaxReassemblySDU*2);
            obj.WaitingForACKBufferContext = -1 * ones(obj.BufferSize, 2);
            % Initialize RLC buffer status structure
            obj.RLCBufferStatus.RNTI = obj.RNTI;
            obj.RLCBufferStatus.LogicalChannelID = obj.LogicalChannelID;
            if obj.SNFieldLength == 12
                obj.PDUHeaderLengthForSDU = 2;
                obj.PDUHeaderLengthForSegmentedSDU = 4;
            else
                obj.PDUHeaderLengthForSDU = 3;
                obj.PDUHeaderLengthForSegmentedSDU = 5;
            end

            % Initialize Rx side configuration
            obj.ReassemblyTimer = rlcBearerConfig.ReassemblyTimer;
            obj.StatusProhibitTimer = rlcBearerConfig.StatusProhibitTimer;
            obj.RxSeqNumFieldLength = obj.SNFieldLength;
            obj.AMRxWindowSize = 2^(obj.RxSeqNumFieldLength - 1);
            obj.TotalRxSeqNum = 2^obj.RxSeqNumFieldLength;
            % Define reassembly buffer and SN map array
            obj.RxBuffer = repmat(nr5g.internal.nrRLCDataReassembly(maxReassemblySDU,obj.MaxPacketSize), maxReassemblySDU, 1);
            for pktIdx = 2:maxReassemblySDU
                obj.RxBuffer(pktIdx) = nr5g.internal.nrRLCDataReassembly(maxReassemblySDU,obj.MaxPacketSize);
            end
            obj.ReassemblySNMap = -1 * ones(maxReassemblySDU, 1);
            obj.RcvdSNList = -1 * ones(maxReassemblySDU, 2);
            obj.RxForwardFcn = rxForwardFcn;

            % Timers in nanoseconds format
            obj.ReassemblyTimerNS = obj.ReassemblyTimer*1e6;
            obj.PollRetransmitTimerNS = obj.PollRetransmitTimer*1e6;
            obj.StatusProhibitTimerNS = obj.StatusProhibitTimer*1e6;
        end

        function nextInvokeTime = run(obj, currentTime)
            %run Run the RLC entity and return the next invoke time of RLC entity
            %
            %   NEXTINVOKETIME = run(OBJ, CURRENTTIME) runs the RLC entity and returns
            %   the next invoke time of RLC entity.
            %
            %   NEXTINVOKETIME indicates the time (in nanoseconds) at which the run
            %   method should be invoked again.
            %
            %   OBJ is an object of type nrRLCAM.
            %
            %   CURRENTTIME is an integer indicating the current time (in nanoseconds).

            if (obj.PollRetransmitTimeLeft > 0) || (obj.ReassemblyTimeLeft > 0) || (obj.StatusProhibitTimeLeft > 0)
                elapsedTime = currentTime - obj.LastRunTime; % In nanoseconds

                nextPollTimeInvokeTime = Inf;
                if obj.PollRetransmitTimeLeft > 0
                    % Update the reassembly timer
                    obj.PollRetransmitTimeLeft = obj.PollRetransmitTimeLeft - elapsedTime;
                    if obj.PollRetransmitTimeLeft <= 0
                        obj.PollRetransmitTimeLeft = 0;
                        % Handle the status prohibit timer expiry
                        pollRetransmitTimerExpiry(obj);
                    else
                        nextPollTimeInvokeTime = currentTime + obj.PollRetransmitTimeLeft;
                    end
                end

                nextReassemblyTimeInvokeTime = Inf;
                if obj.ReassemblyTimeLeft > 0
                    % Update the reassembly timer
                    obj.ReassemblyTimeLeft = obj.ReassemblyTimeLeft - elapsedTime;
                    if obj.ReassemblyTimeLeft <= 0
                        obj.ReassemblyTimeLeft = 0;
                        obj.StatReassemblyTimerExpiry = obj.StatReassemblyTimerExpiry + 1;
                        % Handle the reassembly timer expiry
                        reassemblyTimerExpiry(obj);
                    else
                        nextReassemblyTimeInvokeTime = currentTime + obj.ReassemblyTimeLeft;
                    end
                end

                nextStatusProhibitInvokeTime = Inf;
                if obj.StatusProhibitTimeLeft > 0
                    % Update the status prohibit timer
                    obj.StatusProhibitTimeLeft = obj.StatusProhibitTimeLeft - elapsedTime;
                    if obj.StatusProhibitTimeLeft <= 0
                        obj.StatusProhibitTimeLeft = 0;
                        % Handle the status prohibit timer expiry
                        statusProhibitTimerExpiry(obj);
                    else
                        nextStatusProhibitInvokeTime =  currentTime + obj.StatusProhibitTimeLeft;
                    end
                end
                nextInvokeTime = min([nextPollTimeInvokeTime nextReassemblyTimeInvokeTime nextStatusProhibitInvokeTime]);
            else
                nextInvokeTime = Inf;
            end
            obj.LastRunTime = currentTime;
        end
    end

    methods (Hidden)
        function isPacketQueued = enqueueSDU(obj, rlcSDU)
            %enqueueSDU Queue the received SDU from higher layers in the Tx buffer
            %
            %   enqueueSDU(OBJ, RLCSDU) queues the received SDU in the Tx buffer. Also,
            %   it generates and stores the corresponding RLC AM header in a Tx header
            %   storage buffer.
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
            % Store the SDU in the Tx buffer
            isPacketQueued = enqueue(obj.TxBuffer, rlcSDU);
            if ~isPacketQueued
                % Update the statistics if the packet is not queued
                obj.StatTransmitterQueueOverflow = obj.StatTransmitterQueueOverflow + 1;
                return;
            end

            % Increment the required grant size by the sum of expected MAC & RLC header
            % lengths and complete RLC SDU length
            obj.RequiredGrantLength = obj.RequiredGrantLength + ...
                rlcSDU.PacketLength + obj.MaxRLCMACHeadersOH;
            % Send the updated RLC buffer status report to MAC layer
            obj.RLCBufferStatus.BufferStatus = obj.RequiredGrantLength;
            obj.TxBufferStatusFcn(obj.RLCBufferStatus);
        end

        function rlcPacketInfo = sendPDUs(obj, bytesGranted, remainingTBS, currentTime)
            %sendPDUs Send the RLC protocol data units (PDUs) that fit in the grant
            %notified by MAC layer
            %
            %   RLCPDUS = sendPDUs(OBJ, BYTESGRANTED, REMAININGTBS, CURRENTTIME)
            %   sends the RLC PDUs that fit in the grant notified by MAC.
            %
            %   RLCPACKETINFO is a struct array of RLC AM packets to be transmitted
            %   by MAC. Each element represents one AM packet and its associated
            %   information.
            %
            %   BYTESGRANTED is a positive integer scalar, which represents the number
            %   of granted transmission bytes.
            %
            %   REMAININGTBS is a nonnegative integer scalar, which represents the
            %   remaining number of bytes in the transport block size (TBS). This helps
            %   to avoid the segmentation of RLC SDUs in round-1 of MAC logical channel
            %   prioritization (LCP) procedure.
            %
            % CURRENTTIME is the current simulation time in nanoseconds.

            remainingGrant = bytesGranted;
            rlcPacketInfo = obj.RLCPacketInfo;

            % Transmission of control PDUs get high priority
            if obj.IsStatusPDUTriggered && (obj.StatusProhibitTimeLeft == 0)
                numRLCPackets = 1;
                % Construct the status PDU and update the statistics accordingly
                [statusPDU, statusPDULen] = constructStatusPDU(obj, bytesGranted + remainingTBS);
                rlcPacketInfo(numRLCPackets).Packet = statusPDU;
                rlcPacketInfo(numRLCPackets).PacketLength = statusPDULen;
                obj.StatTransmittedControlPackets = obj.StatTransmittedControlPackets + 1;
                obj.StatTransmittedControlBytes = obj.StatTransmittedControlBytes + statusPDULen;
                % Upon construction of status PDU, start status prohibit timer before
                % sending it to the MAC and reset the status PDU delay flag
                obj.IsStatusPDUDelayed = false;
                obj.StatusProhibitTimeLeft = obj.StatusProhibitTimerNS;
                % Update the amount of grant left in the given grant and buffer status of
                % the RLC entity
                macHeaderLength = (statusPDULen > 255) + 2;
                remainingGrant = remainingGrant - statusPDULen - macHeaderLength;
                obj.RequiredGrantLength = obj.RequiredGrantLength - obj.GrantRequiredForStatusReport;
                obj.GrantRequiredForStatusReport = 0;
            end

            [reTxPDUSet, pollInRetx, reTxPollSN, remainingGrant] = retransmitSDUs(obj, remainingGrant, remainingTBS, currentTime);
            [txPDUSet, pollInTx, txPollSN, ~] = transmitSDUs(obj, remainingGrant, remainingTBS, currentTime);
            rlcPacketInfo = [rlcPacketInfo reTxPDUSet txPDUSet];

            % Send the updated RLC buffer status report to MAC layer
            obj.RLCBufferStatus.BufferStatus = obj.RequiredGrantLength;
            obj.TxBufferStatusFcn(obj.RLCBufferStatus);

            % Set POLL_SN to highest SN among the sent AMD PDUs as per Section 5.3.3.2
            % of 3GPP TS 38.322
            if pollInRetx
                obj.PollSN = reTxPollSN;
                % Restart the poll retransmit timer as per Section 5.3.3.3 of 3GPP TS
                % 38.322
                obj.PollRetransmitTimeLeft = obj.PollRetransmitTimerNS;
            end
            if pollInTx
                obj.PollSN = txPollSN;
                % Restart the poll retransmit timer as per Section 5.3.3.3 of 3GPP TS
                % 38.322
                obj.PollRetransmitTimeLeft = obj.PollRetransmitTimerNS;
            end
        end

        function receivePDUs(obj, rlcPacketInfo, currentTime)
            %receivePDUs Receive and process RLC PDU from the MAC layer
            %
            %   receivePDUs(OBJ, RLCPACKETINFO, CURRENTTIME) Receives and
            %   processes RLC PDU from the MAC layer. It reassembles the
            %   SDU received from RLC PDU and delivers the SDU to higher
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

            % Process data and control PDUs separately by distinguish them with
            % data/control bit in the header
            if bitand(rlcPacketInfo.Packet(1), 128) % For data PDU
                [sdu, higherLayerTags] = processDataPDU(obj, rlcPacketInfo);
                % Forward the received SDU to application layer if any callback is
                % registered
                if ~isempty(obj.RxForwardFcn) && ~isempty(sdu)
                    appPacket = obj.HigherLayerPacketFormat;
                    appPacket.NodeID = rlcPacketInfo.NodeID;
                    appPacket.Packet = sdu;
                    appPacket.PacketLength = numel(sdu);
                    appPacket.Tags = higherLayerTags;
                    obj.RxForwardFcn(appPacket, currentTime);
                end
            else % For control PDU
                processStatusPDU(obj, rlcPacketInfo.Packet);
            end
        end
    end

    methods (Access = private)
        function rlcAMHeader = generateDataHeader(~, segmentationInfo, seqNumFieldLength, segmentSeqNum, segmentOffset)
            %generateDataHeader Generate header for RLC AMD PDU

            % Create an AMD PDU header column vector with the maximum RLC AM header
            % length
            amdPDUHeader = zeros(5, 1);

            % Set D/C flag, segmentation information and sequence number fields in the
            % header. D/C flag is always 1 since it is a data PDU
            if seqNumFieldLength == 12
                amdPDUHeaderLen = 2; % In bytes
                amdPDUHeader(1:amdPDUHeaderLen) = [128 + bitshift(segmentationInfo, 4) + bitshift(segmentSeqNum, -8); ...
                    bitand(segmentSeqNum, 255)];
            else
                amdPDUHeaderLen = 3; % In bytes
                amdPDUHeader(1:amdPDUHeaderLen) = [128 + bitshift(segmentationInfo, 4) + bitshift(segmentSeqNum, -16);
                    bitand(bitshift(segmentSeqNum, -8), 255);
                    bitand(segmentSeqNum, 255)];
            end
            % Append segment offset field depending on the value of segmentation
            % information
            if segmentationInfo >= 2
                amdPDUHeader(amdPDUHeaderLen+1:amdPDUHeaderLen+2) = [bitshift(segmentOffset, -8); bitand(segmentOffset, 255)];
                amdPDUHeaderLen = amdPDUHeaderLen + 2;
            end
            rlcAMHeader = amdPDUHeader(1:amdPDUHeaderLen);
        end

        function [decodedPDU, pduDecodeStatus] = decodeDataPDU(obj, rlcPDU)
            %decodeDataPDU Decode the RLC AMD PDU

            seqNumFieldLength = obj.RxSeqNumFieldLength;
            decodedPDU = obj.DataPDUInfo;
            % Extract the poll bit and segmentation information from the PDU header
            decodedPDU.PollBit = bitand(bitshift(rlcPDU(1), -6), 1);
            decodedPDU.SegmentationInfo = bitand(bitshift(rlcPDU(1), -4), 3);
            decodedPDU.PDULength = numel(rlcPDU);
            pduDecodeStatus = true;
            % Extract SN
            if seqNumFieldLength == 12
                % Index to indicate the starting position of the data field in the received
                % PDU. In case of 12 bit sequence number field length, it is 3
                idx = 3;
                first2bytes = bitor(bitshift(rlcPDU(1), 8), rlcPDU(2));
                decodedPDU.SequenceNumber = bitand(first2bytes, 4095); % 4095 is a 12 bit mask
            else
                % Check if the data PDU is erroneous by testing the 3rd and 4th bits of the
                % first byte of PDU. If either is set (bitwise AND with 12 > 0), consider the
                % packet is erroneous
                if bitand(rlcPDU(1), 12) > 0
                    pduDecodeStatus = false;
                    return;
                end
                % Index to indicate the starting position of the data field in the received
                % PDU. In case of 18 bit sequence number field length, it is 4
                idx  = 4;
                first3Bytes = bitor(bitshift(rlcPDU(1), 16), bitor(bitshift(rlcPDU(2),8), rlcPDU(3)));
                decodedPDU.SequenceNumber = bitand(first3Bytes, 262143); % 262143 is a 18 bit mask
            end
            % Extract segment offset and payload from the PDU
            if decodedPDU.SegmentationInfo == 2 || decodedPDU.SegmentationInfo == 3
                decodedPDU.SegmentOffset = bitor(bitshift(rlcPDU(idx), 8), rlcPDU(idx + 1));
                decodedPDU.Data = rlcPDU(idx + 2:end);
            else
                decodedPDU.SegmentOffset = 0;
                % Extract the data fields
                decodedPDU.Data = rlcPDU(idx:end);
            end
        end

        function [rlcPDU, sduLen, remainingGrant] = retransmitSegment(obj, sn, lostSegmentInfo, remainingGrant, remainingTBS)
            %retransmitSegment Retransmit the specified SDU segment

            sduInfo = obj.RetxBuffer{obj.RetxBufferFront + 1};
            sdu = sduInfo.Packet;
            segmentEnd = lostSegmentInfo(2);
            % Update segment offset end with the actual offset since last segment's end
            % offset is always 65535 irrespective of the SDU size
            if lostSegmentInfo(2) == 65535
                lostSegmentInfo(2) = numel(sdu)-1;
            end
            % Construct the AMD PDU for the SDU segment
            [rlcPDU, sduLen, remainingGrant, isSegmented] = constructAMDPDU(obj, sn, lostSegmentInfo(1), lostSegmentInfo(2), sdu, remainingGrant, remainingTBS);

            if isSegmented
                % Update the start offset of segment because of the resegmentation
                soStartIndex = obj.RetxBufferContext(obj.RetxBufferFront + 1, 3:end) == lostSegmentInfo(1);
                obj.RetxBufferContext(obj.RetxBufferFront + 1, logical([0 0 soStartIndex])) = lostSegmentInfo(1) + sduLen;
                % Update the required grant length, including 8 byte MAC and RLC headers
                % overhead, because of the reduced segment size
                remainingSDULen = lostSegmentInfo(2) - (lostSegmentInfo(1) + sduLen) + 1;
                obj.RequiredGrantLength = obj.RequiredGrantLength + remainingSDULen + obj.MaxRLCMACHeadersOH;
            else
                % Remove the segment information from the retransmission context
                soIndexes = obj.RetxBufferContext(obj.RetxBufferFront + 1, 3:end) == lostSegmentInfo(1) | ...
                    obj.RetxBufferContext(obj.RetxBufferFront + 1, 3:end) == segmentEnd;
                obj.RetxBufferContext(obj.RetxBufferFront + 1, logical([0 0 soIndexes])) = -1;
            end
        end

        function [rlcPDU, sduLen, remainingGrant, isSegmented] = constructAMDPDU(obj, sn, soStart, soEnd, sdu, remainingGrant, remainingTBS)
            %constructAMDPDU Construct an AMD PDU that fits in the given grant

            isSegmented = false;
            if soStart == 0
                pduHeaderLength = obj.PDUHeaderLengthForSDU;
            else
                pduHeaderLength = obj.PDUHeaderLengthForSegmentedSDU;
            end
            fullSDULen = numel(sdu);
            pduLength = pduHeaderLength + soEnd - soStart + 1;
            macHeaderLength = (pduLength > 255) + 2;
            % Remove the considered grant size from the required grant length
            obj.RequiredGrantLength = obj.RequiredGrantLength - ...
                (soEnd - soStart + 1 + obj.MaxRLCMACHeadersOH);

            % Check whether the complete/segmented AMD PDU needs to be
            % segmented/resegmented to fit within the notified grant size. The below
            % conditional also takes care of MAC LCP requirement as such avoiding
            % segmentation/resegmentation by using remaining TBS
            if (pduLength + macHeaderLength) > (remainingTBS + remainingGrant)
                % Update the RLC header for segmented/resegmented SDU
                if soStart
                    % Create the RLC header for middle SDU segment since the SDU has been
                    % segmented earlier
                    pduHeader = generateDataHeader(obj, 3, obj.TxSeqNumFieldLength, sn, soStart);
                else
                    % Create the RLC header for first SDU segment since the SDU has not been
                    % segmented earlier
                    pduHeader = generateDataHeader(obj, 1, obj.TxSeqNumFieldLength, sn, soStart);
                end
                % Calculate the segmented SDU length that fit in the current grant
                % excluding the estimated MAC and RLC headers overhead. Estimation of MAC
                % header size should be done by decrementing 2 bytes (minimum MAC header
                % size) from the remaining grant. It avoids the under-estimation of MAC
                % header size for remaining grant values like 258 bytes
                macHeaderLength = ((remainingGrant-2) > 255) + 2;
                headersOverhead = numel(pduHeader) + macHeaderLength;
                sduLen = remainingGrant - headersOverhead;
                % Create the segmented/resegmented AMD PDU
                rlcPDU = [pduHeader; sdu(soStart + 1 : soStart + sduLen)];
                isSegmented = true;
                remainingGrant = remainingGrant - (headersOverhead + sduLen);
            else
                if soStart
                    if soEnd == fullSDULen-1
                        % Transmit the remaining SDU completely
                        pduHeader = generateDataHeader(obj, 2, obj.TxSeqNumFieldLength, sn, soStart);
                    else
                        % Transmit part of the remaining SDU
                        pduHeader = generateDataHeader(obj, 3, obj.TxSeqNumFieldLength, sn, soStart);
                    end
                else
                    if soEnd == fullSDULen-1
                        % Transmit the remaining SDU completely
                        pduHeader = generateDataHeader(obj, 0, obj.TxSeqNumFieldLength, sn, 0);
                    else
                        % Transmit part of the remaining SDU
                        pduHeader = generateDataHeader(obj, 1, obj.TxSeqNumFieldLength, sn, 0);
                    end
                end
                % Create the complete/segmented RLC AMD PDU
                rlcPDU = [pduHeader; sdu(soStart + 1: soEnd + 1)];
                sduLen = soEnd - soStart + 1;
                remainingGrant = remainingGrant - (numel(rlcPDU) + macHeaderLength);
            end
        end

        function pollFlag = getPollStatus(obj, segmentLength)
            %getPollStatus Return the poll flag and reset the poll counters
            % upon the flag set
            %   POLLFLAG = getPollStatus(OBJ) checks all the poll trigger conditions
            %   except poll counters update and returns the poll flag. If the poll flag
            %   is set, it resets the poll counters.
            %
            %   POLLFLAG = getPollStatus(OBJ, SEGMENTLENGTH) checks all the poll
            %   trigger conditions. It increments the poll byte counter by
            %   SEGMENTLENGTH and poll PDU counter by 1 before the check of poll
            %   trigger. If the poll flag is set, it resets the poll counters.

            pollFlag = 0;
            if nargin == 2
                % Increment PDU without poll count by 1 and bytes without poll count by
                % every new byte data carried in the AMD PDU
                obj.PDUsWithoutPoll = obj.PDUsWithoutPoll + 1;
                obj.BytesWithoutPoll = obj.BytesWithoutPoll + segmentLength;
                % If any of the poll PDU counter or the poll byte counter is enabled, check
                % the PDUs sent without poll or the SDU bytes sent without poll exceeds the
                % specified threshold
                if ((obj.PollPDU ~= 0) && (obj.PDUsWithoutPoll >= obj.PollPDU)) || ...
                        ((obj.PollByte ~= 0) && (obj.BytesWithoutPoll >= obj.PollByte))
                    pollFlag = 1;
                end
            end
            % Send the poll request if either the transmission and retransmission
            % buffers are empty after the submission of current AMD PDU or the
            % occurrence of Tx window stall due to the limited buffer size
            if isEmpty(obj.TxBuffer) && (obj.NumRetxBufferSDUs == 0)
                pollFlag = 1;
            elseif (obj.NumSDUsWaitingForACK == obj.BufferSize) || ...
                    (obj.NumSDUsWaitingForACK == obj.AMTxWindowSize) % Tx window stalling condition
                pollFlag = 1;
            end
            % Send the poll request if the poll retransmit timer has expired
            if obj.RetransmitPollFlag
                pollFlag = 1;
                obj.RetransmitPollFlag = false;
            end
            % Upon poll flag set, reset PDUs and bytes without poll counters
            if pollFlag
                obj.PDUsWithoutPoll = 0;
                obj.BytesWithoutPoll = 0;
            end
        end

        function pollRetransmitTimerExpiry(obj)
            %pollRetransmitTimerExpiry Perform the actions required after
            % the expiry of poll retransmit timer

            obj.RetransmitPollFlag = true;
            % Retransmit an SDU, which is awaiting for acknowledgement, when one of the
            % following conditions is met:
            %   - Empty transmission and retransmission buffers - Tx window stall due
            %   to the limited buffer size - Tx window stall due to no acknowledgement
            %   for the SNs of
            %     size Tx window. This condition occurs when Tx buffer size is
            %     configured to be more than the Tx window size
            % For more details, refer 3GPP TS 38.322 Section 5.3.3.4
            if (isEmpty(obj.TxBuffer) && (obj.NumRetxBufferSDUs == 0)) || ...
                    (obj.NumSDUsWaitingForACK == obj.BufferSize) || ...
                    (obj.NumSDUsWaitingForACK == obj.AMTxWindowSize)
                % Consider an SDU for retransmission that was submitted to the MAC layer,
                % since new SDU cannot be transmitted
                highestSNIdx = find(obj.WaitingForACKBufferContext(:, 1) == obj.TxSubmitted, 1);
                % If the latest transmitted PDU receives an acknowledgement, find one of
                % the PDUs waiting for Acknowledgement for Poll retransmission
                if isempty(highestSNIdx)
                    highestSNIdx = find(obj.WaitingForACKBufferContext(:, 1) >= 0, 1);
                end
                % Enqueue the selected SDU into the retransmission buffer and update the
                % retransmission such that the complete SDU was lost
                sduEnqueueIdx = mod(obj.RetxBufferFront + obj.NumRetxBufferSDUs, obj.BufferSize) + 1;
                obj.RetxBuffer{sduEnqueueIdx} = obj.WaitingForACKBuffer{highestSNIdx};
                obj.RetxBufferContext(sduEnqueueIdx) = obj.WaitingForACKBufferContext(highestSNIdx, 1); % SN of the SDU
                obj.RetxBufferContext(sduEnqueueIdx, 2) = obj.WaitingForACKBufferContext(highestSNIdx, 2) + 1; % Increment of the retransmission count
                % Update the statistics on RLC link failure (RLF) error due to the reach of
                % maximum retransmission limit
                if obj.RetxBufferContext(sduEnqueueIdx, 2) == obj.MaxRetxThreshold
                    obj.StatRLF = obj.StatRLF + 1;
                    return;
                end
                obj.RetxBufferContext(sduEnqueueIdx, 3:4) = [0 65535]; % Lost segments information
                obj.WaitingForACKBufferContext(highestSNIdx, 1:2) = -1;
                obj.WaitingForACKBuffer{highestSNIdx} = [];
                obj.NumSDUsWaitingForACK = obj.NumSDUsWaitingForACK - 1;
                % Update the buffer status of the RLC entity to inform the MAC about the
                % grant requirement
                packetInfo = obj.RetxBuffer{sduEnqueueIdx};
                obj.RequiredGrantLength = obj.RequiredGrantLength + packetInfo.PacketLength + obj.MaxRLCMACHeadersOH;
                obj.NumRetxBufferSDUs = obj.NumRetxBufferSDUs + 1;
            end
        end

        function valueAfterModulus = getTxSNModulus(obj, seqNum)
            %getTxSNModulus Return the Tx modulus for the given sequence
            % number

            % For more details about Tx modulus, refer 3GPP TS 38.322 Section 7.1
            valueAfterModulus = mod(seqNum - obj.TxNextAck, obj.TotalTxSeqNum);
        end

        function [sdu, higherLayerTags] = processDataPDU(obj, dataPDU)
            %processDataPDU Process the received data PDU

            sdu = [];
            higherLayerTags = [];
            % Decode the data packet received from the MAC layer. Update the statistics
            % accordingly
            [decodedPDU, pduDecodeStatus] = decodeDataPDU(obj, dataPDU.Packet);
            if pduDecodeStatus
                obj.StatReceivedDataPackets = obj.StatReceivedDataPackets + 1;
                obj.StatReceivedDataBytes = obj.StatReceivedDataBytes + decodedPDU.PDULength;
            else
                % Update the statistics and avoid any further processing
                obj.StatDecodeFailures = obj.StatDecodeFailures + 1;
                return;
            end

            % Discard the received SDU if it falls outside of the Rx window
            if ~(getRxSNModulus(obj, decodedPDU.SequenceNumber) < obj.AMRxWindowSize)
                obj.StatDroppedDataPackets = obj.StatDroppedDataPackets + 1;
                obj.StatDroppedDataBytes = obj.StatDroppedDataBytes + decodedPDU.PDULength;
                return;
            end
            % If the SDU is a duplicate SDU which was completely received earlier
            if isCompleteSDURcvd(obj, decodedPDU.SequenceNumber)
                numBytesDiscarded = numel(decodedPDU.Data);
                isReassembled = false;
            else
                % Adjust remaining tags on removal of RLC header
                higherLayerTags = wirelessnetwork.internal.packetTags.adjust(dataPDU.Tags, ...
                    -(decodedPDU.PDULength-numel(decodedPDU.Data)));
                if decodedPDU.SegmentationInfo == 0 % Reception of complete SDU
                    [numBytesDiscarded, isReassembled, sdu] = processCompleteSDU(obj, decodedPDU);
                else % On the reception of a segmented SDU
                    [numBytesDiscarded, isReassembled, sdu, reassembledSDUTags] = processSegmentedSDU(obj, decodedPDU, higherLayerTags);
                    if isReassembled
                        % Merge the tags from the segments of an SDU
                        higherLayerTags = wirelessnetwork.internal.packetTags.merge(reassembledSDUTags);
                    end
                end
            end

            if numel(decodedPDU.Data) ~= numBytesDiscarded
                % Update the RLC AM state variables based on the received PDU sequence
                % number
                updateRxState(obj, decodedPDU.SequenceNumber, isReassembled);
                if ~decodedPDU.PollBit
                    return;
                end
                pduSeqNum = getRxSNModulus(obj, decodedPDU.SequenceNumber);
                % Trigger status PDU
                if (pduSeqNum < getRxSNModulus(obj, obj.RxHighestStatus)) || ...
                        (pduSeqNum >= obj.AMRxWindowSize)
                    obj.IsStatusPDUTriggered = true;
                    obj.IsStatusPDUDelayed = false;
                else
                    obj.IsStatusPDUDelayed = true;
                end
            else
                % Trigger the status PDU on discarding the received byte segments as per
                % Section 5.2.3.2.2 of 3GPP TS 38.322
                obj.IsStatusPDUTriggered = true;
            end

            % Check if the received PDU contains the duplicate bytes
            if numBytesDiscarded
                % Update the duplicate segment reception statistics
                obj.StatDuplicateDataPackets = obj.StatDuplicateDataPackets + 1;
                obj.StatDuplicateDataBytes = obj.StatDuplicateDataBytes + numBytesDiscarded;
            end

            % Check if the status prohibit is already running when status PDU is triggered
            if obj.IsStatusPDUTriggered && obj.StatusProhibitTimeLeft ~= 0
                obj.IsStatusPDUTriggered = false;
                obj.IsStatusPDUTriggeredOverSPT = true;
            else
                % Update the buffer status report
                addStatusReportInReqGrant(obj);
            end
        end

        function processStatusPDU(obj, statusPDU)
            %processStatusPDU Process the received status PDU and update
            % the retransmission context

            obj.StatReceivedControlPackets = obj.StatReceivedControlPackets + 1;
            obj.StatReceivedControlBytes = obj.StatReceivedControlBytes + numel(statusPDU);
            % Decode the control packet received from MAC
            [nackSNInfo, soInfo, ackSN] = decodeStatusPDU(obj, statusPDU, obj.RxSeqNumFieldLength);
            % Check whether the received ACK SN is valid and falls inside the Tx window
            if ackSN >= 0
                % Update the retransmission context based on the received STATUS PDU
                updateRetransmissionContext(obj, nackSNInfo, soInfo, ackSN);
            end
        end

        function updateRxState(obj, currPktSeqNum, isReassembled)
            %updateRxState Update the RLC AM receiver context

            % Update the upper end of the receiving window
            if getRxSNModulus(obj, currPktSeqNum) >= getRxSNModulus(obj, obj.RxNextHighest)
                obj.RxNextHighest = mod(currPktSeqNum + 1, obj.TotalRxSeqNum);
            end
            % Check whether all bytes of the RLC SDU with SN = x are received
            if isReassembled
                if (currPktSeqNum == obj.RxHighestStatus) || (currPktSeqNum == obj.RxNext)
                    minSN = mod(currPktSeqNum + 1, obj.TotalRxSeqNum);
                    minSNMod = getRxSNModulus(obj, minSN);
                    rcvdSNList = obj.RcvdSNList(obj.RcvdSNList(:, 1) > -1, :);
                    % Update RxNextHighest and RxHighestStatus sequence numbers by finding a
                    % new sequence number which is not yet reassembled and delivered to higher
                    % layer
                    if ~isempty(rcvdSNList)
                        rcvdSNListMod = getRxSNModulus(obj, rcvdSNList);
                        receptionIndex = (rcvdSNListMod(:, 1) <= minSNMod) & ...
                            (rcvdSNListMod(:, 2) >= minSNMod);
                        if any(receptionIndex, "all")
                            minSN = mod(rcvdSNList(logical(sum(receptionIndex, 2)), 2) + 1, obj.TotalRxSeqNum);
                        end
                    end
                    if currPktSeqNum == obj.RxHighestStatus
                        obj.RxHighestStatus = minSN;
                    end
                    if currPktSeqNum == obj.RxNext
                        obj.RxNext = minSN;
                        % Update the reception status array
                        obj.RcvdSNList(getRxSNModulus(obj, obj.RcvdSNList) > obj.AMRxWindowSize) = -1;
                    end
                end
            end
            updateReassemblyTimerContext(obj);
        end

        function updateRetransmissionContext(obj, nackSNInfo, soInfo, ackSN)
            %updateRetransmissionContext Update the retransmission context

            txSubmittedModulus = getTxSNModulus(obj, obj.TxSubmitted);
            nackSNIdx = 0;

            % Iterate through each SN that was transmitted earlier and update the
            % retransmission context based on the status report
            for snRef = 0:getTxSNModulus(obj, ackSN)-1
                sn = mod(obj.TxNextAck + snRef, obj.TotalTxSeqNum);
                if all(sn ~= nackSNInfo)
                    % Remove the SDU context from waiting-for-ack buffer or retransmission
                    % buffer since it was successfully transmitted
                    isSNWaitingForAck = obj.WaitingForACKBufferContext(:, 1) == sn;
                    obj.WaitingForACKBufferContext(isSNWaitingForAck, :) = -1;
                    if any(isSNWaitingForAck)
                        obj.NumSDUsWaitingForACK = obj.NumSDUsWaitingForACK - 1;
                    end
                    isInRetxBuffer = obj.RetxBufferContext(:, 1) == sn;
                    obj.RetxBufferContext(isInRetxBuffer, :) = -1;
                    if any(isInRetxBuffer)
                        obj.NumRetxBufferSDUs = obj.NumRetxBufferSDUs - 1;
                    end
                    continue;
                end
                nackSNIdx = nackSNIdx + 1;
                % Skip the remaining process if the received NACK is not valid
                snModulus = getTxSNModulus(obj, sn);
                if snModulus > txSubmittedModulus
                    continue;
                end
                % If the SN has any previous pending retransmission context, remove it and
                % consider the latest retransmission context
                reTxNackSNIdxList = obj.RetxBufferContext(:, 1) == sn;
                if any(reTxNackSNIdxList)
                    % Remove the grant required, including MAC and RLC headers, for the
                    % obsolete segments
                    currLostSegments = obj.RetxBufferContext(reTxNackSNIdxList, 3:end);
                    % Replace the segment end with actual offset when it is received as the
                    % standard end marker 65535 for an SDU
                    currLostSegments(currLostSegments == 65535) = numel(obj.RetxBuffer{reTxNackSNIdxList}) - 1;
                    numSegments = numel(currLostSegments(currLostSegments >= 0))/2;
                    % Find the lost segment lengths and sum them. Then, add header overhead of
                    % each segment to the sum and add 1 for each segment in the sum to avoid
                    % error in segment length calculation
                    oldReqGrantLength = sum((currLostSegments(2:2:end) - currLostSegments(1:2:end))) + ...
                        (numSegments * (obj.MaxRLCMACHeadersOH + 1));
                    obj.RequiredGrantLength = obj.RequiredGrantLength - oldReqGrantLength;
                    % Update the segments information in the retransmission context
                    obj.RetxBufferContext(reTxNackSNIdxList, 3:end) = soInfo(nackSNIdx, :);
                    % Add the grant required, including MAC and RLC headers, for the latest
                    % segments
                    numSegments = numel(soInfo(nackSNIdx, (soInfo(nackSNIdx, :) >= 0)))/2;
                    reTxPacketInfo = obj.RetxBuffer{reTxNackSNIdxList};
                    soInfo(nackSNIdx, soInfo(nackSNIdx, :) == 65535) = reTxPacketInfo.PacketLength - 1;
                    newReqGrantLength = sum((soInfo(nackSNIdx, 2:2:end) - soInfo(nackSNIdx, 1:2:end))) + ...
                        (numSegments * (obj.MaxRLCMACHeadersOH + 1));
                    obj.RequiredGrantLength = obj.RequiredGrantLength + newReqGrantLength;
                end

                % If the SN has no pending retransmission context and it is waiting for
                % acknowledgment, add its information to the retransmission context
                nackSNIdxList = obj.WaitingForACKBufferContext(:, 1) == sn;
                if any(nackSNIdxList)
                    currentRetxCount = obj.WaitingForACKBufferContext(nackSNIdxList, 2) + 1;
                    % Update the statistics on RLC link failure (RLF) error due to the reach of
                    % maximum retransmission limit
                    if currentRetxCount == obj.MaxRetxThreshold
                        obj.StatRLF = obj.StatRLF + 1;
                        return;
                    end
                    % Enqueue the SDU into retransmission buffer which has received the NACK
                    % and update its retransmission context
                    sduEnqueueIdx = mod(obj.RetxBufferFront + obj.NumRetxBufferSDUs, obj.BufferSize) + 1;
                    obj.RetxBuffer{sduEnqueueIdx} = obj.WaitingForACKBuffer{nackSNIdxList};
                    obj.RetxBufferContext(sduEnqueueIdx, :) = [sn ...
                        currentRetxCount soInfo(nackSNIdx, :)];
                    % Replace the segment end with actual offset when it is received as the
                    % standard end marker 65535 for an SDU
                    reTxPacketInfo = obj.RetxBuffer{sduEnqueueIdx};
                    soInfo(nackSNIdx, soInfo(nackSNIdx, :) == 65535) = reTxPacketInfo.PacketLength - 1;
                    obj.WaitingForACKBufferContext(nackSNIdxList, 1:2) = -1;
                    obj.NumRetxBufferSDUs = obj.NumRetxBufferSDUs + 1;
                    obj.NumSDUsWaitingForACK = obj.NumSDUsWaitingForACK - 1;
                    numSegments = numel(soInfo(nackSNIdx, (soInfo(nackSNIdx, :) >= 0)))/2; % Not the actual segments
                    % Find the lost segment lengths and sum them, add header length for each
                    % segment, and add 1 for each segment to the required grant to compensate
                    % the error in segment length calculation
                    obj.RequiredGrantLength = obj.RequiredGrantLength + ...
                        sum(soInfo(nackSNIdx, 2:2:end) - soInfo(nackSNIdx, 1:2:end)) + ...
                        (numSegments * (obj.MaxRLCMACHeadersOH + 1));
                end
            end

            pollSNModulus = getTxSNModulus(obj, obj.PollSN);
            % Check if POLL_SN received any positive or negative acknowledgment
            if any(getTxSNModulus(obj, nackSNInfo) == pollSNModulus) || ...
                    (pollSNModulus < getTxSNModulus(obj, ackSN))
                % Stop and reset the t-pollRetransmit timer as per Section 5.3.3.3 of 3GPP
                % TS 38.322
                if obj.PollRetransmitTimeLeft ~= 0
                    obj.PollRetransmitTimeLeft = 0;
                end
            end

            % When receiving a positive acknowledgment for an RLC SDU with SN = x, the
            % transmitting side of an AM RLC entity shall:
            %  - set TX_Next_Ack equal to the SN of the RLC SDU with the smallest SN,
            %  whose SN falls within the range TX_Next_Ack <= SN <= TX_Next and for
            %  which a positive acknowledgment has not been received yet.
            minNACKSNReceived = ackSN;
            if ~isempty(nackSNInfo)
                minNACKSNReceived = nackSNInfo(1);
            end
            obj.TxNextAck = mod(obj.TxNextAck + ...
                getTxSNModulus(obj, minNACKSNReceived), obj.TotalTxSeqNum);
        end

        function valueAfterModulus = getRxSNModulus(obj, value)
            %getRxSNModulus Get the modulus value for the given
            % sequence number

            valueAfterModulus = mod(value - obj.RxNext, obj.TotalRxSeqNum);
        end

        function [controlPDU, statusPDULen] = constructStatusPDU(obj, remainingGrant)
            %constructStatusPDU Construct the status PDU as per 3GPP TS
            % 38.322 Section 6.2.2.5

            bytesFilled = 0;
            isPrevSNLost = false;
            range = 0;
            statusPDU = zeros(obj.GrantRequiredForStatusReport, 1);
            statusPDULen = 3; % minimum status PDU length
            grantLeft = min(remainingGrant, obj.GrantRequiredForStatusReport) - statusPDULen; % Set aside 3 bytes for status PDU header and ACK SN
            lastSNOffset = statusPDULen;
            sn = obj.RxNext;
            rxHighestStatus = getRxSNModulus(obj, obj.RxHighestStatus);

            % Include each missing SN information in the status PDU by iterating
            % through each SN between the lower end of the receiving window and the
            % highest status
            for snIdx = 0:rxHighestStatus-1
                sn = snIdx + obj.RxNext;
                if isCompleteSDURcvd(obj, sn) % On a complete reception of SDU
                    % Do not include the SN which is received completely
                    isPrevSNLost = false;
                    sn = mod(sn + 1, obj.TotalRxSeqNum);
                    continue;
                end
                snBufIdx = getSDUReassemblyBufIdx(obj, sn);
                if snBufIdx > -1 % On a partial reception of SDU
                    % Get the lost segments information and its corresponding status PDU
                    % information
                    segmentsLost = getLostSegmentsInfo(obj.RxBuffer(snBufIdx));
                    [subStatusPDU, subStatusPDULen, e1UpdateOffset] = addSegmentsInfoInStatusPDU(obj, sn, segmentsLost);
                    isPrevSNLost = any(segmentsLost == 65535);
                else % On a complete loss of SDU
                    segmentsLost = [];
                    if isPrevSNLost
                        % Update the range field if SNs are lost consecutively. The loss of the
                        % last segment of the previous SN and complete loss of the current SN is
                        % also considered as consecutive loss
                        sn = mod(sn + 1, obj.TotalRxSeqNum);
                        range = range + 1;
                        continue;
                    end
                    isPrevSNLost = true;
                    [subStatusPDU, subStatusPDULen, e1UpdateOffset] = addSegmentsInfoInStatusPDU(obj, sn, segmentsLost);
                end

                % Update the E3 field in the status PDU and add range to the status PDU
                if range && (~isempty(segmentsLost) || (snIdx == rxHighestStatus-1))
                    if obj.RxSeqNumFieldLength == 12
                        statusPDU(lastSNOffset) = bitor(statusPDU(lastSNOffset), 2);
                    else
                        statusPDU(lastSNOffset) = bitor(statusPDU(lastSNOffset), 8);
                    end
                    statusPDU(statusPDULen + 1) = range + 1;
                    grantLeft = grantLeft - 1;
                    bytesFilled = bytesFilled + 1;
                    statusPDULen = statusPDULen + 1;
                    range = 0;
                end
                % Don't add NACK SN information to the status PDU if the grant is not
                % sufficient. This is to avoid any misinterpretation about NACK SN to ACK
                % SN by the peer RLC entity
                if (subStatusPDULen > grantLeft)
                    break;
                end
                % Update the E1 field in the status PDU to denote that there is a lost SN
                % information after this
                if (obj.RxSeqNumFieldLength == 12) && (lastSNOffset ~= 3)
                    statusPDU(lastSNOffset) = bitor(statusPDU(lastSNOffset), 8);
                elseif (obj.RxSeqNumFieldLength == 18) && (lastSNOffset ~= 3)
                    statusPDU(lastSNOffset) = bitor(statusPDU(lastSNOffset), 32);
                end
                statusPDU(statusPDULen + 1: statusPDULen + subStatusPDULen)= subStatusPDU(1:subStatusPDULen);
                grantLeft = grantLeft - subStatusPDULen;
                lastSNOffset = lastSNOffset + bytesFilled + e1UpdateOffset;
                bytesFilled = subStatusPDULen - e1UpdateOffset;
                statusPDULen = statusPDULen + subStatusPDULen;
                sn = mod(sn + 1, obj.TotalRxSeqNum);
            end

            % Add ACK SN information into the status PDU along with D/C and CPT fields
            statusPDU(1:3) = getACKSNBytes(obj, statusPDULen-3, mod(sn,obj.TotalRxSeqNum));
            controlPDU = statusPDU(1:statusPDULen);
        end

        function [nackInfo, soInfo, ackSN] = decodeStatusPDU(obj, rlcPDU, seqNumFieldLength)
            %decodeStatusPDU decode the received status PDU

            % Estimate the number of SNs that can be present in the received status
            % PDU. The max number of NACK SNs per status PDU equals ceil((status PDU
            % length - ACK SN size)/minimum nack SN size)
            pduLen = numel(rlcPDU);
            numSNs = pduLen;
            % Define an array to store lost segment start and end fields
            soInfo = -1 * ones(numSNs, obj.MaxReassemblySDU * 2);
            % Stores the sequence numbers of the PDUs with lost segments
            nackInfo = -1 * ones(numSNs, 1);
            ackSN = -1;
            numSNs = 0;
            numSOsPerSN = 0;

            % Get the control PDU Type
            cpt = bitand(bitshift(rlcPDU(1), -4), 7);
            if cpt ~= 0
                % On the reception of corrupted status PDU, don't do any further processing
                return;
            end
            % Extract ACK SN and extension bit-1 values
            if seqNumFieldLength == 12
                ackSN = bitor(bitshift(bitand(rlcPDU(1), 15), 8), rlcPDU(2));
                e1 = bitand(rlcPDU(3), 128);
            else
                ackSN = bitor(bitor(bitshift(bitand(rlcPDU(1), 15), 14), bitshift(rlcPDU(2), 6)), bitshift(rlcPDU(3), -2));
                e1 = bitand(rlcPDU(3), 2);
            end

            % Check if the ACK SN falls outside of the tx window
            if ~(getTxSNModulus(obj, ackSN-1) < obj.AMTxWindowSize)
                % Don't decode the status PDU further
                ackSN = -1;
                return;
            end

            octetIndex = 4;
            lastSN = -1;
            % Check whether extension bit-1 is set or the entire PDU has been parsed
            while e1 && (octetIndex <= pduLen)
                % Extract NACK SN, extension bit-1, extension bit-2, and extension bit-3
                if seqNumFieldLength == 12
                    if octetIndex + 1 > pduLen
                        break;
                    end
                    nackSN = bitor(bitshift(rlcPDU(octetIndex), 4), bitshift(rlcPDU(octetIndex + 1), -4));
                    e2 = bitand(rlcPDU(octetIndex + 1), 4);
                    e3 = bitand(rlcPDU(octetIndex + 1), 2);
                    e1 = bitand(rlcPDU(octetIndex + 1), 8);
                    octetIndex = octetIndex + 2;
                else
                    if octetIndex + 2 > pduLen
                        break;
                    end
                    nackSN = bitor(bitor(bitshift(rlcPDU(octetIndex), 10), bitshift(rlcPDU(octetIndex + 1), 2)), bitshift(rlcPDU(octetIndex + 2), -6));
                    e2 = bitand(rlcPDU(octetIndex + 2), 16);
                    e3 = bitand(rlcPDU(octetIndex + 2), 8);
                    e1 = bitand(rlcPDU(octetIndex + 2), 32);
                    octetIndex = octetIndex + 3;
                end
                % If the new SN is not same as the last one, add lost segment information
                % of the new SN in a separate row
                if nackSN ~= lastSN
                    numSNs = numSNs + 1;
                    nackInfo(numSNs) = nackSN;
                    numSOsPerSN = 0;
                    lastSN = nackSN;
                    if e2 % Create an array to hold segmentation information
                        soInfo(numSNs, 1:end) = -1 * ones(1, obj.MaxReassemblySDU * 2);
                    end
                end
                % Extract the segment start and end for the NACK SN
                if e2
                    soStart = bitor(bitshift(rlcPDU(octetIndex), 8), rlcPDU(octetIndex + 1));
                    soEnd = bitor(bitshift(rlcPDU(octetIndex + 2), 8), rlcPDU(octetIndex + 3));
                    soInfo(numSNs, numSOsPerSN + 1:numSOsPerSN + 2) = [soStart soEnd];
                    numSOsPerSN = numSOsPerSN + 2;
                    octetIndex = octetIndex + 4;
                else
                    soInfo(numSNs, numSOsPerSN + 1:numSOsPerSN + 2) = [0 65535];
                end
                % Extract NACK SN range field
                if e3
                    % To exclude the current NACK SN from the NACK range, reduce the NACK range
                    % by 1
                    nackRange = rlcPDU(octetIndex) - 1;
                    for sn = 1:nackRange
                        nackInfo(numSNs + 1) = nackSN + sn;
                        numSNs = numSNs + 1;
                        soInfo(numSNs, 1:2)= [0 65535];
                        soInfo(numSNs, 3:end) = -1 * ones(1, (obj.MaxReassemblySDU-1)*2);
                    end
                    octetIndex = octetIndex + 1;
                end
            end
            snInfoIndices = nackInfo >= 0;
            nackInfo = nackInfo(snInfoIndices);
            soInfo = soInfo(snInfoIndices, :);
        end

        function reassemblyTimerExpiry(obj)
            %reassemblyTimerExpiry Perform the actions required after
            % the expiry of reassembly timer

            % Update the Rx highest status SN to the SN >= the reassembly timer
            % triggered SN for which all bytes have not been received
            minSN = obj.RxNextStatusTrigger;
            receptionIndex = (obj.RcvdSNList(:, 1) <= obj.RxNextStatusTrigger) & ...
                (obj.RcvdSNList(:, 2) >= obj.RxNextStatusTrigger);
            if any(receptionIndex)
                minSN = obj.RcvdSNList(receptionIndex, 2) + 1;
            end
            obj.RxHighestStatus = minSN;

            rhsBufIdx = getSDUReassemblyBufIdx(obj, obj.RxHighestStatus);
            rhsModulus = getRxSNModulus(obj, mod(obj.RxHighestStatus + 1, obj.TotalRxSeqNum));
            rxSNModulus = getRxSNModulus(obj, obj.RxNextHighest);
            isRNHEqualsRHS = rxSNModulus == rhsModulus;
            % Start the reassembly timer again if the conditions mentioned in 3GPP TS
            % 38.322 Section 5.2.3.2.4 are met
            if (rxSNModulus > rhsModulus) || (isRNHEqualsRHS && (rhsBufIdx ~= -1) && ...
                    anyLostSegment(obj.RxBuffer(rhsBufIdx)))
                % Start the reassembly timer
                obj.ReassemblyTimeLeft = obj.ReassemblyTimerNS;
                obj.RxNextStatusTrigger = obj.RxNextHighest;
            end
            % Trigger the status report
            obj.IsStatusPDUTriggered = true;
            addStatusReportInReqGrant(obj);
        end

        function statusProhibitTimerExpiry(obj)
            %statusProhibitTimerExpiry Perform the actions required after
            % the expiry of status prohibit timer

            % Trigger the status report requested while status prohibit timer is
            % running
            if obj.IsStatusPDUTriggeredOverSPT
                obj.IsStatusPDUTriggered = true;
                addStatusReportInReqGrant(obj);
            end
            obj.StatusProhibitTimeLeft = 0;
        end

        function addStatusReportInReqGrant(obj)
            %addStatusReportInReqGrant Update the buffer status by the
            % grant required for sending status report

            grantForNACKSNSegment = 8; % Maximum grant required for sending one NACK SN in the status PDU
            grantSize = 3 + getRxSNModulus(obj, obj.RxHighestStatus) * grantForNACKSNSegment; % 3 for status PDU header and ACK field size
            % Subtract the previously estimated status PDU size before adding the
            % status PDU. This is required when status report request comes before
            % sending the status PDU for previous trigger. This occurs because of delay
            % in resource assignment
            obj.RequiredGrantLength = obj.RequiredGrantLength - obj.GrantRequiredForStatusReport + grantSize;
            obj.GrantRequiredForStatusReport = grantSize;
            % Send the updated RLC buffer status report to MAC layer
            obj.RLCBufferStatus.BufferStatus = obj.RequiredGrantLength;
            obj.TxBufferStatusFcn(obj.RLCBufferStatus);
        end

        function snBufIdx = assignReassemblyBufIdx(obj, sn)
            %assignReassemblyBufIdx Find a place to store the specified
            % SN's SDU in the reassembly buffer

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

        function snBufIdx = getSDUReassemblyBufIdx(obj, sn)
            %getSDUReassemblyIdx Return the reassembly buffer index in
            % which SDU is stored

            snBufIdx = -1;
            for bufIdx = 1:obj.MaxReassemblySDU
                if obj.ReassemblySNMap(bufIdx) == sn
                    snBufIdx = bufIdx;
                    break;
                end
            end
        end

        function ackSN = getACKSNBytes(obj, hasNACKSNs, sn)
            %getACKSNBytes Return status PDU header and ACK SN information

            % Generate ACK SN information for the status PDU along with D/C and CPT
            % fields
            if obj.RxSeqNumFieldLength == 12
                if hasNACKSNs
                    ackSN = [bitshift(sn, -8); bitand(sn, 255); 128];
                else
                    ackSN = [bitshift(sn, -8); bitand(sn, 255); 0];
                end
            else
                if hasNACKSNs
                    ackSN = [bitshift(sn, -14); ...
                        bitand(bitshift(sn, -6), 255); ...
                        bitor(bitshift(bitand(sn, 63), 2), 2)];
                else
                    ackSN = [bitshift(sn, -14); ...
                        bitand(bitshift(sn, -6), 255); ...
                        bitshift(bitand(sn, 63), 2)];
                end
            end
        end

        function [numBytesDiscarded, isReassembled, sdu] = processCompleteSDU(obj, pduInfo)
            % processCompleteSDU Process the received complete SDU

            numBytesDiscarded = 0;
            sdu = uint8(pduInfo.Data);
            isReassembled = true;
            % Update the reception status on receiving a new complete SDU. There is no
            % need to update this on the reception of complete SDU for lower edge SN
            % since it moves forward after the complete reception
            if obj.RxNext ~= pduInfo.SequenceNumber
                updateRxGaps(obj, pduInfo.SequenceNumber);
            end
        end

        function [numDupBytes, isReassembled, sdu, higherLayerTags] = processSegmentedSDU(obj, pduInfo, segmentTags)
            %processSegmentedSDU Process the received segmented SDU

            sdu = [];
            higherLayerTags = [];
            numDupBytes = 0;
            isReassembled = false;
            % Find out the index in the reassembly buffer
            snBufIdx = assignReassemblyBufIdx(obj, pduInfo.SequenceNumber);
            if snBufIdx == -1
                obj.StatDroppedDataPackets = obj.StatDroppedDataPackets + 1;
                obj.StatDroppedDataBytes = obj.StatDroppedDataBytes + pduInfo.PDULength;
                return;
            end
            % Check whether the received segment is last segment
            isLastSegment = false;
            if pduInfo.SegmentationInfo == 2
                isLastSegment = true;
            end
            % Perform the duplicate detection and add the new segment bytes to the
            % reassembly buffer
            [numDupBytes, sdu, higherLayerTags] = reassembleSegment(obj.RxBuffer(snBufIdx), ...
                pduInfo.Data, pduInfo.PDULength, pduInfo.SegmentOffset, isLastSegment, segmentTags);
            if numDupBytes == numel(sdu)
                return;
            end
            % On the reception of all byte segments, reassemble the segments and
            % deliver it to higher layer without any further delay
            if ~isempty(sdu)
                isReassembled = true;
                % Update the reception status on receiving a new complete SDU. There is no
                % need to update this on the reception of complete SDU for lower edge SN
                % since it moves forward after the complete reception
                if obj.RxNext ~= pduInfo.SequenceNumber
                    updateRxGaps(obj, pduInfo.SequenceNumber);
                end
                obj.ReassemblySNMap(obj.ReassemblySNMap == pduInfo.SequenceNumber) = -1;
            end
        end

        function updateReassemblyTimerContext(obj)
            %updateReassemblyTimerContext Update the reassembly timer state
            % and Rx state variables as per 3GPP TS 38.322 Section 5.2.3.2.3

            rnBufIdx = getSDUReassemblyBufIdx(obj, obj.RxNext);
            rnModulus = getRxSNModulus(obj, mod(obj.RxNext + 1, obj.TotalRxSeqNum));
            if obj.ReassemblyTimeLeft ~= 0 % Reassembly timer is in running
                if obj.RxNextStatusTrigger == obj.RxNext
                    % Stop the reassembly timer if there is no gaps in reception till the SN
                    % which caused the start of reassembly timer
                    obj.ReassemblyTimeLeft = 0;
                elseif (getRxSNModulus(obj, obj.RxNextStatusTrigger) == rnModulus) && ((rnBufIdx ~= -1) && ~anyLostSegment(obj.RxBuffer(rnBufIdx)))
                    % Stop the reassembly timer if there is no gaps in reception till the
                    % earliest SN that requires reassembly. This applies only when
                    % RxNextStatusTrigger is equal to RxNext + 1
                    obj.ReassemblyTimeLeft = 0;
                elseif (~(getRxSNModulus(obj, obj.RxNextStatusTrigger) < obj.AMRxWindowSize) && ...
                        (obj.RxNextStatusTrigger ~= (obj.RxNext + obj.AMRxWindowSize)))
                    % Stop the reassembly if RxNextStatusTrigger falls outside of the Rx window
                    obj.ReassemblyTimeLeft = 0;
                end
            end

            if obj.ReassemblyTimeLeft == 0 % Reassembly timer is not running
                rnhModulus = getRxSNModulus(obj, obj.RxNextHighest);
                % Start the reassembly timer if any of the following conditions is met:
                %   - At least one missing SN between lower and upper ends of the receiving
                %   window - At least one missing segment between lower and upper ends of
                %   the receiving window when upper end = lower end + 1
                if (rnhModulus > rnModulus) || ...
                        ((rnhModulus == rnModulus) && (rnBufIdx ~= -1) && anyLostSegment(obj.RxBuffer(rnBufIdx)))
                    obj.ReassemblyTimeLeft = obj.ReassemblyTimerNS;
                    obj.RxNextStatusTrigger = obj.RxNextHighest;
                end
            end
        end

        function [rlcPacketInfo, isPollIncluded, sn, remainingGrant] = transmitSDUs(obj, bytesGranted, remainingTBSSize, currentTime)
            %transmitSDU Generate PDUs for SDUs upon being notified of
            % transmission opportunity

            rlcPacketInfo = obj.RLCPacketInfo;
            numRLCPackets = 0;
            isPollIncluded = false;
            remainingGrant = bytesGranted;
            sn = -1;

            txBuffer = obj.TxBuffer;
            % Iterate through each SDU in the Tx buffer
            while ~isEmpty(txBuffer)
                % Check the Tx window stalling arises due to the Tx buffer size limitation
                if obj.NumSDUsWaitingForACK >= obj.BufferSize || (obj.NumSDUsWaitingForACK == obj.AMTxWindowSize)
                    sn = obj.TxSubmitted;
                    break;
                end
                % Check whether the minimum grant length condition is satisfied as per
                % Section 5.4.3.1.3 of 3GPP TS 38.321
                if remainingGrant < obj.MinMACSubPDULength
                    break;
                end
                sn = obj.TxNext;

                rlcSDU = peek(txBuffer);
                sdu = rlcSDU.Packet;
                % Generate an AMD PDU that fits in the given grant
                [rlcPDU, sduLen, remainingGrant, isSegmented] = constructAMDPDU(obj, sn, ...
                    obj.TxSegmentOffset, rlcSDU.PacketLength - 1, sdu, remainingGrant, remainingTBSSize);
                % Increment the number of RLC packets in the list
                numRLCPackets = numRLCPackets + 1;
                rlcPacketInfo(numRLCPackets).PacketLength = numel(rlcPDU);
                % Update existing tags to accommodate for segmentation and
                % header addition
                rlcPacketInfo(numRLCPackets).Tags = wirelessnetwork.internal.packetTags.segment(rlcSDU.Tags, ...
                    [obj.TxSegmentOffset+1 obj.TxSegmentOffset+sduLen]);
                rlcPacketInfo(numRLCPackets).Tags = wirelessnetwork.internal.packetTags.adjust(rlcPacketInfo(numRLCPackets).Tags, numel(rlcPDU)-sduLen);

                if isSegmented
                    % Update the segment start offset for the segmented SDU
                    obj.TxSegmentOffset = obj.TxSegmentOffset + sduLen;
                    % Update the required grant size for this RLC entity Get the remaining RLC
                    % PDU length
                    obj.RequiredGrantLength = obj.RequiredGrantLength + ...
                        rlcSDU.PacketLength - obj.TxSegmentOffset + obj.MaxRLCMACHeadersOH;
                else
                    obj.TxSegmentOffset = 0;
                    obj.TxSubmitted = sn;
                    dequeue(txBuffer);
                    % After the complete transmission, keep the SDU in a buffer where it can
                    % wait for the acknowledgment
                    obj.NumSDUsWaitingForACK = obj.NumSDUsWaitingForACK + 1;
                    emptyTxedBufIdx = find(obj.WaitingForACKBufferContext(:, 1) == -1, 1);
                    obj.WaitingForACKBuffer{emptyTxedBufIdx} = rlcSDU;
                    obj.WaitingForACKBufferContext(emptyTxedBufIdx, :) = [sn, -1];
                    obj.TxNext = mod(obj.TxNext+1, obj.TotalTxSeqNum);
                end
                obj.StatTransmittedDataPackets = obj.StatTransmittedDataPackets + 1;
                obj.StatTransmittedDataBytes = obj.StatTransmittedDataBytes + numel(rlcPDU);
                % Update the poll bit in the PDU header if any of the status report
                % triggering condition is met
                pollBit = getPollStatus(obj, sduLen);
                if pollBit
                    rlcPDU(1) = bitor(rlcPDU(1), bitshift(pollBit, 6));
                    isPollIncluded = true;
                end
                rlcPacketInfo(numRLCPackets).Packet = rlcPDU;
            end
        end

        function [rlcPacketInfo, isPollIncluded, sn, remainingGrant] = retransmitSDUs(obj, bytesGranted, remainingTBSSize, currentTime)
            %retransmitSDUs Generate PDUs for retransmitting SDUs upon
            % notification of transmission opportunity

            rlcPacketInfo = obj.RLCPacketInfo;
            numRLCPackets = 0;
            isPollIncluded = false;
            remainingGrant = bytesGranted;
            sn = -1;

            % Iterate through the SDUs in the retransmission buffer
            for sduIdx = 1:obj.NumRetxBufferSDUs
                sn = obj.RetxBufferContext(obj.RetxBufferFront + 1);
                segmentsInfo = obj.RetxBufferContext(obj.RetxBufferFront + 1, 3:end);
                lostSegments = segmentsInfo(segmentsInfo >= 0);
                % Iterate through the segments in retransmission for the SDU
                for j = 1:2:numel(lostSegments)
                    % Check whether the minimum grant length condition is satisfied as per
                    % Section 5.4.3.1.3 of 3GPP TS 38.321
                    if remainingGrant < obj.MinMACSubPDULength
                        break;
                    end
                    rlcSDU = obj.RetxBuffer{obj.RetxBufferFront + 1};
                    [rlcPDU, transmittedSDULen, remainingGrant] = retransmitSegment(obj, sn, lostSegments(j:j+1), remainingGrant, remainingTBSSize);
                    % Increment the number of RLC packets in the list
                    numRLCPackets = numRLCPackets + 1;
                    rlcPacketInfo(numRLCPackets).PacketLength = numel(rlcPDU);
                    % Update existing tags to accommodate for segmentation
                    % and header addition
                    rlcPacketInfo(numRLCPackets).Tags = wirelessnetwork.internal.packetTags.segment(rlcSDU.Tags, ...
                        [lostSegments(j)+1 lostSegments(j)+transmittedSDULen]);
                    rlcPacketInfo(numRLCPackets).Tags = wirelessnetwork.internal.packetTags.adjust(rlcPacketInfo(numRLCPackets).Tags, numel(rlcPDU)-transmittedSDULen);
                    segmentsLeft = obj.RetxBufferContext(obj.RetxBufferFront + 1, 3:end);
                    obj.StatRetransmittedDataPackets = obj.StatRetransmittedDataPackets + 1;
                    obj.StatRetransmittedDataBytes = obj.StatRetransmittedDataBytes + numel(rlcPDU);
                    numSegmentsLeft = numel(segmentsLeft(segmentsLeft >= 0));
                    if numSegmentsLeft == 0
                        % After completing the retransmission of all the SDU segments, keep the SDU
                        % in a buffer where it can wait for the acknowledgment
                        obj.NumSDUsWaitingForACK = obj.NumSDUsWaitingForACK + 1;
                        emptyTxedBufIdx = find(obj.WaitingForACKBufferContext(:, 1) == -1, 1);
                        obj.WaitingForACKBuffer{emptyTxedBufIdx} = obj.RetxBuffer{obj.RetxBufferFront + 1};
                        obj.WaitingForACKBufferContext(emptyTxedBufIdx, :) = [sn, obj.RetxBufferContext(obj.RetxBufferFront + 1, 2)];
                        % Clear the retransmission context of the SDU
                        obj.RetxBufferContext(obj.RetxBufferFront + 1, 1:2) = -1;
                        obj.RetxBufferFront = mod(obj.RetxBufferFront + 1, obj.BufferSize);
                        obj.NumRetxBufferSDUs = obj.NumRetxBufferSDUs - 1;
                    end
                    % Update the poll bit in the PDU header if any of the status report
                    % triggering condition is met
                    pollBit = getPollStatus(obj);
                    if pollBit
                        rlcPDU(1) = bitor(rlcPDU(1), bitshift(pollBit, 6));
                        isPollIncluded = true;
                    end
                    rlcPacketInfo(numRLCPackets).Packet = rlcPDU;
                    if numSegmentsLeft == 0
                        break;
                    end
                end
            end
        end

        function [statusPDU, statusPDULen, lastSNOffset] = addSegmentsInfoInStatusPDU(obj, sn, segmentsLost)
            %addSegmentsInfoInStatusPDU Add segmented SDUs information in
            % the status PDU

            lastSNOffset = 0;
            % Define the status PDU with the specified size. Maximum number of bytes to
            % represent a segment loss in the status PDU is 7
            statusPDU = zeros(obj.MaxReassemblySDU * 7, 1);
            statusPDULen = 0;

            segmentIdx = 1;
            numSegmentsLost = size(segmentsLost, 1);
            isRxSeqNum12 = obj.RxSeqNumFieldLength == 12;
            bytesFilled = 0;
            % Iterate through the segments lost for the SDU
            while true
                if lastSNOffset ~= 0
                    % Update the E1 field in the status PDU
                    if isRxSeqNum12
                        statusPDU(lastSNOffset) = bitor(statusPDU(lastSNOffset), 8);
                    else
                        statusPDU(lastSNOffset) = bitor(statusPDU(lastSNOffset), 32);
                    end
                end

                % Update the NACK SN in the status PDU
                if isRxSeqNum12
                    statusPDU(statusPDULen + 1) = bitshift(sn, -4);
                    statusPDU(statusPDULen + 2) = bitshift(bitand(sn, 15), 4);
                    lastSNOffset = lastSNOffset + bytesFilled + 2;
                    statusPDULen = statusPDULen + 2;
                else
                    statusPDU(statusPDULen + 1) = bitshift(sn, -10);
                    statusPDU(statusPDULen + 2) = bitand(bitshift(sn, -2), 255);
                    statusPDU(statusPDULen + 3) = bitshift(bitand(sn, 3), 6);
                    lastSNOffset = lastSNOffset + bytesFilled + 3;
                    statusPDULen = statusPDULen + 3;
                end
                % Check if the whole SDU is missing
                if numSegmentsLost == 0
                    break;
                end
                bytesFilled = 0;
                % Add the segment offset information to the status PDU
                segmentStart = segmentsLost(segmentIdx, 1);
                segmentEnd = segmentsLost(segmentIdx, 2);
                % Update the E2 field
                if isRxSeqNum12
                    statusPDU(lastSNOffset) = bitor(statusPDU(lastSNOffset), 4);
                else
                    statusPDU(lastSNOffset) = bitor(statusPDU(lastSNOffset), 16);
                end
                % Add the SO start in the status PDU
                statusPDU(statusPDULen + 1) = bitshift(segmentStart, -8);
                statusPDU(statusPDULen + 2) = bitand(segmentStart, 255);
                % Add the SO end in the status PDU
                statusPDU(statusPDULen + 3) = bitshift(segmentEnd, -8);
                statusPDU(statusPDULen + 4) = bitand(segmentEnd, 255);
                statusPDULen = statusPDULen + 4;
                bytesFilled = bytesFilled + 4;
                % Update the segment index to next segment start
                segmentIdx = segmentIdx + 1;
                if (segmentIdx > numSegmentsLost)
                    break;
                end
            end
        end

        function updateRxGaps(obj, sn)
            %updateRxGaps Update the completely received SDUs context

            % Identify whether this complete SDU reception is an extension for the
            % existing contiguous SDU receptions. This can be checked by finding its
            % previous and following SDUs reception status
            prevSNRxStatus = (obj.RcvdSNList == mod(sn - 1, obj.TotalRxSeqNum));
            nextSNRxStatus = (obj.RcvdSNList == mod(sn + 1, obj.TotalRxSeqNum));
            isPrevSNContigious = any(prevSNRxStatus, 'all');
            isNextSNContigious = any(nextSNRxStatus, 'all');
            if ~isPrevSNContigious && ~isNextSNContigious
                % Create a new contiguous reception since it is not extending any other
                % existing contiguous reception
                indices = find(obj.RcvdSNList == [-1, -1], 1);
                obj.RcvdSNList(indices, 1) = sn;
                obj.RcvdSNList(indices, 2) = sn;
            elseif isPrevSNContigious && ~isNextSNContigious
                obj.RcvdSNList(prevSNRxStatus(:, 2), 2) = sn;
            elseif ~isPrevSNContigious && isNextSNContigious
                obj.RcvdSNList(nextSNRxStatus(:, 1), 1) = sn;
            else
                % Merge the two contiguous receptions since the new SDU makes them one
                % contiguous reception
                obj.RcvdSNList(prevSNRxStatus(:, 2), 2) = obj.RcvdSNList(nextSNRxStatus(:, 1), 2);
                obj.RcvdSNList(nextSNRxStatus(:, 1), 1:2) = -1;
            end
        end

        function rxStatus = isCompleteSDURcvd(obj, sn)
            %isCompleteSDURcvd Check whether the complete SDU is received

            rxStatus = false;
            % Get the contiguous reception starts and ends
            contiguousRxStarts = getRxSNModulus(obj, obj.RcvdSNList(obj.RcvdSNList(:, 1) >= 0, 1));
            % When the contiguous receptions are present, check whether the given SDU
            % SN falls within any of the contiguous reception
            if ~isempty(contiguousRxStarts)
                contiguousRxEnds = getRxSNModulus(obj, obj.RcvdSNList(obj.RcvdSNList(:, 2) >= 0, 2));
                sn = getRxSNModulus(obj, sn);
                rxStatus = any((contiguousRxStarts <= sn) & (contiguousRxEnds >= sn));
            end
        end
    end
end

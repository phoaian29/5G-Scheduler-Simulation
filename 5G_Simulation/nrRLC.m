classdef (Abstract) nrRLC < handle
    %nrRLC Base class for RLC UM and RLC passthrough entities
    %
    %   Note: This is an internal undocumented class and its API and/or
    %   functionality may change in subsequent releases.

    %   Copyright 2022-2024 The MathWorks, Inc.

    properties (SetAccess = private)
        %RNTI Radio network temporary identifier of a UE
        RNTI

        %LogicalChannelID Logical channel identifier
        LogicalChannelID
    end

    properties
        %BufferSize Maximum capacity of the Tx buffer in terms of number of SDUs
        %   Specify the maximum Tx buffer capacity of an RLC entity as a positive
        %   integer.
        BufferSize

        %ReassemblyTimer Timer for SDU reassembly failure detection in milliseconds
        %   Specify the reassembly timer value as one of 0, 5, 10, 15, 20, 25, 30,
        %   35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100, 110, 120, 130,
        %   140, 150, 160, 170, 180, 190, or 200. For more details, refer 3GPP TS
        %   38.331 information element RLC-Config.
        ReassemblyTimer

        %MaxReassemblySDU Maximum capacity of the reassembly buffer in terms of
        %number of SDUs. This is also equal to the maximum segments that can be
        %missing per SDU at any point of time.
        %   Specify the maximum capacity of the reassembly buffer as an integer
        %   scalar. The reassembly buffer capacity depends on the number of HARQ
        %   processes present. If the number of SDUs under reassembly reaches the
        %   limit, the oldest SDU in the buffer will be discarded
        MaxReassemblySDU
    end

    properties (SetAccess = protected, Hidden)
        %StatTransmittedDataPackets Number of data PDUs sent by RLC to MAC layer on
        %Tx side
        StatTransmittedDataPackets = 0

        %StatTransmittedDataBytes Number of data bytes sent by RLC to MAC layer on
        %Tx side
        StatTransmittedDataBytes = 0

        %StatRetransmittedDataPackets Number of data PDUs retransmitted by RLC to
        %MAC layer at Tx
        StatRetransmittedDataPackets = 0

        %StatRetransmittedDataBytes Number of data bytes retransmitted by RLC to
        %MAC layer at Tx
        StatRetransmittedDataBytes = 0

        %StatTransmittedControlPackets Number of control PDUs sent by RLC to MAC
        %layer at Tx
        StatTransmittedControlPackets = 0

        %StatTransmittedControlBytes Number of control bytes sent by RLC to MAC
        %layer at Tx
        StatTransmittedControlBytes = 0

        %StatTransmitterQueueOverflow Number of packets dropped by RLC layer due to
        %Tx buffer overflow
        StatTransmitterQueueOverflow = 0

        %StatReceivedDataPackets Number of RLC data PDUs received by RLC from MAC
        StatReceivedDataPackets = 0

        %StatReceivedDataBytes Number of RLC data bytes received by RLC from MAC on
        %Rx side
        StatReceivedDataBytes = 0

        %StatReceivedControlPackets Number of RLC control PDUs received by RLC from
        %MAC at Rx
        StatReceivedControlPackets = 0

        %StatReceivedControlBytes Number of RLC control bytes received by RLC from
        %MAC at Rx
        StatReceivedControlBytes = 0

        %StatDroppedDataPackets Number of RLC data PDUs dropped due to reassembly
        %failure at Rx
        StatDroppedDataPackets = 0

        %StatDroppedDataBytes Number of RLC data bytes dropped due to reassembly
        %failure at Rx
        StatDroppedDataBytes = 0

        %StatDuplicateDataPackets Number of duplicate RLC data PDUs dropped at Rx
        StatDuplicateDataPackets = 0

        %StatDuplicateDataBytes Number of duplicate RLC data bytes dropped at Rx
        StatDuplicateDataBytes = 0

        %StatDecodeFailures Number of packets dropped due to decode failure at Rx
        StatDecodeFailures = 0

        %StatReassemblyTimerExpiry Number of times reassembly timer has timed-out
        %at Rx
        StatReassemblyTimerExpiry = 0

        %StatRLF Number of times radio link failure (RLF) occured at RLC at Tx
        StatRLF = 0

        %RLCPacketInfo Default RLC packet structure
        RLCPacketInfo
    end

    properties (Access = protected)
        %TxBufferStatusFcn Function handle to send the RLC buffer status to the
        %associated MAC entity
        TxBufferStatusFcn

        %RxForwardFcn Function handle to forward the received RLC SDUs to the
        %application layer
        RxForwardFcn

        %RLCBufferStatus Format of RLC buffer status report that will be sent to
        %MAC
        RLCBufferStatus = struct('RNTI', 0, 'LogicalChannelID', 4, 'BufferStatus', 0)

        %DataPDUInfo Format of RLC data PDU
        DataPDUInfo = struct('Data', [], 'PDULength', 0, 'PollBit', 0, 'SegmentationInfo', 0, ...
            'SequenceNumber', 0, 'SegmentOffset', 0)

        %HigherLayerPacketFormat Defines the packet format of the application layer
        %in Rx chain
        HigherLayerPacketFormat = struct('NodeID', 0, 'Packet', [], 'PacketLength', 0, ...
            'CurrentTime', 0, 'LogicalChannelID', 4, 'RNTI', 0, 'Tags', [])

        %LastRunTime Time (in nanoseconds) at which the RLC entity was invoked last
        %time
        LastRunTime = 0

        %Stats Structure for reporting the statistics
        Stats = struct('RNTI', 0, 'LogicalChannelID', 4, 'TransmittedPackets', 0, ...
            'TransmittedBytes', 0, 'RetransmittedPackets', 0, ...
            'RetransmittedBytes', 0, 'ReceivedPackets', 0, 'ReceivedBytes', 0, ...
            'DroppedPackets', 0, 'DroppedBytes', 0)

        %PacketInfo Defines the RLC packet format
        PacketInfo = struct(Packet=[], PacketLength=0, Tags=[]);
    end

    properties (Access=protected, Constant)
        %MaxPacketSize Maximum size of packet (in bytes) in the queue
        MaxPacketSize = 9000

        %MaxRLCMACHeadersOH Maximum headers overhead for the calculation of buffer volume
        MaxRLCMACHeadersOH = 8

        %MinMACSubPDULength Minimum MAC subPDU length (in bytes) which equals to
        %sum of MAC subheader length and RLC PDU length with at least some data
        %bytes. For more details on minimum MAC subPDU length, refer Section
        %5.4.3.1.3 of 3GPP TS 38.321
        MinMACSubPDULength = 8
    end

    methods
        % Constructor
        function obj = nrRLC(rnti, logicalChannelID)

            obj.RNTI = rnti;
            obj.LogicalChannelID = logicalChannelID;
            obj.Stats.RNTI = rnti;
            obj.Stats.LogicalChannelID = logicalChannelID;
            obj.HigherLayerPacketFormat.RNTI = rnti;
            obj.HigherLayerPacketFormat.LogicalChannelID = logicalChannelID;
            obj.RLCPacketInfo = repmat(obj.PacketInfo, 1, 0);
        end

        function nextInvokeTime = run(~, ~)
            %run Run the RLC entity and return the next invoke time of RLC entity
            %
            %   NEXTINVOKETIME = run(OBJ, CURRENTTIME) runs the RLC entity and returns
            %   the next invoke time of RLC entity.
            %
            %   NEXTINVOKETIME indicates the time (in nanoseconds) at which the run
            %   method should be invoked again.
            %
            %   OBJ is an object of type nrRLC.
            %
            %   CURRENTTIME is an integer indicating the current time (in nanoseconds).

            nextInvokeTime = Inf;
        end

        function stats = statistics(obj)
            %statistics Return the cumulative RLC statistics
            %
            %   STATS = statistics(OBJ) returns the cumulative statistics collected by
            %   the RLC entity.
            %
            %   STATS is a structure with these fields.
            %       TransmittedPackets   - Number of packets transmitted
            %       TransmittedBytes     - Number of bytes transmitted
            %       RetransmittedPackets - Number of packets retransmitted
            %       RetransmittedBytes   - Number of bytes retransmitted
            %       ReceivedPackets      - Number of packets received
            %       ReceivedBytes        - Number of bytes received
            %       DroppedPackets       - Number of packets dropped at Rx
            %       DroppedBytes         - Number of bytes dropped at Rx

            stats = obj.Stats;
            stats.TransmittedPackets = obj.StatTransmittedDataPackets;
            stats.TransmittedBytes = obj.StatTransmittedDataBytes;
            stats.RetransmittedPackets = obj.StatRetransmittedDataPackets;
            stats.RetransmittedBytes = obj.StatRetransmittedDataBytes;
            stats.ReceivedPackets = obj.StatReceivedDataPackets;
            stats.ReceivedBytes = obj.StatReceivedDataBytes;
            stats.DroppedPackets = obj.StatDroppedDataPackets;
            stats.DroppedBytes = obj.StatDroppedDataBytes;
        end
    end
end
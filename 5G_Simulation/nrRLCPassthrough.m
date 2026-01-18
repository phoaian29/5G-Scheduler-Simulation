classdef nrRLCPassthrough < nr5g.internal.nrRLC
    %nrRLCPassthrough Implement full buffer traffic related functionality
    %
    %   Note: This is an internal undocumented class and its API and/or
    %   functionality may change in subsequent releases.
    %
    %   RLCOBJ = nrRLCPassthrough(RNTI, LOGICALCHANNELID,
    %   TXBUFFERSTATUSFCN) implements full buffer traffic related
    %   functionality.
    %
    %   RNTI is radio network temporary identifier.
    %
    %   LOGICLACHANNELID is logical channel identifier.
    %
    %   TXBUFFERSTATUSFCN specifies the function callback that will be used
    %   to send the buffer status report to MAC.

    %   Copyright 2022-2024 The MathWorks, Inc.

    % Properties that won't get modified after their initialization in the
    % constructor
    properties (Access = private)
        %PacketForFullBuffer Packet to be transmitted when MAC notifies the
        %transmission opportunity
        PacketForFullBuffer
    end

    properties (Access = private, Constant)
        %BufferSizeForFullBuffer Buffer size (in bytes) info that will be
        %sent to MAC when full buffer is enabled. It is set to the maximum
        %value mentioned in 3GPP TS 38.321 Table 6.1.3.1-2
        BufferSizeForFullBuffer = 81338368
    end

    methods
        %Constructor
        function obj = nrRLCPassthrough(rnti, logicalChannelID, txBufferStatusFcn)

            obj@nr5g.internal.nrRLC(rnti, logicalChannelID);
            obj.PacketForFullBuffer = ones(obj.MaxPacketSize, 1);
            obj.TxBufferStatusFcn = txBufferStatusFcn;
            % Send the updated buffer status information to MAC
            if ~isempty(txBufferStatusFcn)
                obj.RLCBufferStatus.RNTI = rnti;
                obj.RLCBufferStatus.LogicalChannelID = logicalChannelID;
                obj.RLCBufferStatus.BufferStatus = obj.BufferSizeForFullBuffer;
                obj.TxBufferStatusFcn(obj.RLCBufferStatus);
            end
        end
    end

    methods (Hidden)
        function rlcPacketInfo = sendPDUs(obj, bytesGranted, ~, ~)
            %sendPDUs Send the RLC protocol data units (PDUs) that fit in
            %the grant notified by MAC layer
            %
            %   RLCPACKETINFO = sendPDUs(OBJ, BYTESGRANTED, ~) sends the RLC PDUs
            %   that fit in the grant notified by MAC.
            %
            %   RLCPACKETINFO is a struct array of RLC PDUs to be transmitted
            %   by MAC. Each element represents one RLC packet and its associated information.
            %
            %   BYTESGRANTED is a positive integer scalar, which represents
            %   the number of granted transmission bytes.

            rlcPacketInfo = repmat(obj.PacketInfo, 1, 0);
            numRLCPackets = 0;
            packetSize = obj.MaxPacketSize;

            % Generate PDUs until the grant is fulfilled
            while bytesGranted > 0
                packetSize = min(packetSize, bytesGranted);
                macHeaderLength = (packetSize > 255) + 2;
                % Add new RLC PDU to the array
                numRLCPackets = numRLCPackets + 1;
                rlcPacketInfo(numRLCPackets).Packet = obj.PacketForFullBuffer(1:packetSize-macHeaderLength);
                rlcPacketInfo(numRLCPackets).PacketLength = packetSize - macHeaderLength;

                % Update the grant size
                bytesGranted = bytesGranted - packetSize;

                % Update statistics
                obj.StatTransmittedDataPackets = obj.StatTransmittedDataPackets + 1;
                obj.StatTransmittedDataBytes = obj.StatTransmittedDataBytes + (packetSize-macHeaderLength);
            end

            % Send the updated buffer status information to MAC
            obj.TxBufferStatusFcn(obj.RLCBufferStatus);
        end

        function receivePDUs(obj, rlcPacketInfo, ~)
            %receivePDUs Receive RLC PDU from the MAC layer
            %
            %   receivePDUs(OBJ, RLCPACKETINFO) Receives RLC PDU from the
            %   MAC layer.
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

            % Update statistics
            obj.StatReceivedDataPackets = obj.StatReceivedDataPackets + 1;
            obj.StatReceivedDataBytes = obj.StatReceivedDataBytes + rlcPacketInfo.PacketLength;
        end
    end
end
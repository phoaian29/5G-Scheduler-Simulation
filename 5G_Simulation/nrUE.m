classdef nrUE < wirelessnetwork.internal.nrNode
    %nrUE 5G NR user equipment (UE) node
    %   UE = nrUE creates a default 5G New Radio (NR) UE.
    %
    %   UE = nrUE(Name=Value) creates one or more similar UEs with the
    %   specified property Name set to the specified Value. You can specify
    %   additional name-value arguments in any order as (Name1=Value1, ...,
    %   NameN=ValueN). The number of rows in 'Position' argument defines the
    %   number of UEs created. 'Position' must be an N-by-3 matrix of numbers
    %   where N(>=1) is the number of UEs and each row must contain three
    %   numeric values representing the [X, Y, Z] position of a UE in meters.
    %   The output, UE is a row-vector of nrUE objects containing N UEs.
    %   You can also supply multiple names for 'Name' argument corresponding to
    %   number of UEs created. Multiple names must be supplied either as a
    %   vector of strings or cell array of character vectors. If name is not set
    %   then a default name as 'NodeX' is given to node, where 'X' is ID of the
    %   node. Assuming 'N' nodes are created and 'M' names are supplied, then
    %   if (M>N) then trailing (M-N) names are ignored, and if (N>M) then
    %   trailing (N-M) nodes are set to default names.
    %   You can set the "Position" and "Name" properties corresponding to the
    %   multiple UEs simultaneously when you specify them as N-V arguments in
    %   the constructor. After the node creation, you can set the "Position"
    %   and "Name" property for only one UE object at a time.
    %
    %   nrUE properties (configurable through N-V pair as well as public settable):
    %
    %   Name                 - Node name
    %   Position             - Node position
    %
    %   nrUE properties (configurable through N-V pair only):
    %
    %   NumTransmitAntennas  - Number of transmit antennas
    %   NumReceiveAntennas   - Number of receive antennas
    %   TransmitPower        - Peak transmit power in dBm
    %   PHYAbstractionMethod - Physical layer (PHY) abstraction method
    %   NoiseFigure          - Noise figure in dB
    %   ReceiveGain          - Receiver antenna gain in dB
    %
    %   nrUE properties (read-only):
    %
    %   ID              - Node identifier of the UE
    %   ConnectionState - Connection state as "Idle" or "Connected"
    %   GNBNodeID       - Node identifier of the gNB to which UE is connected
    %   RNTI            - Radio network temporary identifier of the UE
    %
    %   nrUE methods:
    %
    %   addTrafficSource     - Add data traffic source to the UE
    %   statistics           - Get statistics of the UE
    %   addMobility          - Add random waypoint mobility model to the UE
    %
    %
    %   % Example 1:
    %   %  Create 10 similar UEs with 20 dBm transmit power. Place the UEs randomly along X-axis
    %   %  within 1000 meters from origin.
    %   numUEs = 10;
    %   position = [1000*rand(numUEs, 1) zeros(numUEs, 2)];
    %   UEs = nrUE(Position=position, TransmitPower=20)
    %
    %   % Example 2:
    %   % Create a UE node and add a random waypoint mobility model to it with this
    %   % configuration.
    %   % SpeedRange: [1 1000] (Units are in meters per second)
    %   % PauseDuration: 0.1 seconds
    %   ue = nrUE;
    %   addMobility(ue, SpeedRange=[1 1000], PauseDuration=0.1)
    %
    %   See also nrGNB.

    %   Copyright 2022-2024 The MathWorks, Inc.

    properties(SetAccess = private)
        %NoiseFigure Noise figure in dB
        %   Specify the noise figure in dB. The default value is 6.
        NoiseFigure(1,1) {mustBeNumeric, mustBeFinite, mustBeNonnegative} = 6;

        %ReceiveGain Receiver gain in dB
        %   Specify the receiver gain in dB. The default value is 0.
        ReceiveGain(1,1) {mustBeNumeric, mustBeFinite, mustBeNonnegative} = 0;

        %TransmitPower Peak transmit power of a UE in dBm
        %   Peak transmit power, specified as a finite numeric scalar.
        %   Units are in dBm. The maximum value of transmit power you can
        %   specify is 60 dBm. The default value is 23 dBm. The
        %   configureULPowerControl object function (Uplink transmit power
        %   control mechanism) of the nrGNB object determines the actual
        %   transmit power of a UE node, which can be less than or equal
        %   to the peak transmit power.
        TransmitPower (1,1) {mustBeNumeric, mustBeFinite, mustBeLessThanOrEqual(TransmitPower, 60)} = 23

        %NumTransmitAntennas Number of transmit antennas on UE
        %   Specify the number of transmit antennas on UE. The allowed values are
        %   1, 2, 4. The default value is 1.
        NumTransmitAntennas (1, 1) {mustBeMember(NumTransmitAntennas, ...
            [1 2 4])} = 1;

        %NumReceiveAntennas Number of receive antennas on UE
        %   Specify the number of receive antennas on UE. The allowed values are
        %   1, 2, 4. The default value is 1.
        NumReceiveAntennas (1, 1) {mustBeMember(NumReceiveAntennas, ...
            [1 2 4])} = 1;

        %PHYAbstractionMethod PHY abstraction method
        %   Specify the PHY abstraction method as "linkToSystemMapping" or "none".
        %   The value "linkToSystemMapping" represents link-to-system-mapping based
        %   abstract PHY. The value "none" represents full PHY processing. The default
        %   value is "linkToSystemMapping".
        PHYAbstractionMethod = "linkToSystemMapping";
    end

    properties (SetAccess = private)
        %GNBNodeID Node ID of the gNB to which UE is connected
        GNBNodeID;

        %GNBNodeName Name of the gNB to which UE is connected
        GNBNodeName;

        %ConnectionState Connection state of the UE as "Idle" or "Connected"
        ConnectionState = "Idle"

        %RNTI Radio network temporary identifier (integer in the range 1 to 65522, inclusive) of the UE
        RNTI;
    end

    properties(SetAccess = private, Hidden)
        %NCellID Physical layer cell identity
        NCellID;
    end

    methods(Access = public)
        function obj = nrUE(varargin)

            % Name-value pair check
            coder.internal.errorIf(mod(nargin, 2) == 1,'MATLAB:system:invalidPVPairs');

            if nargin > 0
                param = nrNodeValidation.validateUEInputs(varargin);
                names = param(1:2:end);
                % Search the presence of 'position' N-V pair argument to calculate
                % the number of UEs user intends to create
                positionIdx = find(strcmp([names{:}], 'Position'), 1, 'last');
                numUEs = 1;
                if ~isempty(positionIdx)
                    position = param{2*positionIdx}; % Read value of Position N-V argument
                    validateattributes(position, {'numeric'}, {'nonempty', 'ncols', 3, 'finite'}, mfilename, 'Position');
                    if ismatrix(position)
                        numUEs = size(position, 1);
                    end
                end

                % Search the presence of 'Name' N-V pair argument
                nameIdx = find(strcmp([names{:}], 'Name'), 1, 'last');
                if ~isempty(nameIdx)
                    nodeName = param{2*nameIdx}; % Read value of Position N-V argument
                end

                % Create UE(s)
                obj(1:numUEs) = obj;
                className = class(obj(1));
                classFunc = str2func(className);
                for i=2:numUEs
                    % To support vectorization when inheriting "nrUE", instantiate
                    % class based on the object's class
                    obj(i) = classFunc();
                end

                % Set the configuration of UE(s) as per the N-V pairs
                for i=1:2:nargin-1
                    paramName = param{i};
                    paramValue = param{i+1};
                    switch (paramName)
                        case 'Position'
                            % Set position for UE(s)
                            for j = 1:numUEs
                                obj(j).Position = position(j, :);
                            end
                        case 'Name'
                            % Set name for UE(s). If name is not supplied for all UEs then leave the
                            % trailing UEs with default names
                            nameCount = min(numel(nodeName), numUEs);
                            for j=1:nameCount
                                obj(j).Name = nodeName(j);
                            end
                        otherwise
                            % Make all the UEs identical by setting same value for all the configurable
                            % properties, except position and name
                            [obj.(char(paramName))] = deal(paramValue);
                    end
                end
            end

            % Create internal layers for each UE
            phyParam = {'TransmitPower', 'NumTransmitAntennas', 'NumReceiveAntennas', ...
                'NoiseFigure', 'ReceiveGain', 'Position'};

            % Create internal layers for each UE
            for idx=1:numel(obj)
                ue = obj(idx);
                % Use weak-references for cross-linking handle objects
                ueWeakRef = matlab.lang.WeakReference(ue);

                % Set up traffic manager
                ue.TrafficManager = wirelessnetwork.internal.trafficManager(ue.ID, ...
                    [], @(varargin) ueWeakRef.Handle.processEvents(varargin{:}), DataAbstraction=false, ...
                    PacketContext=struct('DestinationNodeID', 0, 'LogicalChannelID', 4, 'RNTI', 0));

                % Set up MAC
                ue.MACEntity = nrUEMAC( ...
                    @(varargin) ueWeakRef.Handle.processEvents(varargin{:}));

                % Set up PHY
                phyInfo = struct();
                for j=1:numel(phyParam)
                    phyInfo.(char(phyParam{j})) = ue.(char(phyParam{j}));
                end
                if strcmp(ue.PHYAbstractionMethod, "none")
                    ue.PhyEntity = nrUEFullPHY(phyInfo, @(varargin) ueWeakRef.Handle.processEvents(varargin{:})); % Full PHY
                    ue.PHYAbstraction = 0;
                else
                    ue.PhyEntity = nrUEAbstractPHY(phyInfo, @(varargin) ueWeakRef.Handle.processEvents(varargin{:})); % Abstract PHY
                    ue.PHYAbstraction = 1;
                end

                % Set inter-layer interfaces
                ue.setLayerInterfaces();

                % Set Rx Info to be returned with packet relevance check
                ue.RxInfo.ID = ue.ID;
                ue.RxInfo.Position = ue.Position;
                ue.RxInfo.Velocity = ue.Velocity;
                ue.RxInfo.NumReceiveAntennas = ue.NumReceiveAntennas;
            end
        end

        function stats = statistics(obj)
            %statistics Return the statistics of the UE
            %
            %   STATS = statistics(OBJ) returns the statistics of the UE, OBJ. 
            %   STATS is a structure with these fields.
            %   ID   - ID of the UE
            %   Name - Name of the UE
            %   App  - Application layer statistics
            %   RLC  - RLC layer statistics
            %   MAC  - MAC layer statistics
            %   PHY  - PHY layer statistics
            %
            %   App is a structure with these fields.
            %   TransmittedPackets  - Total number of packets transmitted to the RLC layer
            %   TransmittedBytes    - Total number of bytes transmitted to the RLC layer
            %   ReceivedPackets     - Total number of packets received from the RLC layer
            %   ReceivedBytes       - Total number of bytes received from the RLC layer
            %
            %   RLC is a structure with these fields.
            %   TransmittedPackets   - Total number of packets transmitted to the MAC layer
            %   TransmittedBytes     - Total number of bytes transmitted to the MAC layer
            %   RetransmittedPackets - Total number of packets retransmitted to the MAC layer
            %   RetransmittedBytes   - Total number of bytes retransmitted to the MAC layer
            %   ReceivedPackets      - Total number of packets received from the MAC layer
            %   ReceivedBytes        - Total number of bytes received from the MAC layer
            %   DroppedPackets       - Total number of received packets dropped due to
            %                          reassembly failure
            %   DroppedBytes         - Total number of received bytes dropped due to
            %                          reassembly failure
            %
            %   MAC is a structure with these fields.
            %   TransmittedPackets  - Total number of packets transmitted to the PHY layer.
            %                         It only corresponds to new transmissions assuming
            %                         that MAC does not send the packet again to the
            %                         PHY for retransmissions. Packets are buffered at
            %                         PHY. MAC only sends the requests for
            %                         retransmission to the PHY layer
            %   TransmittedBytes    - Total number of bytes transmitted to the PHY layer
            %   ReceivedPackets     - Total number of packets received from the PHY layer
            %   ReceivedBytes       - Total number of bytes received from the PHY layer
            %   Retransmissions     - Total number of retransmissions requests sent to the PHY layer
            %   RetransmissionBytes - Total number of MAC bytes which correspond to retransmissions
            %   DLTransmissionRB    - Total number of downlink resource blocks assigned
            %                         for new transmissions
            %   DLRetransmissionRB  - Total number of downlink resource blocks assigned
            %                         for retransmissions
            %   ULTransmissionRB    - Total number of uplink resource blocks assigned
            %                         for new transmissions
            %   ULRetransmissionRB  - Total number of uplink resource blocks assigned
            %                         for retransmissions
            %
            %   PHY is a structure with these fields.
            %   TransmittedPackets  - Total number of packets transmitted
            %   ReceivedPackets     - Total number of packets received
            %   DecodeFailures      - Total number of decode failures
            %
            % You can fetch statistics for multiple UEs at once by calling this
            % function on a vector of UE objects. STATS is a row-vector where an
            % element at the index 'i' of STATS contains the statistics of UE at index
            % 'i' of the UE vector, OBJ.

            for i=numel(obj):-1:1
                stats(i) = statisticsPerUE(obj(i));
            end
        end

        function [flag, rxInfo] = isPacketRelevant(obj, packet)
            %isPacketRelevant Check whether packet is relevant for the node
            %
            %   [FLAG, RXINFO] = isPacketRelevant(OBJ, PACKET) determines
            %   whether the packet is relevant for the node and returns a
            %   flag, FLAG, indicating the decision. It also returns
            %   receiver information, RXINFO, needed for applying channel
            %   on the incoming packet, PACKET.
            %
            %   FLAG is a logical scalar value indicating whether to invoke
            %   channel or not. Value 1 represents that packet is relevant
            %   and channel must be invoked for it.
            %
            %   The function returns the output, RXINFO, and is valid only
            %   when the FLAG value is 1 (true). The structure of this
            %   output contains these fields:
            %
            %   ID - Node identifier of the receiver
            %   Position - Current receiver position in Cartesian coordinates,
            %              specified as a real-valued vector of the form [x
            %              y z]. Units are in meters.
            %   Velocity - Current receiver velocity (v) in the x-, y-, and
            %              z-directions, specified as a real-valued vector
            %              of the form [vx vy vz]. Units are in meters per
            %              second.
            %   NumReceiveAntennas - Number of receive antennas on node
            %
            %   OBJ is an object of type <a href="matlab:help('nrUE')">nrUE</a>
            %
            %   PACKET is the packet received from the channel, specified as
            %   structure of the format <a href="matlab:help('wirelessnetwork.internal.wirelessPacket')">wirelessPacket</a>.

            flag = false;
            rxInfo = [];

            % Check relevance of packet
            if packet.TransmitterID ~= obj.ID && ~isempty(obj.ReceiveFrequency) && packet.CenterFrequency == obj.ReceiveFrequency && ...
                    intracellPacketRelevance(obj, packet) && rxOn(obj.MACEntity, packet)
                flag = true;
                rxInfo = obj.RxInfo;
                rxInfo.Position = obj.Position;
                rxInfo.Velocity = obj.Velocity;
            end
        end
    end

    methods(Hidden)
        function addConnection(obj, connectionConfig)
            %addConnection Add connection context to UE

            obj.NCellID = connectionConfig.NCellID;
            obj.RNTI = connectionConfig.RNTI;
            obj.ConnectionState = "Connected";
            obj.GNBNodeID = connectionConfig.GNBID;
            obj.GNBNodeName = connectionConfig.GNBName;
            obj.ULCarrierFrequency = connectionConfig.ULCarrierFrequency;
            obj.DLCarrierFrequency = connectionConfig.DLCarrierFrequency;
            obj.ReceiveFrequency = connectionConfig.DLCarrierFrequency;
            if isfield(connectionConfig, 'MUMIMOEnabled')
                obj.MUMIMOEnabled = connectionConfig.MUMIMOEnabled;
            end
			
            % Add connection context to MAC
            macConnectionParam = {'RNTI', 'GNBID', 'GNBName' 'NCellID', 'SchedulingType', 'NumHARQ', ...
                'DuplexMode', 'DLULConfigTDD', 'RBGSizeConfiguration', 'CSIRSConfiguration', 'SRSConfiguration', 'NumResourceBlocks', ...
                'BSRPeriodicity', 'CSIReportPeriod', 'InitialCQIDL', 'SubcarrierSpacing'};
            macConnectionInfo = struct();
            for j=1:numel(macConnectionParam)
                macConnectionInfo.(char(macConnectionParam{j})) = connectionConfig.(char(macConnectionParam{j}));
            end
            % Convert the SCS value from Hz to kHz
            subcarrierSpacingInKHZ = connectionConfig.SubcarrierSpacing/1e3;
            macConnectionInfo.SubcarrierSpacing = subcarrierSpacingInKHZ;
            obj.MACEntity.addConnection(macConnectionInfo);

            % Add connection context to PHY
            phyConnectionParam = {'RNTI', 'NCellID', 'DuplexMode', 'NumResourceBlocks', ...
                'NumHARQ', 'ChannelBandwidth', 'PoPUSCH', 'AlphaPUSCH', 'GNBTransmitPower',...
                'DLCarrierFrequency', 'ULCarrierFrequency', ...
                'CSIReportConfiguration', 'SubcarrierSpacing'};
            phyConnectionInfo = struct();
            for j=1:numel(phyConnectionParam)
                phyConnectionInfo.(char(phyConnectionParam{j})) = connectionConfig.(char(phyConnectionParam{j}));
            end
            phyConnectionInfo.SubcarrierSpacing = subcarrierSpacingInKHZ;
            obj.PhyEntity.addConnection(phyConnectionInfo);
            if strcmp(obj.PHYAbstractionMethod, "none")
                obj.PhyEntity.RVSequence = connectionConfig.RVSequence;
            end

            % Add connection context to UE RLC layer using the given full buffer and RLC
            % bearer configurations
            rlcConnectionParam = {'RNTI', 'FullBufferTraffic', 'RLCBearerConfig'};
            % Populate RLC connection info with parameters from connectionConfig
            rlcConnectionInfo = struct();
            for j=1:numel(rlcConnectionParam)
                rlcConnectionInfo.(char(rlcConnectionParam{j})) = connectionConfig.(char(rlcConnectionParam{j}));
            end
            obj.FullBufferTraffic = rlcConnectionInfo.FullBufferTraffic;
            % Add RLC bearer using the RLC connection info
            addRLCBearer(obj, rlcConnectionInfo);
        end

        function updateSRSPeriod(obj, srsPeriod)
            % Update the period of existing SRS configuration

            updateSRSPeriod(obj.MACEntity, srsPeriod);
        end

        function flag = intracellPacketRelevance(obj, packet)
            %intracellPacketRelevance Returns whether the packet is intra-cell and is relevant for the UE

            flag = 1;
            if packet.Type==2 &&  ~obj.MUMIMOEnabled && packet.Metadata.NCellID==obj.NCellID && ...
                    packet.Metadata.PacketType==nrPHY.PXSCHPacketType && ...
                    ~any(packet.Metadata.RNTI == obj.RNTI)
                % If MU-MIMO is disabled then reject any intra-cell PDSCH packet not intended for this UE
                flag = 0;
            end
        end

        function kpiValue = kpi(obj, destinationNode, kpiString, options)
            %kpi Return the key performance indicator (KPI) value for a specified KPI
            %
            %   KPIVALUE = kpi(OBJ, DESTINATIONNODE, KPISTRING, OPTIONS) returns the KPI
            %   value, KPIVALUE, specified by KPISTRING, from the source node(s) represented
            %   by OBJ to the DESTINATIONNODE. The function supports calculations where the
            %   source node can be a vector, allowing for multiple KPI calculations across
            %   different connections. The calculation of the KPI is determined by the
            %   OPTIONS provided.
            %
            %   KPIVALUE        - The calculated value of the specified KPI. If multiple
            %                     source-destination pairs are provided, kpiValue will be a
            %                     row vector containing the KPI value for each pair.
            %
            %   OBJ             - Vector of source node objects from which the KPI is
            %                     measured. Each element in OBJ represents a source node.
            %
            %   DESTINATIONNODE - Instance of the destination node object to which the KPI is measured.
            %
            %   KPISTRING       - Specifies the KPI to be measured. Supported KPIs are
            %                     "latency" and "bler".
            %
            %   OPTIONS         - Structure with a field:
            %                     - Layer: Specifies the layer at which the KPI should be measured.
            %                              Supported layers are "App" and "PHY".

            arguments
                obj (1,:)
                destinationNode (1,1) nrGNB
                kpiString (1,1) string
                options.Layer (1,1) string
            end

            numSources = numel(obj);
            % Validate the given inputs
            kpiString = validateKPIInputs(obj, kpiString, options.Layer);

            % Initialize kpiValue(s). If there are multiple sourceNode-destinationNode
            % connections provided as input, the function will populate the kpiValue(s) in a
            % vector
            kpiValue = zeros(1,numSources);

            % Iterate through all sourceNode-destinationNode connections to obtain the
            % requested KPI
            for kpiIdx = 1:numSources
                % Set a default value if the node is not yet simulated
                if isempty(obj(kpiIdx).LastRunTime)
                    kpiValue(kpiIdx) = 0;
                    continue;
                end

                if strcmp(kpiString,"latency")
                    kpiValue(kpiIdx) = calculateLatency(obj(kpiIdx), destinationNode); % Get uplink latency
                elseif strcmp(kpiString,"bler")
                    kpiValue(kpiIdx) = calculateBLER(obj(kpiIdx), destinationNode); % Get uplink BLER
                end
            end
        end
    end

    methods(Access = protected)
        function flag = isInactiveProperty(obj, prop)
            flag = false;
            switch prop
                case "GNBNodeID"
                    flag = isempty(obj.GNBNodeID);
                case "GNBNodeName"
                    flag = isempty(obj.GNBNodeName);
                case "RNTI"
                    flag = isempty(obj.RNTI);
            end
        end

        function setLayerInterfaces(obj)
            %setLayerInterfaces Set inter-layer interfaces

            % Use weak-references for cross-linking handle objects
            phyWeakRef = matlab.lang.WeakReference(obj.PhyEntity);
            macWeakRef = matlab.lang.WeakReference(obj.MACEntity);
            ueWeakRef = matlab.lang.WeakReference(obj);

            % Register Phy interface functions at MAC for:
            % (1) Sending packets to Phy
            % (2) Sending Rx request to Phy
            % (3) Sending DL control request to Phy
            % (4) Sending UL control request to Phy
            registerPhyInterfaceFcn(obj.MACEntity, ...
                @(varargin) phyWeakRef.Handle.txDataRequest(varargin{:}), ...
                @(varargin) phyWeakRef.Handle.rxDataRequest(varargin{:}), ...
                @(varargin) phyWeakRef.Handle.dlControlRequest(varargin{:}), ...
                @(varargin) phyWeakRef.Handle.ulControlRequest(varargin{:}));

            % Register MAC callback function at Phy for:
            % (1) Sending the packets to MAC
            % (2) Sending the measured DL channel quality to MAC
            registerMACHandle(obj.PhyEntity, ...
                @(varargin) macWeakRef.Handle.rxIndication(varargin{:}), ...
                @(varargin) macWeakRef.Handle.csirsIndication(varargin{:}));

            % Register node callback function at Phy and MAC for:
            % (1) Sending the out-of-band packets from MAC
            % (2) Sending the in-band packets from Phy
            registerOutofBandTxFcn(obj.MACEntity, ...
                @(varargin) ueWeakRef.Handle.addToTxBuffer(varargin{:}));
            registerTxHandle(obj.PhyEntity, ...
                @(varargin) ueWeakRef.Handle.addToTxBuffer(varargin{:}));
        end

        function stats = statisticsPerUE(obj)
            % Return the statistics of UE

            if ~isempty(obj.RNTI) % Check if UE is connected
                stats = struct('ID', obj.ID, 'Name', obj.Name, 'App', [], ...
                    'RLC', [], 'MAC', [], 'PHY', []);
                % Fetch per-layer stats
                stats.App = statistics(obj.TrafficManager);
                stats.RLC = cellfun(@(x) statistics(x), obj.RLCEntity)';
                stats.MAC = statistics(obj.MACEntity);
                stats.PHY = statistics(obj.PhyEntity);

                stats.App = rmfield(stats.App, 'TrafficSources');
                % RLC stats structure
                rlcStat = struct('TransmittedPackets', 0, 'TransmittedBytes', 0, ...
                    'RetransmittedPackets', 0, 'RetransmittedBytes', 0, ...
                    'ReceivedPackets', 0, 'ReceivedBytes', 0, 'DroppedPackets', 0, ...
                    'DroppedBytes', 0);
                % Form RLC stats
                fieldNames = fieldnames(rlcStat);
                for i=1:size(stats.RLC,1)
                    logicalChannelStat = stats.RLC(i);
                    for j=1:numel(fieldNames)
                        % Create cumulative stats
                        rlcStat.(char(fieldNames{j})) = rlcStat.(char(fieldNames{j})) + ...
                            logicalChannelStat.(char(fieldNames{j}));
                    end
                end
                stats.RLC = rlcStat;
            else
                % Create stats structure
                appStat = struct('TransmittedPackets', 0, 'TransmittedBytes', 0, ...
                    'ReceivedPackets', 0, 'ReceivedBytes', 0);
                rlcStat = struct('TransmittedPackets', 0, 'TransmittedBytes', 0, ...
                    'RetransmittedPackets', 0, 'RetransmittedBytes', 0, ...
                    'ReceivedPackets', 0, 'ReceivedBytes', 0, 'DroppedPackets', 0, ...
                    'DroppedBytes', 0);
                macStat = struct('TransmittedPackets', 0, 'TransmittedBytes', 0, ...
                    'ReceivedPackets', 0, 'ReceivedBytes', 0, 'Retransmissions', 0, ...
                    'RetransmissionBytes', 0, 'DLTransmissionRB', 0, 'DLRetransmissionRB', 0, ...
                    'ULTransmissionRB', 0, 'ULRetransmissionRB', 0);
                phyStat = struct('TransmittedPackets', 0, 'ReceivedPackets', 0, ...
                    'DecodeFailures', 0);
                stats = struct('ID', obj.ID, 'Name', obj.Name, 'App', appStat, ...
                    'RLC', rlcStat, 'MAC', macStat, 'PHY', phyStat);
            end
        end
    end

    methods (Access = private)
        function kpiString = validateKPIInputs(~, kpiString, layer)
            %validateKPIInputs Validate the inputs for KPI method

            % Validate the KPI type against the specified layer
            if strcmpi(kpiString, "latency")
                % Latency KPI is only valid at the application layer
                if ~strcmpi(layer, "App")
                    error(message("nr5g:nrUE:InvalidKPIForLayer", "App", kpiString));
                end
                kpiString = "latency";
            elseif strcmpi(kpiString, "bler")
                % BLER KPI is only valid at the PHY layer
                if ~strcmpi(layer, "PHY")
                    error(message("nr5g:nrUE:InvalidKPIForLayer", "PHY", kpiString));
                end
                kpiString = "bler";
            else
                error(message("nr5g:nrUE:InvalidKPI"));
            end
        end

        function latency = calculateLatency(obj, destinationNode)
            %calculateLatency Return the packet latency (in seconds) for the connection
            %between the sourceNode and the destinationNode

            % Initialize latency to a default value of 0
            latency = 0;

            % Access detailed statistics of the destination node's traffic
            dstStats = statistics(destinationNode.TrafficManager, true);

            % Iterate over each destination's statistics to find the matching
            % source node
            for idx = 1:numel(dstStats.Destinations)
                if dstStats.Destinations(idx).NodeID == obj.ID
                    % Update latency with the average packet latency and exit
                    % loop
                    latency = dstStats.Destinations(idx).AveragePacketLatency;
                    return; % Using return instead of break for immediate exit
                end
            end
        end

        function bler = calculateBLER(obj, destinationNode)
            %calculateBLER Return the Block Error Rate (BLER) between the
            %source node and the destination node.

            bler = 0;
            % Access PHY statistics for the destination node
            phyStats = statistics(destinationNode.PhyEntity);

            for idx = 1:size(phyStats,1)
                if obj.ID == phyStats(idx).UEID
                    % Calculate BLER if there are received packets
                    if phyStats(idx).ReceivedPackets > 0
                        bler = phyStats(idx).DecodeFailures / phyStats(idx).ReceivedPackets;
                    end
                    break;
                end
            end
        end
    end
end
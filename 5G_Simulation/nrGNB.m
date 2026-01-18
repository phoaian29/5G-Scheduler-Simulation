classdef nrGNB < wirelessnetwork.internal.nrNode
    %nrGNB 5G NR base station node (gNodeB or gNB)
    %   GNB = nrGNB creates a default 5G New Radio (NR) gNB.
    %
    %   GNB = nrGNB(Name=Value) creates one or more similar gNBs with the
    %   specified property Name set to the specified Value. You can specify
    %   additional name-value arguments in any order as (Name1=Value1, ...,
    %   NameN=ValueN). The number of rows in 'Position' argument defines the
    %   number of gNBs created. 'Position' must be an N-by-3 matrix where
    %   N(>=1) is the number of gNBs, and each row must contain three numeric
    %   values representing the [X, Y, Z] position of a gNB in meters. The
    %   output, GNB is a row-vector of gNB objects containing N gNBs. You can also
    %   supply multiple names for 'Name' argument corresponding to number of
    %   gNBs created. Multiple names must be supplied either as a vector of
    %   strings or cell array of character vectors. If name is not set then a
    %   default name as 'NodeX' is given to node, where 'X' is ID of the node.
    %   Assuming 'N' nodes are created and 'M' names are supplied, if (M>N)
    %   then trailing (M-N) names are ignored, and if (N>M) then trailing (N-M)
    %   nodes are set to default names. You can set the "Position" and "Name"
    %   properties corresponding to the multiple gNBs simultaneously when you
    %   specify them as N-V arguments in the constructor. After the node
    %   creation, you can set the "Position" and "Name" property for only one
    %   gNB object at a time.
    %
    %   nrGNB properties (configurable through N-V pair as well as public settable):
    %
    %   Name                 - Node name
    %   Position             - Node position
    %
    %   nrGNB properties (configurable through N-V pair only):
    %
    %   DuplexMode           - Duplexing mode as FDD or TDD
    %   CarrierFrequency     - Carrier frequency at which gNB is operating
    %   ChannelBandwidth     - Bandwidth of the carrier gNB is serving
    %   SubcarrierSpacing    - Subcarrier spacing used across the cell
    %   NumResourceBlocks    - Number of resource blocks in channel bandwidth
    %   NumTransmitAntennas  - Number of transmit antennas
    %   NumReceiveAntennas   - Number of receive antennas
    %   TransmitPower        - Transmit power in dBm
    %   PHYAbstractionMethod - Physical layer (PHY) abstraction method
    %   DLULConfigTDD        - Downlink (DL) and uplink (UL) TDD configuration
    %   NoiseFigure          - Noise figure in dB
    %   ReceiveGain          - Receiver antenna gain in dB
    %   NumHARQ              - Number of hybrid automatic repeat request (HARQ)
    %                          processes for each user equipment (UE) which
    %                          connects to the gNB
    %   SRSPeriodicityUE     - SRS transmission periodicity for the connected UEs
    %
    %   nrGNB properties (read-only):
    %
    %   ID                   - Node identifier
    %   ConnectedUEs         - Radio network temporary identifier (RNTI) of the
    %                          connected UEs
    %   UENodeIDs            - ID of the connected UEs
    %   UENodeNames          - Name of the connected UEs
    %
    %   Constant properties:
    %
    %   MCSTable             - MCS table used for DL and UL as per 3GPP TS
    %                          38.214 - Table 5.1.3.1-2
    %
    %   nrGNB methods:
    %
    %   connectUE               - Connect UE(s) to the gNB
    %   addTrafficSource        - Add data traffic source to the gNB
    %   statistics              - Get statistics of the gNB
    %   configureScheduler      - Configure scheduler at the gNB
    %   configureULPowerControl - Configure uplink power control at gNB
    %
    %   The scheduler at the gNB assigns the resources based on the configured
    %   scheduling strategy. To configure a scheduler strategy, use the
    %   <a href="matlab:help('nrGNB.configureScheduler')">configureScheduler</a> function. For DL channel measurements,
    %   the gNB transmits full bandwidth channel state information reference
    %   signal (CSI-RS) which all the UEs in the cell use. For frequency
    %   division duplex (FDD) mode, the periodicity for CSI-RS transmission is
    %   10 slots. For time division duplex (TDD) mode, the periodicity is 'M'
    %   slots. 'M' is the smallest integer which is greater than or equal to 10
    %   and a multiple of the length of the DL-UL pattern (in slots). You can
    %   set different CSI reporting periodicity for the UEs using
    %   'CSIReportPeriodicity' parameter of <a href="matlab:help('nrGNB.connectUE')">connectUE</a> method of this class.
    %   For UL channel measurements, gNB reserves one symbol for the sounding
    %   reference signal (SRS) periodically, across the entire bandwidth. For
    %   FDD, one SRS symbol is reserved every 5 slots. For TDD, one SRS symbol
    %   is reserved every 'N' slots. 'N' is the smallest integer which is
    %   greater than or equal to 5 and a multiple of the length of the DL-UL
    %   pattern (in slots). The gNB configures the UEs to share the reservation
    %   of SRS bandwidth by differing the comb offset, cyclic shift, or
    %   transmission time. Comb size and maximum cyclic shift are both assumed
    %   as 4. You can set different SRS transmission periodicity for the UEs
    %   using 'SRSPeriodicityUE' N-V argument of <a href="matlab:help('nrGNB.SRSPeriodicityUE')">nrGNB</a> class.
    %
    %   % Example 1:
    %   %  Create two gNBs with name as "gNB1" and "gNB2" positioned at
    %   %  [100 100 0] and [5000 100 0], respectively.
    %   gNBs = nrGNB(Name=["gNB1" "gNB2"], Position=[100 100 0; 5000 100 0])
    %
    %   % Example 2:
    %   %  Create a gNB serving a 20 MHz TDD carrier (or cell). Specify SCS as
    %   %  30e3 Hz with following DL-UL configuration:
    %   %  Periodicity of DL-UL pattern = 2 milliseconds
    %   %  Number of full DL slots = 2
    %   %  Number of DL symbols = 12
    %   %  Number of UL symbols = 0
    %   %  Number of full UL slots = 1
    %   tddConfig = struct('DLULPeriodicity', 2, ...
    %               'NumDLSlots', 2, 'NumDLSymbols', 12, 'NumULSymbols', 0, ...
    %               'NumULSlots', 1)
    %   gNB = nrGNB(ChannelBandwidth=20e6, DuplexMode="TDD", ...
    %   SubcarrierSpacing=30e3, NumResourceBlocks=51, DLULConfigTDD=tddConfig)
    %
    %   % Example 3:
    %   %  Create a gNB and configure its scheduler with these parameters:
    %   %  Resource allocation type = 0
    %   %  Maximum number of scheduled users per TTI = 4
    %   %  Fixed MCS index for DL resource allocation = 10
    %   %  Fixed MCS index for UL resource allocation = 10
    %   gNB = nrGNB();
    %   configureScheduler(gNB, ResourceAllocationType=0, ...
    %       MaxNumUsersPerTTI=4, FixedMCSIndexDL=10, FixedMCSIndexUL=10);
    %
    %   See also nrUE.

    %   Copyright 2022-2024 The MathWorks, Inc.

    properties (SetAccess = private)
        %NoiseFigure Noise figure in dB
        %   Specify the noise figure in dB. The default value is 6.
        NoiseFigure(1,1) {mustBeNumeric, mustBeFinite, mustBeNonnegative} = 6;

        %ReceiveGain Receiver gain in dB
        %   Specify the receiver gain in dB. The default value is 6.
        ReceiveGain(1,1) {mustBeNumeric, mustBeFinite, mustBeNonnegative} = 6;

        %TransmitPower Transmit power of gNB in dBm
        %   Specify the transmit power of gNB. Units are in dBm.
        %   The default value is 34.
        TransmitPower (1,1) {mustBeNumeric, mustBeGreaterThanOrEqual(TransmitPower, -202), mustBeLessThanOrEqual(TransmitPower, 60)} = 34

        %NumTransmitAntennas Number of transmit antennas on gNB
        %   Specify the number of transmit antennas on gNB. The allowed values are
        %   1, 2, 4, 8, 16, 32. The default value is 1.
        NumTransmitAntennas (1, 1) {mustBeMember(NumTransmitAntennas, [1 2 4 8 16 32])} = 1;

        %NumReceiveAntennas Number of receive antennas on gNB
        %   Specify the number of receive antennas on gNB. The allowed values are
        %   1, 2, 4, 8, 16, 32. The default value is 1.
        NumReceiveAntennas (1, 1) {mustBeMember(NumReceiveAntennas, [1 2 4 8 16 32])} = 1;

        %PHYAbstractionMethod PHY abstraction method
        %   Specify the PHY abstraction method as "linkToSystemMapping" or "none".
        %   The value "linkToSystemMapping" represents link-to-system-mapping based
        %   abstract PHY. The value "none" represents full PHY processing. The default
        %   value is "linkToSystemMapping".
        PHYAbstractionMethod = "linkToSystemMapping";

        %DuplexMode Duplex mode as FDD or TDD
        %   Specify the duplex mode either as frequency division duplexing
        %   (FDD) or time division duplexing (TDD). The allowed values
        %   are "FDD" or "TDD". The default value is "FDD".
        DuplexMode {mustBeNonempty, mustBeTextScalar} = "FDD";

        %CarrierFrequency Frequency of the carrier served by gNB in Hz
        %   Specify the carrier frequency in Hz. The default value is 2.6e9 Hz.
        CarrierFrequency (1,1) {mustBeNumeric, mustBeFinite, mustBeGreaterThanOrEqual(CarrierFrequency, 600e6)} = 2.6e9;

        %ChannelBandwidth Bandwidth of the carrier served by gNB in Hz
        %   Specify the carrier bandwidth in Hz. In FDD mode, each of the
        %   DL and UL operations happen in separate bands of this size. In
        %   TDD mode, both DL and UL share single band of this size. The
        %   default value is 5e6 Hz.
        ChannelBandwidth (1,1) {mustBeMember(ChannelBandwidth, ...
            [5e6 10e6 15e6 20e6 25e6 30e6 35e6 40e6 45e6 50e6 60e6 70e6 80e6 90e6 100e6 200e6 400e6])} = 5e6;

        %SubcarrierSpacing Subcarrier spacing (SCS) used across the cell
        %   Specify the subcarrier spacing for the cell in Hz. All the UE(s)
        %   connecting to the gNB operates in this SCS. The allowed values are
        %   15e3, 30e3, 60e3 and 120e3. The default value is 15e3.
        SubcarrierSpacing (1, 1) {mustBeMember(SubcarrierSpacing, [15e3 30e3 60e3 120e3])} = 15e3;

        %NumResourceBlocks Number of resource blocks in carrier bandwidth
        %   Specify the number of resource blocks in carrier bandwidth. In FDD
        %   mode, each of the DL and UL bandwidth contains these many resource
        %   blocks. In TDD mode, both DL and UL bandwidth share these resource
        %   blocks. If you do not set this value then the gNB derives it
        %   automatically from channel bandwidth and subcarrier spacing. The
        %   default value is 25 which corresponds to the default 5e6 Hz channel
        %   bandwidth and 15e3 Hz SCS. The minimum value is 4 which is the minimum
        %   required transmission bandwidth for SRS as per 3GPP TS 38.211 Table
        %   6.4.1.4.3-1.
        NumResourceBlocks (1,1) {mustBeInteger, mustBeFinite, mustBeGreaterThanOrEqual(NumResourceBlocks, 4)} = 25;

        %DLULConfigTDD Downlink and uplink time configuration (relevant only for TDD mode)
        %   Specify the DL and UL time configuration for TDD mode. Set this
        %   property only if you have set the 'DuplexMode' as "TDD", otherwise
        %   the set value is ignored. This property corresponds to
        %   tdd-UL-DL-ConfigurationCommon parameter as described in Section 11.1 of
        %   3GPP TS 38.213. Specify it as a structure with following fields:
        %   DLULPeriodicity    - DL-UL pattern periodicity in milliseconds
        %   NumDLSlots         - Number of full DL slots at the start of DL-UL pattern
        %   NumDLSymbols       - Number of DL symbols after full DL slots
        %   NumULSymbols       - Number of UL symbols before full UL slots
        %   NumULSlots         - Number of full UL slots at the end of DL-UL pattern
        %
        %   The reference subcarrier spacing for DL-UL pattern is assumed
        %   to be same as 'SubcarrierSpacing' property of this class. The
        %   configuration supports one 'S' slot after full DL slots and
        %   before full UL slots. The 'S' slot comprises of 'NumDLSymbols'
        %   at the start and 'NumULSymbols' at the end. The symbol count
        %   '14 - (NumDLSymbols + NumULSymbols)' is assumed to be guard
        %   period between DL and UL time. 'NumULSymbols' can be set to 0
        %   or 1. If set to 1 then this UL symbol is utilized for sounding
        %   reference signal (SRS). The default values for the structure
        %   fields are: DLULPeriodicity = 5, NumDLSlots = 2, NumDLSymbols =
        %   12, NumULSymbols = 1, NumULSlots = 2. The default value
        %   corresponds to 15e3 Hz SCS. If you specify SCS as 30e3 Hz, 60e3
        %   Hz, or 120e3 Hz then the default value of DLULPeriodicity field
        %   becomes 2.5 milliseconds, 1.25 milliseconds, 0.625
        %   milliseconds, respectively.
        DLULConfigTDD = struct('DLULPeriodicity', 5, 'NumDLSlots', 2, ...
            'NumDLSymbols', 12, 'NumULSymbols', 1, 'NumULSlots', 2);

        %NumHARQ Number of HARQ processes used for each UE in DL and UL direction
        %   Specify the number of HARQ processes used for each UE in DL and
        %   UL direction. The default value is 16.
        NumHARQ(1, 1) {mustBeInteger, mustBeInRange(NumHARQ, 1, 16)} = 16;

        %ULPowerControlParameters Specify the uplink power control configuration parameters
        %   This property is used to configure the uplink power control.
        %   This property corresponds to parameter PUSCH-PowerControl described
        %   in 3GPP TS 38.213 section 7.1. It is a structure containing the following fields
        %   PoPUSCH     - Nominal transmit power of the UE in dBm per resource block
        %   AlphaPUSCH  - Fractional power control multiplier of an UE specified at gNB
        %
        %   The range of PoPUSCH is [-202, 24]. Allowed values of AlphaPUSCH are
        %   0 0.4 0.5 0.6 0.7 0.8 0.9, or 1. The default values for the structure
        %   fields are PoPUSCH = -60 dBm, AlphaPUSCH = 1 which corresponds to
        %   conventional power control scheme which will  maintain a constant
        %   signal to interference and noise ratio (SINR) at the receiver.
        ULPowerControlParameters = struct('PoPUSCH', -60, 'AlphaPUSCH', 1);

        %SRSPeriodicityUE SRS transmission periodicity for the connected UEs
        %   Specify the SRS transmission periodicity for a UE node, which must be
        %   the same for all connected UE nodes and can be one of the following
        %   values: 5, 8, 10, 16, 20, 32, 40, 64, 80, 160, 320, 640, 1280, or 2560
        %   slots. This periodicity must also be an integer multiple of L, where L
        %   is the interval in slots at which the gNB reserves one symbol for the
        %   SRS resource across the entire bandwidth. The minimum value of L is 5.
        %   For FDD, the nrGNB object fixes the value of L at 5. For TDD, L must be
        %   a multiple of the DL-UL pattern length, and it must be one of these
        %   values: 5, 8, 10, 16, 20, 32, 40, 64, 80, 160, 320, 640, 1280, or 2560
        %   slots.
        SRSPeriodicityUE(1, 1) {mustBeMember(SRSPeriodicityUE, ...
            [5 8 10 16 20 32 40 64 80 160 320 640 1280 2560])} = 5;
    end

    properties(SetAccess = protected)
        %ConnectedUEs RNTI of the UEs connected to the gNB, returned as vector of integers
        ConnectedUEs;

        %UENodeIDs ID of the UEs connected to the gNB, returned as vector of integers
        UENodeIDs;

        %UENodeNames Name of the UEs connected to the gNB, returned as vector of strings
        UENodeNames = strings(0,1);
    end

    properties(SetAccess = protected, Hidden)
        %NCellID Physical layer cell identity of the carrier gNB is serving
        NCellID;
    end

    properties(Hidden)
        %GuardBand Gap between DL and UL bands in Hz (Only valid for FDD)
        GuardBand = 140e6;

        %SRSReservedResource SRS reservation occurrence period as [symbolNumber slotPeriodicity slotOffset]
        SRSReservedResource;

        %SRSConfiguration Row vector of SRS configuration of all the connected UEs.
        % Vector containing the SRS configuration. Each element is an object of type
        % nrSRSConfig. It is a vector of length equal to number of UEs connected to
        % the gNB. An element at index 'i' stores the SRS configuration of UE with
        % RNTI 'i'.
        SRSConfiguration;

        %CSIRSConfiguration Default CSI-RS configuration for the UEs (if not configured for the UE)
        CSIRSConfiguration;

        %CQITable CQI table (TS 38.214 - Table 5.2.2.1-3) used for channel quality measurements
        CQITable = 'table2';

        %InitialMCSIndexUL Initial MCS index for UL
        InitialMCSIndexUL = 0;

        %InitialMCSIndexUL Initial MCS index for DL
        InitialMCSIndexDL = 0;
    end

    properties(SetAccess = protected, Hidden)
        %CSIReportType CSI report type that takes values 1 and 2 to indicate type-I
        %and type-II, respectively
        CSIReportType = 2;
    end

    properties(Access = protected)
        %SchedulerDefaultConfig Flag, specified as true or false, indicating the
        %scheduler with a default or custom configuration. A flag value of true
        %indicates that the scheduler has a default configuration, and a flag value
        %of false indicates that the scheduler has a custom configuration.
        SchedulerDefaultConfig = true;

        %CSIMeasurementSignalDLType The value of "CSIMeasurementSignalDLType" is 1 if the
        % specified value of CSIMeasurementSignalDL is 'SRS'.  It is 0 if the specified
        % value of "CSIMeasurementSignalDL" is 'CSI-RS'.
        CSIMeasurementSignalDLType = 0;

        %RVSequence RV sequence
        RVSequence = [0 3 2 1];
    end

    events(Hidden)
        %ScheduledResources Event of resource scheduling
        %   This event is triggered when scheduler runs to schedule
        %   resources. It passes the event notification along with
        %   structure containing these fields to the registered callback:
        %   CurrentTime    - Current simulation time in seconds
        %   TimingInfo     - Timing information as vector of 3 elements of the form
        %                    [NFrame NSlot NSymbol]
        %   DLGrants       - Structure vector containing scheduled downlink
        %                    assignments. Each structure element has these fields:
        %                    RNTI,Type,HARQID,ResourceAllocationType,FrequencyAllocation,
        %                    StartSymbol,NumSymbols,SlotOffset,MCSIndex,NDI,
        %                    DMRSLength,MappingType,NumLayers,NumCDMGroupsWithoutData,
        %                    W,FeedbackSlotOffset,RV,PRBSet.
        %   ULGrants       - Structure vector containing scheduled uplink
        %                    grants. Each structure element has these fields:
        %                    RNTI,Type,HARQID,ResourceAllocationType,FrequencyAllocation,
        %                    StartSymbol,NumSymbols,SlotOffset,MCSIndex,NDI,
        %                    DMRSLength,MappingType,NumLayers,NumCDMGroupsWithoutData,
        %                    NumAntennaPorts,TPMI,RV,PRBSet.
        ScheduledResources;
    end

    properties(Constant)
        %MCSTable MCS table to be used for DL and UL as per 3GPP TS 38.214 - Table 5.1.3.1-2
        MCSTable = nrGNB.getMCSTable;
    end

    % Constant, hidden properties
    properties (Constant,Hidden)
        DuplexMode_Values  = ["FDD","TDD"];

        SubcarrierSpacing_Values = ["15000","30000","60000","120000"];

        %MinSRSResourcePeriodicity Minimum SRS resource occurrence periodicity (in slots)
        MinSRSResourcePeriodicity = 5;
    end

    methods
        function obj = nrGNB(varargin)

            % Name-value pair check
            coder.internal.errorIf(mod(nargin, 2) == 1,'MATLAB:system:invalidPVPairs');

            if nargin > 0
                % Validate inputs
                param = nrNodeValidation.validateGNBInputs(obj, varargin);
                names = param(1:2:end);
                % Search the presence of 'Position' N-V argument to
                % calculate the number of gNBs user intends to create
                positionIdx = find(strcmp([names{:}], 'Position'), 1, 'last');
                numGNBs = 1;
                if ~isempty(positionIdx)
                    position = param{2*positionIdx}; % Read value of Position N-V argument
                    validateattributes(position, {'numeric'}, {'nonempty', 'ncols', 3, 'finite'}, mfilename, 'Position');
                    if ismatrix(position)
                        numGNBs = size(position, 1);
                    end
                end

                % Search the presence of 'Name' N-V pair argument
                nameIdx = find(strcmp([names{:}], 'Name'), 1, 'last');
                if ~isempty(nameIdx)
                    nodeName = param{2*nameIdx}; % Read value of Position N-V argument
                end

                % Create gNB(s)
                obj(1:numGNBs) = obj;
                className = class(obj(1));
                classFunc = str2func(className);
                for i=2:numGNBs
                    % To support vectorization when inheriting "nrGNB", instantiate
                    % class based on the object's class
                    obj(i) = classFunc();
                end

                % Set the configuration of gNB(s) as per the N-V pairs
                numArgs = numel(param);
                for i=1:2:numArgs-1
                    name = param{i};
                    value = param{i+1};
                    switch (name)
                        case 'Position'
                            % Set position for gNB(s)
                            for j = 1:numGNBs
                                obj(j).Position = position(j, :);
                            end
                        case 'Name'
                            % Set name for gNB(s). If name is not supplied for all gNBs then leave the
                            % trailing gNBs with default names
                            nameCount = min(numel(nodeName), numGNBs);
                            for j=1:nameCount
                                obj(j).Name = nodeName(j);
                            end
                        otherwise
                            % Make all the gNBs identical by setting same value for all the
                            % configurable properties, except position and name
                            [obj.(char(name))] = deal(value);
                    end
                end
            end

            coder.internal.errorIf(any([obj.SubcarrierSpacing] > 30e3) && any(strcmp([obj.PHYAbstractionMethod], 'none')),...
                'nr5g:nrGNB:InvalidSCSWithPhyAbstractionMethod', "gNB")
            % Set physical cell ID same as node ID
            [obj.NCellID] = deal(obj.ID);

            if strcmp(obj(1).DuplexMode, "FDD") % FDD
                % UL band starts guardBand*0.5 Hz below the carrier frequency and it is
                % channelBandwidth Hz wide. UL carrier frequency is calculated as center of
                % UL band
                [obj.ULCarrierFrequency] = deal(obj(1).CarrierFrequency-(obj(1).GuardBand/2)-(obj(1).ChannelBandwidth/2));
                % DL band starts guardBand*0.5 Hz above the carrier frequency and is
                % channelBandwidth Hz wide. DL carrier frequency is calculated as center of
                % DL band
                [obj.DLCarrierFrequency] = deal(obj(1).CarrierFrequency+(obj(1).GuardBand/2)+(obj(1).ChannelBandwidth/2));
            else % TDD
                [obj.ULCarrierFrequency] = deal(obj(1).CarrierFrequency);
                [obj.DLCarrierFrequency] = deal(obj(1).CarrierFrequency);
            end
            % Create internal layers for each gNB
            macParam = ["NCellID", "NumHARQ", "SubcarrierSpacing", ...
                "NumResourceBlocks", "DuplexMode","DLULConfigTDD"];
            phyParam = ["NCellID", "DuplexMode", "ChannelBandwidth", "DLCarrierFrequency", ...
                "ULCarrierFrequency", "NumResourceBlocks", "TransmitPower", ...
                "NumTransmitAntennas", "NumReceiveAntennas", "NoiseFigure", ...
                "ReceiveGain", "Position", "SubcarrierSpacing", "CQITable", "MCSTable"];
            for idx=1:numel(obj) % For each gNB
                gNB = obj(idx);

                % Use weak-references for cross-linking handle objects
                gnbWeakRef = matlab.lang.WeakReference(gNB);

                % Set up traffic manager
                gNB.TrafficManager = wirelessnetwork.internal.trafficManager(gNB.ID, ...
                    [], @(varargin) gnbWeakRef.Handle.processEvents(varargin{:}), DataAbstraction=false, ...
                    PacketContext=struct('DestinationNodeID', 0, 'LogicalChannelID', 4, 'RNTI', 0));

                % Set up MAC
                macInfo = struct();
                for j=1:numel(macParam)
                    macInfo.(macParam(j)) = gNB.(macParam(j));
                end
                % Convert the SCS value from Hz to kHz
                subcarrierSpacingInKHZ = gNB.SubcarrierSpacing/1e3;
                macInfo.SubcarrierSpacing = subcarrierSpacingInKHZ;
                gNB.MACEntity = nrGNBMAC(macInfo, ...
                    @(varargin) gnbWeakRef.Handle.processEvents(varargin{:}));

                % Identify SRS resource occurrence. Scheduler is conveyed SRS resource
                % occurrence periodicity so that it can reserve SRS symbols to exclude them
                % for data scheduling
                gNB.SRSReservedResource = nrConfigureSRSReservedResource(gNB);

                % Create default CSI-RS configuration for the UEs (if not configured for the UE)
                gNB.CSIRSConfiguration = gNB.createCSIRSConfiguration();

                % Set up PHY
                phyInfo = struct();
                for j=1:numel(phyParam)
                    phyInfo.(phyParam(j)) = gNB.(phyParam(j));
                end
                phyInfo.SubcarrierSpacing = subcarrierSpacingInKHZ;
                if strcmp(gNB.PHYAbstractionMethod, "none")
                    gNB.PhyEntity = nrGNBFullPHY(phyInfo, @(varargin) gnbWeakRef.Handle.processEvents(varargin{:})); % Full PHY
                    gNB.PHYAbstraction = 0;
                else
                    gNB.PhyEntity = nrGNBAbstractPHY(phyInfo, @(varargin) gnbWeakRef.Handle.processEvents(varargin{:})); % Abstract PHY
                    gNB.PHYAbstraction = 1;
                end

                % Set up default scheduler
                configureScheduler(gNB);
                gNB.SchedulerDefaultConfig = true;

                % Set inter-layer interfaces
                gNB.setLayerInterfaces();
                gNB.ReceiveFrequency = gNB.ULCarrierFrequency;

                % Set Rx Info to be returned with packet relevance check
                gNB.RxInfo.ID = gNB.ID;
                gNB.RxInfo.Position = gNB.Position;
                gNB.RxInfo.Velocity = gNB.Velocity;
                gNB.RxInfo.NumReceiveAntennas = gNB.NumReceiveAntennas;
            end
        end

        function connectUE(obj, UE, varargin)
            %connectUE Connect one or more UEs to the gNB
            %
            %   connectUE(OBJ, UE, Name=Value) connects one or more UEs to gNB as per
            %   the connection configuration parameters specified in name-value
            %   arguments. UE is a row-vector of objects of type <a
            %   href="matlab:help('nrUE')">nrUE</a> and represents one or more UEs
            %   getting connected to gNB. You can set connection parameter using
            %   name-value arguments in any order as (Name1=Value1,...,NameN=ValueN).
            %   When a name-value argument corresponding to a connection parameter is
            %   not specified, the method uses a default value for it. All the nodes in
            %   the object row-vector, UE, connect using same specified value of the connection
            %   parameter. Use these name-value arguments to set connection parameters.
            %
            %   BSRPeriodicity       - UL buffer status reporting periodicity in
            %                          terms of the number of subframes (1 subframe
            %                          is 1 millisecond). The default value is 5.
            %
            %   CSIReportPeriodicity - CSI-RS reporting periodicity in terms of
            %                          the number of slots. UE reports rank indicator
            %                          (RI), precoding matrix indicator (PMI), and CQI
            %                          based on the measurements done on configured
            %                          CSI-RS. Specify this parameter as a value
            %                          greater than or equal to CSI-RS transmission
            %                          periodicity. For TDD, this parameter must also
            %                          be a multiple of length of DL-UL pattern in
            %                          slots. The default value for reporting
            %                          periodicity is same as the CSI-RS transmission
            %                          periodicity.
            %
            %   FullBufferTraffic    - Enable full buffer traffic in DL and/or UL
            %                          direction for the UE. Possible values: "off",
            %                          "on", "DL" and "UL". Value "on" configures full
            %                          buffer traffic for both DL and UL direction.
            %                          Value "DL" configures full buffer traffic only
            %                          for DL direction. Value "UL" configures full
            %                          buffer traffic only for UL direction. Default
            %                          value is "off" which means that full buffer
            %                          traffic is disabled in DL and UL direction. Use
            %                          this configuration parameter as an alternative
            %                          to <a
            %                          href="matlab:help('nrGNB.addTrafficSource')">addTrafficSource</a> for easily setting up traffic
            %                          during connection configuration itself.
            %
            %   RLCBearerConfig      - RLC bearer configuration, specified as
            %                          an <a
            %                          href="matlab:help('nrRLCBearerConfig')">nrRLCBearerConfig</a> object or a vector of
            %                          <a href="matlab:help('nrRLCBearerConfig')">nrRLCBearerConfig</a> objects. Use this option when
            %                          full buffer is not enabled. If you enable
            %                          the full buffer on the DL or UL direction,
            %                          the object ignores this value. If you do not
            %                          enable the full buffer and you do not specify
            %                          this value, the object uses a default RLC
            %                          bearer configuration.
            %
            %   CustomContext        - Custom UE context, specified as a structure. Use
            %                          this name-value argument to specify any custom
            %                          information regarding a UE node, which you can
            %                          then utilize for custom scheduling. The
            %                          CustomContext property of UEContext, a property
            %                          of the nrScheduler class, reflects the specified
            %                          value for the corresponding UE node. The
            %                          connectUE call, invokable for each UE node,
            %                          utilizes the CustomContext name-value argument
            %                          to provide each UE with a unique context
            %                          structure, enabling variation in fields or
            %                          identical fields with distinct values.
            %
            %   CSIRSConfig          - Channel State Information Reference Signal(CSI-RS) configuration for the UE,
            %                          specified as an object of type <a
            %                          href="matlab:help('nrCSIRSConfig')">nrCSIRSConfig</a>.
            %                          When you specify this argument, the function will use the specified
            %                          value to configure the UE node. If not provided, the gNB node assigns
            %                          a default CSI-RS configuration to the UE node. To disable the CSI-RS
            %                          configuration, set this argument to empty ([]).
            %
            %                          Configurable fields in 'nrCSIRSConfig' object:
            %                          - CSIRSPeriod: [periodicity, offset]
            %                            - The periodicity value must be 4, 5, 8, 10, 16, 20, 32, 40,
            %                              64, 80, 160, 320, or 640, and the offset must be a nonnegative integer.
            %                            - Additionally for TDD, the periodicity must also be a multiple of the number of slots
            %                              in the DL-UL pattern as specified in the DLULConfigTDD property of nrGNB.
            %                          - Offset must be a positive integer.
            %                            - For TDD, Offset value must result in DL slot or special slot in the TDD DL-UL pattern.
            %                              - DL slots: Slots dedicated to downlink transmission.
            %                              - Special slot: A slot that can contains both DL and UL symbols.
            %                          - RowNumber: Row number depends on the number of transmit antennas on the gNB node.
            %                            - 1 Tx antenna: 1 or 2
            %                            - 2 Tx antennas: 3
            %                            - 4 Tx antennas: 4 or 5
            %                            - 8 Tx antennas: 6, 7, or 8
            %                            - 16 Tx antennas: 11 or 12
            %                            - 32 Tx antennas: 16, 17, or 18
            %                          - Density: Frequency density of CSI-RS resource, specified as "one" (default),
            %                            "three", "dot5even", or "dot5odd"
            %
            %                          The gNB node automatically sets the other fields of 'nrCSIRSConfig' object.

            % First argument must be scalar object
            validateattributes(obj, {'nrGNB'}, {'scalar'}, mfilename, 'obj');
            validateattributes(UE, {'nrUE'}, {'vector'}, mfilename, 'UE');

            coder.internal.errorIf(~isempty(obj.LastRunTime), 'nr5g:nrNode:NotSupportedOperation', 'ConnectUE');

            % Name-value pair check
            coder.internal.errorIf(mod(nargin-2, 2) == 1, 'MATLAB:system:invalidPVPairs');
            numUEs = size(UE,2);
            srsResourcePeriodicity = obj.SRSReservedResource(2);
            % Maximum number of the connected UEs with the default SRS periodicity is 16 i.e., ktc(4)*ncsMax(4)
            maxUEWithSRSPeriodicity = 16*(obj.SRSPeriodicityUE/srsResourcePeriodicity);

            validSRSPeriodicity = [5 8 10 16 20 32 40 64 80 160 320 640 1280 2560];
            totalConnectedUEs = size(obj.ConnectedUEs,2)+numUEs;
            % Calculate the minimum SRS transmission periodicity for the connected UEs
            minSRSPeriodicityForGivenUEs = ceil(totalConnectedUEs/16)*srsResourcePeriodicity;
            % Calculate the set of SRS transmission periodicity which is a multiple of
            % SRS resource periodicity and valid for the given number of connected UEs
            validSet = validSRSPeriodicity(validSRSPeriodicity>=minSRSPeriodicityForGivenUEs & ~mod(validSRSPeriodicity,srsResourcePeriodicity));
            if totalConnectedUEs > maxUEWithSRSPeriodicity
                % SRS periodicity must be one of the elements in the validSet
                messageString = ".";
                if ~isempty(validSet)
                    formattedValidSRSSetStr = [sprintf('{') (sprintf(repmat('%d, ', 1, length(validSet)-1)', validSet(1:end-1) )) sprintf('%d}', validSet(end))];
                    messageString = " or increase the SRS periodicity to one of these values: " + formattedValidSRSSetStr + ".";
                end
                coder.internal.error('nr5g:nrGNB:InvalidNumUEWithSRSPeriodicityUE',maxUEWithSRSPeriodicity,obj.SRSPeriodicityUE,messageString);
            end

            connectionConfigStruct = struct('RNTI', 0, 'GNBID', obj.ID, 'GNBName', ...
                obj.Name, 'UEID', 0, 'UEName', [], 'NCellID', obj.NCellID, ...
                'SubcarrierSpacing', obj.SubcarrierSpacing, 'SchedulingType', ...
                0, 'NumHARQ', obj.NumHARQ, 'DuplexMode', obj.DuplexMode, ...
                'CSIReportConfiguration', [],  'SRSConfiguration', ...
                [], 'SRSSubbandSize', [], 'NumResourceBlocks', obj.NumResourceBlocks, ...
                'ChannelBandwidth', obj.ChannelBandwidth, 'DLCarrierFrequency', ...
                obj.DLCarrierFrequency, 'ULCarrierFrequency', obj.ULCarrierFrequency, ...
                'BSRPeriodicity', 5, 'CSIReportPeriodicity', [], 'CSIReportPeriodicityRSRP', ...
                1, 'RBGSizeConfiguration', 1, 'DLULConfigTDD', obj.DLULConfigTDD, ...
                'NumTransmitAntennas', 1, 'NumReceiveAntennas', 1,  'InitialMCSIndexDL', obj.InitialMCSIndexDL,...
                'PoPUSCH', obj.ULPowerControlParameters.PoPUSCH, 'AlphaPUSCH', obj.ULPowerControlParameters.AlphaPUSCH,...
                'GNBTransmitPower', [], 'CSIMeasurementSignalDLType', obj.CSIMeasurementSignalDLType,  "MUMIMOEnabled", obj.MUMIMOEnabled, ...
                'InitialMCSIndexUL', obj.InitialMCSIndexUL, 'InitialCQIDL', [], 'FullBufferTraffic', "off", ...
                'RLCBearerConfig', [], 'RVSequence', obj.RVSequence, 'CustomContext', struct(), 'CSIReportPeriod', []);

            % Initialize connection configuration array for UEs
            connectionConfigList = repmat(connectionConfigStruct, numUEs, 1);

            % Form array of connection configuration (1 for each UE)
            for idx=1:2:nargin-2
                name = varargin{idx};
                value = nrNodeValidation.validateConnectUEInputs(name, varargin{idx+1});
                % Set same value per connection
                [connectionConfigList(:).(char(name))] = deal(value);
            end

            % Information to configure connection information at gNB MAC
            macConnectionParam = ["RNTI", "UEID", "UEName", "CSIRSConfiguration", "SRSConfiguration"];
            % Information to configure connection information at gNB PHY
            phyConnectionParam = ["RNTI", "UEID", "UEName", "SRSSubbandSize", "NumHARQ", "DuplexMode", "CSIMeasurementSignalDLType"];
            % Information to configure connection information at gNB scheduler
            schedulerConnectionParam = ["RNTI", "UEID", "UEName", "NumTransmitAntennas", "NumReceiveAntennas", ...
                "SRSConfiguration", "CSIRSConfiguration", "CSIReportConfiguration", "SRSSubbandSize", ...
                "InitialCQIDL", "InitialMCSIndexUL", "CustomContext", "RLCBearerConfig"];
            % Information to configure connection information at gNB RLC
            rlcConnectionParam = ["RNTI", "FullBufferTraffic", "RLCBearerConfig"];

            % Lookup table for valid RowNumbers based on NumTransmitAntennas
            rowNumberLookup = containers.Map('KeyType', 'double', 'ValueType', 'any');
            rowNumberLookup(1) = [1, 2];
            rowNumberLookup(2) = 3;
            rowNumberLookup(4) = [4, 5];
            rowNumberLookup(8) = [6, 7, 8];
            rowNumberLookup(16) = [11, 12];
            rowNumberLookup(32) = [16, 17, 18];

            % Initialize csirsConfig with default CSI-RS configuration
            csirsConfig = obj.CSIRSConfiguration;

            % Check if CSIRSConfig parameter is provided
            if isfield(connectionConfigList, 'CSIRSConfig')
                if isempty(connectionConfigList(1).CSIRSConfig)
                    % Check if CSIReportPeriodicity parameter is provided
                    if ~isempty(connectionConfigList(1).CSIReportPeriodicity)
                        error('nr5g:nrGNB:InvalidReportPeriodicityWithDisabledCSIRS', ...
                            'Unable to set the CSIReportPeriodicity value when CSIRSConfig is disabled');
                    end
                    csirsConfig = []; % Disable CSI-RS
                else
                    % Validate CSI-RS periodicity and offset for TDD
                    if strcmp(obj.DuplexMode, "TDD")
                        numSlotsDLULPattern = obj.DLULConfigTDD.DLULPeriodicity * (obj.SubcarrierSpacing / 15e3);
                        % Validate CSIRSPeriod value to be a two-element vector of positive integers
                        coder.internal.errorIf(~isnumeric(connectionConfigList(1).CSIRSConfig.CSIRSPeriod) || numel(connectionConfigList(1).CSIRSConfig.CSIRSPeriod) ~= 2, 'nr5g:nrGNB:InvalidCSIRSPeriod');
                        % Validate CSIRS Periodicity to be a multiple of the number of DL slots in the TDD DL-UL pattern
                        coder.internal.errorIf(mod(connectionConfigList(1).CSIRSConfig.CSIRSPeriod(1), numSlotsDLULPattern) ~= 0, 'nr5g:nrGNB:InvalidCSIRSPeriodicity', connectionConfigList(1).CSIRSConfig.CSIRSPeriod(1), numSlotsDLULPattern);
                        % Validate Offset
                        dlSlots = 0:obj.DLULConfigTDD.NumDLSlots; % Covering full DL slots and the special slot
                        offsetMod = mod(connectionConfigList(1).CSIRSConfig.CSIRSPeriod(2), numSlotsDLULPattern);
                        coder.internal.errorIf(~ismember(offsetMod, dlSlots), 'nr5g:nrGNB:InvalidCSIRSOffset', connectionConfigList(1).CSIRSConfig.CSIRSPeriod(2));
                    end
                    csirsConfig.CSIRSPeriod = connectionConfigList(1).CSIRSConfig.CSIRSPeriod;
                    % Validate RowNumber
                    if isKey(rowNumberLookup, obj.NumTransmitAntennas)
                        validRowNumbers = rowNumberLookup(obj.NumTransmitAntennas);
                    else
                        validRowNumbers = [];
                    end
                    % Format valid row numbers as a string for error message
                    formattedValidRowNumbersStr = '{}';
                    if ~isempty(validRowNumbers)
                        formattedValidRowNumbersStr = [sprintf('{') (sprintf(repmat('%d, ', 1, length(validRowNumbers)-1)', validRowNumbers(1:end-1) )) sprintf('%d}', validRowNumbers(end))];
                    end
                    coder.internal.errorIf(~ismember(connectionConfigList(1).CSIRSConfig.RowNumber, validRowNumbers), 'nr5g:nrGNB:InvalidRowNumber',...
                        connectionConfigList(1).CSIRSConfig.RowNumber, obj.NumTransmitAntennas, formattedValidRowNumbersStr);
                    csirsConfig.RowNumber = connectionConfigList(1).CSIRSConfig.RowNumber;
                    % Fill Density value
                    csirsConfig.Density = connectionConfigList(1).CSIRSConfig.Density;
                    % Set the NID value
                    csirsConfig.NID = obj.NCellID;
                end
            end

            % Set connection for each UE
            for i=1:numUEs
                if numUEs == 1
                    coder.internal.errorIf(strcmpi(UE(i).ConnectionState, "Connected") && ismember(UE(i).RNTI, obj.ConnectedUEs), 'nr5g:nrGNB:AlreadyConnectedScalar');
                    coder.internal.errorIf(strcmpi(UE(i).ConnectionState, "Connected") && ~isempty(UE(i).GNBNodeID), 'nr5g:nrGNB:InvalidConnectionScalar', UE(i).GNBNodeID);
                else
                    coder.internal.errorIf(strcmpi(UE(i).ConnectionState, "Connected") && ismember(UE(i).RNTI, obj.ConnectedUEs), 'nr5g:nrGNB:AlreadyConnected', i);
                    coder.internal.errorIf(strcmpi(UE(i).ConnectionState, "Connected") && ~isempty(UE(i).GNBNodeID), 'nr5g:nrGNB:InvalidConnection', i, UE(i).GNBNodeID);
                end
                coder.internal.errorIf(UE(i).NumTransmitAntennas~=UE(i).NumReceiveAntennas && obj.CSIMeasurementSignalDLType, 'nr5g:nrGNB:InvalidNumUETxRxAntennas');
                coder.internal.errorIf(obj.SubcarrierSpacing > 30e3 && strcmp(UE(i).PHYAbstractionMethod, 'none'),...
                    'nr5g:nrGNB:InvalidSCSWithPhyAbstractionMethod', "UE")

                % Update connection information
                rnti = size(obj.ConnectedUEs,2)+1;
                connectionConfig = connectionConfigList(i); % UE specific configuration
                % Create SRS Configuration for a UE
                srsConfig = nrCreateSRSConfiguration(obj,UE(i),rnti);
                % Fill connection configuration
                connectionConfig.SRSConfiguration = srsConfig;
                connectionConfig.RNTI = rnti;
                connectionConfig.UEID = UE(i).ID;
                connectionConfig.UEName = UE(i).Name;
                connectionConfig.NumTransmitAntennas = UE(i).NumTransmitAntennas;
                connectionConfig.NumReceiveAntennas = UE(i).NumReceiveAntennas;
                connectionConfig.CSIRSConfiguration = csirsConfig;
                connectionConfig.CSIReportType = obj.CSIReportType;

                % Remove the CSIRSConfig field from connectionConfig
                if isfield(connectionConfig, 'CSIRSConfig')
                    connectionConfig = rmfield(connectionConfig, 'CSIRSConfig');
                end

                % Validate connection information
                connectionConfig = nrNodeValidation.validateConnectionConfig(connectionConfig);
                if ~isempty(connectionConfig.CSIRSConfiguration)
                    connectionConfig.CSIReportConfiguration.CQITable = obj.CQITable;
                end
                connectionConfig.InitialCQIDL = nrGNB.getCQIIndex(connectionConfig.InitialMCSIndexDL);

                obj.SRSConfiguration = [obj.SRSConfiguration srsConfig];

                % Update list of connected UEs
                obj.ConnectedUEs(end+1) = rnti;
                obj.UENodeIDs(end+1) = UE(i).ID;
                obj.UENodeNames(end+1) = UE(i).Name;

                % Add connection context to gNB MAC
                macConnectionInfo = struct();
                for j=1:numel(macConnectionParam)
                    macConnectionInfo.(macConnectionParam(j)) = connectionConfig.(macConnectionParam(j));
                end
                obj.MACEntity.addConnection(macConnectionInfo);

                % Add connection context to gNB PHY
                phyConnectionInfo = struct();
                for j=1:numel(phyConnectionParam)
                    phyConnectionInfo.(phyConnectionParam(j)) = connectionConfig.(phyConnectionParam(j));
                end
                obj.PhyEntity.addConnection(phyConnectionInfo);
                connectionConfig.GNBTransmitPower = obj.PhyEntity.scaleTransmitPower;

                % Add connection context to gNB scheduler
                schedulerConnectionInfo = struct();
                for j=1:numel(schedulerConnectionParam)
                    schedulerConnectionInfo.(schedulerConnectionParam(j)) = connectionConfig.(schedulerConnectionParam(j));
                end
                obj.MACEntity.Scheduler.addConnectionContext(schedulerConnectionInfo);

                % Add connection context to gNB RLC layer using the given full buffer and RLC
                % bearer configurations
                rlcConnectionInfo = struct();
                % Populate RLC connection info with parameters from connectionConfig
                for j=1:numel(rlcConnectionParam)
                    rlcConnectionInfo.(rlcConnectionParam(j)) = connectionConfig.(rlcConnectionParam(j));
                end
                obj.FullBufferTraffic(rnti) = rlcConnectionInfo.FullBufferTraffic;
                % Add RLC bearer using the RLC connection info
                addRLCBearer(obj, rlcConnectionInfo)

                % Set up connection on UE
                UE(i).addConnection(connectionConfig);
            end
        end

        function stats = statistics(obj, type)
            %statistics Return the statistics of gNB
            %
            %   STATS = statistics(OBJ) returns the statistics of the gNB, OBJ.
            %   STATS is a structure with these fields.
            %   ID   - ID of the gNB
            %   Name - Name of the gNB
            %   App  - Application layer statistics
            %   RLC  - RLC layer statistics
            %   MAC  - MAC layer statistics
            %   PHY  - PHY layer statistics
            %
            %   STATS = statistics(OBJ, TYPE) returns the per-destination
            %   categorization of stats in addition to the output from the previous
            %   syntax. This syntax additionally returns a structure field
            %   'Destinations' for each layer to show per-destination statistics.
            %
            %   TYPE - Specified as 'all' which additionally returns a structure field
            %   'Destinations' for each layer to show per-destination statistics.
            %
            %   App is a structure with these fields.
            %   TransmittedPackets  - Total number of packets transmitted to the RLC layer
            %   TransmittedBytes    - Total number of bytes transmitted to the RLC layer
            %   ReceivedPackets     - Total number of packets received from the RLC layer
            %   ReceivedBytes       - Total number of bytes received from the RLC layer
            %   Destinations        - Row vector of structures of length 'N' where 'N' is
            %                         the number of connected UEs. Each structure
            %                         element corresponds to a connected UE and has
            %                         fields: UEID, UEName, RNTI, TransmittedPackets,
            %                         TransmittedBytes.
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
            %   Destinations         - Row vector of structures of length 'N' where 'N' is
            %                          the number of connected UEs. Each structure
            %                          element corresponds to a connected UE and has
            %                          fields: UEID, UEName, RNTI, TransmittedPackets,
            %                          TransmittedBytes, RetransmittedPackets,
            %                          RetransmittedBytes, ReceivedPackets,
            %                          ReceivedBytes, DroppedPackets, DroppedBytes.
            %
            %   MAC is a structure with these fields.
            %   TransmittedPackets  - Total number of packets transmitted to the PHY layer.
            %                         It only corresponds to new transmissions assuming
            %                         that MAC does not send the packet again to the
            %                         PHY for retransmissions. Packets are buffered at
            %                         PHY. MAC only sends the requests for
            %                         retransmission to the PHY layer.
            %   TransmittedBytes    - Total number of bytes transmitted to the PHY layer
            %   ReceivedPackets     - Total number of packets received from the PHY layer
            %   ReceivedBytes       - Total number of bytes received from the PHY layer
            %   Retransmissions     - Total number of retransmissions requests sent to the PHY layer
            %   RetransmissionBytes - Total number of MAC bytes which correspond to the retransmissions
            %   Destinations        - Row vector of structures of length 'N' where 'N' is
            %                         the number of connected UEs. Each structure
            %                         element corresponds to a connected UE and has
            %                         fields: UEID, UEName, RNTI, TransmittedPackets,
            %                         TransmittedBytes, ReceivedPackets, ReceivedBytes,
            %                         Retransmissions, RetransmissionBytes.
            %
            %   PHY is a structure with these fields.
            %   TransmittedPackets  - Total number of packets transmitted
            %   ReceivedPackets     - Total number of packets received
            %   DecodeFailures      - Total number of decode failures
            %   Destinations        - Row vector of structures of length 'N' where 'N' is
            %                         the number of connected UEs. Each structure
            %                         element corresponds to a connected UE and has
            %                         fields: UEID, UEName, RNTI, TransmittedPackets,
            %                         ReceivedPackets, DecodeFailures.
            %
            % You can fetch statistics for multiple gNBs at once by calling this
            % function on a vector of gNB objects. STATS is a row-vector where an
            % element at the index 'i' of STATS contains the statistics of gNB at index
            % 'i' of the gNB vector, OBJ.

            narginchk(1, 2);
            if nargin == 1
                for i=numel(obj):-1:1
                    stats(i) = statisticsPerGNB(obj(i));
                end
            else
                validateattributes(type, {'char','string'}, {'nonempty', 'scalartext'}, 'statistics','',2);
                coder.internal.errorIf(~any(strcmpi(type, ["all" "a" "al"])), ...
                    'nr5g:nrGNB:InvalidStringInputStatistic',type);
                for i=numel(obj):-1:1
                    stats(i) = statisticsPerGNB(obj(i),"all");
                end
            end
        end

        function configureULPowerControl(obj, nameValuePairs)
            %configureULPowerControl Configure the uplink power control parameters
            %
            %   configureULPowerControl(OBJ, Name=Value) configures uplink power control
            %   parameters at one or more gNB nodes, represented by the scalar or vector of
            %   nrGNB objects, OBJ. This object function sets the power control
            %   configuration parameters using one or more optional name-value arguments. If
            %   you do not specify a name-value argument corresponding to a configuration
            %   parameter, the function assigns a default value to it. You can configure UL
            %   power control for a vector of gNB nodes in a single configureULPowerControl
            %   function call. To calculate the UL transmit power, the UE nodes connected to
            %   a gNB node use the same power control parameter values specified in the
            %   name-value arguments. To set the uplink power control parameters, use these
            %   name-value arguments.
            %
            %   PoPUSCH     - Nominal transmit power of a UE node per resource
            %                 block, specified as a numeric scalar in the range [-202,
            %                 24]. Units are in dBm. The default value is -60 dBm. The
            %                 uplink transmit power tends towards the maximum
            %                 attainable value with an increase in PoPUSCH.
            %   Alpha       - Fractional power control multiplier,
            %                 specified as 0, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, or 1. The
            %                 default value is 1. The uplink transmit power tends
            %                 towards the maximum attainable value with an increase in
            %                 Alpha.
            % For more information about these name-value arguments, see 3GPP TS 38.213
            % Section 7.1. To disable the uplink power control, set Alpha to 1 and
            % PoPUSCH to 24 dBm.

            arguments
                obj {mustBeVector, mustBeA(obj, 'nrGNB')}
                nameValuePairs.PoPUSCH (1, 1) {mustBeNumeric, mustBeGreaterThanOrEqual(nameValuePairs.PoPUSCH, -202), ....
                    mustBeLessThanOrEqual(nameValuePairs.PoPUSCH, 24)} = -60
                nameValuePairs.Alpha (1, 1) {mustBeNumeric, mustBeMember(nameValuePairs.Alpha, [0, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1])} = 1
            end
            coder.internal.errorIf(~isempty([obj.LastRunTime]), 'nr5g:nrNode:NotSupportedOperation', 'configureULPowerControl');
            for nodeIdx = 1:numel(obj)
                coder.internal.errorIf(~isempty(obj(nodeIdx).ConnectedUEs), 'nr5g:nrGNB:ConfigULPowerControlAfterConnectUE', obj(nodeIdx).ID);
                obj(nodeIdx).ULPowerControlParameters.PoPUSCH = nameValuePairs.PoPUSCH;
                obj(nodeIdx).ULPowerControlParameters.AlphaPUSCH = nameValuePairs.Alpha;
            end
        end

        function configureScheduler(obj, varargin)
            %configureScheduler Configure scheduler at gNB
            %
            %   configureScheduler(OBJ, Name=Value) configures a scheduler at a gNB node.
            %   The function sets the scheduling parameters using one or more optional
            %   name-value arguments. If you do not specify a name-value argument
            %   corresponding to a configuration parameter, the function assigns a default
            %   value to it. You can configure schedulers for a vector of gNB nodes in a
            %   single configureScheduler function call, but these schedulers must all use
            %   same configuration parameter values specified in the name-value arguments.
            %   To set the configuration parameters, use these name-value arguments.
            %
            %   Scheduler         - Scheduler strategy, specified as "RoundRobin",
            %                       "ProportionalFair", "BestCQI", or a
            %                       user-implemented custom scheduler which is an
            %                       object of subclass of nrScheduler. The default
            %                       value is "RoundRobin". The RoundRobin scheduler
            %                       provides equal scheduling opportunities to all the
            %                       UE nodes. The BestCQI scheduler, on the other hand,
            %                       gives priority to the UE node with the best channel
            %                       quality indicator (CQI). The BestCQI scheduler
            %                       strategy, therefore, achieves better cell
            %                       throughput. The ProportionalFair scheduler is a
            %                       compromise between the RoundRobin and BestCQI
            %                       schedulers. Round-robin, proportional-fair, and
            %                       best-CQI scheduling strategies try to schedule
            %                       'MaxNumUsersPerTTI' UE nodes in each slot. For your
            %                       custom scheduler implementation, you can choose the
            %                       scheduled UE nodes in the TTI.
            %   PFSWindowSize     - Time constant of an exponential moving average,
            %                       in number of slots. The proportional fair (PF)
            %                       scheduler uses this time constant to calculate the
            %                       average data rate. This name-value argument applies
            %                       when you set the value of the Scheduler argument to
            %                       "ProportionalFair". The default value is 20.
            %   ResourceAllocationType - Specify the resource allocation type as
            %                       0 (resource allocation type 0) or 1 (resource
            %                       allocation type 1). The default value is 1.
            %   MaxNumUsersPerTTI - The allowed maximum number of users per
            %                       transmission time interval (TTI). It is an integer
            %                       scalar that starts from 1. The default value is 8.
            %   FixedMCSIndexDL   - Use modulation and coding scheme (MCS) index for DL
            %                       transmissions without considering any channel
            %                       quality information. The MCS index in the range
            %                       [0, 27] and corresponds to a row in the table TS
            %                       38.214 - Table 5.1.3.1-2. The MCS table is stored
            %                       as static property MCSTable of this class. Use
            %                       'MCSIndex' column of the table to set this
            %                       parameter. The default value is empty which means
            %                       that gNB selects the MCS based on CSI-RS
            %                       measurement report.
            %   FixedMCSIndexUL   - Use modulation and coding scheme (MCS) index for UL
            %                       transmissions without considering any channel
            %                       quality information. The MCS index in the range
            %                       [0-27] and corresponds to a row in the table TS
            %                       38.214 - Table 5.1.3.1-2. The MCS table is stored
            %                       as static property MCSTable of this class. Use
            %                       'MCSIndex' column of the table to set this
            %                       parameter. The default value is empty which means
            %                       that gNB selects the MCS based on SRS measurements.
            %   CSIMeasurementSignalDL - DL channel state information measurement signal,
            %                       specified as "SRS" or "CSI-RS".
            %   MUMIMOConfigDL    - Set this parameter to enable DL multi-user
            %                       multiple-input and multiple-output (MU-MIMO).
            %                       Specify the parameter as a structure with these
            %                       fields.
            %       MaxNumUsersPaired - Maximum number of users that scheduler can pair
            %                           for a MU-MIMO transmission. It is an integer
            %                           scalar in the range [2, 4]. The default value is
            %                           2.
            %       MinNumRBs         - The minimum number of resource blocks (RBs) the
            %                           scheduler must allocate to a UE node for
            %                           considering the UE as an MU-MIMO candidate,
            %                           specified as an integer in the range [1,
            %                           NumResourceBlocks]. The scheduler calculates
            %                           numRBs based on the buffer occupancy and
            %                           channel state information (CSI) reported by the
            %                           UE node. The default value is 6.
            %       MinCQI            - Minimum channel quality indicator (CQI)
            %                           required for considering a UE as an MU-MIMO
            %                           candidate. This field is relevant only for CSI-RS-based
            %                           DL MU-MIMO. It is an integer scalar in the range
            %                           [1, 15]. The default value is 7. For the
            %                           associated CQI table, refer 3GPP TS 38.214
            %                           Table 5.2.2.1-2.
            %       SemiOrthogonalityFactor
            %                         - Inter-user interference (IUI) orthogonality
            %                           factor. Scheduler uses it to decide whether to
            %                           pair up the UEs for MU-MIMO or not. It is a
            %                           numeric scalar in the range [0, 1]. Value 0 for
            %                           a pair of UEs means that they are
            %                           non-orthogonal and value 1 means mutual
            %                           orthogonality between them. The orthogonality
            %                           among the MU-MIMO candidates must be greater
            %                           than this parameter for MU-MIMO eligibility.
            %                           The default value is 0.75. This
            %                           field is relevant only for
            %                           CSI-RS-based DL MU-MIMO.
            %       MaxNumLayers      - Maximum number of layers that can be supported by
            %                           the MU-MIMO DL transmission. It is an integer scalar
            %                           in the range [2, 16]. The default value is 8.
            %       MinSINR           - Minimum SINR in dB for a UE to be considered as an
            %                           MU-MIMO candidate. This field is relevant only for
            %                           SRS-based DL MU-MIMO. It is a double value
            %                           in the range [-7, 25]. The default value is 10.
            %   LinkAdaptationConfigDL
            %                     - Link adaptation (LA) configuration structure for
            %                       downlink transmissions. To enable LA for downlink
            %                       transmissions, create an LA configuration structure
            %                       and pass it as a name-value argument to the
            %                       configureScheduler function. If not set, LA remains
            %                       disabled. The LinkAdaptationConfigDL value of []
            %                       enables Link Adaptation with default values for its
            %                       parameters. Avoid configuring FixedMCSIndexDL along
            %                       with LA. MCS table used for DL and UL LA is as per
            %                       3GPP TS 38.214 - Table 5.1.3.1-2. The LA structure
            %                       contains these fields.
            %       InitialOffset     - Initial MCS offset applied to all UEs. This
            %                           value considers the errors in channel
            %                           measurements at the UE node. Upon receiving the
            %                           CSI report, the gNB node resets the MCS offset
            %                           (referred to as MCSOffset), to the
            %                           InitialOffset. The scheduler then determines
            %                           the MCS for the physical downlink shared
            %                           channel (PDSCH) transmission by subtracting
            %                           MCSOffset from the MCS obtained from the
            %                           channel measurements. It is an integer in the
            %                           range [-27, 27]. The default value is 0.
            %       StepUp            - Incremental value for the MCS offset when a
            %                           packet reception fails. LA considers only the
            %                           failure of new transmissions while ignoring any
            %                           re-transmission feedback. It is a numeric
            %                           scalar in the range [0, 27]. The default value
            %                           is 0.27.
            %       StepDown          - Decremental value for the MCS offset when a
            %                           packet reception is successful. LA considers
            %                           only the success of new transmissions while
            %                           ignoring any re-transmission feedback. It is a
            %                           numeric scalar in the range [0, 27]. The
            %                           default value is 0.03.
            %   LinkAdaptationConfigUL
            %                     - Link adaptation (LA) configuration structure for
            %                       uplink transmissions. To enable LA for uplink
            %                       transmissions, create an LA configuration structure
            %                       and pass it as a name-value argument to the
            %                       configureScheduler function. If not set, LA remains
            %                       disabled. The LinkAdaptationConfigUL value of []
            %                       enables Link Adaptation with default values for its
            %                       parameters. Avoid configuring FixedMCSIndexUL along
            %                       with LA. This structure has identical fields to the
            %                       configuration structure used for LA in downlink
            %                       transmissions.
            %   RVSequence        - Redundancy version sequence, specified as a vector
            %                       limited to a maximum of 4 elements, each uniquely
            %                       taking on values from 0 to 3. To disable
            %                       retransmissions, set RVSequence to a scalar value.

            coder.internal.errorIf(mod(nargin-1, 2) == 1,'MATLAB:system:invalidPVPairs');
            validateattributes(obj, {'nrGNB'}, {'vector'}, mfilename, 'obj');
            coder.internal.errorIf(any(~cellfun(@isempty, {obj.LastRunTime})), 'nr5g:nrNode:NotSupportedOperation', 'configureScheduler');
            coder.internal.errorIf(any(~cellfun(@isempty, {obj.ConnectedUEs})), 'nr5g:nrGNB:ConfigSchedulerAfterConnectUE');
            coder.internal.errorIf(any(~[obj.SchedulerDefaultConfig]),'nr5g:nrGNB:MultipleConfigureSchedulerCalls')

            schedulerInfo = struct(Scheduler='RoundRobin', PFSWindowSize=20, ResourceAllocationType=1, ...
                FixedMCSIndexUL=[], FixedMCSIndexDL=[], MaxNumUsersPerTTI=8, ...
                MUMIMOConfigDL=[], LinkAdaptationConfigDL=[], LinkAdaptationConfigUL=[], RVSequence=obj(1).RVSequence, CSIMeasurementSignalDL="CSI-RS");
            % Default values for DL MU-MIMO config parameter
            mumimoConfigDLCSIRS = struct(MaxNumUsersPaired=2, SemiOrthogonalityFactor=0.75, ...
                MinNumRBs=6, MinCQI=7, MaxNumLayers=8);
            mumimoConfigDLSRS = struct(MaxNumUsersPaired=2, MinNumRBs=6, ...
                MaxNumLayers=8, MinSINR=10);
            % Default values for LA config parameter
            defaultLAConfigDL = struct(StepUp=0.27, StepDown=0.03, InitialOffset=0);
            defaultLAConfigUL = struct(StepUp=0.27, StepDown=0.03, InitialOffset=0);
            enableType2Report = true;

            isCustomScheduler = false;
            % Get the user specified parameters for scheduler
            for idx=1:2:nargin-1
                name = varargin{idx};
                if name == "MUMIMOConfigDL"
                    % Enable Type II feedback for CSI-RS-based MU-MIMO transmission
                    csiMeasurementSignalDLNVPairIdx=find(cellfun(@(x) strcmp(x, "CSIMeasurementSignalDL"), varargin));
                    if isempty(csiMeasurementSignalDLNVPairIdx)
                        csiMeasurementSignalDL="CSI-RS";
                        enableType2Report = 1;
                    else
                        csiMeasurementSignalDL=varargin{csiMeasurementSignalDLNVPairIdx+1};
                        enableType2Report = strcmpi(csiMeasurementSignalDL,"CSI-RS");
                    end
                    schedulerInfo.MUMIMOConfigDL = nrNodeValidation.validateConfigureSchedulerMUMIMOInputs(obj, varargin{idx+1}, csiMeasurementSignalDL, mumimoConfigDLCSIRS, mumimoConfigDLSRS);
                    [obj.MUMIMOEnabled] = deal(true);
                elseif name == "LinkAdaptationConfigDL"
                    schedulerInfo.LinkAdaptationConfigDL = nrNodeValidation.validateConfigureSchedulerLAInputs(varargin{idx+1}, defaultLAConfigDL);
                elseif name == "LinkAdaptationConfigUL"
                    schedulerInfo.LinkAdaptationConfigUL = nrNodeValidation.validateConfigureSchedulerLAInputs(varargin{idx+1}, defaultLAConfigUL);
                elseif name == "CSIMeasurementSignalDL"
                    schedulerInfo.CSIMeasurementSignalDL = nrNodeValidation.validateConfigureSchedulerCSIMeasurementSignalDL(obj, varargin{idx+1});
                else
                    schedulerInfo.(char(name)) = nrNodeValidation.validateConfigureSchedulerInputs(name, varargin{idx+1});
                    if name=="Scheduler" && isa(schedulerInfo.Scheduler, 'nrScheduler')
                        isCustomScheduler = 1;
                        validateattributes(obj, {'nrGNB'}, {'scalar'}, mfilename, 'obj');
                    elseif name=="RVSequence"
                        [obj.RVSequence] = deal(schedulerInfo.RVSequence);
                    end
                end
            end

            coder.internal.errorIf(~isempty(schedulerInfo.LinkAdaptationConfigDL) && ~isempty(schedulerInfo.FixedMCSIndexDL),'nr5g:nrGNB:InvalidLinkAdaptationConfig','LinkAdaptationConfigDL','FixedMCSIndexDL');
            coder.internal.errorIf(~isempty(schedulerInfo.LinkAdaptationConfigUL) && ~isempty(schedulerInfo.FixedMCSIndexUL),'nr5g:nrGNB:InvalidLinkAdaptationConfig','LinkAdaptationConfigUL','FixedMCSIndexUL');

            if enableType2Report
                [obj.CSIReportType] = deal(2); % Type II feedback for CSI-RS-based MU-MIMO transmission
            end
            % If user has supplied a custom scheduler then keep only
            % relevant configuration
            if isCustomScheduler
                % Read user-supplied scheduler object
                scheduler = schedulerInfo.Scheduler;
                customSchedulerParam = struct(Scheduler=[], ResourceAllocationType=1, MaxNumUsersPerTTI=8, RVSequence=[0 3 2 1], CSIMeasurementSignalDL="CSI-RS");
                fields = fieldnames(customSchedulerParam);
                for i=1:size(fields,1)
                    customSchedulerParam.(char(fields{i})) = schedulerInfo.(char(fields{i}));
                end
                schedulerInfo = customSchedulerParam;
            else
                % Create scheduler object(s) if method is called on multiple gNBs
                scheduler(1:numel(obj)) = nrScheduler(); % Preallocate
                for  i=2:numel(obj)
                    scheduler(i) = nrScheduler();
                end
            end
            % Get the required parameters for scheduler from node
            schedulerParam = ["DuplexMode", "NumResourceBlocks", "DLULConfigTDD", ...
                "NumHARQ", "NumTransmitAntennas", "NumReceiveAntennas", "SRSReservedResource", "SubcarrierSpacing"];
            for nodeIdx = 1:numel(obj)
                gNB = obj(nodeIdx);
                for idx=1:numel(schedulerParam)
                    schedulerInfo.(schedulerParam(idx)) = gNB.(schedulerParam(idx));
                end
                obj(nodeIdx).CSIMeasurementSignalDLType = strcmpi(schedulerInfo.CSIMeasurementSignalDL, "SRS");
                coder.internal.errorIf(strcmpi(obj(nodeIdx).PHYAbstractionMethod,"none") ...
                    && strcmpi(schedulerInfo.CSIMeasurementSignalDL, "SRS"),'nr5g:nrGNB:InvalidSRSDLCSIFullPHY');
                % Convert the SCS value from Hz to kHz
                schedulerInfo.SubcarrierSpacing = gNB.SubcarrierSpacing/1e3;
                % Get the DMRSTypeAPosition from gNB MAC
                schedulerInfo.DMRSTypeAPosition = gNB.MACEntity.DMRSTypeAPosition;
                addScheduler(gNB.MACEntity, scheduler(nodeIdx));
                configureScheduler(scheduler(nodeIdx), schedulerInfo);
                if strcmp(gNB.PHYAbstractionMethod, "none")
                    gNB.PhyEntity.RVSequence = schedulerInfo.RVSequence;
                end
                gNB.SchedulerDefaultConfig = false;
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
            %   OBJ is an object of type <a href="matlab:help('nrGNB')">nrGNB</a>
            %
            %   PACKET is the packet received from the channel, specified as
            %   structure of the format <a href="matlab:help('wirelessnetwork.internal.wirelessPacket')">wirelessPacket</a>.

            flag = false;
            rxInfo = [];

            % Check packet relevance
            if packet.TransmitterID ~= obj.ID && ~isempty(obj.ReceiveFrequency) && packet.CenterFrequency == obj.ReceiveFrequency && ...
                    intracellPacketRelevance(obj, packet) && rxOn(obj.MACEntity, packet)
                % gNB is assumed to be stationary. Hence position and
                % velocity are not re-evaluated.
                flag = true;
                rxInfo = obj.RxInfo;
            end
        end
    end

    methods (Access = protected)
        function flag = isInactiveProperty(obj, prop)
            flag = false;
            switch prop
                % DLULConfigTDD is applicable only for TDD
                case "DLULConfigTDD"
                    flag = ~any(strcmpi(obj.DuplexMode, "TDD"));
                case "ConnectedUEs"
                    flag = isempty(obj.ConnectedUEs);
                case "UENodeIDs"
                    flag = isempty(obj.UENodeIDs);
                case "UENodeNames"
                    flag = isempty(obj.UENodeNames);
            end
        end

        function setLayerInterfaces(obj)
            %setLayerInterfaces Set inter-layer interfaces

            % Use weak-references for cross-linking handle objects
            phyWeakRef = matlab.lang.WeakReference(obj.PhyEntity);
            macWeakRef = matlab.lang.WeakReference(obj.MACEntity);
            gnbWeakRef = matlab.lang.WeakReference(obj);

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
            % (2) Sending the measured UL channel quality to MAC
            registerMACHandle(obj.PhyEntity, ...
                @(varargin) macWeakRef.Handle.rxIndication(varargin{:}), ...
                @(varargin) macWeakRef.Handle.srsIndication(varargin{:}));

            % Register node callback function at MAC and Phy for:
            % (1) Sending the out-of-band packets from MAC
            % (2) Sending the in-band packets from Phy
            registerOutofBandTxFcn(obj.MACEntity, ...
                @(varargin) gnbWeakRef.Handle.addToTxBuffer(varargin{:}));
            registerTxHandle(obj.PhyEntity, ...
                @(varargin) gnbWeakRef.Handle.addToTxBuffer(varargin{:}));
        end

        function stats = statisticsPerGNB(obj, ~)
            % Return the statistics for a gNB

            % Create stats structure
            appStat = struct('TransmittedPackets', 0, 'TransmittedBytes', 0, ...
                'ReceivedPackets', 0, 'ReceivedBytes', 0);
            rlcStat = struct('TransmittedPackets', 0, 'TransmittedBytes', 0, ...
                'RetransmittedPackets', 0, 'RetransmittedBytes', 0, ...
                'ReceivedPackets', 0, 'ReceivedBytes', 0, 'DroppedPackets', 0, ...
                'DroppedBytes', 0);
            macStat = struct('TransmittedPackets', 0, 'TransmittedBytes', 0, ...
                'ReceivedPackets', 0, 'ReceivedBytes', 0, 'Retransmissions', 0, ...
                'RetransmissionBytes', 0);
            phyStat = struct('TransmittedPackets', 0, 'ReceivedPackets', 0, ...
                'DecodeFailures', 0);
            stats = struct('ID', obj.ID, 'Name', obj.Name, 'App', appStat, ...
                'RLC', rlcStat, 'MAC', macStat, 'PHY', phyStat);

            if ~isempty(obj.ConnectedUEs) % Check if any UE is connected
                layerStats = struct( 'App', statistics(obj.TrafficManager), ...
                    'RLC', cellfun(@(x) statistics(x), obj.RLCEntity)', 'MAC', statistics(obj.MACEntity), ...
                    'PHY', statistics(obj.PhyEntity));
                destinationIDs = [layerStats.MAC(:).UEID];
                destinationNames = [layerStats.MAC(:).UEName];
                destinationRNTIs = [layerStats.MAC(:).RNTI];
                numDestination = size(destinationIDs,2);

                % Form application stats
                stats.App = rmfield(layerStats.App, 'TrafficSources');
                if nargin == 2 % "all" option
                    stats.App.Destinations = repmat(struct('UEID', [], 'UEName', [], ...
                        'RNTI', [], 'TransmittedPackets', 0, 'TransmittedBytes', 0 ), ...
                        1, numDestination);
                    for i=1:size(destinationIDs,2)
                        stats.App.Destinations(i).UEID = destinationIDs(i);
                        stats.App.Destinations(i).UEName = destinationNames(i);
                        stats.App.Destinations(i).RNTI = destinationRNTIs(i);
                    end

                    % Loop over each traffic source and add the stats number to
                    % the corresponding UE
                    for i=1:size(layerStats.App.TrafficSources,2)
                        trafficSourceStat = layerStats.App.TrafficSources(i);
                        appDestinationID = trafficSourceStat.DestinationNodeID;
                        index = find(destinationIDs == appDestinationID);
                        stats.App.Destinations(index).TransmittedPackets = ...
                            stats.App.Destinations(index).TransmittedPackets + trafficSourceStat.TransmittedPackets;
                        stats.App.Destinations(index).TransmittedBytes = ...
                            stats.App.Destinations(index).TransmittedBytes + trafficSourceStat.TransmittedBytes;
                    end
                end

                % Form RLC stats
                fieldNames = fieldnames(rlcStat);
                if nargin == 2 % "all" option
                    stats.RLC.Destinations = repmat(struct('UEID', [], 'UEName', [], ...
                        'RNTI', [], 'TransmittedPackets', 0, 'TransmittedBytes', 0, ...
                        'RetransmittedPackets', 0, 'RetransmittedBytes', 0, ...
                        'ReceivedPackets', 0, 'ReceivedBytes', 0, 'DroppedPackets', 0, ...
                        'DroppedBytes', 0), 1, numDestination);
                end
                for i=1:size(layerStats.RLC,1)
                    logicalChannelStat = layerStats.RLC(i);
                    nextRLCIndex = 1;
                    for j=1:numel(fieldNames)
                        % Create cumulative stats
                        stats.RLC.(char(fieldNames{j})) = stats.RLC.(char(fieldNames{j})) + ...
                            logicalChannelStat.(char(fieldNames{j}));
                        if nargin == 2 % "all" option
                            if nextRLCIndex
                                nextRLCIndex = 0;  % Execute this block only once for RLC entity
                                rlcDestinationRNTI = logicalChannelStat.RNTI;
                                index = find(destinationRNTIs == rlcDestinationRNTI);
                                stats.RLC.Destinations(index).UEID = destinationIDs(index);
                                stats.RLC.Destinations(index).UEName = destinationNames(index);
                                stats.RLC.Destinations(index).RNTI = rlcDestinationRNTI;
                            end
                            % Set per-destination stats
                            stats.RLC.Destinations(index).(char(fieldNames{j})) = stats.RLC.Destinations(index).(char(fieldNames{j})) + ...
                                logicalChannelStat.(char(fieldNames{j}));
                        end
                    end
                end

                % Form MAC stats
                fieldNames = fieldnames(macStat);
                if nargin == 2 % "all" option
                    stats.MAC.Destinations = repmat(struct('UEID', [], 'UEName', [], ...
                        'RNTI', [], 'TransmittedPackets', 0, 'TransmittedBytes', 0, ...
                        'ReceivedPackets', 0, 'ReceivedBytes', 0, 'Retransmissions', 0, ...
                        'RetransmissionBytes', 0), 1, numDestination);
                end
                for i=1:size(layerStats.MAC,1)
                    ueMACStats = layerStats.MAC(i);
                    for j=1:numel(fieldNames)
                        % Create cumulative stats
                        stats.MAC.(char(fieldNames{j})) = stats.MAC.(char(fieldNames{j})) + ...
                            ueMACStats.(char(fieldNames{j}));
                        if nargin == 2 % "all" option
                            macDestinationID = ueMACStats.UEID;
                            index = find(destinationIDs == macDestinationID);
                            stats.MAC.Destinations(index).UEID = macDestinationID;
                            stats.MAC.Destinations(index).UEName = destinationNames(index);
                            stats.MAC.Destinations(index).RNTI = destinationRNTIs(index);
                            % Set per-destination stats
                            stats.MAC.Destinations(index).(char(fieldNames{j})) = stats.MAC.Destinations(index).(char(fieldNames{j})) + ...
                                ueMACStats.(char(fieldNames{j}));
                        end
                    end
                end

                % Form PHY stats
                fieldNames = fieldnames(phyStat);
                if nargin == 2 % "all" option
                    stats.PHY.Destinations = repmat(struct('UEID', [], 'UEName', [], ...
                        'RNTI', [], 'TransmittedPackets', 0,'ReceivedPackets', 0, ...
                        'DecodeFailures', 0), 1, numDestination);
                end
                for i=1:size(layerStats.PHY,1)
                    uePHYStats = layerStats.PHY(i);
                    for j=1:numel(fieldNames)
                        % Create cumulative stats
                        stats.PHY.(char(fieldNames{j})) = stats.PHY.(char(fieldNames{j})) + ...
                            uePHYStats.(char(fieldNames{j}));
                        if nargin == 2 % "all" option
                            % Set per-destination stats
                            phyDestinationID = uePHYStats.UEID;
                            index = find(destinationIDs == phyDestinationID);
                            stats.PHY.Destinations(index).UEID = phyDestinationID;
                            stats.PHY.Destinations(index).UEName = destinationNames(index);
                            stats.PHY.Destinations(index).RNTI = destinationRNTIs(index);
                            stats.PHY.Destinations(index).(char(fieldNames{j})) = stats.PHY.Destinations(index).(char(fieldNames{j})) + ...
                                uePHYStats.(char(fieldNames{j}));
                        end
                    end
                end
            end
        end

        function csirsConfiguration = createCSIRSConfiguration(obj)
            %createCSISRSConfiguration Return common CSI-RS configuration
            %for the cell

            % The default CSI-RS configuration is full-bandwidth. The
            % number of CSI-RS ports equals the number of Tx antennas at
            % gNB. The function sets the periodicity of CSI-RS to 10 slots
            % for FDD. For TDD, the periodicity is a multiple of the length
            % of the DL-UL pattern ( in slots). In this case, the least
            % value of periodicity is 10.

            % Each row contains: AntennaPorts, NumSubcarriers(Max k_i), and
            % NumSymbols(Max l_i) as per TS 38.211 Table 7.4.1.5.3-1
            csirsRowNumberTable = [
                1 1 1; % This row has density 3. Only density as '1' are used.
                1 1 1;
                2 1 1;
                4 1 1;
                4 1 1;
                8 4 1;
                8 2 1;
                8 2 1;
                12 6 1;
                12 3 1;
                16 4 1;
                16 4 1;
                24 3 2;
                24 3 2;
                24 3 1;
                32 4 2;
                32 4 2;
                32 4 1;
                ];

            subcarrierSet = [1 3 5 7 9 11]; % k0 k1 k2 k3 k4 k5
            symbolSet = [0 4]; % l0 l1

            csirsConfiguration = nrCSIRSConfig(CSIRSType="nzp", NumRB=obj.NumResourceBlocks);
            csirsConfiguration.RowNumber = find(csirsRowNumberTable(2:end, 1) == obj.NumTransmitAntennas, 1)+1;
            csirsConfiguration.SubcarrierLocations = subcarrierSet(1:csirsRowNumberTable(csirsConfiguration.RowNumber, 2));
            csirsConfiguration.SymbolLocations = symbolSet(1:csirsRowNumberTable(csirsConfiguration.RowNumber, 3));
            minCSIRSPeriodicity = 10; % Slots
            if strcmp(obj.DuplexMode, "TDD") % TDD
                dlULConfigTDD = obj.DLULConfigTDD;
                numSlotsDLULPattern = dlULConfigTDD.DLULPeriodicity*(obj.SubcarrierSpacing/15e3);
                % Select periodicity such that it is at least 10 and
                % multiple of DL-UL pattern length in slots
                allowedCSIRSPeriodicity = [4,5,8,10,16,20,32,40,64,80,160,320,640];
                allowedCSIRSPeriodicity = allowedCSIRSPeriodicity(allowedCSIRSPeriodicity>=minCSIRSPeriodicity & ...
                    ~mod(allowedCSIRSPeriodicity, numSlotsDLULPattern));
                minCSIRSPeriodicity = allowedCSIRSPeriodicity(1);
            end
            csirsConfiguration.CSIRSPeriod = [minCSIRSPeriodicity 0];
        end
    end

    methods(Hidden)
        function flag = intracellPacketRelevance(~, ~)
            %intracellPacketRelevance Returns whether the packet is relevant for the gNB

            % gNB does not reject any intra-cell packet
            flag = 1;
        end

        function kpiValue = kpi(obj, destinationNode, kpiString, options)
            %kpi Return the key performance indicator (KPI) value for a specified KPI
            %
            %   KPIVALUE = kpi(OBJ, DESTINATIONNODE, KPISTRING, OPTIONS) returns the KPI
            %   value, KPIVALUE, specified by KPISTRING, from the source node(s) represented
            %   by OBJ to the DESTINATIONNODE. The function supports calculations where
            %   either the source node or the destination node can be a vector, allowing for
            %   multiple KPI calculations across different connections. Additionally, if
            %   DESTINATIONNODE is empty, the function calculates KPIs at the cell level.
            %   The calculation of the KPI is determined by the OPTIONS provided.
            %
            %   KPIVALUE        - The calculated value of the specified KPI. If multiple
            %                     source-destination pairs are provided, kpiValue will be a
            %                     row vector containing the KPI value for each connection.
            %
            %   OBJ             - Vector of source node objects from which the KPI is
            %                     measured. Each element in OBJ represents a source node.
            %
            %   DESTINATIONNODE - Vector of destination node objects to which the KPI is
            %                     measured. Each element in DESTINATIONNODE represents a
            %                     destination node. If empty, the KPI is calculated at the
            %                     cell level.
            %
            %   KPISTRING       - Specifies the KPI to be measured. Supported KPIs are
            %                     "latency", "bler", and "prbUsage".
            %
            %   OPTIONS         - Structure with fields:
            %                     - Layer: Specifies the layer at which the KPI should be measured.
            %                              Supported layers are "App", "PHY", and "MAC".
            %                     - LinkType: Specifies the type of link ("DL" for downlink or "UL"
            %                                 for uplink). Default is "DL".

            arguments
                obj (1,:)
                destinationNode (1,:)
                kpiString (1,1) string
                options.Layer (1,1) string
                options.LinkType (1,1) {mustBeMember(options.LinkType,["DL","UL"])} = "DL"
            end

            numSources = size(obj,2);
            numDestinations = size(destinationNode,2);
            if numSources>1 && numDestinations>1
                error(message("nr5g:nrGNB:KPIInvalidSignature"));
            end
            % Validate inputs
            kpiString = validateKPIInputs(obj, destinationNode, kpiString, options.Layer);

            % Initialize kpiValue(s). If there are multiple sourceNode-destinationNode
            % connections provided as input, the function will populate the kpiValue(s) in
            % a row vector
            if numSources > numDestinations
                numKPIs = numSources;
            else
                numKPIs = numDestinations;
            end
            kpiValue = zeros(1,numKPIs);

            currentSourceNode = obj;
            currentDestinationNode = destinationNode;
            % Iterate through all sourceNode-destinationNode connections to obtain the
            % requested KPI
            for kpiIdx = 1:numKPIs
                % Determine the current source and destination nodes
                if numSources > 1
                    currentSourceNode = obj(kpiIdx);
                else
                    if numDestinations == 0
                        currentDestinationNode = [];
                    else
                        currentDestinationNode = destinationNode(kpiIdx);
                    end
                end

                % Set a default value if the node is not yet simulated
                if isempty(currentSourceNode.LastRunTime)
                    continue;
                end

                % Calculate the KPI based on the specified kpiString
                if kpiString == "latency"
                    kpiValue(kpiIdx) = calculateLatency(currentSourceNode, currentDestinationNode);
                elseif kpiString == "bler"
                    kpiValue(kpiIdx) = calculateBLER(currentSourceNode, currentDestinationNode, options.LinkType);
                elseif kpiString == "prbUsage"
                    kpiValue(kpiIdx) = kpi(currentSourceNode.MACEntity, kpiString, options.LinkType);
                end
            end
        end
    end

    methods (Static, Hidden)
        function cqiIndex = getCQIIndex(mcsIndex)
            %getCQIIndex Returns the CQI row index based on MCS index

            mcsRow = MACConstants.MCSTable(mcsIndex + 1, 1:2);
            % Gets row indices matching the modulation scheme corresponding
            % to mcsIndex
            modSchemeMatch = find(MACConstants.CQITable(:, 1)  == mcsRow(1));
            % Among rows with modulation scheme match, find the closet match for code
            % rate (without exceeding the coderate corresponding to
            % mcsIndex)
            cqiRow = find(MACConstants.CQITable(modSchemeMatch, 2) > ...
                mcsRow(2),1); % Find the first row with higher coderate
            if ~isempty(cqiRow)
                cqiRow = modSchemeMatch(cqiRow)-1; % Previous row
            else
                cqiRow = modSchemeMatch(end);
            end
            cqiIndex = cqiRow-1; % 0-based indexing
        end

        function mcsTable = getMCSTable()
            % Create TS 38.214 - Table 5.1.3.1-2 MCS table

            tableArray = [(0:27)'  MACConstants.MCSTable(1:28, :)];
            columnNames = ["MCS Index", "Modulation Order", "Code Rate x 1024", "Bit Efficiency"];

            % Package array in a table
            mcsTable = array2table(tableArray,"VariableNames",columnNames);
            mcsTable.Properties.VariableNames = columnNames;
            mcsTable.Properties.Description = 'TS 38.214 - Table 5.1.3.1-2: MCS Table';
        end
    end

    methods (Access = private)
        function kpiString = validateKPIInputs(~, destinationNode, kpiString, layer)
            %validateKPIInputs Validate the inputs for KPI method

            % Validate the KPI type against the specified layer
            if strcmpi(kpiString, "latency")
                % Latency KPI is only valid at the application layer
                if ~strcmpi(layer, "App")
                    error(message("nr5g:nrGNB:InvalidKPIForLayer", "App", kpiString));
                end
                % Latency KPI requires a destination node
                if isempty(destinationNode)
                    error(message("nr5g:nrGNB:InvalidCellLevelKPI", kpiString));
                end
                kpiString = "latency";
            elseif strcmpi(kpiString, "prbUsage")
                % PRB Usage KPI is only valid at the MAC layer
                if ~strcmpi(layer, "MAC")
                    error(message("nr5g:nrGNB:InvalidKPIForLayer", "MAC", kpiString));
                end
                % PRB Usage KPI is calculated at the cell level, so destination node should be
                % empty
                if ~isempty(destinationNode)
                    error(message("nr5g:nrGNB:InvalidConnectionLevelKPI", kpiString));
                end
                kpiString = "prbUsage";
            elseif strcmpi(kpiString, "bler")
                % BLER KPI is only valid at the PHY layer
                if ~strcmpi(layer, "PHY")
                    error(message("nr5g:nrGNB:InvalidKPIForLayer", "PHY", kpiString));
                end
                kpiString = "bler";
            else
                error(message("nr5g:nrGNB:InvalidKPI"));
            end
        end

        function latency = calculateLatency(obj, destinationNode)
            %calculateLatency Return the packet latency (in seconds) for the connection
            %between the sourceNode and the destinationNode

            % Initialize latency to a default value of 0
            latency = 0;

            % Access detailed statistics of the destination node's traffic
            dstStats = statistics(destinationNode.TrafficManager, true);

            % Iterate over each destination's statistics to find the matching source node
            for idx = 1:numel(dstStats.Destinations)
                if dstStats.Destinations(idx).NodeID == obj.ID
                    % Update latency with the average packet latency and exit loop
                    latency = dstStats.Destinations(idx).AveragePacketLatency;
                    return; % Using return instead of break for immediate exit
                end
            end
        end

        function bler = calculateBLER(obj, destinationNode, linkType)
            %calculateBLER Return the Block Error Rate (BLER) between the source node and
            %the destination node. If the destination node is empty, the BLER is calculated
            %at the cell level.

            bler = 0;
            % If destinationNode is empty, calculate BLER at the cell level
            if isempty(destinationNode)
                % Access PHY statistics for the source node (gNB)
                phyStats = statistics(obj.PhyEntity);
                if linkType == "DL"
                    numPDSCHPackets = sum([phyStats.TransmittedPackets]);
                    if numPDSCHPackets > 0
                        bler = obj.MACEntity.NumPDSCHNACKs/numPDSCHPackets;
                    end
                else
                    numPUSCHPackets = sum([phyStats.ReceivedPackets]);
                    % Calculate BLER as the ratio of decode failures to received packets
                    if numPUSCHPackets > 0
                        bler = sum([phyStats.DecodeFailures])/numPUSCHPackets;
                    end
                end
            else
                % Access PHY statistics for the destination node
                phyStats = statistics(destinationNode.PhyEntity);

                % Calculate BLER if there are received packets
                if phyStats.ReceivedPackets > 0
                    bler = phyStats.DecodeFailures / phyStats.ReceivedPackets;
                end
            end
        end
    end
end
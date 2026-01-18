classdef nrSchedulerConfig < handle
    %nrSchedulerConfig Scheduler configuration
    %
    %   Note: This is an internal undocumented class and its API and/or
    %   functionality may change in subsequent releases.

    %   Copyright 2024 The MathWorks, Inc.

    properties (SetAccess = private)
        %ResourceAllocationType Type for resource allocation type (RAT). Value 0
        %means RAT-0 and value 1 means RAT-1
        ResourceAllocationType

        %MaxNumUsersPerTTI Maximum users that can be scheduled per TTI
        MaxNumUsersPerTTI

        %PFSWindowSize Specifies the time constant (in number of slots) of
        %exponential moving average equation for average data rate calculation in
        %proportional fair scheduler (PFS)
        PFSWindowSize

        %FixedMCSIndexDL MCS index that will be used to allocate DL without
        %considering any channel quality information
        FixedMCSIndexDL

        %FixedMCSIndexUL MCS index that will be used to allocate UL resources
        %without considering any channel quality information
        FixedMCSIndexUL

        %MUMIMOConfigDL Structure that contains DL MU-MIMO parameters i.e.,
        %MaxNumUsersPaired, MinNumRBs and SemiOrthogonalityFactor fields
        MUMIMOConfigDL

        %LinkAdaptationConfigDL Structure that contains DL link adaptation
        %parameters i.e., InitialOffset, StepUp and StepDown fields
        LinkAdaptationConfigDL

        %LinkAdaptationConfigUL Structure that contains UL link adaptation
        %parameters. The structure has same fields as 'LinkAdaptationConfigDL'
        LinkAdaptationConfigUL

        %CSIMeasurementSignalDLType The value of "CSIMeasurementSignalDLType" is 1 if the
        % specified value of CSIMeasurementSignalDL is 'SRS'.  It is 0 if the specified
        % value of "CSIMeasurementSignalDL" is 'CSI-RS'.
        CSIMeasurementSignalDLType (1,1) = 0;
    end

    properties (SetAccess = private, Hidden)
        %AlphaPFS Scheduler weight for instantaneous throughput
        AlphaPFS

        %BetaPFS Scheduler weight for historical/served throughput
        BetaPFS

        %RVSequence Redundancy version (RV) sequence
        RVSequence
    end

    properties (Hidden)
        %SchedulingType Type of scheduling (slot based or symbol based)
        % Value 0 means slot based and value 1 means symbol based. The default
        % value is 0
        SchedulingType = 0;

        %SchedulerStrategy Defines a scheduling strategy for the UEs selection
        % Value 0 means round robin (RR), 1 means proportional fair (PF) and 2
        % means BestCQI scheduler. The default value is 0
        SchedulerStrategy = 0;

        %SchedulerPeriodicity Periodicity at which the schedulers (DL and UL) run
        %in terms of number of slots (for FDD mode). Default value is 1 slot.
        %Maximum number of slots in a frame is 160 (i.e SCS 240 kHz)
        SchedulerPeriodicity = 1;

        %SlotsSinceSchedulerRunDL Number of slots elapsed since last DL scheduler run (for FDD mode)
        % It is incremented every slot and when it reaches the
        % 'SchedulerPeriodicity', it is reset to zero and DL scheduler runs
        SlotsSinceSchedulerRunDL = 0;

        %SlotsSinceSchedulerRunUL Number of slots elapsed since last UL scheduler run (for FDD mode)
        % It is incremented every slot and when it reaches the
        % 'SchedulerPeriodicity', it is reset to zero and UL scheduler runs
        SlotsSinceSchedulerRunUL = 0;

        %PUSCHMappingType PUSCH mapping type (A or B)
        PUSCHMappingType = 'A';

        %PDSCHMappingType PDSCH mapping type (A or B)
        PDSCHMappingType = 'A';
    end

    methods
        function set.SchedulerPeriodicity(obj, value)
            obj.SchedulerPeriodicity = value;
            % Initialization to make sure that schedulers run in the
            % very first slot of simulation run
            obj.SlotsSinceSchedulerRunDL = value - 1;
            obj.SlotsSinceSchedulerRunUL = value - 1;
        end
    end

    methods (Hidden)
        function obj = nrSchedulerConfig(param)
            %schedulerConfig Construct a scheduler configuration object
            %
            % PARAM is a structure including the following fields:
            %   Scheduler                     - Scheduler type (Field only present for in-built schedulers: RR, PF and BestCQI)
            %   ResourceAllocationType        - RAT-0 (value 0) or RAT-1 (value 1)
            %   MaxNumUsersPerTTI             - The allowed maximum number of users per TTI
            %   PFSWindowSize                 - Time constant of an exponential moving average in number of slots
            %   FixedMCSIndexDL               - Use MCS index for DL transmissions without considering any CQI
            %   FixedMCSIndexUL               - Use MCS index for DL transmissions without considering any CQI
            %   MUMIMOConfigDL                - Enable DL multi-user multiple-input and multiple-output (MU-MIMO)
            %   LinkAdaptationConfigDL        - LA configuration structure for downlink transmissions
            %   LinkAdaptationConfigUL        - LA configuration structure for uplink transmissions
            %   RVSequence                    - Redundancy version (RV) sequence
            %   CSIMeasurementSignalDLType    - DL measurement signal type as SRS (value 1) or CSI-RS (value 0)

            if isfield(param,'Scheduler') % In-built scheduler
                % Set the weights for scheduling strategy
                % AlphaPFS defines the scheduler weights which is exponentially applied to
                % instantaneous throughput. It has a value of either '0' or '1', where '0'
                % means instantaneous throughput is not considered. Similarly BetaPFS
                % defines the scheduler weight which is exponentially applied to historical
                % average throughput. It has a value of either '0' or '1
                switch param.Scheduler
                    case 'ProportionalFair'
                        obj.AlphaPFS = 1;
                        obj.BetaPFS = 1;
                        obj.SchedulerStrategy = 1;
                    case 'BestCQI'
                        obj.AlphaPFS = 1;
                        obj.BetaPFS = 0;
                        obj.SchedulerStrategy = 2;
                    otherwise % RR Scheduler
                        obj.AlphaPFS = 0;
                        obj.BetaPFS = 0;
                        obj.SchedulerStrategy = 0;
                end
                param = rmfield(param,'Scheduler');
            end
            % Set supplied context in the object
            fields = fieldnames(param);
            for i=1:size(fields,1)
                obj.(char(fields{i})) = param.(char(fields{i}));
            end
        end
    end
end

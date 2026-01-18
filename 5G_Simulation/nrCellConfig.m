classdef nrCellConfig < handle
    %nrCellConfig Configuration of a cell at gNB
    %
    %   Note: This is an internal undocumented class and its API and/or
    %   functionality may change in subsequent releases.

    %   Copyright 2024 The MathWorks, Inc.

    properties (SetAccess = private)
        %SubcarrierSpacing Subcarrier spacing used across the cell
        SubcarrierSpacing

        %NumResourceBlocks Number of resource blocks in channel bandwidth
        NumResourceBlocks

        %DuplexMode Duplex mode as FDD or TDD
        DuplexMode

        %DLULConfigTDD DL and UL time division configuration for TDD mode
        % For more information, refer the <a
        % href="matlab:help('nrGNB.DLULConfigTDD')">DLULConfigTDD</a> structure.
        DLULConfigTDD
    end

    properties (SetAccess = private, Hidden)
        %SlotDuration Slot duration in ms
        SlotDuration

        %NumSlotsFrame Number of slots in a 10 ms frame
        NumSlotsFrame

        %DuplexModeNumber Stores the duplex mode type as integer, for improved
        %performance. Value 0 means FDD and 1 means TDD.
        DuplexModeNumber

        %NumDLULPatternSlots Number of slots in DL-UL pattern (for TDD mode)
        NumDLULPatternSlots

        %DLULSlotFormat Format of the slots in DL-UL pattern (for TDD mode)
        % N-by-14 matrix where 'N' is number of slots in DL-UL pattern. Each row
        % contains the symbol type of the 14 symbols in the slot. Value 0, 1 and 2
        % represent DL symbol, UL symbol and guard symbol respectively.
        DLULSlotFormat

        %DMRSTypeAPosition Position of DM-RS in type A transmission (2 or 3)
        DMRSTypeAPosition

        %ULReservedResource Reserved resources information for UL direction
        % Array of three elements: [symNum slotPeriodicity slotOffset].
        % These symbols are not available for PUSCH scheduling as per the
        % slot offset and periodicity. Currently, it is used for SRS
        % resources reservation
        ULReservedResource

        %NumTransmitAntennas Number of transmit antennas used by gNB on this cell
        NumTransmitAntennas

        %NumReceiveAntennas Number of received antennas used by gNB on this cell
        NumReceiveAntennas
    end

    properties(Hidden)
        %CurrDLULSlotIndex Slot index of the current running slot in the DL-UL pattern at the time of scheduler invocation (for TDD mode)
        CurrDLULSlotIndex = 0;

        %NextULSchedulingSlot Slot to be scheduled next by UL scheduler
        % Slot number in the 10 ms frame whose resources will be scheduled
        % when UL scheduler runs next (for TDD mode)
        NextULSchedulingSlot
    end

    methods (Hidden)
        function obj = nrCellConfig(param)
            %nrCellConfig Construct a cell configuration object
            %
            % PARAM is a structure including the following fields:
            %   SubcarrierSpacing  - Subcarrier spacing used across the cell
            %   NumResourceBlocks  - Number of resource blocks in channel bandwidth
            %   DuplexMode         - Duplexing mode as FDD or TDD
            %   DLULConfigTDD      - Downlink (DL) and uplink (UL) TDD configuration
            %   DMRSTypeAPosition  - Position of DM-RS in type A transmission
            %   ULReservedResource - UL reserved resource as [symbolNum slotPeriodicity slotOffset]
            %   NumTransmitAntennas - Number of transmit antennas used by gNB on this cell
            %   NumReceiveAntennas  - Number of receive antennas used by gNB on this cell

            % Initialize the properties
            inputParam = ["SubcarrierSpacing", "NumResourceBlocks", "DuplexMode", ...
                "DMRSTypeAPosition", "ULReservedResource", "NumTransmitAntennas", "NumReceiveAntennas"];
            for idx=1:numel(inputParam)
                obj.(inputParam(idx)) = param.(inputParam(idx));
            end

            obj.SlotDuration = 1/(obj.SubcarrierSpacing/15);
            obj.NumSlotsFrame = 10*(obj.SubcarrierSpacing/15);
            if strcmp(param.DuplexMode, 'FDD')
                obj.DuplexModeNumber = 0;
            else
                obj.DuplexModeNumber = 1;
                configTDD = param.DLULConfigTDD;
                obj.DLULConfigTDD = configTDD;
                numDLULPatternSlots = configTDD.DLULPeriodicity/obj.SlotDuration;
                obj.NumDLULPatternSlots = numDLULPatternSlots;
                numDLSlots = configTDD.NumDLSlots;
                numDLSymbols = configTDD.NumDLSymbols;
                numULSlots = configTDD.NumULSlots;
                numULSymbols = configTDD.NumULSymbols;
                % All the remaining symbols in DL-UL pattern are assumed to be guard symbols
                guardDuration = (numDLULPatternSlots * 14) - ...
                    (((numDLSlots + numULSlots)*14) + ...
                    numDLSymbols + numULSymbols);

                dlType = nr5g.internal.MACConstants.DLType;
                ulType = nr5g.internal.MACConstants.ULType;
                guardType = nr5g.internal.MACConstants.GuardType;
                % Set format of slots in the DL-UL pattern. Value 0, 1 and 2 means symbol
                % type as DL, UL and guard, respectively
                obj.DLULSlotFormat = guardType * ones(numDLULPatternSlots, 14);
                obj.DLULSlotFormat(1:numDLSlots, :) = dlType; % Mark all the symbols of full DL slots as DL
                obj.DLULSlotFormat(numDLSlots + 1, 1 : numDLSymbols) = dlType; % Mark DL symbols following the full DL slots
                obj.DLULSlotFormat(numDLSlots + floor(guardDuration/14) + 1, (numDLSymbols + mod(guardDuration, 14) + 1) : end)  ...
                    = ulType; % Mark UL symbols at the end of slot before full UL slots
                obj.DLULSlotFormat((end - numULSlots + 1):end, :) = ulType; % Mark all the symbols of full UL slots as UL type
            end
        end
    end
end
function srsReservedResource = nrConfigureSRSReservedResource(gNB)
%nrConfigureSRSReservedResource Configure the SRS reserved resource
%
% SRSRESERVEDRESOURCE = nrConfigureSRSReservedResource(GNB) returns the SRS
% reserved resource for a GNB. When the UEs connect, the scheduler
% assigns SRS configuration to them.
% GNB                 - gNB node object
% SRSRESERVEDRESOURCE - SRS resource occurrence period as [symbolNumber slotPeriodicity slotOffset]

% Copyright 2024 The MathWorks, Inc.

% Set the SRS resource periodicity and offset in terms of slots
minSRSResourcePeriodicity = gNB.MinSRSResourcePeriodicity;
if gNB.DuplexMode=="FDD" % FDD
    srsResourcePeriodicity = minSRSResourcePeriodicity;
    srsResourceOffset = 0;
else % TDD
    dlULConfigTDD = gNB.DLULConfigTDD;
    numSlotsDLULPattern = dlULConfigTDD.DLULPeriodicity*(gNB.SubcarrierSpacing/15e3);
    % Set SRS resource periodicity as minimum value such that it is at least 5
    % slots and integer multiple of numSlotsDLULPattern
    allowedSRSPeriodicity = [1 2 4 5 8 10 16 20 32 40 64 80 160 320 640 1280 2560];
    allowedSRSPeriodicity = allowedSRSPeriodicity(allowedSRSPeriodicity>=minSRSResourcePeriodicity & ...
        ~mod(allowedSRSPeriodicity, numSlotsDLULPattern));
    srsResourcePeriodicity = allowedSRSPeriodicity(1);
    % SRS slot offset depends on the occurrence of first slot in TDD pattern
    % with UL symbol. If 'S' slot does not have a UL symbol then SRS is
    % transmitted in the slot after 'S' slot. Otherwise, it is transmitted in
    % the 'S' slot
    srsResourceOffset = dlULConfigTDD.NumDLSlots + (dlULConfigTDD.NumULSymbols==0 && dlULConfigTDD.NumDLSymbols>0);
end

% Set SRS resource occurrence period as [symbolNumber slotPeriodicity slotOffset]
srsReservedResource = [13 srsResourcePeriodicity srsResourceOffset]; % SRS on last symbol
end
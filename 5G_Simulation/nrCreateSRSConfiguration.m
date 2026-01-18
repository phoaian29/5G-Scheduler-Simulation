function srsConfig = nrCreateSRSConfiguration(gNB, UE, rnti)
%nrCreateSRSConfiguration Create and return the SRS configuration for the UE
%
% SRSCONFIG = nrCreateSRSConfiguration(GNB,UE,RNTI) returns the SRS
% configuration for a UE identified by the RNTI and connected to the GNB.
% This function is called while connecting a UE. When the UE connect, the
% scheduler assigns this configuration to the UE.
% GNB        - gNB node object
% UE         - UE node object
% RNTI       - RNTI of the connected UE
% SRSCONFIG  - nrSRSConfig object

% Copyright 2024 The MathWorks, Inc.

% SRS resource periodicity and offset in terms of slots
srsResourcePeriodicity = gNB.SRSReservedResource(2);
srsResourceOffset = gNB.SRSReservedResource(3);

ktc = 4; % Comb size
ncsMax = 4; % Maximum cyclic shift (Not using maximum value which could be 12 for ktc=4)
srsPeriodicityPerUE = gNB.SRSPeriodicityUE; % SRS transmission periodicity

% Calculate csrs for full bandwidth (or as close as possible to it)
srsBandwidthMapping = nrSRSConfig.BandwidthConfigurationTable{:,2};
csrs = find(srsBandwidthMapping <= gNB.NumResourceBlocks, 1, 'last') - 1;

% Create SRS configuration. gNB populates unique SRS configuration using
% slotOffset, comb offset and cyclic shift. gNB fills the number of SRS
% ports later as the UEs connect, based on Tx antenna count on the UE.
srsConfig = nrSRSConfig;
srsConfig.CSRS = csrs;
srsConfig.BSRS = 0;
srsConfig.KTC = ktc;
slotOffset = srsResourceOffset + srsResourcePeriodicity*mod(floor((rnti-1)/(ktc*ncsMax)), (srsPeriodicityPerUE/srsResourcePeriodicity));
srsConfig.SRSPeriod = [srsPeriodicityPerUE slotOffset];
srsConfig.KBarTC = mod(rnti-1, ktc);
srsConfig.CyclicShift = mod(floor((rnti-1)/ktc), ncsMax);
srsConfig.NumSRSPorts = UE.NumTransmitAntennas;
srsConfig.NSRSID = gNB.NCellID;
end
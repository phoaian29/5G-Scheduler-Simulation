function channels = hNRCreateCDLChannels(channelConfig, gNB, UEs)
    %hNRCreateCDLChannels Creates CDL channels for gNB-to-UE and UE-to-gNB
    %links in a cell for single-user or multi-user scenarios.
    %   
    %   CHANNELS = hNRCreateCDLChannels(CHANNELCONFIG, GNB, UES) creates
    %   uplink (UL) and downlink (DL) channels.
    %
    %   CHANNELS      - An N-by-N array, where N represents the number of
    %                   nodes in the cell.
    %
    %   CHANNELCONFIG - A structure with these fields:
    %       DelayProfile - CDL delay profile
    %       PathDelays   - Discrete path delay row vector (s)
    %   GNB           - An nrGNB object.
    %   UES           - A vector of nrUE objects.

    % Copyright 2024 The MathWorks, Inc.
  
    numUEs = numel(UEs);
    numNodes = numel(gNB) + numUEs;
    channels = cell(numNodes, numNodes);
    % Get waveform sample rate from gNB configuration
    waveformInfo = nrOFDMInfo(gNB.NumResourceBlocks, gNB.SubcarrierSpacing / 1e3);
    sampleRate = waveformInfo.SampleRate;

    % Create a CDL channel model object with the desired configuration,
    % including the delay profile (CDL-A, B, C, D, or E), delay spread, and
    % Doppler frequency.
    cdlChannel = nrCDLChannel(CarrierFrequency=gNB.CarrierFrequency, ...
        TransmitArrayOrientation = [0 12 0]', SampleRate = sampleRate);
    for fname=fieldnames(channelConfig)'
        cdlChannel.(fname{1})=channelConfig.(fname{1});
    end
    cdlChannel.TransmitAntennaArray.Element = 'isotropic';
    cdlChannel.ChannelFiltering = strcmp(gNB.PHYAbstractionMethod,'none');

    isFDD = gNB.DuplexMode == "FDD";
    % For each UE, set DL and UL channel instances
    for ueIdx = 1:numUEs
        % Create DL channel with custom delay profile
        dlChannel = hMakeCustomCDL(cdlChannel);
        % Configure the channel seed based on the UE number
        % (results in independent fading for each UE)
        dlChannel.Seed = 73 + (ueIdx - 1);
        % Set antenna panel
        dlChannel = hArrayGeometry(dlChannel, gNB.NumTransmitAntennas, UEs(ueIdx).NumReceiveAntennas, "downlink");
        
        % Compute the LOS angle from gNB to UE
        [~, depAngleDL] = rangeangle(UEs(ueIdx).Position', gNB.Position');
        % Configure the azimuth and zenith departure angle offsets for this UE
        dlChannel.AnglesAoD(:) = dlChannel.AnglesAoD(:) + depAngleDL(1);
        % Convert elevation angle to zenith angle
        dlChannel.AnglesZoD(:) = dlChannel.AnglesZoD(:) - dlChannel.AnglesZoD(1) + (90 - depAngleDL(2));

        % Compute the LOS angle from UE to gNB
        [~, arrAngleDL] = rangeangle(gNB.Position', UEs(ueIdx).Position');
        % Configure the azimuth and zenith arrival angle offsets for this UE
        dlChannel.AnglesAoA(:) = dlChannel.AnglesAoA(:) - dlChannel.AnglesAoA(1) + arrAngleDL(1);
        % Convert elevation angle to zenith angle
        dlChannel.AnglesZoA(:) = dlChannel.AnglesZoA(:) - dlChannel.AnglesZoA(1) + (90 - arrAngleDL(2));

        % For FDD, assign appropriate DL carrier frequency
        if isFDD
            dlChannel.CarrierFrequency = gNB.DLCarrierFrequency;
        end

        channels{gNB.ID, UEs(ueIdx).ID} = dlChannel;
        
        % Create UL channel
        ulChannel = clone(dlChannel);
        ulChannel.swapTransmitAndReceive();
        ulChannel = hArrayGeometry(ulChannel, UEs(ueIdx).NumTransmitAntennas, gNB.NumReceiveAntennas, "uplink");
        % For FDD, assign appropriate UL carrier frequency
        if isFDD
            ulChannel.CarrierFrequency = gNB.ULCarrierFrequency;
        end

        channels{UEs(ueIdx).ID, gNB.ID} = ulChannel;
    end
end
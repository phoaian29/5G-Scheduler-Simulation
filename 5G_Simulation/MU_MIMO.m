%% ====================== 1. SYSTEM CONFIGURATION ======================

% --- gNB CONFIGURATION (128T128R) ---
gNBConfig = struct();
gNBConfig.Position = [0 0 30];             % Vị trí [x,y,z]
gNBConfig.TransmitPower = 60;             % Công suất phát (dBm) ~ theo yêu cầu trung bình
gNBConfig.SubcarrierSpacing = 30000;      % 30 kHz
gNBConfig.CarrierFrequency = 4.9e9;       % 4.9 GHz
gNBConfig.ChannelBandwidth = 100e6;       % 100 MHz
% Với 100MHz @ 30kHz SCS, số lượng RB chuẩn là 273
gNBConfig.NumResourceBlocks = 273;        
gNBConfig.NumTransmitAntennas = 32;      % 128 Anten phát
gNBConfig.NumReceiveAntennas = 32;       % 128 Anten thu
gNBConfig.ReceiveGain = 32;               % Gain thu 32 dBi
gNBConfig.DuplexMode = "TDD";             % TDD
gNBConfig.SRSPeriodicity = 5;             % SRS Periodicity = 5 slots

% --- UE CONFIGURATION ---
ueConfig = struct();
ueConfig.NumUEs = 2;     
ueConfig.NumTransmitAntennas = 4;
ueConfig.NumReceiveAntennas = 4;          % 4 Anten thu mỗi UE
ueConfig.ReceiveGain = 0;                 % Gain 0 dBi
ueConfig.MaxDistance = 1200;              % Bán kính 1200m
ueConfig.MinDistance = 10;                
ueConfig.AzimuthRange = [-30 30];         % Góc phương vị +/- 30 độ
ueConfig.ElevationAngle = 0;    
ueConfig.NoiseFigureMin = 20;              
ueConfig.NoiseFigureMax = 30;             

% --- MU-MIMO & SCHEDULER ---
muMIMOConfig = struct();
muMIMOConfig.MaxNumUsersPaired = 4;      % Ghép tối đa 4 UE
muMIMOConfig.MinNumRBs = 3;               % Min 3 RBs
muMIMOConfig.SemiOrthogonalityFactor = 0.9; 
muMIMOConfig.MinCQI = 1;                % Tương ứng CQI 1
muMIMOConfig.MaxNumLayers = 16;
schedulerConfig = struct();
schedulerConfig.ResourceAllocationType = 0; % RB-based
schedulerConfig.MaxNumUsersPerTTI = 64;     % Max 64 UE/TTI
schedulerConfig.SignalType = "SRS";         

% --- CHANNEL MODEL ---
channelConfig = struct();
channelConfig.DelayProfile = "CDL-D";       
channelConfig.DelaySpread = 450e-9;         % 450ns
channelConfig.MaxDopplerShift = 136;        % ~136 Hz
channelConfig.Orientation = [60; 0; 0];     % Hướng anten gNB

% --- SIMULATION CONTROL ---
simConfig = struct();
simConfig.NumFrameSimulation = 2;           % Chạy thử 2 khung (20ms) vì 512 UE rất nặng
simConfig.EnableTraces = true;              

%% ====================== 2. INITIALIZATION ======================

wirelessnetworkSupportPackageCheck
rng("default");
networkSimulator = wirelessNetworkSimulator.init;

% --- Create gNB (32T32R) ---
gNB = nrGNB('Position', gNBConfig.Position, ...
    'TransmitPower', gNBConfig.TransmitPower, ...
    'SubcarrierSpacing', gNBConfig.SubcarrierSpacing, ...
    'CarrierFrequency', gNBConfig.CarrierFrequency, ...
    'ChannelBandwidth', gNBConfig.ChannelBandwidth, ...
    'NumTransmitAntennas', gNBConfig.NumTransmitAntennas, ... 
    'NumReceiveAntennas', gNBConfig.NumReceiveAntennas, ...   
    'DuplexMode', gNBConfig.DuplexMode, ...
    'ReceiveGain', gNBConfig.ReceiveGain, ...
    'SRSPeriodicityUE', gNBConfig.SRSPeriodicity, ... 
    'NumResourceBlocks', gNBConfig.NumResourceBlocks);

% --- Configure Scheduler ---
% Chuẩn bị struct config cho Scheduler
muMIMOStruct = struct(...
    'MaxNumUsersPaired', muMIMOConfig.MaxNumUsersPaired, ...
    'MinNumRBs', muMIMOConfig.MinNumRBs, ...
    'SemiOrthogonalityFactor', muMIMOConfig.SemiOrthogonalityFactor, ...
    'MinCQI', muMIMOConfig.MinCQI, ...
    'MaxNumLayers', muMIMOConfig.MaxNumLayers); 

% Khởi tạo SchedulerDRL (Yêu cầu bạn phải có file class SchedulerDRL.m)
% Nếu chưa có, hãy dùng scheduler mặc định bằng cách comment dòng dưới
drlScheduler = SchedulerDRL(); 

configureScheduler(gNB, ...
    'Scheduler', drlScheduler, ...       
    'ResourceAllocationType', schedulerConfig.ResourceAllocationType, ... 
    'MaxNumUsersPerTTI', schedulerConfig.MaxNumUsersPerTTI, ...
    'MUMIMOConfigDL', muMIMOStruct, ...
    'CSIMeasurementSignalDL', schedulerConfig.SignalType);

% --- Create UEs ---
UEs = nrUE.empty(0, ueConfig.NumUEs); 
rng(42); 

% Tạo vị trí UE theo Sector (-30 đến 30 độ)
ueAzimuths = ueConfig.AzimuthRange(1) + (ueConfig.AzimuthRange(2) - ueConfig.AzimuthRange(1)) * rand(ueConfig.NumUEs, 1);
ueElevations = zeros(ueConfig.NumUEs, 1);
ueDistances = ueConfig.MinDistance + (ueConfig.MaxDistance - ueConfig.MinDistance) * rand(ueConfig.NumUEs, 1);

[xPos, yPos, zPos] = sph2cart(deg2rad(ueAzimuths), deg2rad(ueElevations), ueDistances);
uePositions = [xPos yPos zPos] + gNBConfig.Position;

fprintf('Khoi tao %d UEs (gNB: 32T32R, Ptx: %ddBm)...\n', ueConfig.NumUEs, gNBConfig.TransmitPower);

for i = 1:ueConfig.NumUEs
    currentNoise = ueConfig.NoiseFigureMin + (ueConfig.NoiseFigureMax - ueConfig.NoiseFigureMin) * rand();
    UEs(i) = nrUE('Name', "UE-" + string(i), ...
                  'Position', uePositions(i, :), ...
                  'NumReceiveAntennas', ueConfig.NumReceiveAntennas, ... 
                  'NoiseFigure', currentNoise, ... 
                  'ReceiveGain', ueConfig.ReceiveGain, ...
                  'NumTransmitAntennas',ueConfig.NumTransmitAntennas);           
end

connectUE(gNB, UEs, FullBufferTraffic="DL", CSIReportPeriodicity=10);

addNodes(networkSimulator, gNB);
addNodes(networkSimulator, UEs);

%% ====================== 3. CHANNEL MODEL ======================

cdlConfig = struct(...
    'DelayProfile', channelConfig.DelayProfile, ...
    'DelaySpread', channelConfig.DelaySpread, ...
    'MaximumDopplerShift', channelConfig.MaxDopplerShift, ...
    'TransmitArrayOrientation', channelConfig.Orientation);

% Tạo kênh truyền CDL
channels = hNRCreateCDLChannels(cdlConfig, gNB, UEs);
customChannelModel  = hNRCustomChannelModel(channels);
addChannelModel(networkSimulator, @customChannelModel.applyChannelModel);

%% ====================== 4. RUN SIMULATION ======================

if simConfig.EnableTraces
    simSchedulingLogger = helperNRSchedulingLogger(simConfig.NumFrameSimulation, gNB, UEs);
    simPhyLogger = helperNRPhyLogger(simConfig.NumFrameSimulation, gNB, UEs);
end

% Visualizer
metricsVisualizer = helperNRMetricsVisualizer(gNB, UEs, ...
    'RefreshRate', 20, ... % Update ít lại để đỡ lag
    'PlotSchedulerMetrics', true, ...
    'PlotPhyMetrics', false, ...
    'PlotCDFMetrics', true, ...
    'LinkDirection', 0);

simulationLogFile = "simulationLogs_128T128R"; 
simulationTime = simConfig.NumFrameSimulation * 1e-2;

fprintf('Dang chay mo phong 128T128R trong %.2f giay...\n', simulationTime);
run(networkSimulator, simulationTime);

%% ====================== 5. LOGS & METRICS ======================
displayPerformanceIndicators(metricsVisualizer);

if simConfig.EnableTraces
    simulationLogs = cell(1, 1);
    % Logic lấy log TDD/FDD
    if gNB.DuplexMode == "FDD"
        logInfo = struct('DLTimeStepLogs',[], 'ULTimeStepLogs',[], 'SchedulingAssignmentLogs',[], 'PhyReceptionLogs',[]);
        [logInfo.DLTimeStepLogs, logInfo.ULTimeStepLogs] = getSchedulingLogs(simSchedulingLogger);
    else 
        logInfo = struct('TimeStepLogs',[], 'SchedulingAssignmentLogs',[], 'PhyReceptionLogs',[]);
        logInfo.TimeStepLogs = getSchedulingLogs(simSchedulingLogger);
    end
    
    logInfo.SchedulingAssignmentLogs = getGrantLogs(simSchedulingLogger);
    logInfo.PhyReceptionLogs = getReceptionLogs(simPhyLogger);
    save(simulationLogFile, "simulationLogs");
    
    % Plot Histrogram UE/RB
    avgNumUEsPerRB = calculateAvgUEsPerRBDL(logInfo, gNB.NumResourceBlocks, ...
        schedulerConfig.ResourceAllocationType, gNBConfig.DuplexMode);
    
    figure; theme("light");
    histogram(avgNumUEsPerRB, 'BinWidth', 0.1);
    title('Distribution of Avg UEs per RB (128T128R Configuration)');
    xlabel('Average Number of UEs per RB');
    ylabel('Frequency');
    grid on;
end

%% ====================== HELPER FUNCTION ======================
function avgUEsPerRB = calculateAvgUEsPerRBDL(logInfo, numResourceBlocks, ratType, duplexMode)
    % (Giữ nguyên hàm tính toán như phiên bản trước)
    if strcmp(duplexMode, 'TDD')
        timeStepLogs = logInfo.TimeStepLogs;
        freqAllocations = timeStepLogs(:, 5);
    elseif strcmp(duplexMode, 'FDD')
        timeStepLogs = logInfo.DLTimeStepLogs;
        freqAllocations = timeStepLogs(:, 4);
    end
    numOfSlots = size(timeStepLogs, 1) - 1;
    if ~ratType
        numRBG = size(freqAllocations{2}, 2);
        P = ceil(numResourceBlocks / numRBG);
        numRBsPerRBG = P * ones(1, numRBG);
        if mod(numResourceBlocks, P) > 0, numRBsPerRBG(end) = mod(numResourceBlocks, P); end
    end
    avgUEsPerRB = zeros(1, numOfSlots);
    for slotIdx = 1:numOfSlots
        if strcmp(duplexMode, 'TDD')
            slotType = timeStepLogs{slotIdx + 1, 4};
            if ~strcmp(slotType, 'DL'), continue; end
        end
        freqAllocation = freqAllocations{slotIdx + 1};
        if ~ratType
            totalUniqueUEs = sum(arrayfun(@(rbgIdx) nnz(freqAllocation(:, rbgIdx) > 0) * numRBsPerRBG(rbgIdx), 1:length(numRBsPerRBG)));
            avgUEsPerRB(slotIdx) = totalUniqueUEs / numResourceBlocks;
        else
            ueRBUsage = zeros(1, numResourceBlocks);
            for ueIdx = 1:size(freqAllocation, 1)
                startRB = freqAllocation(ueIdx, 1);
                ueRBUsage(startRB + 1:(startRB + freqAllocation(ueIdx, 2))) = ueRBUsage(startRB + 1:(startRB + freqAllocation(ueIdx, 2))) + 1;
            end
            avgUEsPerRB(slotIdx) = mean(ueRBUsage(ueRBUsage > 0));
        end
    end
    avgUEsPerRB = avgUEsPerRB(avgUEsPerRB > 0);
end
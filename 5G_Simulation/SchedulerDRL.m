% classdef SchedulerDRL < nrScheduler
%     properties (Access = public)
%         DRL_IP = "127.0.0.1";
%         DRL_Port = 5555;
%         DRL_Socket = [];
%         AvgThroughputMBps = ones(1, 4) * 1e-6; 
%         Rho = 0.9;
%         LastServedBytes = zeros(1, 4);
%         LastAllocRatio = zeros(1, 4);
%     end
% 
%     methods (Access = public)
%         function obj = SchedulerDRL(varargin)
%         end
% 
%         function success = connectToDRLAgent(obj)
%             success = false;
%             if ~isempty(obj.DRL_Socket), delete(obj.DRL_Socket); obj.DRL_Socket = []; end
%             try
%                 disp('[MATLAB] Connecting to PPO Trainer...');
%                 obj.DRL_Socket = tcpclient(obj.DRL_IP, obj.DRL_Port, 'Timeout', 60);
%                 disp('[MATLAB] Connected!');
%                 success = true;
%             catch
%                 disp('[MATLAB] Connection Failed.');
%             end
%         end
%     end
% 
%     methods (Access = protected)
%         function dlAssignments = scheduleNewTransmissionsDL(obj, timeFrequencyResource, schedulingInfo)
%             eligibleUEs = schedulingInfo.EligibleUEs;
%             if isempty(eligibleUEs), dlAssignments = struct([]); return; end
% 
%             % 1. CSI Calculation
%             numEligibleUEs = size(eligibleUEs,2);
%             W = cell(numEligibleUEs, 1);
%             rank = zeros(numEligibleUEs, 1);
%             channelQuality = zeros(obj.NumUEs, obj.CellConfig.NumResourceBlocks);
%             cqiSizeArray = ones(obj.CellConfig.NumResourceBlocks, 1);
% 
%             for i=1:numEligibleUEs
%                 rnti = eligibleUEs(i);
%                 ueCtx = obj.UEContext(rnti);
%                 carrierCtx = ueCtx.ComponentCarrier(1);
%                 csiMeasurement = carrierCtx.CSIMeasurementDL;
%                 csiMeasurementCQI = max(csiMeasurement.CSIRS.CQI(:)) * cqiSizeArray(1);
%                 channelQuality(rnti, :) = csiMeasurementCQI;
% 
%                 numCSIRSPorts = obj.CellConfig(1).NumTransmitAntennas;
%                 if ~isempty(carrierCtx.CSIRSConfiguration)
%                     numCSIRSPorts = carrierCtx.CSIRSConfiguration.NumCSIRSPorts;
%                 end
%                 [rank(i), W{i}] = obj.selectRankAndPrecodingMatrixDL(rnti, csiMeasurement, numCSIRSPorts);
%             end
% 
%             schedulerInput = obj.SchedulerInputStruct;
%             schedulerInput.eligibleUEs = eligibleUEs;
%             schedulerInput.channelQuality = channelQuality(eligibleUEs, :);
%             schedulerInput.selectedRank = rank;
%             schedulerInput.W = W;
%             schedulerInput.bufferStatus = [obj.UEContext(eligibleUEs).BufferStatusDL];
%             schedulerInput.freqOccupancyBitmap = timeFrequencyResource.FrequencyResource;
% 
%             % 2. RUN DRL STRATEGY
%             [allottedUEs, freqAllocation, mcsIndex, W_final] = obj.runSchedulingStrategyDRL(schedulerInput);
% 
%             % 3. Update Metrics
%             servedBytes = zeros(1, 4);
%             currentAlloc = zeros(1, 4);
%             rbgSize = obj.getRBGSize();
%             numRBGTotal = floor(obj.CellConfig.NumResourceBlocks / rbgSize);
% 
%             for k = 1:length(allottedUEs)
%                 ueID = allottedUEs(k);
%                 if ueID <= 4
%                     numRBG = sum(freqAllocation(k,:));
%                     if numRBG > 0
%                         mcs = mcsIndex(k);
%                         bpp = obj.getBytesPerPRB(mcs);
%                         servedBytes(ueID) = numRBG * rbgSize * bpp;
%                         currentAlloc(ueID) = numRBG / numRBGTotal;
%                     end
%                 end
%             end
% 
%             obj.LastServedBytes = servedBytes;
%             obj.LastAllocRatio = currentAlloc;
%             instRateMbps = (servedBytes * 8) / 1e6; 
%             obj.AvgThroughputMBps = obj.Rho * obj.AvgThroughputMBps + (1 - obj.Rho) * instRateMbps;
% 
%             % 4. Final Output
%             numNewTxs = min(size(eligibleUEs,2), schedulingInfo.MaxNumUsersTTI);
%             dlAssignments = obj.DLGrantArrayStruct(1:numNewTxs);
%             numAllottedUEs = numel(allottedUEs);
%             for index = 1:numAllottedUEs
%                 selectedUE = allottedUEs(index);
%                 dlAssignments(index).RNTI = selectedUE;
%                 dlAssignments(index).GNBCarrierIndex = 1;
%                 dlAssignments(index).FrequencyAllocation = freqAllocation(index, :);
%                 carrierCtx = obj.UEContext(selectedUE).ComponentCarrier(1);
%                 mcsOffset = fix(carrierCtx.MCSOffset(obj.DLType+1));
%                 dlAssignments(index).MCSIndex = min(max(mcsIndex(index) - mcsOffset, 0), 27);
%                 dlAssignments(index).W = W_final{index};
%             end
%             dlAssignments = dlAssignments(1:numAllottedUEs);
%         end
% 
%         function [allottedUEs, freqAllocation, mcsIndex, W_final] = runSchedulingStrategyDRL(obj, schedulerInput)
%             if isempty(obj.DRL_Socket)
%                 if ~obj.connectToDRLAgent(), allottedUEs=[]; freqAllocation=[]; mcsIndex=[]; W_final={}; return; end
%             end
% 
%             % --- 1. PREPARE 5 FEATURES ---
%             featuresMatrix = zeros(4, 5);
%             for u = 1:4
%                 if ismember(u, schedulerInput.eligibleUEs)
%                     ueCtx = obj.UEContext(u);
%                     idx = find(schedulerInput.eligibleUEs == u, 1);
%                     featuresMatrix(u, 1) = ueCtx.BufferStatusDL;
%                     featuresMatrix(u, 2) = obj.AvgThroughputMBps(u);
%                     featuresMatrix(u, 3) = mean(schedulerInput.channelQuality(idx, :));
%                     featuresMatrix(u, 4) = schedulerInput.selectedRank(idx);
%                     featuresMatrix(u, 5) = obj.LastAllocRatio(u);
%                 end
%             end
% 
%             % --- SYNC PRB BUDGET ---
%             payload.features = featuresMatrix;
%             payload.last_served = obj.LastServedBytes;
%             payload.prb_budget = obj.CellConfig.NumResourceBlocks; 
% 
%             try
%                 jsonStr = jsonencode(payload);
%                 write(obj.DRL_Socket, uint8(jsonStr));
%                 write(obj.DRL_Socket, uint8(10));
% 
%                 while obj.DRL_Socket.NumBytesAvailable == 0, end
%                 data = read(obj.DRL_Socket, obj.DRL_Socket.NumBytesAvailable);
%                 response = jsondecode(char(data));
%                 prbCounts = response.prbs;
%             catch
%                 allottedUEs=[]; freqAllocation=[]; mcsIndex=[]; W_final={}; return;
%             end
% 
%             % --- MAP TO RBG ---
%             rbgSize = obj.getRBGSize();
%             numRBG = size(schedulerInput.freqOccupancyBitmap, 2);
%             tempFreqAlloc = zeros(4, numRBG); 
%             tempMCS = zeros(4, 1);
%             tempW = cell(4, 1);
%             currentRBGIndex = 1;
% 
%             for ueID = 1:4
%                 numPRB = prbCounts(ueID);
%                 if numPRB > 0 && ismember(ueID, schedulerInput.eligibleUEs)
%                     numRBG_Needed = ceil(numPRB / rbgSize);
%                     endRBG = min(currentRBGIndex + numRBG_Needed - 1, numRBG);
%                     if endRBG >= currentRBGIndex
%                         tempFreqAlloc(ueID, currentRBGIndex:endRBG) = 1;
%                         idx = find(schedulerInput.eligibleUEs == ueID, 1);
%                         avgCQI = mean(schedulerInput.channelQuality(idx, :));
%                         tempMCS(ueID) = min(27, floor(avgCQI * 1.8)); 
%                         tempW{ueID} = schedulerInput.W{idx};
%                         currentRBGIndex = endRBG + 1;
%                     end
%                 end
%             end
% 
%             finalUEs = []; finalFreqAlloc = []; finalMCS = []; finalW = {};
%             for u = 1:4
%                 if sum(tempFreqAlloc(u, :)) > 0
%                     finalUEs = [finalUEs, u];
%                     finalFreqAlloc = [finalFreqAlloc; tempFreqAlloc(u, :)];
%                     finalMCS = [finalMCS; tempMCS(u)];
%                     finalW = [finalW; tempW{u}];
%                 end
%             end
%             allottedUEs = finalUEs; freqAllocation = finalFreqAlloc; mcsIndex = finalMCS; W_final = finalW;
%         end
% 
%         function rbgSize = getRBGSize(obj)
%             numRBs = obj.CellConfig.NumResourceBlocks;
%             if numRBs <= 36, rbgSize = 2;
%             elseif numRBs <= 72, rbgSize = 4;
%             elseif numRBs <= 144, rbgSize = 8;
%             else, rbgSize = 16;
%             end
%         end
% 
%         function bpp = getBytesPerPRB(~, mcs)
%             effs = [0.15 0.23 0.38 0.60 0.88 1.18 1.48 1.91 2.40 2.73 3.32 3.90 4.52 5.12 5.55 6.07 6.23 6.50 6.70 6.90 7.00 7.10 7.20 7.30 7.35 7.40 7.45 7.48 7.50];
%             if mcs<0,mcs=0;end; if mcs>28,mcs=28;end
%             bpp = (effs(mcs+1) * 12 * 14 * 0.9) / 8;
%         end
% 
%         function [rank, W] = selectRankAndPrecodingMatrixDL(obj, rnti, csi, ports)
%             carrierCtx = obj.UEContext(rnti).ComponentCarrier(1);
%             numRBGs = carrierCtx.NumRBGs; 
%             report = csi.CSIRS; rank = report.RI;
%             if ports == 1 || isempty(report.W)
%                 W = 1;
%             else
%                 if ismatrix(report.W), W = repmat(report.W.', 1, 1, numRBGs);
%                 else, wBase = permute(report.W, [2 1 3]); curr = size(wBase, 3);
%                     if curr < numRBGs, W = cat(3, wBase, repmat(wBase(:,:,end), 1, 1, numRBGs-curr));
%                     else, W = wBase; end
%                 end
%             end
%         end
%     end
% end



classdef SchedulerDRL < nrScheduler
    properties (Access = public)
        DRL_IP = "127.0.0.1";
        DRL_Port = 5555;
        DRL_Socket = [];
        
        % Cấu hình hệ thống
        MaxUEs = 2;           % Hỗ trợ tối đa 64 UE
        SubbandSize = 16;       % Số PRB mỗi Subband
        MaxNumLayers = 16;      % Số layer MU-MIMO tối đa
        
        % Metrics theo dõi
        AvgThroughputMBps = []; 
        Rho = 0.9;
        LastServedBytes = [];
        LastAllocRatio = [];
        LastSRSReport = [];
        LastSRSUpdateTime = [];
        
        % Mapping Table: CQI to Spectral Efficiency (Approx)
        CQIToSE = [0.15 0.23 0.38 0.60 0.88 1.18 1.48 1.91 2.40 2.73 3.32 3.90 4.52 5.12 5.55 6.00];
    end

    methods (Access = public)
        function obj = SchedulerDRL(varargin)
            % Constructor
        end

        function success = connectToDRLAgent(obj)
            success = false;
            if ~isempty(obj.DRL_Socket), delete(obj.DRL_Socket); obj.DRL_Socket = []; end
            try
                disp('🔌 [MATLAB] Connecting to PPO Trainer (Max 64 UEs)...');
                obj.DRL_Socket = tcpclient(obj.DRL_IP, obj.DRL_Port, 'Timeout', 60);
                disp('✅ [MATLAB] Connected!');
                success = true;
            catch
                disp('❌ [MATLAB] Connection Failed.');
            end
        end
    end

    methods (Access = protected)
        function dlAssignments = scheduleNewTransmissionsDL(obj, timeFrequencyResource, schedulingInfo)
            eligibleUEs = schedulingInfo.EligibleUEs;
            if isempty(eligibleUEs), dlAssignments = struct([]); return; end

            % --- 0. KHỞI TẠO METRICS NẾU CHƯA CÓ ---
            if isempty(obj.AvgThroughputMBps)
                obj.AvgThroughputMBps = zeros(1, obj.MaxUEs);
                obj.LastServedBytes = zeros(1, obj.MaxUEs);
                obj.LastAllocRatio = zeros(1, obj.MaxUEs);
                obj.LastSRSReport = cell(1, obj.MaxUEs);
                obj.LastSRSUpdateTime = nan(1, obj.MaxUEs);
            end

            % --- 1. CHUẨN BỊ DỮ LIỆU (7 FEATURES) ---
            % FIX: Lấy NumResourceBlocks trực tiếp từ CellConfig
            numRBs = obj.CellConfig.NumResourceBlocks;
            rbgSize = obj.getRBGSize();
            numSubbands = ceil(numRBs / obj.SubbandSize); 
            
            % FIX: Tính thời gian hiện tại từ Frame/Slot (tránh lỗi obj.CurrentTime)
            slotDur = obj.CellConfig(1).SlotDuration; 
            currentTime = (double(obj.CurrFrame) * 10e-3) + (double(obj.CurrSlot) * slotDur * 1e-3);

            % Feature Matrix: [MaxUEs x FeatureDim]
            featDim = 5 + 2 * numSubbands;
            exportMatrix = zeros(obj.MaxUEs, featDim);
            
            % precodingMatrixMap = cell(obj.MaxUEs, 1);
            
            for i = 1:length(eligibleUEs)
                rnti = eligibleUEs(i);
                if rnti > obj.MaxUEs, continue; end
                
                ueCtx = obj.UEContext(rnti);
                carrierCtx = ueCtx.ComponentCarrier(1);
                % --- DECODE CSI (Ưu tiên SRS khi bật CSIMeasurementSignalDL = "SRS") ---
                if obj.SchedulerConfig.CSIMeasurementSignalDLType
                    [dlRank, wbCQI, sbCQI, W_out] = obj.decodeSRS(carrierCtx, numSubbands, rnti, currentTime);
                else
                    csirsConfig = carrierCtx.CSIRSConfiguration;
                    if ~isempty(csirsConfig)
                       [dlRank, ~, wbCQI, sbCQI, W_out, ~] = obj.decodeCSIRS(...
                            csirsConfig, currentTime, currentTime + 1e-3, carrierCtx, numRBs);
                    else
                        disp("FallBack")
                        dlRank = 1; wbCQI = 1; sbCQI = ones(1, numSubbands); W_out = 1;
                    end
                end
                
                precodingMatrixMap{rnti} = W_out;

                % --- TÍNH TOÁN 7 FEATURES ---
                f_R = obj.AvgThroughputMBps(rnti);
                f_h = dlRank / obj.MaxNumLayers;
                % Normalized number of already allocated RBGs across MU-MIMO layers (not delay)
                f_d = obj.LastAllocRatio(rnti);
                f_b = ueCtx.BufferStatusDL;
                
                cqiIdx = min(max(round(wbCQI), 1), 16);
                f_o = obj.CQIToSE(cqiIdx) / 6.0;
                
                if length(sbCQI) ~= numSubbands
                    sbCQI = ones(1, numSubbands) * wbCQI;
                end
                sb_cqi_idx = min(max(round(sbCQI), 1), 16);
                f_g_vec = obj.CQIToSE(sb_cqi_idx) / 6.0;
                
                f_rho_vec = obj.computeCrossCorrelation(precodingMatrixMap, eligibleUEs, rnti, numSubbands);

                exportMatrix(rnti, :) = [f_R, f_h, f_d, f_b, f_o, f_g_vec, f_rho_vec];
                % --- [NEW LOG] IN RA 7 FEATURES CỦA UE ĐẦU TIÊN (Active) ---
                if f_b > 0 && i == 1 % Chỉ in log cho UE đầu tiên có buffer
                    fprintf('\n--- [MATLAB SENDING UE %d] ---\n', rnti);
                    fprintf('1. Tput (f_R):   %.4f\n', f_R);
                    fprintf('2. Rank (f_h):   %.4f\n', f_h);
                    fprintf('3. Alloc RBG (f_d):  %.4f\n', f_d);
                    fprintf('4. Buffer (f_b): %.0f\n', f_b);
                    fprintf('5. WB CQI (f_o): %.4f\n', f_o);
                    fprintf('6. SB CQI (Vec): [%.2f, %.2f, ... size=%d]\n', f_g_vec(1), f_g_vec(2), length(f_g_vec));
                    fprintf('7. Corr (Vec):   [%.2f, ...]\n', f_rho_vec(1));
                    fprintf('-----------------------------\n');
                end
            end


            % --- 2. GỬI SANG PYTHON ---
            payload.features = exportMatrix;
            payload.num_subbands = numSubbands;
            payload.prb_budget = numRBs;
            payload.last_served = obj.LastServedBytes;
            
            prbCounts = obj.communicateWithPython(payload);
            
            % --- 3. THỰC HIỆN LẬP LỊCH (MAPPING) ---
            [allottedUEs, freqAllocation, mcsIndex, W_final] = obj.performRBGMapping(...
                prbCounts, eligibleUEs, precodingMatrixMap, exportMatrix, rbgSize, schedulingInfo.MaxNumUsersTTI);

            % --- 4. UPDATE METRICS ---
            servedBytes = zeros(1, obj.MaxUEs);
            currentAlloc = zeros(1, obj.MaxUEs);
            numRBGTotal = ceil(numRBs / rbgSize) * obj.MaxNumLayers;
            
            for k = 1:length(allottedUEs)
                ueID = allottedUEs(k);
                numRBG = sum(freqAllocation(k,:));
                numLayers = obj.getNumLayersFromW(W_final{k});
                if numRBG > 0
                    mcs = mcsIndex(k);
                    bpp = obj.getBytesPerPRB(mcs);
                    servedBytes(ueID) = numRBG * rbgSize * bpp;
                    allocatedRBGs = numRBG * numLayers;
                    currentAlloc(ueID) = allocatedRBGs / numRBGTotal;
                end
            end
            
            obj.LastServedBytes = servedBytes;
            obj.LastAllocRatio = currentAlloc;
            instRateMbps = (servedBytes * 8) / 1e6; 
            obj.AvgThroughputMBps = obj.Rho * obj.AvgThroughputMBps + (1 - obj.Rho) * instRateMbps;

            % --- 5. TẠO OUTPUT ---
            numAllotted = length(allottedUEs);
            dlAssignments = obj.DLGrantArrayStruct(1:numAllotted);
            
            for idx = 1:numAllotted
                selectedUE = allottedUEs(idx);
                dlAssignments(idx).RNTI = selectedUE;
                dlAssignments(idx).GNBCarrierIndex = 1;
                dlAssignments(idx).FrequencyAllocation = freqAllocation(idx, :);
                
                carrierCtx = obj.UEContext(selectedUE).ComponentCarrier(1);
                mcsOffset = fix(carrierCtx.MCSOffset(obj.DLType+1));
                dlAssignments(idx).MCSIndex = min(max(mcsIndex(idx) - mcsOffset, 0), 27);
                dlAssignments(idx).W = W_final{idx};
            end
        end

        function prbCounts = communicateWithPython(obj, payload)
            if isempty(obj.DRL_Socket)
                if ~obj.connectToDRLAgent(), prbCounts=zeros(1, obj.MaxUEs); return; end
            end
            try
                jsonStr = jsonencode(payload);
                write(obj.DRL_Socket, uint8(jsonStr));
                write(obj.DRL_Socket, uint8(10));
                
                while obj.DRL_Socket.NumBytesAvailable == 0, end
                data = read(obj.DRL_Socket, obj.DRL_Socket.NumBytesAvailable);
                response = jsondecode(char(data));
                prbCounts = response.prbs;
            catch
                disp('⚠️ Socket Error/Timeout');
                prbCounts = zeros(1, obj.MaxUEs);
            end
        end

        function [finalUEs, finalFreqAlloc, finalMCS, finalW] = performRBGMapping(obj, prbCounts, eligibleUEs, pMap, feats, rbgSize, maxUsers)
            numRBs = obj.CellConfig.NumResourceBlocks;
            numRBGs = ceil(numRBs / rbgSize);
            
            tempFreqAlloc = zeros(obj.MaxUEs, numRBGs); 
            tempMCS = zeros(obj.MaxUEs, 1);
            tempW = cell(obj.MaxUEs, 1);
            currentRBGIndex = 1;
            
            for ueID = 1:obj.MaxUEs
                numPRB = prbCounts(ueID);
                if numPRB > 0 && ismember(ueID, eligibleUEs)
                    numRBG_Needed = ceil(numPRB / rbgSize);
                    endRBG = min(currentRBGIndex + numRBG_Needed - 1, numRBGs);
                    
                    if endRBG >= currentRBGIndex
                        tempFreqAlloc(ueID, currentRBGIndex:endRBG) = 1;
                        se = feats(ueID, 5) * 6.0;
                        tempMCS(ueID) = min(27, floor(se * 4.5)); 
                        tempW{ueID} = pMap{ueID};
                        currentRBGIndex = endRBG + 1;
                    end
                end
            end
            
            finalUEs = []; finalFreqAlloc = []; finalMCS = []; finalW = {};
            count = 0;
            for u = 1:obj.MaxUEs
                if sum(tempFreqAlloc(u, :)) > 0
                    if count >= maxUsers, break; end
                    count = count + 1;
                    finalUEs = [finalUEs, u];
                    finalFreqAlloc = [finalFreqAlloc; tempFreqAlloc(u, :)];
                    finalMCS = [finalMCS; tempMCS(u)];
                    finalW = [finalW; tempW{u}];
                end
            end
        end
        
        % --- HELPER FUNCTIONS ---
        % FIX: Thêm tham số numRBs vào hàm decodeCSIRS để tránh lỗi NSizeGrid
function [dlRank, pmiSet, widebandCQI, cqiSubband, precodingMatrix, sinrEffSubband] = decodeCSIRS(obj, csirsConfig, pktStartTime, pktEndTime, carrierCtx, numRBs)
             
             % 1. Lấy thông tin cấu hình ăng-ten
             numTx = obj.CellConfig.NumTransmitAntennas;
             numSB = ceil(numRBs / obj.SubbandSize);
             
             % 2. Lấy báo cáo CSI thực tế từ UE Context (Được cập nhật bởi PHY)
             % carrierCtx được truyền vào chính là obj.UEContext(rnti).ComponentCarrier(1)
             csiReport = carrierCtx.CSIMeasurementDL.CSIRS;

             % 3. Kiểm tra xem đã có báo cáo CSI chưa
             if isempty(csiReport) || all(isnan(csiReport.CQI(:)))
                 % --- FALLBACK (Nếu chưa có báo cáo nào - dùng giá trị mặc định) ---
                 disp("Fallback")
                 dlRank = 1; 
                 pmiSet = []; 
                 widebandCQI = 15; 
                 cqiSubband = ones(1, numSB) * 15; 
                 precodingMatrix = ones(1, numTx) ./ sqrt(numTx); 
                 sinrEffSubband = [];
             else
                 % --- REAL DATA (Dùng dữ liệu thực) ---
                 disp("RealData")
                 % A. Xử lý CQI
                 rawCQI = csiReport.CQI;
                 if isempty(rawCQI)
                     widebandCQI = 1; 
                     cqiSubband = ones(1, numSB);
                 elseif isscalar(rawCQI)
                     % Nếu cấu hình Wideband CQI
                     widebandCQI = rawCQI;
                     cqiSubband = ones(1, numSB) * rawCQI;
                 else
                     % Nếu cấu hình Subband CQI
                     widebandCQI = mean(rawCQI, 'all');
                     cqiSubband = rawCQI(:).'; % Đảm bảo là row vector
                     
                     % Resize nếu kích thước không khớp (do cấu hình Subband khác nhau)
                     if length(cqiSubband) ~= numSB
                         % Đơn giản nhất là resize/resample
                         % Ở đây dùng nội suy nearest neighbor để khớp kích thước
                         cqiSubband = imresize(cqiSubband, [1 numSB], 'nearest');
                     end
                 end
                 
                 % B. Xử lý Rank (RI)
                 if isempty(csiReport.RI)
                     dlRank = 1;
                 else
                     dlRank = csiReport.RI;
                 end
                 
                 % C. Xử lý Precoding Matrix (W)
                 if isfield(csiReport, 'W') && ~isempty(csiReport.W)
                     % W thường có kích thước (NumTx, NumLayers, NumSubbands) hoặc tương tự
                     % Scheduler cần (NumLayers, NumTx) cho băng rộng hoặc RBG
                     % Lấy W của subband đầu tiên hoặc trung bình (Simplified)
                     W_raw = csiReport.W; 
                     
                     % Xử lý chiều (Dimensions)
                     % Giả sử W_raw là (NumTx x NumLayers) cho Wideband
                     if size(W_raw, 1) == numTx && size(W_raw, 2) == dlRank
                         precodingMatrix = W_raw.'; % Chuyển vị thành (NumLayers x NumTx)
                     elseif size(W_raw, 2) == numTx && size(W_raw, 1) == dlRank
                         precodingMatrix = W_raw;   % Đã đúng chiều
                     else
                         % Fallback W nếu kích thước lạ
                         precodingMatrix = ones(dlRank, numTx) ./ sqrt(numTx);
                     end
                 else
                     % Fallback W nếu không có PMI
                     precodingMatrix = ones(dlRank, numTx) ./ sqrt(numTx);
                 end
                 
                 pmiSet = [];
                 sinrEffSubband = [];
             end
        end

        function rbgSize = getRBGSize(obj)
            numRBs = obj.CellConfig.NumResourceBlocks;
            if numRBs <= 36, rbgSize = 2; elseif numRBs <= 72, rbgSize = 4;
            elseif numRBs <= 144, rbgSize = 8; else, rbgSize = 16; end
        end

        function bpp = getBytesPerPRB(~, mcs)
            effs = [0.15 0.23 0.38 0.60 0.88 1.18 1.48 1.91 2.40 2.73 3.32 3.90 4.52 5.12 5.55 6.07 6.23 6.50 6.70 6.90 7.00 7.10 7.20 7.30 7.35 7.40 7.45 7.48 7.50];
            if mcs<0,mcs=0;end; if mcs>28,mcs=28;end
            bpp = (effs(mcs+1) * 12 * 14 * 0.9) / 8;
        end

        function [dlRank, wbCQI, sbCQI, W_out] = decodeSRS(obj, carrierCtx, numSubbands, rnti, currentTime)
            srsReport = carrierCtx.CSIMeasurementDL.SRS;
            numTx = obj.CellConfig.NumTransmitAntennas;
            hasCached = ~isempty(obj.LastSRSReport{rnti});
            if isempty(srsReport)
                if hasCached
                    srsReport = obj.LastSRSReport{rnti};
                    fprintf('[SRS] UE %d reuse cached SRS at t=%.6f (last update t=%.6f)\n', ...
                        rnti, currentTime, obj.LastSRSUpdateTime(rnti));
                else
                    fprintf('[SRS] UE %d no SRS yet at t=%.6f, fallback defaults\n', rnti, currentTime);
                    dlRank = 1;
                    wbCQI = 1;
                    sbCQI = ones(1, numSubbands);
                    W_out = ones(1, numTx) ./ sqrt(numTx);
                    return
                end
            elseif ~hasCached || ~isequaln(srsReport, obj.LastSRSReport{rnti})
                obj.LastSRSReport{rnti} = srsReport;
                obj.LastSRSUpdateTime(rnti) = currentTime;
                fprintf('[SRS] UE %d update SRS at t=%.6f\n', rnti, currentTime);
            else
                fprintf('[SRS] UE %d SRS unchanged at t=%.6f (last update t=%.6f)\n', ...
                    rnti, currentTime, obj.LastSRSUpdateTime(rnti));
            end

            if isfield(srsReport, 'RI') && ~isempty(srsReport.RI)
                dlRank = srsReport.RI;
            else
                dlRank = 1;
            end

            if isfield(srsReport, 'W') && ~isempty(srsReport.W)
                W_raw = srsReport.W;
                W_out = obj.normalizeSRSW(W_raw, dlRank, numTx);
            else
                W_out = ones(dlRank, numTx) ./ sqrt(numTx);
            end

            if isfield(srsReport, 'MCSIndex') && ~isempty(srsReport.MCSIndex)
                wbCQI = obj.mcsToCQI(srsReport.MCSIndex);
            else
                wbCQI = 1;
            end
            sbCQI = ones(1, numSubbands) * wbCQI;
        end

        function cqi = mcsToCQI(~, mcsIndex)
            mcsIndex = min(max(round(mcsIndex), 0), 28);
            cqi = max(1, min(15, ceil((mcsIndex / 28) * 15)));
        end

        function W_out = normalizeSRSW(~, W_raw, dlRank, numTx)
            if isempty(W_raw)
                W_out = ones(dlRank, numTx) ./ sqrt(numTx);
                return
            end

            if isscalar(W_raw)
                W_out = ones(dlRank, numTx) ./ sqrt(numTx);
                return
            end

            if ndims(W_raw) >= 3
                W_raw = W_raw(:, :, 1);
            end

            if size(W_raw, 1) == dlRank && size(W_raw, 2) == numTx
                W_out = W_raw;
            elseif size(W_raw, 1) == numTx && size(W_raw, 2) == dlRank
                W_out = W_raw.';
            else
                W_out = ones(dlRank, numTx) ./ sqrt(numTx);
            end
        end

        function rho_vec = computeCrossCorrelation(obj, precodingMap, eligibleUEs, rnti, numSubbands)
            rho_vec = zeros(1, numSubbands);
            if isempty(eligibleUEs) || numel(eligibleUEs) < 2
                return
            end

            candidateW = precodingMap{rnti};
            if isempty(candidateW)
                return
            end

            for m = 1:numSubbands
                Pm_u = obj.getPrecodingSubband(candidateW, m);
                if isempty(Pm_u)
                    continue
                end
                maxCorr = 0;
                for idx = 1:numel(eligibleUEs)
                    otherUE = eligibleUEs(idx);
                    if otherUE == rnti
                        continue
                    end
                    otherW = precodingMap{otherUE};
                    if isempty(otherW)
                        continue
                    end
                    Pm_c = obj.getPrecodingSubband(otherW, m);
                    if isempty(Pm_c)
                        continue
                    end
                    corrMatrix = Pm_u' * Pm_c;
                    colSum = sum(abs(corrMatrix), 1);
                    kappa = max(colSum);
                    maxCorr = max(maxCorr, kappa);
                end
                rho_vec(m) = maxCorr;
            end
        end

        function Pm = getPrecodingSubband(~, W, subbandIdx)
            if isempty(W)
                Pm = [];
                return
            end
            if isscalar(W)
                Pm = W;
                return
            end
            if ndims(W) >= 3
                maxIdx = size(W, 3);
                subbandIdx = min(subbandIdx, maxIdx);
                Pm = W(:, :, subbandIdx);
                return
            end
            Pm = W;
        end

        function numLayers = getNumLayersFromW(~, W)
            if isempty(W)
                numLayers = 1;
                return
            end
            if isnumeric(W)
                if isscalar(W)
                    numLayers = 1;
                else
                    numLayers = size(W, 1);
                end
                return
            end
            numLayers = 1;
        end
    end
end

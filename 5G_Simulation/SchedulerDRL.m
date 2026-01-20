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
        MaxUEs = 64;           % Hỗ trợ tối đa 64 UE
        SubbandSize = 16;       % Số PRB mỗi Subband
        
        % Metrics theo dõi
        AvgThroughputMBps = []; 
        Rho = 0.9;
        LastServedBytes = [];
        LastAllocRatio = [];
        
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
            
            precodingMatrixMap = cell(obj.MaxUEs, 1);

            for i = 1:length(eligibleUEs)
                rnti = eligibleUEs(i);
                if rnti > obj.MaxUEs, continue; end
                
                ueCtx = obj.UEContext(rnti);
                carrierCtx = ueCtx.ComponentCarrier(1);
                csirsConfig = carrierCtx.CSIRSConfiguration;

                % --- DECODE CSI ---
                % Truyền thêm numRBs vào hàm decode để fallback nếu cần
                if ~isempty(csirsConfig)
                   [dlRank, ~, wbCQI, sbCQI, W_out, ~] = obj.decodeCSIRS(...
                        csirsConfig, currentTime, currentTime + 1e-3, carrierCtx, numRBs);
                else
                    dlRank = 1; wbCQI = 1; sbCQI = ones(1, numSubbands); W_out = 1;
                end
                
                precodingMatrixMap{rnti} = W_out;

                % --- TÍNH TOÁN 7 FEATURES ---
                f_R = obj.AvgThroughputMBps(rnti);
                f_h = dlRank / 4.0;
                f_d = 0;
                f_b = ueCtx.BufferStatusDL;
                
                cqiIdx = min(max(round(wbCQI), 1), 16);
                f_o = obj.CQIToSE(cqiIdx) / 6.0;
                
                if length(sbCQI) ~= numSubbands
                    sbCQI = ones(1, numSubbands) * wbCQI;
                end
                sb_cqi_idx = min(max(round(sbCQI), 1), 16);
                f_g_vec = obj.CQIToSE(sb_cqi_idx) / 6.0;
                
                f_rho_vec = zeros(1, numSubbands);

                exportMatrix(rnti, :) = [f_R, f_h, f_d, f_b, f_o, f_g_vec, f_rho_vec];
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
            numRBGTotal = floor(numRBs / rbgSize);
            
            for k = 1:length(allottedUEs)
                ueID = allottedUEs(k);
                numRBG = sum(freqAllocation(k,:));
                if numRBG > 0
                    mcs = mcsIndex(k);
                    bpp = obj.getBytesPerPRB(mcs);
                    servedBytes(ueID) = numRBG * rbgSize * bpp;
                    currentAlloc(ueID) = numRBG / numRBGTotal;
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
        function [dlRank, pmiSet, widebandCQI, cqiSubband, precodingMatrix, sinrEffSubband] = decodeCSIRS(obj, csirsConfig, pktStartTime, pktEndTime, carrierConfigInfo, numRBs)
             
             % Fallback logic nếu chưa có implementation PHY đầy đủ
             % FIX: Dùng numRBs được truyền vào thay vì carrierConfigInfo.NSizeGrid
             numSB = ceil(numRBs / obj.SubbandSize);
             
             % Logic mặc định giả lập
             dlRank = 1; 
             pmiSet = []; 
             widebandCQI = 15; % Giả lập kênh tốt
             cqiSubband = ones(1, numSB) * 15; 
             precodingMatrix = 1; 
             sinrEffSubband = [];
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
    end
end
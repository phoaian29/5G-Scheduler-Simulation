% function updatedUserPairingMatrix = updateUserPairingMatrix(userPairingMatrix, ueContext, muMIMOConfigDL)
% %updateUserPairingMatrix Update orthogonality matrix for all UEs based on
% % the DL CSI Type II feedback
% %
% %   Note: This is an internal undocumented function and its API and/or
% %   functionality may change in subsequent releases.
% %
% %   UPDATEDUSERPAIRINGMATRIX = nr5g.internal.updateUserPairingMatrix(...
% %   USERPAIRINGMATRIX, UECONTEXT, MUMIMOCONFIGDL) returns the updated orthogonality matrix for
% %   all UEs based on the DL CSI Type II feedback.
% %
% %   USERPAIRINGMATRIX contains the current orthogonality matrix for all
% %   UEs based on the DL CSI Type II feedback.
% %
% %   UECONTEXT contains the context of all the connected UEs.
% %   This is a vector of nr5g.internal.nrUEContext objects of length
% %   equal to the number of UEs connected to the gNB.
% %   Value at index 'i' stores the context of a UE with RNTI 'i'.
% %
% %   MUMIMOCONFIGDL structure contains the constraints that the user pairing
% %   algorithm needs for determining the MU-MIMO candidacy among UEs.
% %
% %   UPDATEDUSERPAIRINGMATRIX contains the updated orthogonality matrix for all
% %   UEs based on the DL CSI Type II feedback.
% 
% %   Copyright 2024 The MathWorks, Inc.
% 
% % Check if all UEs have reported Type II feedback at least once
% ueCSIMeasurementDLArray = [ueContext.CSIMeasurementDL];
% ueCSIMeasurementDL = [ueCSIMeasurementDLArray.CSIRS];
% if ~isreal(ueCSIMeasurementDL(end).W)
%     W = [ueCSIMeasurementDL.W];
%     pOrth = abs(W'*W);
%     pOrth = pOrth/max(pOrth,[],'all');
%     updatedUserPairingMatrix = pOrth <= 1-muMIMOConfigDL.SemiOrthogonalityFactor;
%     usersRank = [ueCSIMeasurementDL.RI];
%     usersRNTI = [ueContext.RNTI];
%     rankIndices = zeros(1, sum(usersRank));
%     count = 1;
%     for i = 1:numel(usersRank)
%         rankIndices(count:count+usersRank(i)-1) = usersRNTI(i);
%         count = count+usersRank(i);
%     end
%     updatedUserPairingMatrix = [rankIndices' updatedUserPairingMatrix.*repmat(rankIndices,size(rankIndices,2),1)];
% else
%     updatedUserPairingMatrix = userPairingMatrix;
% end




function updatedUserPairingMatrix = updateUserPairingMatrix(userPairingMatrix, ueContext, muMIMOConfigDL)
    %updateUserPairingMatrix Update orthogonality matrix for all UEs based on
    % the DL CSI Type II feedback
    %
    %   Note: This is an internal undocumented function and its API and/or
    %   functionality may change in subsequent releases.
    
    %   Copyright 2024 The MathWorks, Inc.
    
    % Check if all UEs have reported Type II feedback at least once
    ueCSIMeasurementDLArray = [ueContext.CSIMeasurementDL];
    ueCSIMeasurementDL = [ueCSIMeasurementDLArray.CSIRS];
    
    if ~isreal(ueCSIMeasurementDL(end).W)
        % Gom tất cả ma trận W của các UE lại
        % Nếu là Subband, W sẽ có kích thước: [NumTx x Tổng_Layers x NumSubbands]
        W = [ueCSIMeasurementDL.W];
        
        % --- BẮT ĐẦU ĐOẠN SỬA ---
        if ismatrix(W)
            % Trường hợp 1: Wideband (W là ma trận 2D) -> Dùng cách cũ
            pOrth = abs(W'*W);
        else
            % Trường hợp 2: Subband (W là ma trận 3D) -> Xử lý từng subband
            [~, totalLayers, numSubbands] = size(W);
            pOrth = zeros(totalLayers, totalLayers);
            
            % Duyệt qua từng subband
            for i = 1:numSubbands
                wSub = W(:,:,i); % Lấy lát cắt 2D của subband thứ i
                
                % Tính độ trực giao tại subband này và cập nhật giá trị lớn nhất (tệ nhất)
                % để đảm bảo an toàn cho việc ghép cặp
                pOrth = max(pOrth, abs(wSub' * wSub));
            end
        end
        % --- KẾT THÚC ĐOẠN SỬA ---
    
        pOrth = pOrth/max(pOrth,[],'all');
        updatedUserPairingMatrix = pOrth <= 1-muMIMOConfigDL.SemiOrthogonalityFactor;
        usersRank = [ueCSIMeasurementDL.RI];
        usersRNTI = [ueContext.RNTI];
        rankIndices = zeros(1, sum(usersRank));
        count = 1;
        for i = 1:numel(usersRank)
            rankIndices(count:count+usersRank(i)-1) = usersRNTI(i);
            count = count+usersRank(i);
        end
        updatedUserPairingMatrix = [rankIndices' updatedUserPairingMatrix.*repmat(rankIndices,size(rankIndices,2),1)];
    else
        updatedUserPairingMatrix = userPairingMatrix;
    end
end
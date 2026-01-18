%nrPMISelect Select precoder matrix indicator
%   [PMI,SINR,SUBBANDIDX] = nrPMISelect(NLAYERS, HEST, NOISEEST, BANDSIZE)
%   selects a vector PMI of precoder matrix indicators used for
%   codebook-based transmission of a number of layers NLAYERS over a MIMO
%   channel. Codebook-based transmission is defined in TS 38.211 Section
%   6.3.1.5. The MIMO channel estimates HEST must be an array of dimensions
%   K-by-L-by-R-by-P where K is the number of subcarriers, L is the number
%   of OFDM symbols, R is the number of receive antennas and P is the
%   number of reference signal ports. Each element of PMI contains a
%   precoding matrix indicator for each frequency subband of size BANDSIZE
%   in resource blocks. This function uses a linear minimum mean squared
%   error (LMMSE) SINR metric for PMI selection.
%
%   SINR is a matrix of dimensions NSB-by-(MaxTPMI+1) containing the
%   signal-to-interference-plus-noise ratio per subband after precoding
%   with all possible precoder matrices for a number of ports P and number
%   of layers NLAYERS. NSB is the number of subbands of size BANDSIZE and
%   MaxTPMI the largest precoder matrix indicator for P and NLAYERS.
%
%   SUBBANDIDX is a NSB-by-2 matrix containing the first and last
%   subcarrier indices of HEST for each of the subbands of size BANDSIZE
%   used for PMI selection.
%
%   SINRPERRETPMI is NSRSINDICES-by-nlayers matrix containing the
%   signal-to-interference-plus-noise ratio per RE after precoding with
%   precoder matrix corresponds to selected TPMI for a number of ports P
%   and number of layers NLAYERS. NSRSINDICES is number of srs indices
%   available in the slot.
% 
%   See also nr5g.internal.nrPrecodedSINR, nr5g.internal.nrMaxPUSCHPrecodingMatrixIndicator.

%   Copyright 2019-2024 The MathWorks, Inc.

function [pmi,sinr,subbandIndices,sinrPerRETPMI] = nrPMISelect(nlayers, hest, noiseest, bandSize)

    nports = size(hest,4);
    maxTPMI = nr5g.internal.nrMaxPUSCHPrecodingMatrixIndicator(nlayers,nports);
    [numSC,numSymb] = size(hest,[1 2]);
    sinrRE = zeros(numSC,numSymb,nlayers,maxTPMI+1);
    % Indices where channel estimation information is available
    ind = find( sum(hest,3:4)~=0 );
    [sc,symb]= ind2sub([numSC numSymb],ind);
    subs = [sc,symb];
    sinrPerRE = zeros(length(sc),nlayers,maxTPMI+1);
    sinrPerRETPMI = zeros(length(sc),nlayers);

    if ~isempty(subs) && noiseest ~= 0
        % Rearrange the channel matrix dimensions from K-by-L-by-R-by-P to
        % R-by-P-by-K-by-L
        H = permute(hest,[3 4 1 2]);
        Htemp = reshape(H,size(H,1),size(H,2),[]);
        Hsrs = Htemp(:,:,ind);

        for tpmi = 0:maxTPMI
            W = nrPUSCHCodebook(nlayers,nports,tpmi).';
            % Get the SINR after precoding with W for each RE
            precodedSINR = nr5g.internal.nrPrecodedSINR(Hsrs,noiseest,W);
            sinrPerRE(:,:,tpmi+1) = precodedSINR;
        end
        % Map the calculated SINR values to corresponding RE indices
        for reIdx = 1:length(ind)
            k = sc(reIdx);
            l = symb(reIdx);
            sinrRE(k,l,:,:) = sinrPerRE(reIdx,:,:);
        end

        % Get the sum of SINR across all layers
        totalSINR = reshape(sum(sinrRE,3),numSC,numSymb,[]);
        [sinrBands,subbandIndices] = nr5g.internal.nrSINRPerSubband(totalSINR, bandSize);
        [~,pmi]  = max(sinrBands,[],2);
        pmi(isnan(sinrBands(:,1))) = NaN;
        pmi = pmi-1; % PMI is 0-based
        sinr = sinrBands; % Return SINR per subband
        subbandStart = 0;
        for i = 1:length(pmi)
            % Average SINR per subband
            srsSubbandIndLogical = ((sc>=subbandIndices(i,1)) & (sc<=subbandIndices(i,2)));
            srsSubbandIndices = sc(srsSubbandIndLogical);

            for indLoop = 1:length(srsSubbandIndices)
                sinrPerRETPMI(subbandStart+indLoop,:) = sinrRE(srsSubbandIndices(indLoop),symb(subbandStart+indLoop),:,pmi(i)+1);
            end
            subbandStart = subbandStart+length(srsSubbandIndices);
        end
    else % If there are no channel estimates available
        pmi = NaN;
        sinr = NaN;
        subbandIndices = NaN;
        sinrPerRETPMI = NaN;
    end
end
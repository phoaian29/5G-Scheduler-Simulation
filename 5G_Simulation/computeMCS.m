function mcs = computeMCS(l2smSRS,carrier,pdsch,sinr,mcsTable)
%   computeMCS returns the Modulation and Coding scheme (MCS) index
% 
%   Note: This is an internal undocumented function and its API and/or
%   functionality may change in subsequent releases
% 
%   MCS = nr5g.internal.computeMCS(L2SMSRS,CARRIER,PDSCH,SINR,MCSTABLE)
%   returns the MCS index, for the specified carrier configuration CARRIER,
%   PDSCH configuration PDSCH, downlink post equalized signal to
%   interference plus noise ratio(sinr) SINR, and type of mcs table
%   MCSTABLE.
% 
%   L2SMSRS is link to system mapping interface object
% 
%   MCS is modulation and coding scheme index calculated using
%   MCSTable and it is of size 1-by-numCodewords.
% 
%   CARRIER is a carrier-specific configuration object as described in
%   <a href="matlab:help('nrCarrierConfig')">nrCarrierConfig</a>
%
%   PDSCH is the physical downlink shared channel configuration object as
%   described in <a href="matlab:help('nrPDSCHConfig')">nrPDSCHConfig</a>.
% 
%   SINR is linear SINR values at SRS locations for all the layers for the
%   selected precoding matrix. It is of size srsIndLen-by-nLayers 
%
%   MCSTABLE is PDSCH MCS Table, as defined in TS 38.214 Table 5.1.3.1-1 to
%   5.1.3.1-4. It must be string input and member of the following set
%   {'qam64','qam256','qam64LowSE','qam1024'}, and each member represents
%   PDSCH MCS tables as defined below :
%
%   'qam64'      - MCS index table 1,
%                     corresponding to TS 38.214 Table 5.1.3.1-1
%   'qam256'     - MCS index table 2,
%                     corresponding to TS 38.214 Table 5.1.3.1-2
%   'qam64LowSE' - MCS index table 3,
%                     corresponding to TS 38.214 Table 5.1.3.1-3
%   'qam1024'    - MCS index table 4,
%                     corresponding to TS 38.214 Table 5.1.3.1-4

%   Copyright 2024 The MathWorks, Inc.

% Get MCS table values
mcsTableValues =  getMCSTable(mcsTable);

% Compute the DL MCS
overhead = 0;
blerThreshold = 0.1;

[~, mcs] = nr5g.internal.L2SM.cqiSelect(l2smSRS,carrier,pdsch,overhead,pow2db(sinr),mcsTableValues(:,2:3),blerThreshold);

end

function MCSTableValues = getMCSTable(tableName)

persistent tables;

if isempty(tables)
    mcsTableClass = nrPDSCHMCSTables;
    props = ["QAM64Table","QAM256Table","QAM64LowSETable","QAM1024Table"];
    numProps = numel(props);
    for i = 1:numProps
        tmpTable = mcsTableClass.(props(i));
        % l2sm accepts cqitable transmit code rate(tcr) values as tcr/1024
        tmpArray = [tmpTable.MCSIndex tmpTable.Qm (tmpTable.TargetCodeRate)*1024 tmpTable.SpectralEfficiency];
        tables{i} = tmpArray(~isnan(tmpArray(:,3)),:);
    end
end
tabNames = ["qam64","qam256","qam64LowSE","qam1024"];
MCSTable = tables(strcmpi(tableName,tabNames));
coder.internal.errorIf(isempty(MCSTable),'nr5g:nrSRSDLChannelMeasurements:InvalidMCSTable');
MCSTableValues = MCSTable{1};
end
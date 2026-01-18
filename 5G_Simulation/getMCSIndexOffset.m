function mcsOffset = getMCSIndexOffset(laConfig,offset,rxResult)
%getMCSIndexOffset returns the updated Modulation and Coding
% Scheme (MCS) offset.
%
%   Note: This is an internal undocumented function and its API and/or
%   functionality may change in subsequent releases
%
%   MCSOFFSET = nr5g.internal.getMCSIndexOffset(LACONFIG, OFFSET, RXRESULT)
%   returns the MCS offset upon receiving PDSCH or PUSCH feedback.
% 
%   MCSOFFSET           - Updated MCS offset after accounting for the
%                         downlink(DL) or uplink(UL) feedback
%   LACONFIG              - Link adaptation (LA) configuration structure
%                           for DL or UL transmissions.
%       InitialOffset     - Initial MCS offset applied to all UEs. This
%                           value factors in the channel measurements
%                           errors at the UE node. To determine the final
%                           MCS for the DL transmission, the scheduler
%                           applies this offset to the channel measurement
%                           based MCS. It is an integer in the range [-27,
%                           27]. The default value is 0.
%       StepUp            - Incremental value for the MCS offset when a
%                           packet reception fails. It is a numeric
%                           scalar in the range [0, 27]. The default value
%                           is 0.27.
%       StepDown          - Decremental value for the MCS offset when a
%                           packet reception is successful. It is a
%                           numeric scalar in the range [0, 27]. The
%                           default value is 0.03.
%   OFFSET              - Current MCS offset
%   RXRESULT            - Reception feedback where '1' means successful
%                         and '0' means failure
%
% The function implements the link adaptation algorithm outlined in
% reference [1], with a notable modification. In contrast to the original
% approach, which applies an offset to the Signal-to-Noise Ratio (SNR), the
% adopted method applies the offset directly to the MCS index.
% 
% The adopted link adaptation algorithm consists of these steps:
%   1. Define the LA configuration parameters StepUp and StepDown.
%      The target Block error rate, TargetBLER, is defined as StepDown /
%      (StepDown + StepUp).
%   2. Upon receiving channel measurements feedback (CQI) from the UE node,
%      the gNB maps the reported CQI to an appropriate MCS value. The MCS
%      table used for the DL LA complies with 3GPP TS 38.214, Table
%      5.1.3.1-2.
%   3. Reset MCSOffset to InitialOffset on each CSI reporting
%      periodicity, considering any channel measurement errors.
%   4. Determine the MCS for physical downlink shared channel (PDSCH)
%      transmission as the MCS reported by the UE node minus the MCS offset
%      (PDSCH_MCS = UEReportedMCS - MCSOffset)
%   5. If the UE reports a successful PDSCH reception, decrease the
%      MCSOffset by StepDown.
%   6. If the UE reports a failed PDSCH reception, increase the
%      MCSOffset by StepUp.
%   
% Similarly, for UL link adaptation, the gNB calculates the  UL MCS
% based on the sounding reference signal (SRS) channel measurements.
% Determine the physical uplink shared channel (PUSCH) as the MCS
% calculated by the gNB node minus the MCS offset (PUSCH_MCS =
% gNBMeasuredMCS - MCSOffset)
%   
% [1] M. G. Sarret, D. Catania, F. Frederiksen, A. F. Cattoni, G.
% Berardinelli, and P. Mogensen,  "Dynamic outer loop link adaptation for
% the 5G centimeterwave concept", in Proc. Eur. Wireless, 2015, pp. 1–6

%   Copyright 2023 The MathWorks, Inc.


% As per 3GPP TS 38.214 - Table 5.1.3.1-2, '27' is the maximum
% configurable MCS index.
if rxResult == 1
    % Reception is successful
    mcsOffset = max(-27, offset-laConfig.StepDown);
else
    % Reception is failure 
    mcsOffset = min(27, offset+laConfig.StepUp);
end
end


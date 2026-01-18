classdef MACConstants
    %MACConstants Class holds information about various MAC layer constants
    %
    %   Note: This is an internal undocumented class and its API and/or
    %   functionality may change in subsequent releases.

    %   Copyright 2021-2024 The MathWorks, Inc.

    %#codegen

    properties(Constant)
        %BufferSizeLevelFiveBit Buffer size level table for buffer status report (BSR) control element (CE) as per 3GPP TS 38.321 Table 6.1.3.1-1
        %   This table is applicable for short and short truncated BSR
        %   formats. The table has 32 buffer levels with each level
        %   represented by a 5-bit value. The buffer size level identifies
        %   the total amount of data available across all logical channels
        %   of a logical channel group.
        BufferSizeLevelFiveBit = [0;10;14;20;28;38;53;74;
            102;142;198;276;384;535;745;1038;
            1446;2014;2806;3909;5446;7587;10570;
            14726;20516;28581;39818;55474;77284;
            107669;150000;150000];
        %BufferSizeLevelEightBit Buffer size level table for BSR CE as per 3GPP TS 38.321 Table 6.1.3.1-2
        %   This table is applicable for long and long truncated BSR
        %   formats. The table has 256 buffer levels with each level
        %   represented by a 8-bit value. The last buffer level is reserved
        %   for future use.
        BufferSizeLevelEightBit = [0;10;11;12;13;14;15;16;17;18;19;20;22;23;25;26;28;30;32;34;36;38;
            40;43;46;49;52;55;59;62;66;71;75;80;85;91;97;103;110;117;124;
            132;141;150;160;170;181;193;205;218;233;248;264;281;299;318;339;
            361;384;409;436;464;494;526;560;597;635;677;720;767;817;870;926;987;
            1051;1119;1191;1269;1351;1439;1532;1631;1737;1850;1970;2098;2234;2379;
            2533;2698;2873;3059;3258;3469;3694;3934;4189;4461;4751;5059;5387;5737;
            6109;6506;6928;7378;7857;8367;8910;9488;10104;10760;11458;12202;12994;
            13838;14736;15692;16711;17795;18951;20181;21491;22885;24371;25953;
            27638;29431;31342;33376;35543;37850;40307;42923;45709;48676;51836;
            55200;58784;62599;66663;70990;75598;80505;85730;91295;97221;103532;
            110252;117409;125030;133146;141789;150992;160793;171231;182345;194182;
            206786;220209;234503;249725;265935;283197;301579;321155;342002;364202;
            387842;413018;439827;468377;498780;531156;565634;602350;641449;683087;
            727427;774645;824928;878475;935498;996222;1060888;1129752;1203085;
            1281179;1364342;1452903;1547213;1647644;1754595;1868488;1989774;
            2118933;2256475;2402946;2558924;2725027;2901912;3090279;3290873;
            3504487;3731968;3974215;4232186;4506902;4799451;5110989;5442750;
            5796046;6172275;6572925;6999582;7453933;7937777;8453028;9001725;
            9586039;10208280;10870913;11576557;12328006;13128233;13980403;
            14887889;15854280;16883401;17979324;19146385;20389201;21712690;
            23122088;24622972;26221280;27923336;29735875;31666069;33721553;
            35910462;38241455;40723756;43367187;46182206;49179951;52372284;
            55771835;59392055;63247269;67352729;71724679;76380419;81338368;81338368;inf];

        %CQITable CQI table as per TS 38.214 - Table 5.2.2.1-3. This table
        %is used to indicate channel quality for both UL and DL
        % Modulation CodeRate Efficiency
        CQITable = [0  0   0
            2 	78      0.1523
            2 	193 	0.3770
            2 	449 	0.8770
            4 	378 	1.4766
            4 	490 	1.9141
            4 	616 	2.4063
            6 	466 	2.7305
            6 	567 	3.3223
            6 	666 	3.9023
            6 	772 	4.5234
            6 	873 	5.1152
            8 	711 	5.5547
            8 	797 	6.2266
            8 	885 	6.9141
            8 	948 	7.4063];

        %MCSTable MCS table as per TS 38.214 - Table 5.1.3.1-2
        %This table is used to indicate MCS for both UL and DL
        % Modulation CodeRate Efficiency
        MCSTable = [2	120	0.2344
            2	193     0.3770
            2	308     0.6016
            2	449     0.8770
            2	602     1.1758
            4	378     1.4766
            4	434     1.6953
            4	490     1.9141
            4	553     2.1602
            4	616     2.4063
            4	658     2.5703
            6	466     2.7305
            6	517     3.0293
            6	567     3.3223
            6	616     3.6094
            6	666     3.9023
            6	719     4.2129
            6	772     4.5234
            6	822     4.8164
            6	873     5.1152
            8	682.5	5.3320
            8	711     5.5547
            8	754     5.8906
            8	797     6.2266
            8	841     6.5703
            8	885     6.9141
            8	916.5	7.1602
            8	948     7.4063
            2    0       0
            4    0       0
            6    0       0
            8    0       0];

        %DLType Value to specify downlink direction or downlink symbol type
        DLType = 0;

        %ULType Value to specify uplink direction or uplink symbol type
        ULType = 1;

        %GuardType Value to specify guard symbol type
        GuardType = 2;

        %NominalRBGSizePerBW Nominal RBG size for the specified bandwidth in accordance with 3GPP TS 38.214, Section 5.1.2.2.1
        NominalRBGSizePerBW = [
            36   2   4
            72   4   8
            144  8   16
            275  16  16 ];

        %txBandwidthConfig Channel bandwidth and resource blocks table as per 3GPP TS 38.104 -
        % Section 5.3.2. Each row has 3 columns: bandwidth(Hz), SCS (Hz), maximum
        % number of RBs in the transmission bandwidth. Value 0 in 3rd column means
        % invalid combination of bandwidth and SCS. Only SCSs upto 120e3 are
        % considered
        txBandwidthConfig = [5e6    15e3  25;
            5e6    30e3  11;
            5e6    60e3  0;
            5e6    120e3 0;
            10e6   15e3  52;
            10e6   30e3  24;
            10e6   60e3  11;
            10e6   120e3 0;
            15e6   15e3  79;
            15e6   30e3  38;
            15e6   60e3  18;
            15e6   120e3 0;
            20e6   15e3  106;
            20e6   30e3  51;
            20e6   60e3  24;
            20e6   120e3 0;
            25e6   15e3  133;
            25e6   30e3  65;
            25e6   60e3  31;
            25e6   120e3 0;
            30e6   15e3  160;
            30e6   30e3  78;
            30e6   60e3  38;
            30e6   120e3 0;
            35e6   15e3  188;
            35e6   30e3  92;
            35e6   60e3  44;
            35e6   120e3 0;
            40e6   15e3  216;
            40e6   30e3  106;
            40e6   60e3  51;
            40e6   120e3 0;
            45e6   15e3  242;
            45e6   30e3  119;
            45e6   60e3  58;
            45e6   120e3 0;
            50e6   15e3  270;
            50e6   30e3  133;
            50e6   60e3  66;
            50e6   120e3 32;
            60e6   15e3  0;
            60e6   30e3  162;
            60e6   60e3  79;
            60e6   120e3 0;
            70e6   15e3  0;
            70e6   30e3  189;
            70e6   60e3  93;
            70e6   120e3 0;
            80e6   15e3  0;
            80e6   30e3  217;
            80e6   60e3  107;
            80e6   120e3 0;
            90e6   15e3  0;
            90e6   30e3  245;
            90e6   60e3  121;
            90e6   120e3 0;
            100e6  15e3  0;
            100e6  30e3  273;
            100e6  60e3  135;
            100e6  120e3 66;
            200e6  15e3  0;
            200e6  30e3  0;
            200e6  60e3  264;
            200e6  120e3 132;
            400e6  15e3  0;
            400e6  30e3  0;
            400e6  60e3  0;
            400e6  120e3 264;
            ];
    end
end
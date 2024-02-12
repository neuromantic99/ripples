
clear

load('\\128.40.224.64\marcbusche\Jana\Matlab Code\Relevant_Code_Trajectories\overview Tables\overview_table_A_onlyRecsWithRippleData.mat')

static_vars.genotype = overview_table_A.Genotype;
static_vars.trajectory = overview_table_A.TrajectoryPrefix;
static_vars.exp_date = overview_table_A.AnimalID;
static_vars.timepoint = overview_table_A.Timepoint;
static_vars.save_dir = '\\128.40.224.64\marcbusche\Jana\Neuropixels\Trajectories\Processed data';   %%%%This is the directory where your pre-proceed data will be saved
static_vars.exp_name = overview_table_A.BaselineNo;
static_vars.probe_num =overview_table_A.Imec_1_Traj;





%%%%%
load (['\\128.40.224.64\MarcBusche\Matlab Code\NPX Config\long_linear_shank_ChanMapBank01.mat']);
swr_vits.depth_axis = [20:20:7680];
%%%%%
swr_vits.swr_freq_range = [80 250]; %% Dupret preprint 10/2023
swr_vits.supra_freq_range = [250 500]; %%%supra-ripple, in Dupret preprint from 10/2023 [200 500]
swr_vits.swr_max_length = 0.4; %%%Buzsaki states ~40-120ms duration for SWRs
%although https://www.science.org/doi/10.1126/science.aax0758 [Buzsaki]
%uses 400ms as maximum length for detection (default)
swr_vits.swr_cluster_gap = 0.12; %% 120ms as default
swr_vits.medLevel = 2; %% 5 as default, I've tried to use a higher medLevel but it would be too restrictive
swr_vits.detect_type = 'Dupret'; %% median absolute deviation (MAD) as default but otherwise 'Dupret' method using medLevel.*median
swr_vits.do_multichan = 'Yes'; %% Yes as default
swr_vits.onset_offset = 0.5; % when the envelope falls below x times the detections threshold, default =0.5
swr_vits.restTimeWindow = 6; %in seconds, minimal duration of resting periods to be included in SWR detection
swr_vits.CARtype = 'Original'; %% either 'Original' (all channels) or 'CA1' for referencing


%%
for anim = 1
    for experiment = 2 % [1:length(static_vars.exp_name{anim})]
        for imec_probe = static_vars.probe_num{anim}
            clallbut imec_probe genotype anim experiment...
                do_spec swr_vits static_vars baseline

            rootZ = [static_vars.save_dir '\' static_vars.genotype{anim} static_vars.trajectory{anim} static_vars.exp_date{anim} static_vars.timepoint{anim}];
            cd (rootZ)

            %%% get probe trajectory
            idcs   = strfind(pwd,'\');
            newdir = rootZ(1:idcs(end));
            load(['probe_trajectory_details_0603_imec'  num2str(imec_probe)]);%loads in trajectory for each probe

            %%% load in bandpower resting state data
            cd([rootZ '\Baseline' num2str(static_vars.exp_name{anim}(experiment))])
            load(['lfp_frequency_bands_resting_imec' num2str(imec_probe) '.mat'])
            SWR_pow=freq.SWR_pow_rest;
            SRP_pow=freq.SRP_pow_rest;

            %%% find CA1 channels based on trajectory explorer
            %%% and implantation coordinates
            idxCA1 = find(~cellfun(@isempty,regexp(probe_details.alignedregions,['|Field CA1|'])));
            CAchans = 384-idxCA1;
            %%%find strongest power ripple channel and 2 channels
            %%%before and after
            SWR_pow_CA1= SWR_pow(CAchans);
            [~,b1] = max(SWR_pow_CA1);
            maxSWRchan=CAchans(b1);
            SWR_chans=[(CAchans(b1)-2):(CAchans(b1)+2)];


            %%load in raw LFP data and synchronise to behavioural data
            load (['lfp_data_CAR_fin_imec' num2str(imec_probe) '.mat']);
            load (['synch_channel_data_AP_LP_imec' num2str(imec_probe)]);
            %%%need to synchronise LP data to behavioural recording times
            TF_LF = islocalmax(double(sync_dat_LF),'FlatSelection','first');
            sync_dat_idxLF = find(TF_LF==1);
            onset_syncLF = sync_dat_idxLF(1);
            offset_syncLF = sync_dat_idxLF(end-4);
            %%%chop LFP to match behaviour recording range
            dataArray_lfp=dataArray_lfp(:,onset_syncLF:offset_syncLF);
            lfp_tim = [0:size(dataArray_lfp,2)-1]./lfp_vits.sampling_rate;
            % need to make bad/ignored channels finite
            dataArray_lfp (isnan(sum(dataArray_lfp,2)),:)=0;
            %%%optionally filter lfps for 50Hz noise
            %%dataArray_lfp=avgXnpxMainsFilt(dataArray_lfp,1,lfp_vits,1);

            %%%%load behavioural data
            load behavioural_state_classifiers_final
            %%%get resting state indices and ignore those with
            %%%movement/locomotion
            resting_data = behavioural_details.restTS{imec_probe,1};
            locomotion_period = find(resting_data == 0);

            %%%mask out time-points in LFP data and time vector corresponding
            %%%to locomotion
            disp('Masking out movement/locomotion in LFP data')
            restLFP = dataArray_lfp;
            restLFP(:,locomotion_period)=NaN;
            restLFPtim = lfp_tim;
            restLFPtim(locomotion_period)=NaN;

            %%% exclude resting periods below a certain threshold (see
            %%% above)

            resting_data = behavioural_details.restTS{1,1};
            vec=[1:length(resting_data)];
            resting_timepoints=vec(resting_data);
            restingperiods_pre=SplitVec(resting_timepoints,'consecutive');
            length_RestingPeriods=cellfun(@length,restingperiods_pre);
            idxLongRest=find(length_RestingPeriods>= swr_vits.restTimeWindow*lfp_vits.sampling_rate); %% only resting periods longer then 6s (to be set at the beginning of the code)
            restingperiods=cell2mat(restingperiods_pre(idxLongRest));            

            rest_time=length(restingperiods)/lfp_vits.sampling_rate;%% rest time after removing too short periods
            swr_dat.restTimeBehaviour=length(resting_timepoints)/lfp_vits.sampling_rate;

            % %             %%%find periods with high theta/delta ratio in
            % %                 CA1, excluding periods where there is high
            % theta power present based on the assumption that SWR
            % frequencies and theta are mutually exclusive, I found that it
            % was too restrictive
            % %
            % %             theta=bandpass(dataArray_lfp(maxSWRchan,:),freq_vits.theta_range,lfp_vits.sampling_rate);
            % %             dataEnvtheta = abs(hilbert(theta));
            % %             delta=bandpass(dataArray_lfp(maxSWRchan,:),freq_vits.delta_range,lfp_vits.sampling_rate);
            % %             dataEnvdelta = abs(hilbert(delta));
            % %             ratio=abs(dataEnvtheta)./abs(dataEnvdelta);
            % %             ratio(:,locomotion_period)=NaN;
            % %             high_theta=find(ratio>swr_vits.ThetaDeltaThreshold);
            % %
            % %             rest_time=(size(dataArray_lfp,2)-length(high_theta)-length(locomotion_period))/lfp_vits.sampling_rate;
            % %             swr_dat.restTimeBehaviour=(size(dataArray_lfp,2)-length(locomotion_period))/lfp_vits.sampling_rate;


            %%  chop CAR data - either original or use CA data

            CARdat=[];
            if strcmp(swr_vits.CARtype,'Original')
                CARdat=mean(dataArray_lfp,1); % is this the correct way of using all channels for referencing?
                %%% or get hippocampus only channels and average for common reference
            elseif strcmp(swr_vits.CARtype,'CA1')
                CARdat = dataArray_lfp(CAchans,:);   %%%use average of CA1 channels for CAR
                CARdat = mean(CARdat,1,'omitnan');
            end


            CARdat_SWR_pre = bandpass(CARdat,swr_vits.swr_freq_range,lfp_vits.sampling_rate);
            CARdat_SWR = sgolayfilt(abs(hilbert(CARdat_SWR_pre)),4,101); %% needed further down

            %
            figure;plot(swr_vits.depth_axis,SWR_pow)
            hold on;plot(swr_vits.depth_axis,SRP_pow)
            %             yline(2*CARdat_SWR)
            legend('SWR Power','SRP Power','CAR Power');xlabel ('Depth (microns)')
            ylabel('Power (AU)')
            drawnow

            swr_dat.SWR_pow = SWR_pow;
            swr_dat.SRP_pow = SRP_pow;
            swr_dat.CARdat_SWR = CARdat_SWR;
            swr_dat.CARdat = CARdat;





            %%
            %%%%%engine to identify ripples base on Dupret Lab
            %%%%%https://doi.org/10.1038/s41593-021-00804-w; modified +
            %%%%%using October 2023 preprint by Dupret lab as a basis
            %%% 1. filter the data into ripple band 80-250 Hz
            %%% 2. Also calculate instantaneous amplitude (envelope) in ripple band using Hilbert
            %%% transform and smooth the envelope using a 4th order
            %%% sgolayfilt over 101 data points, if one of the 5 Detection
            %%% channels is noisy add another channel

            %%%pre-allocate for speed
            dataEnv=NaN(size(dataArray_lfp));
            dataLFP_SWR=NaN(size(dataArray_lfp));
            dataEnv_unsmoothed=NaN(size(dataArray_lfp));

            for chan= SWR_chans
                [dataLFP_SWR(chan,:), ~]=bandpass(dataArray_lfp(chan,:),swr_vits.swr_freq_range,lfp_vits.sampling_rate);%%%sharp wave ripple range
                dataEnv(chan,:) = sgolayfilt(abs(hilbert(dataLFP_SWR(chan,:))),4,101); %%% smooth data
                dataEnv_unsmoothed(chan,:)=abs(hilbert(dataLFP_SWR(chan,:)));
                if sum(dataLFP_SWR(chan,:))==0 || isnan(sum(dataLFP_SWR(chan,:)))
                    disp('Noisy channel in CA1, include more channels')
                    SWR_chans=[SWR_chans,(max(SWR_chans)+1)];
                end
            end


            disp('Spectrum calculated...')



            %% ACTUAL PROCESSING BIT

            %%% set threshold for events that are too brief < 4 ripple cycles
            %%% (see Gava (Dupret Lab paper)
            swr_thresh_length_short = floor(4/swr_vits.swr_freq_range(2)*lfp_vits.sampling_rate);   %%%min number of datapoints

            %%% set threshold for events that are too long > 120ms 4 ripple cycles
            %%% see https://www.ncbi.nlm.nih.gov/pmc/articles/PMC6310484/ (Buzsaki)
            swr_thresh_length_long = swr_vits.swr_max_length./(1/lfp_vits.sampling_rate);

            if strcmp(swr_vits.do_multichan,'No')
                %%%find strongest power ripple channel
                [~,b1] = max(SWR_pow(SWR_chans));
                SWR_chans = SWR_chans(b1);
            end

            %% cycle through channel(s)
            filt_swr=[]; %bandpass filtered
            unfilt_swr=[]; %broadband lfp
            envelope_swr=[]; %envelope for each event
            swr_chan=[];

            ev_onset=[]; % last time the event has been below 0.5/0.75 of detection threshold
            ev_offset=[];% first time the event drops below 0.5/0.75 of detection threshold
            ev_time=[]; % duration from onset to offset
            ev_peak=[]; % timepoint of max Amplitude of the smoothed hilbert transfrom envelope
            peak_amp=[];  % max Amplitude of the smoothed hilbert transfrom envelope

            ev_count=0;

            for ch=SWR_chans(1):SWR_chans(end)
                disp(['Identifying SWRs in channel ' num2str(ch) ' of ' num2str(SWR_chans(end))])
                SWR_chan_data=[];pot_evs_len=[];pot_evs_end=[];pot_evs_val=[];ind_evs=[];
                SWR_chan_data = dataEnv(ch,:);


                if sum(SWR_chan_data)>0 %% exclude broken channels

                    %%%only include resting periods above minimum length
                    %%%threshold
                    SWR_chan_data(setdiff(1:end,restingperiods)) = NaN;

                    %%% next remove data when ripple band envelope signals are below threshold
                    if strcmp(swr_vits.detect_type,'MAD')
                        detection_threshold = swr_vits.medLevel*mad(SWR_chan_data,1);
                    else
                        %%% X *median of the envelope values of that channel (Dupret approach)
                        detection_threshold = swr_vits.medLevel*median(SWR_chan_data,'omitnan');
                    end

                    SWR_chan_data = double((SWR_chan_data > detection_threshold));
                    SWR_chan_data(SWR_chan_data==0)=NaN;

                    %%%identify potential events
                    [pot_evs_len,pot_evs_st,pot_evs_end,pot_evs_val] = SplitVec(SWR_chan_data,'equal','length','first','last','firstval');
                    pot_evs_val=pot_evs_val';
                    ind_evs = find(~isnan(pot_evs_val) & pot_evs_len >= swr_thresh_length_short & pot_evs_len <= swr_thresh_length_long);
                    for ev= 1:length(ind_evs)
                        ev_count=ev_count+1;
                        ev_off_tmp=[];ev_on_tmp=[];

                        ev_on_tmp = find(dataEnv(ch,1:pot_evs_st(ind_evs(ev))) <= (swr_vits.onset_offset*detection_threshold),1,'last');
                        if ~isempty(ev_on_tmp)
                            ev_onset(ev_count) = ev_on_tmp;
                        else
                            ev_onset(ev_count) = NaN;
                        end

                        ev_off_tmp = find(dataEnv(ch,pot_evs_end(ind_evs(ev)):end) < (swr_vits.onset_offset*detection_threshold),1,'first');
                        if ~isempty(ev_off_tmp)
                            ev_offset(ev_count) = pot_evs_end(ind_evs(ev)) + ev_off_tmp;
                        else
                            ev_offset(ev_count) = NaN;
                        end

                        %%%calculate envelope peak for each ripple 
                        
                        if ~isnan(ev_onset(ev_count)) && ~isnan(ev_offset(ev_count))
                            [peak_amp_pre,ev_peak_tmp1] = max(dataEnv(ch,ev_onset(ev_count):ev_offset(ev_count)));
                            ev_peak(ev_count)=ev_peak_tmp1+ev_onset(ev_count);
                            peak_amp(ev_count)=peak_amp_pre;
                        else
                            ev_peak(ev_count)=NaN;
                            peak_amp(ev_count)=NaN;
                        end


                        %%%catch in case at start or end of recording...
                        if isnan(ev_onset(ev_count)) || isnan(ev_offset(ev_count))
                            filt_swr{ev_count}=NaN;
                            swr_dat.unfilt_swr{ev_count}=NaN;
                            envelope_swr{ev_count}=NaN;
                        else
                            filt_swr{ev_count}=dataLFP_SWR(ch,ev_onset(ev_count):ev_offset(ev_count));
                            unfilt_swr{ev_count}=dataArray_lfp(ch,ev_onset(ev_count)-200:ev_offset(ev_count)+200);
                            envelope_swr{ev_count}=dataEnv(ch,ev_onset(ev_count):ev_offset(ev_count));

                        end
                        swr_chan(ev_count)=ch;
                        ev_time(ev_count) = ev_offset(ev_count) - ev_onset(ev_count);
                    end
                else
                    disp('Noisy channel in CA1')
                end
            end


            if ev_count > 0
                %%%evaluate number of clusters:
                %%%cluster data using k nearest neighbour

                disp('Identifying SWR clusters')
                Idx = rangesearch(ev_peak',ev_peak',swr_vits.swr_cluster_gap./(1/lfp_vits.sampling_rate),"SortIndices",true); %% difference is no longer caculted between onset times but between peaks
                for j=1:length(Idx)
                    if j <= length(Idx)
                        tmp=[];ind_dups=[];
                        for k = 1:length(Idx)
                            tmp(k)=isequal(sort(Idx{j}),sort(Idx{k}));
                        end
                        ind_dups = find(tmp==1);
                        ind_dups = ind_dups(ind_dups ~= j);
                        Idx(ind_dups)=[]; %%%remove duplicates
                    end
                end

                disp(['Detected' num2str(length(Idx))])

                %%%ignore any events that dont occur on at least two channels
                if strcmp(swr_vits.do_multichan,'Yes')
                    disp('Ignoring SWRs not occuring on multiple channels..')
                    Idx(cellfun(@length,Idx) < 2)=[];
                end

               



                %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

                for j=1:length(Idx)
                    swr_on_ev=[];
                    %%% get the earlierst possible onset time an latest
                    %%% possible offset time of each cluster for checking
                    %%% for potential overlap
                    swr_evs_onset_times(j) = min(ev_onset(Idx{j}))./lfp_vits.sampling_rate;
                    swr_evs_offset_times(j) = max(ev_offset(Idx{j}))./lfp_vits.sampling_rate;
                end

                %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

                %%% find overlapping clusters and merge them

                Onset_offset = [swr_evs_onset_times' swr_evs_offset_times'];
                [Onset_offset,idx] = sortrows(Onset_offset);

                %%% find clusters with same onset time and only keep the shortest one
                [~,ind_sorted_uq,~] = unique(swr_evs_onset_times);
                vec1=[1:size(Onset_offset,1)];
                not_uq=~ismember(vec1,ind_sorted_uq);
                not_uq=unique(swr_evs_onset_times(not_uq));
                for u=1:length(not_uq)
                    temp1=find(Onset_offset(:,1)==not_uq(u));
                    offset_tmp1=min(Onset_offset(temp1,2));
                    Onset_offset(temp1(1),1) = not_uq(u);
                    Onset_offset(temp1(1),2) = offset_tmp1;
                    Onset_offset([temp1(2):temp1(length(temp1))],:) = NaN;
                    idx([temp1(2):temp1(length(temp1))],:) = NaN;
                end

                %%% find clusters with same offset time and keep the shortest one
                [~,ind_sorted_uq,~] = unique(Onset_offset(:,2));
                vec1=[1:size(Onset_offset,1)];
                not_uq=~ismember(vec1,ind_sorted_uq);
                not_uq=unique(Onset_offset(not_uq,2));
                for u=1:length(not_uq)
                    temp1=find(Onset_offset(:,2)==not_uq(u));
                    onset_tmp1=max(Onset_offset(temp1,1));
                    Onset_offset(temp1(1),2) = not_uq(u);
                    Onset_offset(temp1(1),1) = offset_tmp1;
                    Onset_offset([temp1(2):temp1(length(temp1))],:) = NaN;
                    idx([temp1(2):temp1(length(temp1))],:) = NaN;
                end

                %%% remove NaNs in indexing vector and onset offset vector
                Onset_offset=Onset_offset(~isnan(Onset_offset(:,1)),:);
                idx=idx(~isnan(idx(:,1)),:);

                %%% find overlapping clusters (offset time of preceeding cluster is greater
                %%% then onset of the next cluster)

                for v=1:length(Onset_offset)
                    if v==1
                        Onset_offset(v,3)=1;
                    else
                        Onset_offset(v,3)=Onset_offset(v,1)>Onset_offset((v-1),2);
                    end
                end

                %%% find clusters in a row that are overlapping
                tmp1split=find(cellfun(@sum,SplitVec(Onset_offset(:,3)))==0)';
                idx_split=SplitVec(Onset_offset(:,3),[],'index');

                cl_tmp1=[];

                %%% merge overlapping clusters and replace others with NaNs
                for x=tmp1split

                    cl_tmp = idx_split{1,x};
                    cl_tmp1 = [min(cl_tmp)-1 cl_tmp];
                    onset_1= max(Onset_offset(cl_tmp1,1));
                    offset_1= min(Onset_offset(cl_tmp1,2));
                    Onset_offset(cl_tmp1(1),1) = onset_1;
                    Onset_offset(cl_tmp1(1),2) = offset_1;
                    %%%replace onset and offset for the remaining cluster IDs
                    %%%with NaNs
                    Onset_offset([cl_tmp1(2):cl_tmp1(length(cl_tmp1))],:) = NaN;
                    idx([cl_tmp1(2):cl_tmp1(length(cl_tmp1))],:) = NaN;

                end

                %%% remove NaNs in indexing vector and onset offset vector
                Onset_offset=Onset_offset(~isnan(Onset_offset(:,1)),:);
                idx_final=idx(~isnan(idx(:,1)),:);

                %%% remove merged/excluded clusers
                disp(['Removed ' num2str(size(Idx,1)-size(idx_final,1)) ' clusters after merging'])
                Idx= {Idx{idx_final',1}};


                %%%%%%%%%%%%%%%%%%%%%%%%%%%%


                %% plot the clustering
                h1=figure;
                hold on;plot([1:length(resting_data)]./lfp_vits.sampling_rate,resting_data.*250)
                ylabel('Channel');xlabel('Time(s)')
                legend ('Resting Periods (AU)','autoupdate','off')
                swr_dat.swr_no_events = length(Idx);
                swr_dat.swr_per_event = cellfun(@length,Idx);
                swr_dat.events_per_cluster = Idx;
                swr_dat.unfilt_LFP_CA1 = restLFP([min(CAchans)-10:max(CAchans)+10],:); %% saving unfiltered lfp data for CA1 channels +/- 10 channels during resting state
                swr_dat.CSDcheck = [];
                swr_dat.CSD = [];
                h2=figure;

                low_lim=min(CAchans)-10;
                upper_lim=max(CAchans)+10;
                [~,idx]=max(swr_dat.SWR_pow(low_lim:upper_lim));
                max_power_channel = idx*2;

                for j=1:length(Idx)


                    [swr_dat.peak_amp_Env(j),swr_dat.idx_maxRipple(j)]= max(peak_amp(Idx{j})); %% find ripple with the highest Amplitude per Cluster and save its characteristics

                    swr_dat.swr_evs_onset_times(j) = ev_onset(Idx{j}(swr_dat.idx_maxRipple(j)))./lfp_vits.sampling_rate;
                    swr_dat.swr_evs_offset_times(j) = ev_offset(Idx{j}(swr_dat.idx_maxRipple(j)))./lfp_vits.sampling_rate;
                    swr_dat.swr_evs_peak_times(j) = ev_peak((Idx{j}(swr_dat.idx_maxRipple(j))))./lfp_vits.sampling_rate;
                    swr_dat.all_swr_evs_peak_times{j} = {ev_peak(Idx{j})./lfp_vits.sampling_rate};
                    swr_dat.all_peak_amp_Env{j}= {peak_amp(Idx{j})};
                   

                    figure(h1);
                    hold on;plot(ev_onset(Idx{j})'./lfp_vits.sampling_rate,swr_chan(Idx{j})','o')
                    [~,ev_ind] = min(ev_onset(Idx{j}));  %%%get the first swr in cluster

                    
                    swr_dat.swr_evs_filt_swr{j} = filt_swr{(Idx{j}(swr_dat.idx_maxRipple(j)))};
                    swr_dat.swr_evs_unfilt_swr{j} = unfilt_swr{(Idx{j}(swr_dat.idx_maxRipple(j)))};
                    swr_dat.swr_evs_envelope{j} = envelope_swr{(Idx{j}(swr_dat.idx_maxRipple(j)))};

                    if j <=25
                        figure(h2);
                        subplot(5,5,j)
                        hold on;plot(swr_dat.swr_evs_filt_swr{j});axis off
                    end

                    %% further quality controls for each putative ripple event adapted from Dupret lab preprint (October 2023)
                    % (1) The ripple band power (derived from squaring the mean ripple amplitude) in the detection channel should exceed twice
                    % the magnitude obtained for the reference channel.
                    % (2) The mean frequency of the event should surpass 100 Hz.
                    % (3) The event must comprise a minimum of 4 complete
                    % ripple cycles. (already checked this above!)
                    % (4) The power in the ripple band should be at least double compared to the control high frequency band.


                    %(1)
                    ev_ripple_band_power = mean(swr_dat.swr_evs_envelope{j})^2;
                    ref_ripple_band_power = mean(swr_dat.CARdat_SWR(ev_onset(Idx{j}(swr_dat.idx_maxRipple(j))):ev_offset(Idx{j}(swr_dat.idx_maxRipple(j)))))^2; %swr_dat.CARdat_SWR is the envelope of the CARdat
                    ref_check=ev_ripple_band_power>(2*ref_ripple_band_power);
                    %(2)
                    ev_mean_freq = mean(instfreq(swr_dat.swr_evs_filt_swr{j},lfp_vits.sampling_rate)); %% double check how to do it correctly
                    freq_check= ev_mean_freq>100;
                    %(4)
                    long_lfp =swr_dat.swr_evs_unfilt_swr{j}; % unfiltered LFP saved with additional timepoints 
                    correct_lfp = long_lfp(201:end-200); % need to remove 200 timepoints before and after the ripple as they have been additionally saved above
                    pow_swr_range=bandpower(correct_lfp,lfp_vits.sampling_rate, swr_vits.swr_freq_range);
                    pow_supraRippleRange =bandpower(correct_lfp,lfp_vits.sampling_rate, swr_vits.supra_freq_range);
                    pow_check=pow_swr_range>2*pow_supraRippleRange;

                    all_checks = ref_check*freq_check*pow_check;

                    if all_checks==1
                        swr_dat.qualitycheck(j)=1;
                    else
                        swr_dat.qualitycheck(j)=0;
                    end




                    %% check if the CSD profile confirms that it is a
                    %%% Ripple event rather than fast gamma
                    
                    %get LFP data around the ripple and run CSD analysis
                    
                    t=round(swr_dat.swr_evs_peak_times(j)*2500);
                    timevec=[t-200:t+200];
                    dataCSD_SWR=swr_dat.unfilt_LFP_CA1(:,timevec);
                    CSD = getCSDfromLFP(dataCSD_SWR);
                    CSDoutput =CSD.csd;

                    sm_CSD=imgaussfilt(interp2(CSDoutput),5); % smoothing 
                    vec=1:size(sm_CSD,1);
                    depth_profile=sm_CSD(:,401); % depth profil at maximum ripple amplitude
                    [a]=islocalmax(sm_CSD(:,401)); % find strongest source
                    if sum(a)>0
                        max_channels=vec(a);
                        [~,idx2]=min(abs(max_channels-max_power_channel)); % source that is closest to the channel with highest SWR power
                        max_source(j)=depth_profile(max_channels(idx2));
                        [c]=islocalmin(depth_profile); % find sinks
                        vec1=vec(c);
                        min_channels=vec1(vec(c)<max_channels(idx2));
                        [~,idx3]=min(abs(min_channels-max_channels(idx2))); % find first sink below the source
                        if isempty(idx3)
                            max_sink(j)=NaN;
                        else
                            max_sink(j)=depth_profile(min_channels(idx3));
                        end
                        sink_source_diff(j) = max_source(j) - max_sink(j); % calculate sink-source difference to use for thresholding


                        if max_sink(j)<0 && max_source(j)>0 && sink_source_diff(j)>1.2 %% hard threshold picked after testing it out with a few recordings
                            swr_dat.CSDcheck(j)=1;
                            swr_dat.CSD(:,:,j) = imgaussfilt(CSDoutput,1);
                            swr_dat.sm_CSD(:,:,j)=sm_CSD;
                        else
                            swr_dat.CSDcheck(j)=0;
                            swr_dat.CSD(:,:,j) = imgaussfilt(CSDoutput,1);
                            swr_dat.sm_CSD(:,:,j)=sm_CSD;
                        end

                    else
                        swr_dat.CSDcheck(j)=0;
                        swr_dat.CSD(:,:,j) = imgaussfilt(CSDoutput,1);
                        swr_dat.sm_CSD(:,:,j)=sm_CSD;
                    end

                end
                drawnow
                swr_dat.notMultChan = k-j; %% ripple that didn't meet the multchan criterion


            else
                swr_dat.swr_no_events = NaN;
                swr_dat.swr_evs_onset_times = NaN;
                swr_dat.swr_evs_filt_swr = NaN;
                swr_dat.swr_evs_unfilt_swr =NaN;
                swr_dat.notMultChan = NaN;
            end




            swr_dat.no_chans = length(SWR_chans);   %%%total number of channels examined
            swr_dat.rest_time = rest_time;  %%%overall duration of resting period (s)
            swr_dat.swr_freq = swr_dat.swr_no_events ./ swr_dat.rest_time; %%%frequency of swr events during resting
            swr_dat.swr_freq_chan_norm = swr_dat.swr_no_events ./ swr_dat.no_chans ./ swr_dat.rest_time; %%%swr frequency normalised by number of channels examined and rest time
            swr_dat.CA1chans = CAchans;
            swr_dat.swr_dur = ev_time./lfp_vits.sampling_rate;
            swr_dat.chan_for_each_ev = swr_chan;
            swr_dat.detection_threshold=detection_threshold;




%             tmp_save=(['SWR_data_071123test_imec' num2str(imec_probe)]);
%             save(tmp_save,'swr_dat','swr_vits');



        end


    end
end

%% plot CSD analysis 
figure;
timevec=[-400:400];

for n=1:10%length(swr_dat.CSDcheck)

    data=swr_dat.sm_CSD(:,:,n);
    nexttile
    s=pcolor(timevec,[1:size(swr_dat.sm_CSD(:,:,n),1)],swr_dat.sm_CSD(:,:,n));
    s.FaceColor = 'interp';s.LineStyle='none';
    set(gca,'YDir','normal');colormap redblue;axis square; caxis([min(min(data(20:60,:))) max(max(data(20:60,:)))]);
    if swr_dat.CSDcheck(n)==1
        title([num2str(swr_dat.swr_evs_peak_times(n))],'Color','g')
    else
        title([num2str(swr_dat.swr_evs_peak_times(n))],'Color','r')
    end
    
    nexttile; plot(data(:,401),[1:size(swr_dat.sm_CSD(:,:,n),1)])
   
       
end









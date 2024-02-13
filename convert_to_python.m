clear
% overview = load('/Volumes/MarcBusche/Jana/Matlab Code/Relevant_Code_Trajectories/overview Tables/overview_table_A_onlyRecsWithRippleData.mat');
% writetable(overview.overview_table_A, 'overview.csv')

behaviour_state_classifiers = load("/Volumes/MarcBusche/Jana/Neuropixels/Trajectories/Processed data/NLGF_A_1393311_3M/Baseline3/behavioural_state_classifiers_final.mat");
% writecell(behaviour_state_classifiers.behavioural_details.restTS, "restTS.csv")
resty = behaviour_state_classifiers.behavioural_details.restTS;
save('restTS.mat', 'resty');


   
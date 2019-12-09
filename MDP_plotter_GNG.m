
%Script plots behavioural information alongside plots of state-action
%prediction error, and simulated spiking information. 

%This is set up for the version of the EE game where the high probability
%arm simply shifts

% If there are a large number of trials, the script will plot the outcomes
% in separate figures in batches of 100.

% Anna Sales Jan 2018 / Nov 2019 UoB.


%% Plot in batches of 100, or just the entire MDP - whichever is smaller.
%  This stops the plots from getting too crowded if there are a lot of
%  trials.

nn=length(MDP_OUT);
if nn>99
    batch=100;
else
    batch=nn;
end


starts=(0:batch:nn-1)+1;  %define start and end of each batch
ends=starts+batch-1;
if mod(nn,batch)~=0
    ends(end)=nn
end

n_exc=[starts', ends'];  %this matrix holds the start and end numbers of each batch
titleSt='Go/no-go task';  %a string to use as the title for the plots

%% Now do the plotting
for o=1:size(n_exc, 1)  %Go through each batch of MDP in turn
   
    first_=n_exc(o,1);  %start and end of this batch
    last_=n_exc(o,2);       
    n_trials=n_exc(2)-n_exc(1)+1;
    MDP=MDP_OUT([first_:last_]);  %pull out the MDPs in this batch
    n=length(MDP);
    %create some empty matrices to hold info extracted from the MDP 
    DAlog=[]; %dopamine
    obs_all=[];  %observations
    actions=[];  %actions
    pols=zeros(n, 1);  %policies chosen
    SAPE=[];  %State action prediction error
    df_track=[];    %decay factor

    %pull out some data from the first MDP to check the size of different variables.  
    
    oddballs=MDP(1).oddballs;  %trials which were oddballs
    prefs=exp(MDP(1).C);  %agent's preferences.
    V=MDP(1).V;  %policies allowed
    T=size(V, 1)+1; %Time points per trial
    np=size(V,2);  %number of policies
    n=length(MDP);  %number of MDPs in this batch 
    trials=1:n;
     
    for m=1:n    

         %extract useful quantities from MDP, store in vectors defined
         %above.

         u=MDP(m).u;                                 %extract actions for this MDP
         pols(m)=find(ismember(V', u, 'rows'));      %work out which policy was chosen    
         actions(m,:)  = MDP(m).u;                   %store actions, collate with those from other MDPs
         final_states(m,:)=MDP(m).s;                 %store final state
         SAPE=[SAPE, MDP(m).SAPE];           %extract SAPE, store
         df_track=[df_track, MDP(m).df];     %extract decay factor, store
         DAlog=[DAlog, MDP(m).w];            %extract dopamine (precision), stor
         obs_all=[obs_all, MDP(m).o ];       %extract observations, store
   
    end
    

 %% Simulated spiking : generate poisson spike series using SAPE as prob(spike)
 % Define the time base for this batch:
    if T>2
        t_first=(T-1)*first_-1;
        t_last=(T-1)*last_;
    else
        t_first=(T-1)*first_ ; %for one step trials there's only one time point in each MDP
        t_last=(T-1)*last_;
    end
    
    time_epoch=1;  %assume one second per decision
    t=[t_first-time_epoch/2, t_last+time_epoch/2]; 

    %Use a 10Hz max firing rate (physiologically realistic for LC)
    dt = 0.1; %split each second into 10 bins (this will set the max firing rate as P=1 corresponds to firing in EVERY bin   
    n_bins=time_epoch/dt;
    baseline_firing=1; %min firing rate (when prob<<1), in hz. Represents baseline LC firing.
    n_bins_bl=(1./baseline_firing)  / dt; %number of bins we need to get one spike event at baseline rate.
    bl_prob=1/n_bins_bl;
    range_SAPE=max(SAPE)-min(SAPE);  %define the range of prediction errors, so that we can convert to a probability of spiking
    
    % sigmoid activation to convert prediction error to firing probability. 
    PEsig=repelem(SAPE,n_bins)-min(SAPE);
    abs_prob=sig( PEsig-0.6*range_SAPE , 6); % mean of sigmoid is set by  PEsig
    PE_prob=bl_prob + ((1-bl_prob)* (abs_prob)); %number between bl_prob and 1 for each time point, based on SAPE
    
    tvec = t(1):dt:t(2)-dt;
    rng default; % reset random number generator 
    spk_poiss = rand(size(tvec)); % random numbers between 0 and 1
    
    %define bins in which there is a spike:
    spk_poiss_idx = find(spk_poiss < PE_prob); % If the random number is below the prob in a given bin, it will contain a spike
    spk_poiss_t = tvec(spk_poiss_idx)'; % use idxs to get corresponding spike time

    %%    Plot outcomes, spiking, SAPE

    %format observations for plotting
    no_rew_ind=find(obs_all==5); %find index of times when agent chose 'go' but in error
    rew_ind=find(obs_all==4); %find index of times when agent chose 'go' & was rewarded
    cue_rew_ind=find(obs_all==2); %indices of trials when a reward cue was given.
    nullc_ind=find(obs_all==3); %indices of trials when a non-reward cue was given.
    home=find(obs_all==1); %indices when agent was at the 'home' point
    
    %format actions for plotting as a heatmap
    a2=reshape(actions', length(SAPE),1 );
    C = num2cell(a2);
    
    %set up the figures
    fig2=figure('Units', 'normalized', 'Position', [0,0,1,1], 'Color', 'w');
    hold on
    figure(fig2)
    title(['Trials ' num2str(first_) 'to ' num2str(last_)])
    SAPEplot=subplot(3,1,1, 'Parent', fig2);
    SAPEplot.Position=[0.1, 0.65, 0.82, 0.28];
    spikePlot=subplot(3,1,2, 'Parent', fig2);
    spikePlot.Position=[0.1, 0.42, 0.82, 0.1];
    heatmapPlot=subplot(3,1,3, 'Parent', fig2);
    heatmapPlot.Position=[0.1, 0.21, 0.82, 0.1];

    %plot SAPE 
    subplot(SAPEplot)
    plot(SAPEplot, t_first:t_last, SAPE)
    xlim([t_first, t_last])
    xlabel('Time (s)', 'Fontsize', 16);
    ylabel({'State-action prediction error';'\Sigma (\Delta p)'}, 'FontSize', 16)
    hold on
    title(['State-action prediction errors: ' titleSt ], 'FontSize', 16)
    xticks(0:10:t_last);
    set(gca,'fontsize',13);

    %plot spiking
    subplot(spikePlot)
    plot(spikePlot, repmat(spk_poiss_t,1,2), [-1,1], 'k') 
    hold on
    ylim([-3,3])
    xlim([t_first, t_last])
    set(gca,'fontsize',12);
    xlabel('Time (s) ', 'FontSize', 16)     
    title('Simulated LC firing' , 'FontSize', 16)
    spikePlot.YAxis.Visible = 'off';   % remove y-axis
    xticks(0:10:length(SAPE));

    %plot actions as a heatmap
    subplot(heatmapPlot)  
    heatmap=false(length(unique(V)), n*(T-1));  %create boolean heatmap
    acts=reshape(actions', [1,numel(actions)]);% get one long vector of actions
    obs=reshape(obs_all', [1, numel(obs_all)] );  %and observations

    %set the locations where the agent actually goes as true in the boolean
    %matrix
    for t=1:size(heatmap, 2)
        heatmap(acts(t),t)=true; %NB the heatmap is plotted upsidedown at this point!
    end

    %now set all the true squares to the value of the observations
    obs(1:3:size(obs,2))=[];
    big_obs=repmat(obs,3,1); %reformat so that it'll overlay the boolean array
    heatmap=heatmap.*repmat(obs,3,1); %set squares corresponding to observations to the right value
    heatmap=flipud(heatmap);  %now heatmap is correctly laid out.
    
    %sort out the colours on the heatmap:
    heatmap(heatmap==0)=0.2;
    heatmap_little=heatmap(:,1:10);

    %observation colours: white (0), black (1), (2)green, (3)red, (4) yellow (5) magenta
    cmap=[ 1 1 1 ;0 0 0 ;0 1 0.4 ;1 0 0;1 1 0; 1 0.3 1 ] 
    cmap=repelem(cmap, 2,1)
    
    %plot the heatmap
    imagesc(heatmap);
    yticks([1:3])
    set(gca,'YTickLabel',{'Location 3' 'Location 2' 'Location 1'});
    colormap(cmap)

    caxis([0 5.5])
    hold on
    new_trial=2.5:2:n*T
    plot( repelem(new_trial, 2, 1), [0,4], '--k' )
    xticks([1:2:n*T])
    xlabs=split(num2str(1:n), ' ');
    xlabs(strcmp('',xlabs)) = [];
    xtickprops = get(gca,'XTickLabel');
    set(gca,'XTickLabel',[]);
    set(gca, 'FontSize', 13);
    xlabel('Actions and observations')
      
    %sort out a key
    dim2 = [heatmapPlot.Position(1)+0.26, heatmapPlot.Position(2)-0.15, .3 .1];
    mybox=annotation('rectangle',dim2)

    l1=[mybox.Position(1)+0.01, mybox.Position(1)+0.03];
    l2=[mybox.Position(2)+0.08, mybox.Position(2)+0.08];

    redline=annotation('line',l1,l2, 'Color', 'r', 'LineWidth', 6 );
    t1dim=[l1(1)+0.025,l2(1)-0.017, 0.2, 0.03];
    text1=annotation('textbox', t1dim, 'String', 'No-go cue', 'LineStyle', 'none', 'VerticalAlignment', 'bottom','FontSize', 12);

    greenline=annotation('line',l1,l2-0.025, 'Color', 'g', 'LineWidth', 6 );
    t2dim=[l1(1)+0.025,l2(1)-0.044, 0.2, 0.03];
    text2=annotation('textbox', t2dim, 'String', 'Go! Cue', 'LineStyle', 'none', 'VerticalAlignment', 'bottom','FontSize', 12);

    yelline=annotation('line',l1,l2-0.05, 'Color', 'y', 'LineWidth', 6 );
    t3dim=[l1(1)+0.025,l2(1)-0.071, 0.2, 0.03];
    text3=annotation('textbox', t3dim, 'String', 'Reward obtained', 'LineStyle', 'none', 'VerticalAlignment', 'bottom','FontSize', 12);

    blkline=annotation('line',l1+0.11,l2, 'Color', 'k', 'LineWidth', 6 );
    t4dim=[l1(1)+0.135,l2(1)-0.0171, 0.2, 0.03];
    text4=annotation('textbox', t4dim, 'String', 'At location 1', 'LineStyle', 'none', 'VerticalAlignment', 'bottom','FontSize', 12);

    magline=annotation('line',l1+0.11,l2-0.024, 'Color', 'm', 'LineWidth', 6 );
    t5dim=[l1(1)+0.135,l2(1)-0.017, l1(1)+0.01, l2(2)-0.13];
    text5=annotation('textbox', t5dim, 'String', 'Reward sought unsuccessfully', 'LineStyle', 'none', 'FontSize', 12);

    trialline=annotation('line',l1+0.11,l2-0.050, 'Color', 'k', 'LineWidth', 1, 'LineStyle', '--' );
     t6dim=[l1(1)+0.135,l2(1)-0.045, l1(1)+0.01, l2(2)-0.13];
    text5=annotation('textbox', t6dim, 'String', 'New trial', 'LineStyle', 'none', 'FontSize', 12);

    heatmapPlot.Box='off';
    SAPEplot.Box='off';
    spikePlot.Box='off'        

end   


%% define a sigmoid function, used in code above.
function [y] = sig(x, grad);  %x is the mean of the sigmoid, grad is a gradient.
y=1./ ( 1+exp(-grad* x) );  %return the outcome of the sigmoid.
end
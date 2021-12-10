%Author : Mahmut Ağralı
%Code is used to train the DQN agent
clear;clc;

% load system
mdl = 'DQN_VTOL_Plant_Simulink' ; 
open_system(mdl);


%initialize observation
obsInfo = rlNumericSpec([3 1]);
    %'LowerLimit',-inf,...
    %'UpperLimit',inf);
obsInfo.Name = 'observations';
obsInfo.Description = 'error';
numObservations = obsInfo.Dimension(1);

%initialize action space
actInfo = rlFiniteSetSpec(0.4:0.01:0.8);
actInfo.Name = 'Voltage';
actInfo.Description = 'Z Altitude';
numActions = actInfo.Dimension(1);

%simulate system
env = rlSimulinkEnv('DQN_VTOL_Plant_Simulink','DQN_VTOL_Plant_Simulink/RL Agent',obsInfo,actInfo);

% set funtction
env.ResetFcn = @(in)localResetFcn(in);

%set time step and max simulation time
Ts = 0.01;
Tf = 20;
rng(0);

%desing network
statePath = [ 
    featureInputLayer(numObservations,'Normalization','none','Name','observation')
    fullyConnectedLayer(400,'Name','CriticStateFC1')
    reluLayer('Name', 'CriticRelu1')
    fullyConnectedLayer(300,'Name','CriticStateFC2')];
actionPath = [
    featureInputLayer(numActions,'Normalization','none','Name','action')
    fullyConnectedLayer(300,'Name','CriticActionFC1','BiasLearnRateFactor',0)];
commonPath = [
   additionLayer(2,'Name','add')
    reluLayer('Name','CriticCommonRelu')
    fullyConnectedLayer(1,'Name','CriticOutput')];

criticNetwork = layerGraph();
criticNetwork = addLayers(criticNetwork,statePath);
criticNetwork = addLayers(criticNetwork,actionPath);
criticNetwork = addLayers(criticNetwork,commonPath);
    
criticNetwork = connectLayers(criticNetwork,'CriticStateFC2','add/in1');
criticNetwork = connectLayers(criticNetwork,'CriticActionFC1','add/in2');

figure
plot(criticNetwork)

%initialize critic options
criticOpts = rlRepresentationOptions('LearnRate',1e-03,'GradientThreshold',1,'UseDevice','cpu');

critic = rlQValueRepresentation(criticNetwork,obsInfo,actInfo,'Observation',{'observation'},'Action',{'action'},criticOpts);

%initialize DQN agent
agentOpts = rlDQNAgentOptions(...
    'SampleTime',Ts, ...
    'UseDoubleDQN',false, ...
    'ExperienceBufferLength',1e6, ...
    'DiscountFactor',0.99, ...
    'MiniBatchSize',256); 

%generate DQN agent
agent = rlDQNAgent(critic,agentOpts);

%set training options
maxepisodes = 5000;
maxsteps = ceil(Tf/Ts);
trainOpts = rlTrainingOptions(...
     'MaxEpisodes',maxepisodes,...
    'MaxStepsPerEpisode',maxsteps,...
    'ScoreAveragingWindowLength',5,...
    'Verbose',false,...
    'Plots','training-progress',...
    'StopTrainingCriteria','EpisodeReward',...
    'StopTrainingValue',-0.5,...
    'SaveAgentCriteria','EpisodeReward',...
    'SaveAgentValue',-50);

doTraining = true;

if doTraining
    % Train the agent.
    trainingStats = train(agent,env,trainOpts);
    save 'DQN_VTOL_Plant_Simulink.mat' agent;
    

else
    % Load the pretrained agent for the example.
    load('DQN_VTOL_Plant_Simulink.mat','agent')
end

simOpts = rlSimulationOptions('MaxSteps',maxsteps,'StopOnError','on');
experiences =  sim(env,agent,simOpts);



function in = localResetFcn(in)
%reset value
 in=setVariable(in,'pitch0',0);
end 

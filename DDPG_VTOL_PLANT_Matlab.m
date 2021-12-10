%Author : Mahmut Ağralı
%Code is used to train the DDPG agent

%load system
mdl = 'DDPG_VTOL_PLANT_Simulink';
open_system(mdl)

%initialize observation and action
obsInfo = rlNumericSpec([2 1],...
    'LowerLimit',[-2*pi -2*pi]',...
    'UpperLimit',[2*pi 2*pi]');
obsInfo.Name = 'observations';
obsInfo.Description = 'pitchangleerror,pitchdot';
numObservations = obsInfo.Dimension(1);

actInfo1 = rlNumericSpec([1 1],'LowerLimit',[-6],'UpperLimit',[6]);
actInfo.Name = 'voltage';

%initialize 
numActions = actInfo1.Dimension(1);
blks = ["DDPG_VTOL_PLANT_Simulink/RL Agent"];
env = rlSimulinkEnv('DDPG_VTOL_PLANT_Simulink',blks,...
    obsInfo,actInfo1);
env.ResetFcn = @(in)localResetFcn(in);

Ts = 0.01;
Tf = 10;
rng(0)

%design network
statePath = [
    featureInputLayer(2,'Normalization','none','Name','observation')
    fullyConnectedLayer(400,'Name','CriticStateFC1')
    reluLayer('Name', 'CriticRelu1')
    fullyConnectedLayer(300,'Name','CriticStateFC2')];
actionPath = [
    featureInputLayer(1,'Normalization','none','Name','action')
    fullyConnectedLayer(300,'Name','CriticActionFC1','BiasLearnRateFactor',0)];
commonPath = [
    additionLayer(2,'Name','add')
    reluLayer('Name','CriticCommonRelu')
    fullyConnectedLayer(1,'Name','CriticOutput')];

%set critic network
criticNetwork = layerGraph();
criticNetwork = addLayers(criticNetwork,statePath);
criticNetwork = addLayers(criticNetwork,actionPath);
criticNetwork = addLayers(criticNetwork,commonPath);
    
criticNetwork = connectLayers(criticNetwork,'CriticStateFC2','add/in1');
criticNetwork = connectLayers(criticNetwork,'CriticActionFC1','add/in2');

criticOpts = rlRepresentationOptions('LearnRate',1e-03,'GradientThreshold',1,'UseDevice','gpu');

%initialize critic 
criticA = rlQValueRepresentation(criticNetwork,obsInfo,actInfo1,'Observation',{'observation'},'Action',{'action'},criticOpts);

%design actor network
actorNetwork = [
    featureInputLayer(2,'Normalization','none','Name','observation')
    fullyConnectedLayer(400,'Name','ActorFC1')
    reluLayer('Name','ActorRelu1')
    fullyConnectedLayer(300,'Name','ActorFC2')
    reluLayer('Name','ActorRelu2')
    fullyConnectedLayer(1,'Name','ActorFC3')
    tanhLayer('Name','ActorTanh')
    scalingLayer('Name','ActorScaling','Scale',max(actInfo1.UpperLimit))];

%initialize actor network
actorOpts = rlRepresentationOptions('LearnRate',1e-04,'GradientThreshold',1,'UseDevice','gpu');

actorA = rlDeterministicActorRepresentation(actorNetwork,obsInfo,actInfo1,'Observation',{'observation'},'Action',{'ActorScaling'},actorOpts);

%initialize DDPG agent
agentOpts = rlDDPGAgentOptions(...
    'SampleTime',Ts,...
    'TargetSmoothFactor',1e-3,...
    'ExperienceBufferLength',1e6,...
    'DiscountFactor',0.99,...
    'MiniBatchSize',128);

agentOpts.NoiseOptions.Variance = 0.6;
agentOpts.NoiseOptions.VarianceDecayRate = 1e-5;

agentA = rlDDPGAgent(actorA,criticA,agentOpts);

%initialize training options
maxepisodes = 400000;
maxsteps = 1001;
trainOpts = rlTrainingOptions(...
    'MaxEpisodes',maxepisodes,...
    'MaxStepsPerEpisode',maxsteps,...
    'ScoreAveragingWindowLength',[5],...
    'Verbose',false,...
    'Plots','training-progress',...
    'StopTrainingCriteria','EpisodeReward',...
    'StopTrainingValue',[-0.5],...
    'SaveAgentCriteria','EpisodeReward',...
    'SaveAgentValue',[-50]);

    stats = train(agentA,env,trainOpts);
 
 function in=localResetFcn(in)
 %reset value
 in=setVariable(in,'pitch0',0)
 end
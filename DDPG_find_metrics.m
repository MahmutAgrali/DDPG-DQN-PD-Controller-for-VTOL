%Author : Mahmut Ağralı
%Code is used for find the metrics for DDPG algorithm

%initialize
Tf=10;
Ts=0.01;
pitch0=0;

sAgent = 'DDPG_Agent701';
load(sAgent);

%open system
open_system('DDPG_VTOL_PLANT_Simulink');
open_system('DDPG_VTOL_PLANT_Simulink/Scope_Phi');

%simulate
Gain = 1;
ScopeData = sim('DDPG_VTOL_PLANT_Simulink');

%get data from scope
DDPG_sig = reshape(ScopeData.ScopeData{1}.Values.Data,1,1004);
ref = reshape(ScopeData.ScopeData{2}.Values.Data,1,1004);
error = ref - DDPG_sig;

%imms = immse(sin(0.2)*ones(1,1004),DDPG_sig)

%get metrics
MSE = mean(error.^2);
ISE = sum(error.^2);
IAE = sum(abs(error));
% if(best > MSE)
%    best = MSE;
%end
fprintf("MSE : "+MSE+" ISE : "+ISE+" IAE : "+IAE + " at "+sAgent+" - Gain : "+Gain);%+" - Best MSE :"+best);
%end
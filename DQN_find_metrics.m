%Author : Mahmut Ağralı
%Code is used for find the metrics for DQN algorithm

Tf=10;
Ts=0.01;

sAgent = 'DQN_Agent120';
load(sAgent);

open_system('DQN_VTOL_Plant_Simulink/Scope_Phi');
ScopeData = sim('DQN_VTOL_Plant_Simulink');

DQN_sig = reshape(ScopeData.ScopeData{1}.Values.Data,1,1001);
ref = reshape(ScopeData.ScopeData{2}.Values.Data,1,1001);
error = ref- DQN_sig;
MSE = mean(error.^2);
ISE = sum(error.^2);
IAE = sum(abs(error));

fprintf("MSE : "+MSE+" ISE : "+ISE+" IAE : "+IAE + " at "+sAgent);
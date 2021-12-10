%Author : Mahmut Ağralı
%Code is used for find the metrics for PID algorithm

%load system
open_system('PID_VTOL/Scope_Phi');
ScopeData = sim('PID_VTOL');

%get scope data
PID_sig = reshape(ScopeData.ScopeData{1}.Values.Data,1,1001);
ref = reshape(ScopeData.ScopeData{2}.Values.Data,1,1001);
error = ref - PID_sig;
MSE = mean(error.^2);
ISE = sum(error.^2);
IAE = sum(abs(error));

fprintf("MSE : "+MSE+" ISE : "+ISE+" IAE : "+IAE + " at P : 17.9012745706573, I : 22.4780602564623, D : 3.11340256685474");
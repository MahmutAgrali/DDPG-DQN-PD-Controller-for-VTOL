%Author : Mahmut Ağralı
%Code is used for compare all algorithms

figure
hold on

plot(0:0.01:10,ref(1:1001),'black','LineWidth',2)
plot(0:0.01:10,DQN_sig(1:1001),'blue','LineWidth',2)
plot(0:0.01:10,DDPG_sig(1:1001),'red','LineWidth',2)
plot(0:0.01:10,PID_sig,'green','LineWidth',2)

legend('Reference','DQN','DDPG','PID')
xlabel({'Time(s)'})
ylabel({'Pitch Angle(rad)'})
%title({'The step response of the algorithms'})
hold off

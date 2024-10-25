%%%%%%%%%%%%%%%%%%%%%%% Esempio accesso dati %%%%%%%%%%%%%%%%%%%%%%%%%
clc
clear
close all

%{
subplot(2,1,1)
geoplot(Position.latitude,Position.longitude)
geobasemap streets
subplot(2,1,2)
plot(Acceleration.Timestamp,Acceleration.Z)
hold on
plot(Acceleration.Timestamp,Acceleration.Y)
plot(Acceleration.Timestamp,Acceleration.X)
hold off
legend('Z acc','Y acc','X acc')
%}
for U =1:2 
    for S= 1:2
        for M= 1:2
            str = 'U'+string(U)+'_S'+string(S)+'_M'+string(M)+'.mat'
            load(str)
            Acceleration.Timestamp=Acceleration.Timestamp-Acceleration.Timestamp(1);

            s=minutes(Acceleration.Timestamp)*60+seconds(Acceleration.Timestamp);

            acceleration = [s, Acceleration.X,Acceleration.Y,Acceleration.Z];
            str1 = 'U'+string(U)+'_S'+string(S)+'_M'+string(M)+'load.mat'
            save(str1, 'acceleration');
           
        end
    end
end


fprintf("Done!");
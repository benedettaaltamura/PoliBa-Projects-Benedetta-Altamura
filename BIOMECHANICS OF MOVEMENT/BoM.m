%% PREPARAZIONE
clc
clear all
close all

%% LOAD DATA
opts = detectImportOptions('anca.txt', 'Delimiter', '\t','ReadVariableNames', false);
opts = setvartype(opts, 'double');
opts = setvaropts(opts, 'DecimalSeparator', ',');

file_list = dir('*.txt');
data = cell(numel(file_list), 1);

for i = 1:numel(file_list)
    current_table = readtable(file_list(i).name, opts);
    data{i} = table2array(current_table);
    nome=strrep(file_list(i).name, '.txt', '');
    eval([nome ' = data{i};']);
end

%% VERIFYING FRAMES
for i = 1:size(ginocchio,1)
    if caviglia(i,1) ~= ginocchio(i,1) || caviglia(i,1) ~= anca(i,1) || caviglia(i,1) ~= spalla(i,1) || ginocchio(i,1) ~= anca(i,1) || ginocchio(i,1) ~= spalla(i,1) || anca(i,1) ~= spalla(i,1)
        error(['I frames acquisiti non sono coerenti in posizione ',num2str(i)]);
    end
end
disp('I frames acquisiti sono coerenti')
tempo = ginocchio(:,1);

%% POSITION
% X Axis
pxtallone= zeros(size(tempo,1),1);
pxcaviglia = caviglia(:,2);
pxginocchio = ginocchio(:,2);
pxanca = anca(:,2);
pxspalla = spalla(:,2);

% Y Axis
pytallone= zeros(size(tempo,1),1);
pycaviglia = caviglia(:,3);
pyginocchio = ginocchio(:,3);
pyanca = anca(:,3);
pyspalla = spalla(:,3);

%% VELOCITÀ GIUNTI
vxcaviglia = [];
vycaviglia = [];
vxginocchio = [];
vyginocchio = [];
vxanca = [];
vyanca = [];
vxspalla = [];
vyspalla = [];

for i=1:(length(pxginocchio)-1)
    vxcaviglia(i) = (pxcaviglia(i+1)-pxcaviglia(i))/(tempo(i+1)-tempo(i));
    vycaviglia(i) = (pycaviglia(i+1)-pycaviglia(i))/(tempo(i+1)-tempo(i));
    vxginocchio(i) = (pxginocchio(i+1)-pxginocchio(i))/(tempo(i+1)-tempo(i));
    vyginocchio(i) = (pyginocchio(i+1)-pyginocchio(i))/(tempo(i+1)-tempo(i));
    vxanca(i) = (pxanca(i+1)-pxanca(i))/(tempo(i+1)-tempo(i));
    vyanca(i) = (pyanca(i+1)-pyanca(i))/(tempo(i+1)-tempo(i));
    vxspalla(i) = (pxspalla(i+1)-pxspalla(i))/(tempo(i+1)-tempo(i));
    vyspalla(i) = (pyspalla(i+1)-pyspalla(i))/(tempo(i+1)-tempo(i));
end

vxcaviglia(end+1) = vxcaviglia(end);
vycaviglia(end+1) = vycaviglia(end);
vxginocchio(end+1) = vxginocchio(end);
vyginocchio(end+1) = vyginocchio(end);
vxanca(end+1) = vxanca(end);
vyanca(end+1) = vyanca(end);
vxspalla(end+1) = vxspalla(end);
vyspalla(end+1) = vyspalla(end);



%modifica per grafici 
vxcaviglia(42) = vxcaviglia(43);
vycaviglia(42) = vycaviglia(43);
vxginocchio(42) = vxginocchio(43);
vyginocchio(42) = vyginocchio(43);
vxanca(42) = vxanca(43);
vyanca(42) = vyanca(43);
vxspalla(42) = vxspalla(43);
vyspalla(42) = vyspalla(43);
vxcaviglia(91) = vxcaviglia(92);
vycaviglia(91) = vycaviglia(92);
vxginocchio(91) = vxginocchio(92);
vyginocchio(91) = vyginocchio(92);
vxanca(91) = vxanca(92);
vyanca(91) = vyanca(92);
vxspalla(91) = vxspalla(92);
vyspalla(91) = vyspalla(92);

figure(1)
hold on
plot(tempo,vxcaviglia)
plot(tempo,vycaviglia)
xlabel('tempo[s]')
ylabel('velocità [m/s]')
title('velocità caviglia')
legend("velocità caviglia su asse x","velocità caviglia su asse y",'Location','best')
hold off

figure(2)
hold on
plot(tempo,vxginocchio)
plot(tempo,vyginocchio)
xlabel('tempo[s]')
ylabel('velocità [m/s]')
title('velocità ginocchio')
legend("velocità ginocchio su asse x","velocità ginocchio su asse y",'Location','best')
hold off

figure(3)
hold on
plot(tempo,vxanca)
plot(tempo,vyanca)
xlabel('tempo[s]')
ylabel('velocità [m/s]')
title('velocità anca')
legend("velocità anca su asse x","velocità anca su asse y",'Location','best')
hold off

figure(4)
hold on
plot(tempo,vxspalla)
plot(tempo,vyspalla)
xlabel('tempo[s]')
ylabel('velocità [m/s]')
title('velocità spalla')
legend("velocità spalla su asse x","velocità spalla su asse y",'Location','best')
hold off

%% ACCELERAZIONE GIUNTI
axcaviglia = [];
aycaviglia = [];
axginocchio = [];
ayginocchio = [];
axanca = [];
ayanca = [];
axspalla = [];
ayspalla = [];

for i=1:(length(pxginocchio)-1)
    axcaviglia(i) = (vxcaviglia(i+1)-vxcaviglia(i))/(tempo(i+1)-tempo(i));
    aycaviglia(i) = (vycaviglia(i+1)-vycaviglia(i))/(tempo(i+1)-tempo(i));
    axginocchio(i) = (vxginocchio(i+1)-vxginocchio(i))/(tempo(i+1)-tempo(i));
    ayginocchio(i) = (vyginocchio(i+1)-vyginocchio(i))/(tempo(i+1)-tempo(i));
    axanca(i) = (vxanca(i+1)-vxanca(i))/(tempo(i+1)-tempo(i));
    ayanca(i) = (vyanca(i+1)-vyanca(i))/(tempo(i+1)-tempo(i));
    axspalla(i) = (vxspalla(i+1)-vxspalla(i))/(tempo(i+1)-tempo(i));
    ayspalla(i) = (vyspalla(i+1)-vyspalla(i))/(tempo(i+1)-tempo(i));
end

axcaviglia(end+1) = axcaviglia(end);
aycaviglia(end+1) = aycaviglia(end);
axginocchio(end+1) = axginocchio(end);
ayginocchio(end+1) = ayginocchio(end);
axanca(end+1) = axanca(end);
ayanca(end+1) = ayanca(end);
axspalla(end+1) = axspalla(end);
ayspalla(end+1) = ayspalla(end);

%modifica per plot
axcaviglia(41) = axcaviglia(42);
aycaviglia(41) = aycaviglia(42);
axginocchio(41) = axginocchio(42);
ayginocchio(41) = ayginocchio(42);
axanca(41) = axanca(42);
ayanca(41) = ayanca(42);
axspalla(41) = axspalla(42);
ayspalla(41) = ayspalla(42);
axcaviglia(90) = axcaviglia(91);
aycaviglia(90) = aycaviglia(91);
axginocchio(90) = axginocchio(91);
ayginocchio(90) = ayginocchio(91);
axanca(90) = axanca(91);
ayanca(90) = ayanca(91);
axspalla(90) = axspalla(91);
ayspalla(90) = ayspalla(91);

figure(5)
hold on
plot(tempo,axcaviglia)
plot(tempo,aycaviglia)
xlabel('tempo[s]')
ylabel('accelerazione [m/s^2]')
title('accelerazione caviglia')
legend("accelerazione caviglia su asse x","accelerazione caviglia su asse y",'Location','best')
hold off

figure(6)
hold on
plot(tempo,axginocchio)
plot(tempo,ayginocchio)
xlabel('tempo[s]')
ylabel('accelerazione [m/s^2]')
title('accelerazione ginocchio')
legend("accelerazione ginocchio su asse x","accelerazione ginocchio su asse y",'Location','best')
hold off

figure(7)
hold on
plot(tempo,axanca)
plot(tempo,ayanca)
xlabel('tempo[s]')
ylabel('accelerazione [m/s^2]')
title('accelerazione anca')
legend("accelerazione anca su asse x","accelerazione anca su asse y",'Location','best')
hold off

figure(8)
hold on
plot(tempo,axspalla)
plot(tempo,ayspalla)
xlabel('tempo[s]')
ylabel('accelerazione [m/s^2]')
title('accelerazione spalla')
legend("accelerazione spalla su asse x","accelerazione spalla su asse y",'Location','best')
hold off

%% VELOCITÀ ANGOLARE PIEDE
alpha_deg = atand((pycaviglia-pytallone)./(pxcaviglia-pxtallone));
alpha_rad = atan((pycaviglia-pytallone)./(pxcaviglia-pxtallone));

for i=1:length(alpha_deg)
    if alpha_deg(i)<0
        alpha_deg(i) = 180+alpha_deg(i);
    end
end

figure(9)

plot(tempo,alpha_deg) 
xlabel("tempo [s]")
ylabel("alpha [deg]")

walpha = [];
walpha_rad = [];

for i=1:(length(pxginocchio)-1)
    walpha(i) = (alpha_deg(i+1)-alpha_deg(i))/(tempo(i+1)-tempo(i));
    walpha_rad(i) = (alpha_rad(i+1)-alpha_rad(i))/(tempo(i+1)-tempo(i));
end

walpha(end+1) = walpha(end);
walpha_rad(end+1) = walpha_rad(end);

wpiede = [];

for i=1:(length(pxginocchio))
    wpiede(i) = sqrt(vxcaviglia(i)^2+vycaviglia(i)^2)/sqrt(pxcaviglia(i)^2+pycaviglia(i)^2);
    if walpha_rad(i)<0
        wpiede(i)= -wpiede(i);
    end
end

figure(10)
hold on
walpha(42)=walpha(43);
walpha(91)=walpha(92);
walpha_rad(42)=walpha_rad(43);
walpha_rad(91)=walpha_rad(92);
wpiede(42)=wpiede(43);
wpiede(91)=wpiede(92);
plot(tempo,walpha_rad)
plot(tempo,wpiede)
legend("variazione di angolo","formula inversa equazione di Galileo ")
ylabel("velocità angolare [rad/s]")
xlabel("tempo [s]")
title("velocità angolare piede")
hold off

%% VELOCITÀ ANGOLARE GAMBA
beta_deg = [];
beta_deg = atand((pyginocchio-pycaviglia)./(pxginocchio-pxcaviglia));
for i=1:length(beta_deg)
    if beta_deg(i)<0
        beta_deg(i) = 180+beta_deg(i);
    end
end
beta_rad = [];
beta_rad = atan((pyginocchio-pycaviglia)./(pxginocchio-pxcaviglia));
for i=1:length(beta_rad)
    if beta_rad(i)<0
        beta_rad(i) = pi+beta_rad(i);
    end
end
figure(11)
plot(tempo,beta_deg)
xlabel("tempo [s]")
ylabel("beta [deg]")
hold off
wbeta= [];
wbeta_rad = [];
for i=1:(length(pxginocchio)-1)
    wbeta(i) = (beta_deg(i+1)-beta_deg(i))/(tempo(i+1)-tempo(i));
    wbeta_rad(i) = (beta_rad(i+1)-beta_rad(i))/(tempo(i+1)-tempo(i));
end
wbeta(end+1) = wbeta(end);
wbeta_rad(end+1) = wbeta_rad(end);

wgamba = [];
for i=1:(length(pxginocchio))
    wgamba(i) = sqrt(vxginocchio(i)^2+vyginocchio(i)^2)/sqrt((pxginocchio(i)-pxcaviglia(i))^2+(pyginocchio(i)-pycaviglia(i))^2);
    if wbeta_rad(i)<0
        wgamba(i)= -wgamba(i);
    end
end
figure(12)
hold on
wbeta(42)=wbeta(43);
wbeta(91)=wbeta(92);
wbeta_rad(42)=wbeta_rad(43);
wbeta_rad(91)=wbeta_rad(92);
wgamba(42)=wgamba(43);
wgamba(91)=wgamba(92);
plot(tempo,wbeta_rad)
plot(tempo,wgamba)
legend("variazione di angolo","formula inversa equazione di Galileo")
ylabel("velocità angolare [rad/s]")
xlabel("tempo [s]")
title("velocità angolare gamba")
hold off

%% VELOCITÀ ANGOLARE COSCIA
gamma_deg = [];
gamma_deg = atand((pyanca-pyginocchio)./(pxanca-pxginocchio));
for i=1:length(gamma_deg)
    if gamma_deg(i)<0
        gamma_deg(i) = 180+gamma_deg(i);
    end
end
gamma_rad = [];
gamma_rad = atan((pyanca-pyginocchio)./(pxanca-pxginocchio));
for i=1:length(gamma_rad)
    if gamma_rad(i)<0
        gamma_rad(i) = pi+gamma_rad(i);
    end
end
wgamma= [];
wgamma_rad = [];
for i=1:(length(pxanca)-1)
    wgamma(i) = (gamma_deg(i+1)-gamma_deg(i))/(tempo(i+1)-tempo(i));
    wgamma_rad(i) = (gamma_rad(i+1)-gamma_rad(i))/(tempo(i+1)-tempo(i));
end
wgamma(end+1) = wgamma(end);
wgamma_rad(end+1) = wgamma_rad(end);
figure(13)
plot(tempo,gamma_deg)
xlabel("tempo [s]")
ylabel("gamma [deg]")
hold off
wcoscia = [];
for i=1:(length(pxanca))
    wcoscia(i) = sqrt(vxanca(i)^2+vyanca(i)^2)/sqrt((pxanca(i)-pxginocchio(i))^2+(pyanca(i)-pyginocchio(i))^2);
    if wgamma_rad(i)<0
        wcoscia(i)= -wcoscia(i);
    end
end
figure(14)
hold on
wgamma(42)=wgamma(43);
wgamma(91)=wgamma(92);
wgamma_rad(42)=wgamma_rad(43);
wgamma_rad(91)=wgamma_rad(92);
wcoscia(42)=wcoscia(43);
wcoscia(91)=wcoscia(92);
plot(tempo,wgamma_rad)
plot(tempo,wcoscia)
legend("variazione di angolo","formula inversa equazione di Galileo")
ylabel("velocità angolare [rad/s]")
xlabel("tempo [s]")
title("velocità angolare coscia")
hold off

%% VELOCITÀ ANGOLARE HAT
delta_deg = [];
delta_deg = atand((pyspalla-pyanca)./(pxspalla-pxanca));
for i=1:length(delta_deg)
    if delta_deg(i)<0
        delta_deg(i) = 180+delta_deg(i);
    end
end
delta_rad = [];
delta_rad = atan((pyspalla-pyanca)./(pxspalla-pxanca));
for i=1:length(delta_rad)
    if delta_rad(i)<0
        delta_rad(i) = pi+delta_rad(i);
    end
end
figure(15)
plot(tempo,delta_deg)
xlabel("tempo [s]")
ylabel("delta [deg]")
hold off
wdelta= [];
wdelta_rad = [];
for i=1:(length(pxspalla)-1)
    wdelta(i) = (delta_deg(i+1)-delta_deg(i))/(tempo(i+1)-tempo(i));
    wdelta_rad(i) = (delta_rad(i+1)-delta_rad(i))/(tempo(i+1)-tempo(i));
end
wdelta(end+1) = wdelta(end);
wdelta_rad(end+1) = wdelta_rad(end);

wHAT = [];
for i=1:(length(pxspalla))
    wHAT(i) = sqrt(vxspalla(i)^2+vyspalla(i)^2)/sqrt((pxspalla(i)-pxanca(i))^2+(pyspalla(i)-pyanca(i))^2);
    if wdelta_rad(i)<0
        wHAT(i)= -wHAT(i);
    end
end
figure(16)
hold on
wdelta(42)=wdelta(43);
wdelta(91)=wdelta(92);
wdelta_rad(42)=wdelta_rad(43);
wdelta_rad(91)=wdelta_rad(92);
wHAT(42)=wHAT(43);
wHAT(91)=wHAT(92);
plot(tempo,wdelta_rad)
plot(tempo,wHAT)
legend("variazione di angolo","formula inversa equazione di Galileo")
ylabel("velocità angolare [rad/s]")
title("velocità angolare HAT")
xlabel("tempo [s]")
hold off

%% ACCELERAZIONE ANGOLARE PIEDE
a_alpha = [];
a_alpha_rad = [];
for i=1:(length(pxginocchio)-1)
    a_alpha(i) = (walpha(i+1)-walpha(i))/(tempo(i+1)-tempo(i));
    a_alpha_rad(i) = (walpha_rad(i+1)-walpha_rad(i))/(tempo(i+1)-tempo(i));
end
a_alpha(end+1) = a_alpha(end);
a_alpha_rad(end+1) = a_alpha_rad(end);

acaviglia = [];
a_piede = [];

for i=1:(length(pxginocchio))
    acaviglia(i) = sqrt(axcaviglia(i)^2+aycaviglia(i)^2);
    a_piede(i) = (acaviglia(i) + (walpha_rad(i)^2)*(sqrt(pxcaviglia(i)^2+pycaviglia(i)^2)))/(sqrt(pxcaviglia(i)^2+pycaviglia(i)^2));
    if a_alpha_rad(i)<0
        a_piede(i)= -a_piede(i);
    end
end


figure(17)
hold on 
a_alpha(41)=a_alpha(42);
a_alpha(90)=a_alpha(91);
a_alpha_rad(41)=a_alpha_rad(42);
a_alpha_rad(90)=a_alpha_rad(91);
a_piede(41)=a_piede(42);
a_piede(90)=a_piede(91);
plot(tempo,a_alpha_rad)
plot(tempo,a_piede)
legend("acc. angolare definizione","formula inversa equazione di Rivals")
title("accelerazione angolare piede")
ylabel("accelerazione angolare [rad^2/s]")
xlabel("tempo[s]")
hold off

%% ACCELERAZIONE ANGOLARE GAMBA
a_beta = [];
a_beta_rad = [];
for i=1:(length(pxginocchio)-1)
    a_beta(i) = (wbeta(i+1)-wbeta(i))/(tempo(i+1)-tempo(i));
    a_beta_rad(i) = (wbeta_rad(i+1)-wbeta_rad(i))/(tempo(i+1)-tempo(i));
end
a_beta(end+1) = a_beta(end);
a_beta_rad(end+1) = a_beta_rad(end);

aginocchio = [];
a_gamba = [];

for i=1:(length(pxginocchio))
    aginocchio(i) = sqrt(axginocchio(i)^2+ayginocchio(i)^2);
    a_gamba(i) = (aginocchio(i) + (wbeta_rad(i)^2)*(sqrt((pxginocchio(i)-pxcaviglia(i))^2+(pyginocchio(i)-pycaviglia(i))^2)))/(sqrt((pxginocchio(i)-pxcaviglia(i))^2+(pyginocchio(i)-pycaviglia(i))^2));
    if a_beta_rad(i)<0
        a_gamba(i)= -a_gamba(i);
    end
end


figure(18)
hold on 
a_beta(41)=a_beta(42);
a_beta(90)=a_beta(91);
a_beta_rad(41)=a_beta_rad(42);
a_beta_rad(90)=a_beta_rad(91);
a_gamba(41)=a_gamba(42);
a_gamba(90)=a_gamba(91);
plot(tempo,a_beta_rad)
plot(tempo,a_gamba)
legend("acc. angolare definizione","formula inversa equazione di Rivals")
title("accelerazione angolare gamba")
ylabel("accelerazione angolare [rad^2/s]")
xlabel("tempo[s]")
hold off

%% ACCELERAZIONE ANGOLARE COSCIA
a_gamma = [];
a_gamma_rad = [];
for i=1:(length(pxginocchio)-1)
    a_gamma(i) = (wgamma(i+1)-wgamma(i))/(tempo(i+1)-tempo(i));
    a_gamma_rad(i) = (wgamma_rad(i+1)-wgamma_rad(i))/(tempo(i+1)-tempo(i));
end
a_gamma(end+1) = a_gamma(end);
a_gamma_rad(end+1) = a_gamma_rad(end);

aanca = [];
a_coscia = [];

for i=1:(length(pxginocchio))
    aanca(i) = sqrt(axanca(i)^2+ayanca(i)^2);
    a_coscia(i) = (aanca(i) + (wgamma_rad(i)^2)*(sqrt((pxanca(i)-pxginocchio(i))^2+(pyanca(i)-pyginocchio(i))^2)))/(sqrt((pxanca(i)-pxginocchio(i))^2+(pyanca(i)-pyginocchio(i))^2));
    if a_gamma_rad(i)<0
        a_coscia(i)= -a_coscia(i);
    end
end


figure(19)
hold on 
a_gamma(41)=a_gamma(42);
a_gamma(90)=a_gamma(91);
a_gamma_rad(41)=a_gamma_rad(42);
a_gamma_rad(90)=a_gamma_rad(91);
a_coscia(41)=a_coscia(42);
a_coscia(90)=a_coscia(91);
plot(tempo,a_gamma_rad)
plot(tempo,a_coscia)
legend("acc. angolare definizione","formula inversa equazione di Rivals")
title("accelerazione angolare coscia")
ylabel("accelerazione angolare [rad^2/s]")
xlabel("tempo[s]")
hold off

%% ACCELERAZIONE ANGOLARE HAT
a_delta = [];
a_delta_rad = [];
for i=1:(length(pxginocchio)-1)
    a_delta(i) = (wdelta(i+1)-wdelta(i))/(tempo(i+1)-tempo(i));
    a_delta_rad(i) = (wdelta_rad(i+1)-wdelta_rad(i))/(tempo(i+1)-tempo(i));
end
a_delta(end+1) = a_delta(end);
a_delta_rad(end+1) = a_delta_rad(end);

aspalla = [];
a_HAT = [];

for i=1:(length(pxginocchio))
    aspalla(i) = sqrt(axspalla(i)^2+ayspalla(i)^2);
    a_HAT(i) = (aspalla(i) + (wdelta_rad(i)^2)*(sqrt((pxspalla(i)-pxanca(i))^2+(pyspalla(i)-pyanca(i))^2)))/(sqrt((pxspalla(i)-pxanca(i))^2+(pyspalla(i)-pyanca(i))^2));
    if a_delta_rad(i)<0
        a_HAT(i)= -a_HAT(i);
    end
end

figure(20)
hold on 
a_delta(41)=a_delta(42);
a_delta(90)=a_delta(91);
a_delta_rad(41)=a_delta_rad(42);
a_delta_rad(90)=a_delta_rad(91);
a_HAT(41)=a_HAT(42);
a_HAT(90)=a_HAT(91);
plot(tempo,a_delta_rad)
plot(tempo,a_HAT)
legend("acc. angolare definizione","formula inversa equazione di Rivals")
title("accelerazione angolare HAT")
ylabel("accelerazione angolare [rad^2/s]")
xlabel("tempo[s]")
hold off

%% ANGOLI DH IN GRADI
theta1_deg= alpha_deg;
theta2_deg = beta_deg - alpha_deg;
theta3_deg = gamma_deg - beta_deg;
theta4_deg = -gamma_deg + delta_deg;

figure(21)
plot(tempo,theta1_deg)
title("Angolo Theta1")
ylabel("angolo [deg]")
xlabel("tempo[s]")
ylim ([40 75])
hold off

figure(22)
plot(tempo,theta2_deg)
title("Angolo Theta2")
ylabel("angolo [deg]")
xlabel("tempo[s]")
hold off

figure(23)
plot(tempo,theta3_deg)
title("Angolo Theta3")
ylabel("angolo [deg]")
xlabel("tempo[s]")
hold off

figure(24)
plot(tempo,theta4_deg)
title("Angolo Theta4")
ylabel("angolo [deg]")
xlabel("tempo[s]")
hold off

%% MISURE ANTROPOMETRICHE
H= 1.55; %altezza

m_tot = 47; 
m_tot_gamba = m_tot/2; %è  la massa totale del sistema
m_piede = 0.0145*m_tot;
m_gamba = 0.0465*m_tot;
m_coscia = 0.1*m_tot;
m_HAT_gamba = 0.678*m_tot/2;

L1 = sqrt(pxcaviglia.^2 + pycaviglia.^2);
L2 = sqrt((pxginocchio-pxcaviglia).^2 + (pyginocchio-pycaviglia).^2);
L3 = sqrt((pxanca-pxginocchio).^2 + (pyanca-pyginocchio).^2);
L4 = sqrt((pxspalla-pxanca).^2 + (pyspalla-pyanca).^2);
Lpiede = [0.152 * H,0];

% Distanze prossimali
dp_hat=0.610*abs(L4);
dp_coscia=0.433*abs(L3);
dp_gamba=0.433*abs(L2);
dp_piede=0.5*abs(L1);

% Distanze distali
dd_hat=0.390*abs(L4);
dd_coscia=0.567*abs(L3);
dd_gamba=0.567*abs(L2);
dd_piede=0.5*abs(L1);

%% CINEMATICA DEL COM
theta1 = alpha_rad;
theta2 = beta_rad - alpha_rad;
theta3 = gamma_rad - beta_rad;
theta4 = -gamma_rad + delta_rad;

d = 0;
alpha_dh = 0;

cm_piede_s=[];
cm_gamba_s=[];
cm_coscia_s=[];
cm_HAT_s=[];

figure(25);

% Plot della posa statica

T01_s = [cos(theta1(1)) -cos(alpha_dh)*sin(theta1(1)) sin(alpha_dh)*sin(theta1(1)) L1(1)*cos(theta1(1));
sin(theta1(1)) cos(alpha_dh)*cos(theta1(1)) -sin(alpha_dh)*cos(theta1(1)) L1(1)*sin(theta1(1));
0 sin(alpha_dh) cos(alpha_dh) d;
0 0 0 1];

T12_s = [cos(theta2(1)) -cos(alpha_dh)*sin(theta2(1)) sin(alpha_dh)*sin(theta2(1)) L2(1)*cos(theta2(1));
sin(theta2(1)) cos(alpha_dh)*cos(theta2(1)) -sin(alpha_dh)*cos(theta2(1)) L2(1)*sin(theta2(1));
0 sin(alpha_dh) cos(alpha_dh) d;
0 0 0 1];

T23_s = [cos(theta3(1)) -cos(alpha_dh)*sin(theta3(1)) sin(alpha_dh)*sin(theta3(1)) L3(1)*cos(theta3(1));
sin(theta3(1)) cos(alpha_dh)*cos(theta3(1)) -sin(alpha_dh)*cos(theta3(1)) L3(1)*sin(theta3(1));
0 sin(alpha_dh) cos(alpha_dh) d;
0 0 0 1];

T34_s = [cos(theta4(1)) -cos(alpha_dh)*sin(theta4(1)) sin(alpha_dh)*sin(theta4(1)) L4(1)*cos(theta4(1));
sin(theta4(1)) cos(alpha_dh)*cos(theta4(1)) -sin(alpha_dh)*cos(theta4(1)) L4(1)*sin(theta4(1));
0 sin(alpha_dh) cos(alpha_dh) d;
0 0 0 1];

cm_piede_s = T01_s*[-L1(1)*0.5;0; 0; 1];
cm_gamba_s=T01_s*T12_s*[-L2(1)*0.433;0; 0; 1];
cm_coscia_s=T01_s*T12_s*T23_s*[-L3(1)*0.433;0; 0; 1];
cm_HAT_s=T01_s*T12_s*T23_s*T34_s*[-L4(1)*0.390;0; 0; 1]; 

% Sistema di riferimento fisso (0) sul tallone
O0= [0,0,0]';
X0= [1,0,0]';
Y0= [0,1,0]'; 

% Sistema di riferimento locale della caviglia (1)
O1= [0,0,0,1]';
X1= [1,0,0,1]';
Y1= [0,1,0,1]';

% Sistema di riferimento locale del ginocchio(2)
O2= [0,0,0,1]';
X2= [1,0,0,1]';
Y2= [0,1,0,1]';

% Sistema di riferimento locale dell'anca (3)
O3= [0,0,0,1]';
X3= [1,0,0,1]';
Y3= [0,1,0,1]';

% Applicazione della matrice di trasformazione al sdr 1
O1_T_s= T01_s*O1;
X1_T_s= T01_s*X1;
Y1_T_s= T01_s*Y1;

% Applicazione della matrice di trasformazione al sdr 2
O2_T_s= T01_s*T12_s*O2;
X2_T_s= T01_s*T12_s*X2;
Y2_T_s= T01_s*T12_s*Y2;

% Applicazione della matrice di trasformazione al sdr 3
O3_T_s= T01_s*T12_s*T23_s*O3;
X3_T_s= T01_s*T12_s*T23_s*X3;
Y3_T_s= T01_s*T12_s*T23_s*Y3;

% Applicazione della matrice di trasformazione al sdr 4
O4_T_s= T01_s*T12_s*T23_s*T34_s*O3;
X4_T_s= T01_s*T12_s*T23_s*T34_s*X3;
Y4_T_s= T01_s*T12_s*T23_s*T34_s*Y3;

% Disegno la gamba
plot([O0(1) O1_T_s(1)],[O0(2) O1_T_s(2) ], 'Color', '[0.3010 0.7450 0.9330]','LineWidth',2);
hold on
plot([O0(1) Lpiede(1)],[O0(2) Lpiede(2)], 'Color', '[0.3010 0.7450 0.9330]','LineWidth',2);
plot([O1_T_s(1) O2_T_s(1)],[O1_T_s(2) O2_T_s(2)], 'Color', '[0.3010 0.7450 0.9330]','LineWidth',2);
plot([O2_T_s(1) O3_T_s(1)],[O2_T_s(2) O3_T_s(2)], 'Color', '[0.3010 0.7450 0.9330]','LineWidth',2);
plot([O3_T_s(1) O4_T_s(1)],[O3_T_s(2) O4_T_s(2)], 'Color', '[0.3010 0.7450 0.9330]','LineWidth',2);

p1= plot(O1_T_s(1), O1_T_s(2), 'o', 'MarkerFaceColor', '[0.6350 0.0780 0.1840]', 'MarkerEdgeColor', '[0.6350 0.0780 0.1840]', 'MarkerSize',10);
p2= plot(O2_T_s(1), O2_T_s(2), 'o', 'MarkerFaceColor', '[0.5, 0.75, 0.5]', 'MarkerEdgeColor', '[0.5, 0.75, 0.5]','MarkerSize',10);
p3= plot(O3_T_s(1), O3_T_s(2), 'o', 'MarkerFaceColor', '[0 0.4470 0.7410]', 'MarkerEdgeColor', '[0 0.4470 0.7410]','MarkerSize',10);
p4= plot(O4_T_s(1), O4_T_s(2), 'o', 'MarkerFaceColor', '[0.4940 0.1840 0.5560]', 'MarkerEdgeColor', '[0.4940 0.1840 0.5560]','MarkerSize',10);
legend([p1, p2, p3,p4], {'caviglia', 'ginocchio', 'anca', 'spalla'});
title("trial static")

xlim([-0.5 1.5]);
ylim([0 1.8]);

hold off;
    
% Plot movimento

cm_piede=[];
cm_gamba=[];
cm_coscia=[];
cm_HAT=[];

figure(26);

for i=1:size(tempo,1)

    T01 = [cos(theta1(i)) -cos(alpha_dh)*sin(theta1(i)) sin(alpha_dh)*sin(theta1(i)) L1(i)*cos(theta1(i));
    sin(theta1(i)) cos(alpha_dh)*cos(theta1(i)) -sin(alpha_dh)*cos(theta1(i)) L1(i)*sin(theta1(i));
    0 sin(alpha_dh) cos(alpha_dh) d;
    0 0 0 1];

    T12 = [cos(theta2(i)) -cos(alpha_dh)*sin(theta2(i)) sin(alpha_dh)*sin(theta2(i)) L2(i)*cos(theta2(i));
    sin(theta2(i)) cos(alpha_dh)*cos(theta2(i)) -sin(alpha_dh)*cos(theta2(i)) L2(i)*sin(theta2(i));
    0 sin(alpha_dh) cos(alpha_dh) d;
    0 0 0 1];

    T23 = [cos(theta3(i)) -cos(alpha_dh)*sin(theta3(i)) sin(alpha_dh)*sin(theta3(i)) L3(i)*cos(theta3(i));
    sin(theta3(i)) cos(alpha_dh)*cos(theta3(i)) -sin(alpha_dh)*cos(theta3(i)) L3(i)*sin(theta3(i));
    0 sin(alpha_dh) cos(alpha_dh) d;
    0 0 0 1];
 
    T34 = [cos(theta4(i)) -cos(alpha_dh)*sin(theta4(i)) sin(alpha_dh)*sin(theta4(i)) L4(i)*cos(theta4(i));
    sin(theta4(i)) cos(alpha_dh)*cos(theta4(i)) -sin(alpha_dh)*cos(theta4(i)) L4(i)*sin(theta4(i));
    0 sin(alpha_dh) cos(alpha_dh) d;
    0 0 0 1];

    cm_piede(:,i) = T01*[-L1(i)*0.5;0; 0; 1];
    cm_gamba(:,i)=T01*T12*[-L2(i)*0.433;0; 0; 1];
    cm_coscia(:,i)=T01*T12*T23*[-L3(i)*0.433;0; 0; 1];
    cm_HAT(:,i)=T01*T12*T23*T34*[-L4(i)*0.390;0; 0; 1];

    %Applicazione della matrice di trasformazione al sdr 1
    O1_T= T01*O1;
    X1_T= T01*X1;
    Y1_T= T01*Y1;

    %Applicazione della matrice di trasformazione al sdr 2
    O2_T= T01*T12*O2;
    X2_T= T01*T12*X2;
    Y2_T= T01*T12*Y2;
 
    %Applicazione della matrice di trasformazione al sdr 3
    O3_T= T01*T12*T23*O3;
    X3_T= T01*T12*T23*X3;
    Y3_T= T01*T12*T23*Y3;
 
    %Applicazione della matrice di trasformazione al sdr 4
    O4_T= T01*T12*T23*T34*O3;
    X4_T= T01*T12*T23*T34*X3;
    Y4_T= T01*T12*T23*T34*Y3;

     
    %disegno la gamba
    plot([O0(1) O1_T(1)],[O0(2) O1_T(2) ], 'Color', '[0.3010 0.7450 0.9330]','LineWidth',2);
    hold on
    plot([O0(1) Lpiede(1)],[O0(2) Lpiede(2)], 'Color', '[0.3010 0.7450 0.9330]','LineWidth',2);
    plot([O1_T(1) O2_T(1)],[O1_T(2) O2_T(2)], 'Color', '[0.3010 0.7450 0.9330]','LineWidth',2);
    plot([O2_T(1) O3_T(1)],[O2_T(2) O3_T(2)], 'Color', '[0.3010 0.7450 0.9330]','LineWidth',2);
    plot([O3_T(1) O4_T(1)],[O3_T(2) O4_T(2)], 'Color', '[0.3010 0.7450 0.9330]','LineWidth',2);

    p1= plot(O1_T(1), O1_T(2), 'o', 'MarkerFaceColor', '[0.6350 0.0780 0.1840]', 'MarkerEdgeColor', '[0.6350 0.0780 0.1840]', 'MarkerSize',10);
    p2= plot(O2_T(1), O2_T(2), 'o', 'MarkerFaceColor', '[0.5, 0.75, 0.5]', 'MarkerEdgeColor', '[0.5, 0.75, 0.5]','MarkerSize',10);
    p3= plot(O3_T(1), O3_T(2), 'o', 'MarkerFaceColor', '[0 0.4470 0.7410]', 'MarkerEdgeColor', '[0 0.4470 0.7410]','MarkerSize',10);
    p4= plot(O4_T(1), O4_T(2), 'o', 'MarkerFaceColor', '[0.4940 0.1840 0.5560]', 'MarkerEdgeColor', '[0.4940 0.1840 0.5560]','MarkerSize',10);
    legend([p1, p2, p3,p4], {'caviglia', 'ginocchio', 'anca', 'spalla'});
    title("trial dinamico")

    xlim([-0.5 1.5]);
    ylim([0 1.8]);

    pause(0.1); % Rallenta animazione
    if i ~= length(tempo)
      clf(figure(26),'reset'); % Cancella tutto nella figura tranne la scena successiva, elimina cioè la precedente
    end
end
hold off

%% COM TOTALE
cm_totale =[];
for i=1:size(tempo,1)
    cm_totale(:,i) = (cm_piede(:,i)*m_piede + cm_gamba(:,i)*m_gamba + cm_coscia(:,i)*m_coscia + cm_HAT (:,i)*m_HAT_gamba)/(m_tot_gamba);
end

% Variabili per plot COM 
cm_totale_s_x= cm_totale(1,1);
cm_totale_s_y= cm_totale(2,1);
cm_totale_s=[cm_totale_s_x,cm_totale_s_y, 0,1];

%% VELOCITÀ E ACCELERAZIONI COM TOTALE
vcm_totale = [];
vcm_totale_mod = [];
for i=1:(size(cm_totale,2)-1)
    vcm_totale(:,i) = (cm_totale(:,i+1)-cm_totale(:,i))/(tempo(i+1)-tempo(i));
end
vcm_totale(:,end+1) = vcm_totale(:,end);

figure(27)
hold on
for i=1:size(tempo,1)
    vcm_totale_mod(i) = sqrt((vcm_totale(1,i)^2)+(vcm_totale(2,i)^2));
    if (vcm_totale_mod(i)> 1.5) % Soglia presa ad occhio per evitare che siano plottati i valori dovuti al taglio del video
        vcm_totale_mod(i) = 0; 
    end
    
end

plot(tempo,vcm_totale_mod);
title('modulo velocità COM totale nel tempo');
ylabel("velocità [m/s]")
xlabel("tempo [s]")
hold off;

figure(28)
hold on
vcm_totale(:,42)=vcm_totale(:,43);
vcm_totale(:,91)=vcm_totale(:,92);
plot(tempo,vcm_totale(1,:))
plot(tempo,vcm_totale(2,:))
title("velocità del centro di massa totale")
legend("velocità lungo x","velocità lungo y")
ylabel("velocità [m/s]")
xlabel("tempo [s]")
hold off

acm_totale = [];
acm_totale_mod = [];
for i=1:(size(cm_totale,2)-1)
    acm_totale(:,i) = (vcm_totale(:,i+1)-vcm_totale(:,i))/(tempo(i+1)-tempo(i));
end
acm_totale(:,end+1) = acm_totale(:,end);

for i=1:size(tempo,1)
    acm_totale_mod(i) = sqrt((acm_totale(1,i)^2)+(acm_totale(2,i)^2));
    if (acm_totale_mod(i)>1)
        acm_totale_mod(i)= 0;
    end
end

figure(29)
plot(tempo,acm_totale_mod)
title('modulo accelerazione COM totale nel tempo');
ylabel("accelerazione [m/s^2]")
xlabel("tempo [s]")

figure(30)
hold on
acm_totale(:,41)=acm_totale(:,42);
acm_totale(:,90)=acm_totale(:,91);
plot(tempo,acm_totale(1,:))
plot(tempo,acm_totale(2,:))
title("accelerazione del centro di massa totale")
legend("accelerazione lungo x","accelerazione lungo y")
ylabel("accelerazione [m/s^2]")
xlabel("tempo [s]")
hold off

%% VELOCITÀ CENTRI DI MASSA
vxpiede = [];
vypiede = [];
vxgamba = [];
vygamba = [];
vxcoscia = [];
vycoscia = [];
vxHAT = [];
vyHAT = [];

for i=1:(length(pxginocchio)-1)
    vxpiede(i) = (cm_piede(1,i+1)-cm_piede(1,i))/(tempo(i+1)-tempo(i));
    vypiede(i) = (cm_piede(2,i+1)-cm_piede(2,i))/(tempo(i+1)-tempo(i));
    vxgamba(i) = (cm_gamba(1,i+1)-cm_gamba(1,i))/(tempo(i+1)-tempo(i));
    vygamba(i) = (cm_gamba(2,i+1)-cm_gamba(2,i))/(tempo(i+1)-tempo(i));
    vxcoscia(i) = (cm_coscia(1,i+1)-cm_coscia(1,i))/(tempo(i+1)-tempo(i));
    vycoscia(i) = (cm_coscia(2,i+1)-cm_coscia(2,i))/(tempo(i+1)-tempo(i));
    vxHAT(i) = (cm_HAT(1,i+1)-cm_HAT(1,i))/(tempo(i+1)-tempo(i));
    vyHAT(i) = (cm_HAT(2,i+1)-cm_HAT(2,i))/(tempo(i+1)-tempo(i));
end

vxpiede(end+1) = vxpiede(end);
vypiede(end+1) = vypiede(end);
vxgamba(end+1) = vxgamba(end);
vygamba(end+1) = vygamba(end);
vxcoscia(end+1) = vxcoscia(end);
vycoscia(end+1) = vycoscia(end);
vxHAT(end+1) = vxHAT(end);
vyHAT(end+1) = vyHAT(end);

% Modifiche per plot
vxpiede(42) = vxpiede(43);
vypiede(42) = vypiede(43);
vxgamba(42) = vxgamba(43);
vygamba(42) = vygamba(43);
vxcoscia(42) = vxcoscia(43);
vycoscia(42) = vycoscia(43);
vxHAT(42) = vxHAT(43);
vyHAT(42) = vyHAT(43);

vxpiede(91) = vxpiede(92);
vypiede(91) = vypiede(92);
vxgamba(91) = vxgamba(92);
vygamba(91) = vygamba(92);
vxcoscia(91) = vxcoscia(92);
vycoscia(91) = vycoscia(92);
vxHAT(91) = vxHAT(92);
vyHAT(91) = vyHAT(92);

%% ACCELERAZIONE CENTRI DI  MASSA
axpiede = [];
aypiede = [];
axgamba = [];
aygamba = [];
axcoscia = [];
aycoscia = [];
axHAT = [];
ayHAT = [];

for i=1:(length(pxginocchio)-1)
    axpiede(i) = (vxpiede(i+1)-vxpiede(i))/(tempo(i+1)-tempo(i));
    aypiede(i) = (vypiede(i+1)-vypiede(i))/(tempo(i+1)-tempo(i));
    axgamba(i) = (vxgamba(i+1)-vxgamba(i))/(tempo(i+1)-tempo(i));
    aygamba(i) = (vygamba(i+1)-vygamba(i))/(tempo(i+1)-tempo(i));
    axcoscia(i) = (vxcoscia(i+1)-vxcoscia(i))/(tempo(i+1)-tempo(i));
    aycoscia(i) = (vycoscia(i+1)-vycoscia(i))/(tempo(i+1)-tempo(i));
    axHAT(i) = (vxHAT(i+1)-vxHAT(i))/(tempo(i+1)-tempo(i));
    ayHAT(i) = (vyHAT(i+1)-vyHAT(i))/(tempo(i+1)-tempo(i));
end

axpiede(end+1) = axpiede(end);
aypiede(end+1) = aypiede(end);
axgamba(end+1) = axgamba(end);
aygamba(end+1) = aygamba(end);
axcoscia(end+1) = axcoscia(end);
aycoscia(end+1) = aycoscia(end);
axHAT(end+1) = axHAT(end);
ayHAT(end+1) = ayHAT(end);

% Modifiche per plot
axpiede(41) = axpiede(42);
aypiede(41) = aypiede(42);
axgamba(41) = axgamba(42);
aygamba(41) = aygamba(42);
axcoscia(41) = axcoscia(42);
aycoscia(41) = aycoscia(42);
axHAT(41) = axHAT(42);
ayHAT(41) = ayHAT(42);

axpiede(90) = axpiede(91);
aypiede(90) = aypiede(91);
axgamba(90) = axgamba(91);
aygamba(90) = aygamba(91);
axcoscia(90) = axcoscia(91);
aycoscia(90) = aycoscia(91);
axHAT(90) = axHAT(91);
ayHAT(90) = ayHAT(91);

%% STATICA 
% Plot static con centri di massa
g = 9.81;
F_peso = -(m_tot_gamba)*g; 
F_totale_cm_s = F_peso;
R_COP_s = - F_totale_cm_s;

%COPx_s = cm_totale(1,1) - cm_totale(2,1)*(0/F_totale_cm_s); %FORMULA VECCHIA COP  
COPx_s = (- F_peso*cm_totale(1,1))/R_COP_s; %FORMULA ASATTA COP STATICA

figure(31);
plot([O0(1) O1_T_s(1)],[O0(2) O1_T_s(2) ], 'Color', '[0.3010 0.7450 0.9330]','LineWidth',2);
hold on
plot([O0(1) Lpiede(1)],[O0(2) Lpiede(2)], 'Color', '[0.3010 0.7450 0.9330]','LineWidth',2);
plot([O1_T_s(1) O2_T_s(1)],[O1_T_s(2) O2_T_s(2)], 'Color', '[0.3010 0.7450 0.9330]','LineWidth',2);
plot([O2_T_s(1) O3_T_s(1)],[O2_T_s(2) O3_T_s(2)], 'Color', '[0.3010 0.7450 0.9330]','LineWidth',2);
plot([O3_T_s(1) O4_T_s(1)],[O3_T_s(2) O4_T_s(2)], 'Color', '[0.3010 0.7450 0.9330]','LineWidth',2);
 
p1= plot(O1_T_s(1), O1_T_s(2), 'o', 'MarkerFaceColor', '[0.6350 0.0780 0.1840]', 'MarkerEdgeColor', '[0.6350 0.0780 0.1840]', 'MarkerSize',10);
p2= plot(O2_T_s(1), O2_T_s(2), 'o', 'MarkerFaceColor', '[0.5, 0.75, 0.5]', 'MarkerEdgeColor', '[0.5, 0.75, 0.5]','MarkerSize',10);
p3= plot(O3_T_s(1), O3_T_s(2), 'o', 'MarkerFaceColor', '[0 0.4470 0.7410]', 'MarkerEdgeColor', '[0 0.4470 0.7410]','MarkerSize',10);
p4= plot(O4_T_s(1), O4_T_s(2), 'o', 'MarkerFaceColor', '[0.4940 0.1840 0.5560]', 'MarkerEdgeColor', '[0.4940 0.1840 0.5560]','MarkerSize',10);

title("trial static")


cm1=plot(cm_piede_s(1), cm_piede_s(2), 'p', 'MarkerFaceColor', 'm', 'MarkerEdgeColor', 'm', 'MarkerSize',10);
cm2=plot(cm_gamba_s(1), cm_gamba_s(2), 'p', 'MarkerFaceColor', 'b', 'MarkerEdgeColor', 'b', 'MarkerSize',10);
cm3=plot(cm_coscia_s(1),cm_coscia_s(2), 'p', 'MarkerFaceColor', 'g', 'MarkerEdgeColor', 'g', 'MarkerSize',10);
cm4=plot( cm_HAT_s(1), cm_HAT_s(2), 'p', 'MarkerFaceColor', 'r', 'MarkerEdgeColor', 'r', 'MarkerSize',10);
cmTOT= plot(cm_totale_s(1), cm_totale_s(2), '*', 'MarkerFaceColor', 'k', 'MarkerEdgeColor', 'k', 'MarkerSize',10);
COP = plot(COPx_s, 0,'hexagram', 'MarkerFaceColor', '#D95319', 'MarkerEdgeColor', '#D95319', 'MarkerSize',10);


legend([p1, p2, p3, p4, cm1, cm2, cm3,cm4,cmTOT,COP], {'caviglia', 'ginocchio', 'anca', 'spalla','COM piede', 'COM gamba', 'COM coscia', 'COM HAT','COM totale','COP'});
title("Trial statico - COM e COP") 
xlim([-0.5 1.5]);
ylim([0 1.8]);
hold off;


% Forze rispetto al COM
Rx_anca_s=0;
Rx_ginocchio_s=0;
Rx_caviglia_s=0;
Rx_piede_s=0;
Ry_anca_s= m_HAT_gamba*g;
Ry_ginocchio_s= Ry_anca_s + m_coscia*g;
Ry_caviglia_s= Ry_ginocchio_s+ m_gamba*g;
Ry_piede_s= Ry_caviglia_s + m_piede*g;

% Coppie rispetto le articolazioni, controllare BRACCIO PIEDE e se
% implementarlo
C_anca_s= Ry_anca_s*dd_hat(1)*cos(delta_rad(1)) - Rx_anca_s*dd_hat(1)*sin((delta_rad(1)));
C_ginocchio_s= C_anca_s - Rx_ginocchio_s*dd_coscia(1)*sin(gamma_rad(1))+...
    Ry_ginocchio_s*dd_coscia(1)*cos(gamma_rad(1)) - Rx_anca_s*dp_coscia(1)*sin(gamma_rad(1)) +...
    Ry_anca_s*dp_coscia(1)*cos(gamma_rad(1));
C_caviglia_s= C_ginocchio_s - Rx_caviglia_s*dd_gamba(1)*sin(beta_rad(1))+...
    Ry_caviglia_s*dd_gamba(1)*cos(beta_rad(1)) - Rx_ginocchio_s*dp_gamba(1)*sin(beta_rad(1)) +...
    Ry_ginocchio_s*dp_gamba(1)*cos(beta_rad(1));
C_piede_s= C_caviglia_s - Rx_piede_s*cm_totale(2,1)-...
    Ry_piede_s*(COPx_s-cm_totale(1,1)) - Rx_caviglia_s*dp_piede(1)*sin(alpha_rad(1)) +...
    Ry_caviglia_s*dp_piede(1)*cos(alpha_rad(1));

%% DINAMICA
% Forze rispetto le articolazioni
Rx_anca_d=m_HAT_gamba*axHAT;
Rx_ginocchio_d=Rx_anca_d+m_coscia*axcoscia;
Rx_caviglia_d=Rx_ginocchio_d+m_gamba*axgamba;
Rx_piede_d=Rx_caviglia_d+m_piede*axpiede;
Ry_anca_d=m_HAT_gamba*ayHAT+m_HAT_gamba*g;
Ry_ginocchio_d=Ry_anca_d+m_coscia*aycoscia+m_coscia*g;
Ry_caviglia_d=Ry_ginocchio_d+m_gamba*aygamba+m_gamba*g;
Ry_piede_d=Ry_caviglia_d+m_piede*aypiede+m_piede*g;

figure
hold on
plot(tempo,Rx_anca_d);
plot(tempo,Ry_anca_d);
xlabel("Tempo[s]")
ylabel("Forze [N]")
title("Forze sull'anca")
legend("Forza sull'anca su asse x","Forza sull'anca  su asse y",'Location','best')
hold off

figure
hold on
plot(tempo,Rx_ginocchio_d);
plot(tempo,Ry_ginocchio_d);
xlabel("Tempo[s]")
ylabel("Forze [N]")
title("Forze sul ginocchio")
legend("Forza sul ginocchio su asse x","Forza sul ginocchio  su asse y",'Location','best')
hold off

figure
hold on
plot(tempo,Rx_caviglia_d);
plot(tempo,Ry_caviglia_d);
xlabel("Tempo[s]")
ylabel("Forze [N]")
title("Forze sulla caviglia")
legend("Forza sulla caviglia su asse x","Forza sulla caviglia su asse y",'Location','best')
hold off

figure
hold on
plot(tempo,Rx_piede_d);
plot(tempo,Ry_piede_d);
xlabel("Tempo[s]")
ylabel("Forze [N]")
title("Forze sul piede")
legend("Forza sul piede su asse x","Forza sul piede su asse y",'Location','best')
hold off


% La derivata della quantità di moto angolare coincide con la coppia totale del sistema
Hg_deriv=[];
for i=1:length(cm_piede)
    Hg_deriv(i) = ((((cm_piede(1,i)-cm_totale(1,i))*(m_piede*(aypiede(i)-acm_totale(2,i))))-((cm_piede(2,i)-cm_totale(2,i))*(m_piede*(axpiede(i)-acm_totale(1,i))))) + ...
        (((cm_gamba(1,i)-cm_totale(1,i))*(m_gamba*(aygamba(i)-acm_totale(2,i))))-((cm_gamba(2,i)-cm_totale(2,i))*(m_gamba*(axgamba(i)-acm_totale(1,i))))) + ...
        (((cm_coscia(1,i)-cm_totale(1,i))*(m_coscia*(aycoscia(i)-acm_totale(2,i))))-((cm_coscia(2,i)-cm_totale(2,i))*(m_coscia*(axcoscia(i)-acm_totale(1,i))))) + ...
        (((cm_HAT(1,i)-cm_totale(1,i))*(m_HAT_gamba*(ayHAT(i)-acm_totale(2,i))))-((cm_HAT(2,i)-cm_totale(2,i))*(m_HAT_gamba*(axHAT(i)-acm_totale(1,i))))));
end

C_anca_d=[];
C_ginocchio_d=[];
C_caviglia_d=[];
COPx_d=[];

% Coppie rispetto le articolazioni
for i=1:length(dd_hat)
    C_anca_d(i)= Ry_anca_d(i)*dd_hat(i)*cos(delta_rad(i)) - Rx_anca_d(i)*dd_hat(i)*sin((delta_rad(i)));
    C_ginocchio_d(i)= C_anca_d(i) - Rx_ginocchio_d(i)*dd_coscia(i)*sin(gamma_rad(i))+...
        Ry_ginocchio_d(i)*dd_coscia(i)*cos(gamma_rad(i)) - Rx_anca_d(i)*dp_coscia(i)*sin(gamma_rad(i)) +...
        Ry_anca_d(i)*dp_coscia(i)*cos(gamma_rad(i));
    C_caviglia_d(i)= C_ginocchio_d(i) - Rx_caviglia_d(i)*dd_gamba(i)*sin(beta_rad(i))+...
        Ry_caviglia_d(i)*dd_gamba(i)*cos(beta_rad(i)) - Rx_ginocchio_d(i)*dp_gamba(i)*sin(beta_rad(i)) +...
        Ry_ginocchio_d(i)*dp_gamba(i)*cos(beta_rad(i));
end

figure(32)
plot(tempo,C_caviglia_d);
xlabel("Tempo[s]")
ylabel("Coppia [Nm]")
title("Coppia alla caviglia")
hold off

figure(33)
plot(tempo,C_ginocchio_d);
xlabel("Tempo[s]")
ylabel("Coppia [Nm]")
title("Coppia al ginocchio")
hold off

figure(34)
plot(tempo,C_anca_d);
xlabel("Tempo[s]")
ylabel("Coppia [Nm]")
title("Coppia all'anca")
hold off

F_peso = zeros(4,size(acm_totale,2));
for i=1:size(acm_totale,2)
    F_peso(2,i) = -(m_tot_gamba)*g;
    F_inerzia(:,i) = (m_tot_gamba)*acm_totale(:,i);
end
F_totale_cm = F_peso+F_inerzia;

% Calcolo COP
COPx_d = cm_totale(1,:)-(((cm_totale(2,:).*sum(F_totale_cm(1,:)))+Hg_deriv)./sum(F_totale_cm(2,:))); %formula esatta
%COPx_d = cm_totale(1,:) - (cm_totale(2,:) .* sum(F_totale_cm(1,:) + Hg_deriv)) ./ sum(F_totale_cm(2,:)); %formula iniziale 


% Coppia del piede
for i=1:length(dd_hat)
    
    C_piede_d(i)= C_caviglia_d(i) - Rx_piede_d(i)*cm_totale(2,i)-...
        Ry_piede_d(i)*(COPx_d(i)-cm_totale(1,i)) - Rx_caviglia_d(i)*dp_piede(i)*sin(alpha_rad(i)) +...
        Ry_caviglia_d(i)*dp_piede(i)*cos(alpha_rad(i));
end

figure
plot(tempo,C_piede_d);
xlabel("Tempo[s]")
ylabel("Coppia [Nm]")
title("Coppia del piede")
hold off


% Plot dinamic con i centri di massa
figure(35);
% Disegno la gamba
distance_cm_cop = [];
for i=1:size(tempo,1)
 
    T01 = [cos(theta1(i)) -cos(alpha_dh)*sin(theta1(i)) sin(alpha_dh)*sin(theta1(i)) L1(i)*cos(theta1(i));
    sin(theta1(i)) cos(alpha_dh)*cos(theta1(i)) -sin(alpha_dh)*cos(theta1(i)) L1(i)*sin(theta1(i));
    0 sin(alpha_dh) cos(alpha_dh) d;0 0 0 1];
 
    T12 = [cos(theta2(i)) -cos(alpha_dh)*sin(theta2(i)) sin(alpha_dh)*sin(theta2(i)) L2(i)*cos(theta2(i));
    sin(theta2(i)) cos(alpha_dh)*cos(theta2(i)) -sin(alpha_dh)*cos(theta2(i)) L2(i)*sin(theta2(i));
    0 sin(alpha_dh) cos(alpha_dh) d;0 0 0 1];
 
    T23 = [cos(theta3(i)) -cos(alpha_dh)*sin(theta3(i)) sin(alpha_dh)*sin(theta3(i)) L3(i)*cos(theta3(i));
    sin(theta3(i)) cos(alpha_dh)*cos(theta3(i)) -sin(alpha_dh)*cos(theta3(i)) L3(i)*sin(theta3(i));
    0 sin(alpha_dh) cos(alpha_dh) d;0 0 0 1];

    T34 = [cos(theta4(i)) -cos(alpha_dh)*sin(theta4(i)) sin(alpha_dh)*sin(theta4(i)) L4(i)*cos(theta4(i));
    sin(theta4(i)) cos(alpha_dh)*cos(theta4(i)) -sin(alpha_dh)*cos(theta4(i)) L4(i)*sin(theta4(i));
    0 sin(alpha_dh) cos(alpha_dh) d;
    0 0 0 1];
 
    % Applicazione della matrice di trasformazione al sdr 1
    O1_T= T01*O1;
    X1_T= T01*X1;
    Y1_T= T01*Y1;
 
    % Applicazione della matrice di trasformazione al sdr 2
    O2_T= T01*T12*O2;
    X2_T= T01*T12*X2;
    Y2_T= T01*T12*Y2;

    % Applicazione della matrice di trasformazione al sdr 3
    O3_T= T01*T12*T23*O3;
    X3_T= T01*T12*T23*X3;
    Y3_T= T01*T12*T23*Y3;

    % Applicazione della matrice di trasformazione al sdr 4
    O4_T= T01*T12*T23*T34*O3;
    X4_T= T01*T12*T23*T34*X3;
    Y4_T= T01*T12*T23*T34*Y3;

    % Disegno i link
    plot([O0(1) O1_T(1)],[O0(2) O1_T(2) ], 'Color', '[0.3010 0.7450 0.9330]','LineWidth',2);
    hold on
    plot([O0(1) Lpiede(1)],[O0(2) Lpiede(2)], 'Color', '[0.3010 0.7450 0.9330]','LineWidth',2);
    plot([O1_T(1) O2_T(1)],[O1_T(2) O2_T(2)], 'Color', '[0.3010 0.7450 0.9330]','LineWidth',2);
    plot([O2_T(1) O3_T(1)],[O2_T(2) O3_T(2)], 'Color', '[0.3010 0.7450 0.9330]','LineWidth',2);
    plot([O3_T(1) O4_T(1)],[O3_T(2) O4_T(2)], 'Color', '[0.3010 0.7450 0.9330]','LineWidth',2);

    p1= plot(O1_T(1), O1_T(2), 'o', 'MarkerFaceColor', '[0.6350 0.0780 0.1840]', 'MarkerEdgeColor', '[0.6350 0.0780 0.1840]', 'MarkerSize',10);
    p2= plot(O2_T(1), O2_T(2), 'o', 'MarkerFaceColor', '[0.5, 0.75, 0.5]', 'MarkerEdgeColor', '[0.5, 0.75, 0.5]','MarkerSize',10);
    p3= plot(O3_T(1), O3_T(2), 'o', 'MarkerFaceColor', '[0 0.4470 0.7410]', 'MarkerEdgeColor', '[0 0.4470 0.7410]','MarkerSize',10);
    p4= plot(O4_T(1), O4_T(2), 'o', 'MarkerFaceColor', '[0.4940 0.1840 0.5560]', 'MarkerEdgeColor', '[0.4940 0.1840 0.5560]','MarkerSize',10);

   
    cm1=plot(cm_piede(1,i), cm_piede(2,i), 'p', 'MarkerFaceColor', 'm', 'MarkerEdgeColor', 'm', 'MarkerSize',10);
    cm2=plot(cm_gamba(1,i), cm_gamba(2,i), 'p', 'MarkerFaceColor', 'b', 'MarkerEdgeColor', 'b', 'MarkerSize',10);
    cm3=plot(cm_coscia(1,i),cm_coscia(2,i), 'p', 'MarkerFaceColor', 'g', 'MarkerEdgeColor', 'g', 'MarkerSize',10);
    cm4=plot( cm_HAT(1,i), cm_HAT(2,i), 'p', 'MarkerFaceColor', 'r', 'MarkerEdgeColor', 'r', 'MarkerSize',10);
    cmTOT=plot(cm_totale(1,i), cm_totale(2,i), '*', 'MarkerFaceColor', 'k', 'MarkerEdgeColor', 'k', 'MarkerSize',10);
    COP = plot(COPx_d(i), 0,'hexagram', 'MarkerFaceColor', '#D95319', 'MarkerEdgeColor', '#D95319', 'MarkerSize',10);

    distance_cm_cop(i) = cm_totale(1,i)-COPx_d(i);

    legend([p1, p2, p3, p4, cm1, cm2, cm3,cm4,cmTOT,COP], {'caviglia', 'ginocchio', 'anca', 'spalla','COM piede', 'COM gamba', 'COM coscia', 'COM HAT','COM totale','COP'});
    title("Trial dinamico - COM e COP")

    xlim([-0.5 1.5]);
    ylim([0 1.8]);

    pause(0.05); 
        if i ~= length(tempo)
          clf(figure(35),'reset');
        end
end
hold off

%% Plot distanza tra COM e COP 
figure(36)
plot(tempo,distance_cm_cop)
xlabel("Tempo[s]")
ylabel("Distanza [m]")
title("Distanza tra COM e COP")
hold off

% Plot traiettoria COP su x
figure(37)
plot(tempo,COPx_d)
xlabel("Tempo[s]")
ylabel("Distanza [m]")
title("Traiettoria COP")
hold off

% Plot distanza tra COP e tallone (essendo tallone fisso, equivale alla
% traiettoria del COP)
figure(38)
plot(tempo,COPx_d)
xlabel("Tempo[s]")
ylabel("Distanza [m]")
title("Distanza tra COP e tallone")
hold off

% Plot distanza tra COP e punta(0.152 * H) su x
punta = 0.152 * H;
figure(39)
plot(tempo,punta-COPx_d)
xlabel("Tempo[s]")
ylabel("Distanza [m]")
title("Distanza tra COP e punta")
hold off


%% QUANTITÀ DI MOTO 
% Lineare 
qdmlxpiede = m_piede.*vxpiede;
qdmlypiede = m_piede.*vypiede;
qdmlxgamba = m_gamba.*vxgamba;
qdmlygamba = m_gamba.*vygamba;
qdmlxcoscia = m_coscia.*vxcoscia;
qdmlycoscia = m_coscia.*vycoscia;
qdmlxHAT = m_HAT_gamba.*vxHAT;
qdmlyHAT = m_HAT_gamba.*vyHAT;

qdmlxtot=qdmlxpiede+qdmlxgamba+qdmlxcoscia+qdmlxHAT;
qdmlytot=qdmlypiede+qdmlygamba+qdmlycoscia+qdmlyHAT;

qdmapiede = [];
qdmagamba = [];
qdmacoscia = [];
qdmaHAT = [];

% Angolare 
for i=1:length(vypiede)
    qdmapiede(i) = (((cm_piede(1,i)-cm_totale(1,i))*(m_piede*(vypiede(i)-vcm_totale(2,i))))-...
        ((cm_piede(2,i)-cm_totale(2,i))*(m_piede*(vxpiede(i)-vcm_totale(1,i)))));
    qdmagamba(i) = (((cm_gamba(1,i)-cm_totale(1,i))*(m_gamba*(vygamba(i)-vcm_totale(2,i))))-...
        ((cm_gamba(2,i)-cm_totale(2,i))*(m_gamba*(vxgamba(i)-vcm_totale(1,i)))));
    qdmacoscia(i) = (((cm_coscia(1,i)-cm_totale(1,i))*(m_coscia*(vycoscia(i)-vcm_totale(2,i))))-...
        ((cm_coscia(2,i)-cm_totale(2,i))*(m_coscia*(vxcoscia(i)-vcm_totale(1,i)))));
    qdmaHAT(i) = (((cm_HAT(1,i)-cm_totale(1,i))*(m_HAT_gamba*(vyHAT(i)-vcm_totale(2,i))))-...
        ((cm_HAT(2,i)-cm_totale(2,i))*(m_HAT_gamba*(vxHAT(i)-vcm_totale(1,i)))));
end

%% Plot quantità di moto

%Plot quantità di moto lineare totale
qdmltot  = sqrt((qdmlxtot.^2) + (qdmlytot.^2));
figure(40)
% hold on
% plot(tempo,qdmlxtot)
% plot(tempo,qdmlytot)
plot(tempo,qdmltot)
xlabel("Tempo[s]")
ylabel("Quantità di moto lineare  [Kg*m/s]")
title("Quantità di moto lineare totale")
% legend("Quantità di moto lineare su asse x","Quantità di moto lineare su asse y",'Location','best')
hold off

% Plot derivata quantità di moto lineare totale
Dqdml = ones(1,length(vypiede));
Dqdml = (m_piede + m_gamba + m_coscia + m_HAT_gamba ).* acm_totale_mod;
figure(41)
plot(tempo,Dqdml)
xlabel("Tempo[s]")
ylabel("Derivata quantità di moto lineare  [N]")
title("Derivata quantità di moto lineare totale")
hold off

% Plot quantità di moto angolare totale
qdmatot = qdmapiede + qdmagamba + qdmacoscia + qdmaHAT ;
figure(42)
plot(tempo,qdmatot)
xlabel("Tempo[s]")
ylabel("Quantità di moto angolare  [Kg*m^2/s]")
title("Quantità di moto angolare totale")
hold off

% Plot derivata quantità di moto angolare totale (calcolata nella sezione
% dinamica)
figure(43)
plot(tempo,Hg_deriv)
xlabel("Tempo[s]")
ylabel("Derivata quantità di moto angolare  [Nm]")
title("Derivata quantità di moto angolare totale")
hold off


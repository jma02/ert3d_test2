
% plot surface model
dx_main = 30;
dy_main = 15;

Lx = dx_main/2;
Ly = dy_main/2;
k = 0.0025;

[X, Y] = meshgrid(linspace(-dx_main/2, dx_main/2, 50), linspace(-dy_main/2, dy_main/2, 50));

Z = 3*sin(7*X/dx_main).*sin(8*(Y-0.25)/dy_main).*exp(-k*((Lx-X).^2+(Ly-Y).^2));
surf(X, Y, Z)


load sample1.mat

% plot electrode centroids on the ground surface
figure(1)
plot3(u.line1.electrodes_xyz(:,1), u.line1.electrodes_xyz(:,2), u.line1.electrodes_xyz(:,3), 'rx')
hold on
plot3(u.line2.electrodes_xyz(:,1), u.line2.electrodes_xyz(:,2), u.line2.electrodes_xyz(:,3), 'gx')
plot3(u.line3.electrodes_xyz(:,1), u.line3.electrodes_xyz(:,2), u.line3.electrodes_xyz(:,3), 'bx')
legend('line1', 'line2', 'line3')
title('measuring electrodes on each line')


% example injection from each line
figure(2)
injid = 3;
plot3(u.line1.inj(injid).Iin_xyz(1), u.line1.inj(injid).Iin_xyz(2), u.line1.inj(injid).Iin_xyz(3), 'rx')
hold on
plot3(u.line1.inj(injid).Iout_xyz(1), u.line1.inj(injid).Iout_xyz(2), u.line1.inj(injid).Iout_xyz(3), 'bx')

injid = 1;
plot3(u.line2.inj(injid).Iin_xyz(1), u.line2.inj(injid).Iin_xyz(2), u.line2.inj(injid).Iin_xyz(3), 'rs')
plot3(u.line2.inj(injid).Iout_xyz(1), u.line2.inj(injid).Iout_xyz(2), u.line2.inj(injid).Iout_xyz(3), 'bs')

injid = 5;
plot3(u.line3.inj(injid).Iin_xyz(1), u.line3.inj(injid).Iin_xyz(2), u.line3.inj(injid).Iin_xyz(3), 'rd')
plot3(u.line3.inj(injid).Iout_xyz(1), u.line3.inj(injid).Iout_xyz(2), u.line3.inj(injid).Iout_xyz(3), 'bd')

legend('inj line1 (in)', 'inj line1 (out)', 'inj line2 (in)', 'inj line2 (out)', 'inj line3 (in)', 'inj line3 (out)')
title('example injecting electrodes on each line')

% conductivity distribution
figure(3)
scatter3(g(:,1), g(:,2), g(:,3), 5, sigma_model, 'filled')
title('conductivity distribution')

% example data
figure(4)
subplot(3,1,1)
injid = 3;
plot(u.line1.inj(injid).U)
ylabel(['voltage values for line1, inj id ' num2str(injid)])
xlabel('measurement id')
subplot(3,1,2)
injid = 1;
plot(u.line2.inj(injid).U)
ylabel(['voltage values for line2, inj id ' num2str(injid)])
xlabel('measurement id')
subplot(3,1,3)
injid = 4;
plot(u.line3.inj(injid).U)
ylabel(['voltage values for line3, inj id ' num2str(injid)])
xlabel('measurement id')
    
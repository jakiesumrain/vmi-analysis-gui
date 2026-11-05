%% Load Integrals
gData = loadG('G_r512_k512_l10.h5',1);
%% Load Data

data = load('image.txt');
%% Do Inversion
x0 = 512;
y0 = 512;
% Set default values for parameters if not passed from Python
if ~exist('quadrants', 'var')
    quadrants = [1,1,1,1]; % Default if not set
end
if ~exist('momentum_file', 'var')
    momentum_file = 'px-py_l10.txt'; % Default output file for momentum distribution
end
if ~exist('energy_file', 'var')
    energy_file = 'energy_l10.txt'; % Default output file for energy values
end
if ~exist('spectrum_file', 'var')
    spectrum_file = 'energy_spec_l10.txt'; % Default output file for energy spectrum
end
if ~exist('beta_file', 'var')
    beta_file = 'betas10.txt'; % Default output file for beta parameters
end
if ~exist('regularization', 'var')
    regularization = 0; % Default regularization parameter
end
fold = resizeFolded(foldQuadrant(data,x0,y0,quadrants),512);
%%
out = pbasex(fold,gData,1, 0, regularization, 1.0); % Using regularization parameter from config
%% Plot Inversion Data
figure;
subplot(1,1,1)
plot(out.E,out.IE(:)*diag(1./mean(out.IE(:)))) % normalize by image and plot
xlabel('Energy (eV)')
ylabel('Intensity (normalized)')
legend('energy spectrum')
title('Sample CPBASEX Inversion Data')
%ylim([0,5])

writematrix(out.E, energy_file);
writematrix(out.IE, spectrum_file);
writematrix(out.betas, beta_file);
%%
figure;
subplot(1,3,1)
imagesc(data(:,:)); % image raw data
title('Raw Image')
subplot(1,3,2)
imagesc(out.recon(:,:)) % image fit data
title('Non-Inverted Fit')
subplot(1,3,3)
imagesc(out.inv(:,:)) % image inverted fit data
title('Inverted Fit')
%caxis([0,1e2])

writematrix(out.inv, momentum_file);
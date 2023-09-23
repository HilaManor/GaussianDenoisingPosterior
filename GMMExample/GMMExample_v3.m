clc;clear; close all;

set(0,'defaultaxesfontname', 'Computer Modern');
set(0,'defaultaxesfontsize',20);
howmuch = 20;
delta = 0.005;
delta_opt = 0.005;

bit = 0.5;

%% Estimation parameters for theoretical 
vmu1 = [-4.56906402  1.19054303];
vmu2 = [7.45312006 0.93325484];
vmu3 = [-29.929742882262423468, 0.037439587322420805822];
vmu4 = [230.34131759225654054, 2.7227885187301910177];

noisy_sample = [2.357602207535946; 6.3522639145461675];
mmse_sample = [1.9932879400384003; 4.280250212414902];

evs = [-0.63709737 -0.77078333;
       -0.77078333  0.63709737];

% Parameters of the density px
sn2 = 4;
m1 = [1 2];
m2 = [8 10];
sigma1 = [1 0; 0 2];
sigma2 = [2 1; 1 1];
sigmaN = eye(2)*sn2;

i = 1; % ev num
%% Plot figure for the theoretical
figure;

proj_xs = (vmu1(i)-howmuch):delta_opt:(vmu1(i)+howmuch);

fun = @(z)-(z(1) + z(2) * vmu1(i) +  z(3) * vmu2(i) + z(4) * vmu3(i) + z(5) * vmu4(i)) ...
    + delta_opt * sum(exp(z(1) + z(2)* proj_xs(1,:) + z(3) * (proj_xs(1,:) - vmu1(i)).^2 + z(4) * (proj_xs(1,:) - vmu1(i)).^3 + z(5) * (proj_xs(1,:) - vmu1(i)).^4));
options = optimoptions('fmincon', 'MaxFunctionEvaluations', 6000);
z = fmincon(fun, zeros(5,1), [],[],[],[],[],[],[], options);
f = @(x) exp(z(1) + z(2) * x(1,:) + z(3) * (x(1,:) - vmu1(i)).^2 + z(4) * (x(1,:) - vmu1(i)).^3 + z(5)*(x(1,:) - vmu1(i)).^4);

yup3 = (vmu1(i) + 3 * sqrt(vmu2(i)));
ydown3 = (vmu1(i) - 3 * sqrt(vmu2(i)));

abit = bit * sqrt(vmu2(i));

less_proj_noncentered_xs = (ydown3 - abit):delta:(yup3 + abit);
func_pdf = f(less_proj_noncentered_xs) / (delta * sum(f(less_proj_noncentered_xs)));

subplot(2, 1, 1);
draw_xs = linspace(-3 -bit, 3 + bit, numel(less_proj_noncentered_xs));
set(gca,'TickLabelInterpreter','latex')

plot(draw_xs, func_pdf, 'r--');
hold on;

marg_post = x_given_y_v(less_proj_noncentered_xs, noisy_sample, m1, m2, sigma1, sigma2, sigmaN, evs(:, i));
plot(draw_xs, marg_post, 'b-');
set(gca,'TickLabelInterpreter','latex')

xtickformat('%.1g$\\sqrt{\\lambda}$')

legend("Estimated", "GT", 'interpreter', 'latex', 'location', 'northwest', 'FontSize', 19.5);
% legend("Estimated with moments", "True", "$v_1^T\mu_1$", 'interpreter', 'latex', 'location', 'southoutside', 'Orientation', 'horizontal', 'NumColumns', 3);
% legend boxoff
%         legend("Estimated with moments", "True", ...
%             "$v_1^T\mu_1$", ...
%             sprintf("$v_1^T\\mu_1 + 3\\sqrt{\\lambda_%d}$", i), ...
%             sprintf("$v_1^T\\mu_1 - 3\\sqrt{\\lambda_%d}$", i), 'interpreter', 'latex', 'location', 'southoutside', 'Orientation','horizontal', 'NumColumns', 3);

yticks([]);
title(sprintf('Posterior along $v_%d$', i), 'Interpreter', 'latex')
axis tight;
%%
subplot(2, 1, 2);
set(gca,'TickLabelInterpreter','latex')

% Calc true posterior
less_xs_centered = linspace(-3-bit, 3+bit, numel(less_proj_noncentered_xs)) * sqrt(vmu2(i));
% dt = less_xs_centered(2) - less_xs_centered(1);
less_xs_centered_v = less_xs_centered .* evs(:, i);
less_xs = noisy_sample + less_xs_centered_v;

deriv_func = evs(:, i)' * mu1_real(less_xs, m1, m2, sigma1, sigma2, sigmaN)';
plot(draw_xs, deriv_func, 'b-'); hold on;
scatter(0,   vmu1(i), 100, 'ko', 'filled')
title(sprintf('$v_1^T \\mu_1(y+\\alpha v_%d)$', i), 'Interpreter', 'latex');
axis tight;
yticks([]);
[~, objh] = legend(sprintf('$v_1^T \\mu_1(y+\\alpha v_%d)$', i), ...
       "$v_1^T\mu_1(y)$",  'interpreter', 'latex', 'location', 'southeast', 'FontSize', 19.5);
set(gca,'TickLabelInterpreter','latex')
% xlabel('\alpha')
xtickformat('%.1g$\\sqrt{\\lambda}$')

objhl = findobj(objh, 'type', 'patch');
set(objhl, 'MarkerSize', 10);

h = gcf;
width = 5.5206;
height = width;
h.PaperUnits = 'inches';
h.PaperPosition = [0,0,width,height];
h.PaperSize = [width,height];
h.Units = 'inches';
h.Position = [0,0,width,height];



% print('-dpng','-r600','./GMM_Example_theoretical_prob.png')
% print('-demf','-r600','./GMM_Example_theoretical_prob.emf')
print('-dpdf','-r600','./GMM_Example_theoretical_prob.pdf')
% print('-dsvg','-r600','./GMM_Example_theoretical_prob.svg')

%% Estimation parameters for NN

vmu1 = [-4.30736081  1.3081067 ];
vmu2 = [8.00351897 1.23179447];
vmu3 = [-39.45561205402611, -0.4084097949397242];
vmu4 = [359.2421455165662, -134.50419747418377];
noisy_sample = [2.357602207535946; 6.3522639145461675];
mmse_sample = [1.8525186898719517; 4.102764288684162];
evs = [[-0.65860592 -0.75248804]
 [-0.75248804  0.65860592]];


load('nn_vmu1_vals.mat')

%% Plot figure for the NN
% figure('units', 'inch', 'position', []);
figure;

proj_xs = (vmu1(i)-howmuch):delta:(vmu1(i)+howmuch);

fun = @(z)-(z(1) + z(2)* vmu1(i) +  z(3) * vmu2(i) + z(4) * vmu3(i) + z(5) * vmu4(i)) ...
    + delta * sum(exp(z(1) + z(2)* proj_xs(1,:) + z(3) * (proj_xs(1,:) - vmu1(i)).^2 + z(4) * (proj_xs(1,:) - vmu1(i)).^3 + z(5) * (proj_xs(1,:) - vmu1(i)).^4));
options = optimoptions('fmincon', 'MaxFunctionEvaluations', 6000);
z = fmincon(fun, zeros(5,1), [],[],[],[],[],[],[], options);
f = @(x) exp(z(1) + z(2) * x(1,:) + z(3) * (x(1,:) - vmu1(i)).^2 + z(4) * (x(1,:) - vmu1(i)).^3 + z(5)*(x(1,:) - vmu1(i)).^4);

yup3 = (vmu1(i) + 3 * sqrt(vmu2(i)));
ydown3 = (vmu1(i) - 3 * sqrt(vmu2(i)));

abit = bit * sqrt(vmu2(i));

less_proj_noncentered_xs = (ydown3 - abit):delta:(yup3 + abit);
func_pdf = f(less_proj_noncentered_xs) / (delta * sum(f(less_proj_noncentered_xs)));

subplot(2, 1, 1);
draw_xs = linspace(-3 -bit, 3 + bit, numel(less_proj_noncentered_xs));
set(gca,'TickLabelInterpreter','latex')

plot(draw_xs, func_pdf, 'r--');
hold on;

% calc true posterior
marg_post = x_given_y_v(less_proj_noncentered_xs, noisy_sample, m1, m2, sigma1, sigma2, sigmaN, evs(:, i));
plot(draw_xs, marg_post, 'b-');

set(gca,'TickLabelInterpreter','latex')
xtickformat('%.1g$\\sqrt{\\lambda}$')
legend("Estimated", "GT", 'interpreter', 'latex' ,'location', 'northwest', 'FontSize', 19.5);

yticks([]);
title(sprintf('Posterior along $v_%d$', i), 'Interpreter', 'latex')
axis tight;

subplot(2, 1, 2);

set(gca,'TickLabelInterpreter','latex')

% Calc true posterior
less_xs_centered = linspace(-3-bit, 3+bit, numel(less_proj_noncentered_xs)) * sqrt(vmu2(i));
less_xs_centered_v = less_xs_centered .* evs(:, i);
less_xs = noisy_sample + less_xs_centered_v;

plot(draw_xs, nn_vmu1_vals, 'b-'); hold on;
scatter(0,   vmu1(i), 100, 'ko', 'filled')
title(sprintf('$v_1^T \\hat{\\mu}_1(y+\\alpha v_%d)$', i), 'Interpreter', 'latex');
axis tight;
yticks([]);
[~, objh] = legend(sprintf('$v_1^T \\hat{\\mu}_1(y+\\alpha v_%d)$', i), ...
       "$v_1^T\hat{\mu}_1(y)$",  'interpreter', 'latex', 'location', 'southeast', 'FontSize', 19.5);
set(gca,'TickLabelInterpreter','latex')
% xlabel('\alpha')
xtickformat('%.1g$\\sqrt{\\lambda}$')

objhl = findobj(objh, 'type', 'patch');
set(objhl, 'MarkerSize', 10);

h = gcf;
width = 5.5206;
height = width;
h.PaperUnits = 'inches';
h.PaperPosition = [0,0,width,height];
h.PaperSize = [width,height];
h.Units = 'inches';
h.Position = [0,0,width,height];


% print('-dpng','-r600','./GMM_Example_NN_prob.png', '-fillpage')
% print('-demf','-r600','./GMM_Example_NN_prob.emf')
print('-dpdf','-r600','./GMM_Example_NN_prob.pdf')
% print('-dsvg','-r600','./GMM_Example_NN_prob.svg', '-fillpage')


% legend("Estimated with moments for analytical $\mu_1$", ...
%        "True", "$v_1^T\mu_1$", ...
%        "Estimated with moments for NN model", ...
%        "$v_1^T\hat{\mu_1}$", ...
%        'interpreter', 'latex', 'location', 'southoutside', 'Orientation','horizontal', 'NumColumns', 3);
%         legend boxoff

% print('-dpdf','-r500','filename.pdf')
% print('-dpng','-r500','Figures/probs.png')



function out = mu1_real(Yt, m1, m2, sigma1, sigma2, sigmaN)
    Y = Yt';

    e1 = (m1' + sigma1 * inv(sigma1 + sigmaN) * (Y' - m1'))';
    e2 = (m2' + sigma2 * inv(sigma2 + sigmaN) * (Y' - m2'))';

    p_y_given_from1 = mvnpdf(Y, m1, sigma1 + sigmaN);
    p_y_given_from2 = mvnpdf(Y, m2, sigma2 + sigmaN);
    p_from1_given_y = p_y_given_from1 ./ (p_y_given_from1 + p_y_given_from2);
    p_from2_given_y = 1 - p_from1_given_y;
%     p_from2_given_y = p_y_given_from2 ./ (p_y_given_from1 + p_y_given_from2)
    
    out = (e1.* p_from1_given_y) + (e2.* p_from2_given_y);
end 


function out = x_given_y_v(alpha, Yt, m1, m2, sigma1, sigma2, sigmaN, v)
    Y = Yt';
    p_y_given_from1 = mvnpdf(Y, m1, sigma1 + sigmaN);
    p_y_given_from2 = mvnpdf(Y, m2, sigma2 + sigmaN);
    p_from1_given_y = p_y_given_from1 ./ (p_y_given_from1 + p_y_given_from2);
    p_from2_given_y = 1 - p_from1_given_y;
%     p_from2_given_y = p_y_given_from2 ./ (p_y_given_from1 + p_y_given_from2)
    p_x_given_y_from1 = mvnpdf(alpha', v' * (m1' + sigma1 * inv(sigma1 + sigmaN) * (Y' - m1')), v' * (sigma1 - sigma1 * inv(sigma1 + sigmaN) *sigma1) * v);
    p_x_given_y_from2 = mvnpdf(alpha', v' * (m2' + sigma2 * inv(sigma2 + sigmaN) * (Y' - m2')), v' * (sigma2 - sigma2 * inv(sigma2 + sigmaN) *sigma2) * v);

    out = (p_x_given_y_from1 .* p_from1_given_y) + (p_x_given_y_from2 .* p_from2_given_y);
end 


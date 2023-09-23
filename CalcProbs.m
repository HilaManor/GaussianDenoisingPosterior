clc;clear; close all;

%% Define inputs

outputs = fullfile("Outputs", "Rebuttal"); % Inputs folder
% outputs = fullfile("Outputs", "Faces_fix");
% outputs = fullfile("Outputs", "FMD_new", "ConfocalFIsh");
% outputs = fullfile("Outputs", "NaturalImages", "New/");
delta = 0.01;       % how infinitesimal are we going with the integral
howmuch = 20;       % the interval of the optimized integral, depends on the domain
one_and_a_half = 0; % 1= show [-3, -1.5, 0, 1.5, 3]*lambda / 0= show [-3, -1, 0, 1, 3]*lambda
use_bigger = 0;     % Use a different c constant moments
use_poly = 1;       % Use the polynomial fit moments

% delta = 0.1;
% howmuch = 100;

%% Run

set(0,'defaultaxesfontname', 'Computer Modern');
set(0,'defaultaxesfontsize', 16);

subfolders = dir(fullfile(outputs, "*")); 
subfolders = subfolders(~ismember({subfolders.name},{'.','..'}));

for subdir_i=1:numel(subfolders)
    subsubfolders = dir(fullfile(subfolders(subdir_i).folder, subfolders(subdir_i).name, "*"));
    subsubfolders = subsubfolders(~ismember({subsubfolders.name},{'.','..'}));
    subsubfolders = subsubfolders([subsubfolders.isdir]);

    disp(subfolders(subdir_i).name)

    for subsubdir_i=1:numel(subsubfolders)
        curdir = fullfile(subsubfolders(subsubdir_i).folder, subsubfolders(subsubdir_i).name);
    

        if use_poly
            if ~exist(fullfile(curdir, "poly_moments.mat"), 'file')
                    continue
                end
        
                if exist(fullfile(outputs, subfolders(subdir_i).name, sprintf('%s_prob.png', subsubfolders(subsubdir_i).name)), "file")
                    continue
                end
        
                load(fullfile(curdir, "poly_moments.mat"))
        else
            if use_bigger
                if ~exist(fullfile(curdir, "bigger_c_moments.mat"), 'file')
                    continue
                end
        
                if exist(fullfile(outputs, subfolders(subdir_i).name, sprintf('%s_prob_biggerm.png', subsubfolders(subsubdir_i).name)), "file")
                    continue
                end
        
                load(fullfile(curdir, "bigger_c_moments.mat"))
            else
                if ~exist(fullfile(curdir, "moments.mat"), 'file')
                    continue
                end
        
                if exist(fullfile(outputs, subfolders(subdir_i).name, sprintf('%s_prob.png', subsubfolders(subsubdir_i).name)), "file")
                    continue
                end
        
                load(fullfile(curdir, "moments.mat"))
            end
        end
        
        
        f1 = figure('units','inch','position',[0,0,15,12], 'visible','off');
        ncols = 8;
        n_ev = cast(n_ev,"like",ncols);
        nrows = n_ev * 2;
        im = imread(fullfile(curdir, "im.png"));
        npatch = imread(fullfile(curdir, "npatch.png"));
        rpatch = imread(fullfile(curdir, "rpatch.png"));
        ppatch = imread(fullfile(curdir, "ppatch.png"));
        subplot(nrows,ncols, 1:ncols:(nrows*ncols)); imshow(im); title("Original Image"); axis tight;
        subplot(nrows,ncols, 2:ncols:(n_ev*ncols)); imshow(ppatch); title("Original Patch"); axis tight;
        subplot(nrows,ncols, ((ncols*n_ev)+2):ncols:(nrows*ncols)); imshow(npatch); title("Noisy Patch"); axis tight;
        
        for i=1:n_ev
%             figure(f1);
            set(0, 'CurrentFigure', f1);
            xs = (vmu1(i)-howmuch):delta:(vmu1(i)+howmuch);
            eigvec = imread(fullfile(curdir, sprintf("pc%d_eigvec.png", i)));
            if one_and_a_half
                evdown15 = imread(fullfile(curdir, sprintf("pc%d_evdown_1.5.png", i)));
            else
                evdown1 = imread(fullfile(curdir, sprintf("pc%d_evdown_1.png", i)));
            end
            evdown3 = imread(fullfile(curdir, sprintf("pc%d_evdown_3.png", i)));
            if one_and_a_half
                evup15 = imread(fullfile(curdir, sprintf("pc%d_evup_1.5.png", i)));
            else
                evup1 = imread(fullfile(curdir, sprintf("pc%d_evup_1.png", i)));
            end
            evup3 = imread(fullfile(curdir, sprintf("pc%d_evup_3.png", i)));
            subplot(nrows, ncols, (2*(i-1)*ncols) + 3); imshow(eigvec); title(sprintf("EigVec, Eigval$=%0.2g$", vmu2(i)), 'Interpreter','latex'); axis tight;
            subplot(nrows, ncols, (2*(i-1)*ncols) + 4); imshow(evdown3); title(sprintf("$-3\\sqrt{\\lambda_%d}$", i), 'Interpreter','latex'); axis tight;
            if one_and_a_half
                subplot(nrows, ncols, ncols*(i-1)*2 + 5); imshow(evdown15); title(sprintf("$-1.5\\sqrt{\\lambda_%d}$", i), 'Interpreter','latex'); axis tight;
            else
                subplot(nrows, ncols, ncols*(i-1)*2 + 5); imshow(evdown1); title(sprintf("$-1\\sqrt{\\lambda_%d}$", i), 'Interpreter','latex'); axis tight;
            end
            subplot(nrows, ncols, ncols*(i-1)*2 + 6); imshow(rpatch); title("MMSE restoration", 'Interpreter','latex'); axis tight;
            if one_and_a_half    
                subplot(nrows, ncols, ncols*(i-1)*2 + 7); imshow(evup15); title(sprintf("$+1.5\\sqrt{\\lambda_%d}$", i), 'Interpreter','latex'); axis tight;
            else
                subplot(nrows, ncols, ncols*(i-1)*2 + 7); imshow(evup1); title(sprintf("$+1\\sqrt{\\lambda_%d}$", i), 'Interpreter','latex'); axis tight;
            end
            subplot(nrows, ncols, ncols*(i-1)*2 + 8); imshow(evup3); title(sprintf("$+3\\sqrt{\\lambda_%d}$", i), 'Interpreter','latex'); axis tight;
        

            %% The maximum entropy optimization problem
            fun = @(z)-(z(1) + z(2)* vmu1(i) +  z(3) * vmu2(i) + z(4) * vmu3(i) + z(5) * vmu4(i)) ...
                + delta * sum(exp(z(1) + z(2)* xs(1,:) + z(3) * (xs(1,:) - vmu1(i)).^2 + z(4) * (xs(1,:) - vmu1(i)).^3 + z(5) * (xs(1,:) - vmu1(i)).^4));
            z = fmincon(fun, zeros(5,1));
            f = @(x) exp(z(1) + z(2) * x(1,:) + z(3) * (x(1,:) - vmu1(i)).^2 + z(4) * (x(1,:) - vmu1(i)).^3 + z(5)*(x(1,:) - vmu1(i)).^4);
            %     f = exp(z(1) + z(2) * x(1,:) + z(3) * (x(1,:) - vmu1).^2 + z(4) * (x(1,:) - vmu1).^3 + z(5)*(x(1,:) - vmu1).^4);
            
            %% Plot
            yup3 = (vmu1(i) + 3 * sqrt(vmu2(i)));
            if one_and_a_half
                yup15 = (vmu1(i) + 1.5 * sqrt(vmu2(i)));
                ydown15 = (vmu1(i) - 1.5 * sqrt(vmu2(i)));
            else
                yup1 = (vmu1(i) + 1 * sqrt(vmu2(i)));
                ydown1 = (vmu1(i) - 1 * sqrt(vmu2(i)));
            end
%             yup2 = (vmu1(i) + 2 * sqrt(vmu2(i)));
            ydown3 = (vmu1(i) - 3 * sqrt(vmu2(i)));
%             ydown2 = (vmu1(i) - 2 * sqrt(vmu2(i)));
            bit = 0.5;
            abit = bit * sqrt(vmu2(i));

            less_xs = (ydown3 - abit):delta:(yup3 + abit);
            pdf = f(less_xs) / (delta*sum(f(less_xs)));

            est_vmu1 = delta * sum(less_xs .* pdf);
            est_vmu2 = delta * sum((less_xs - est_vmu1).^2 .* pdf);

            if (abs(est_vmu1 - vmu1(i)) / vmu1(i)) > 0.5 || (abs(est_vmu2 - vmu2(i)) / vmu2(i)) > 0.5
                pdf = zeros(size(pdf));
                disp('Uncorrect fit');
            end
            

            subplot(nrows, ncols, ((2*(i-1)*ncols) + ncols + 4):((2*(i-1)*ncols) + ncols + 8))
            draw_xs = linspace(-3 -bit, 3 + bit, numel(less_xs));
            set(gca,'TickLabelInterpreter','latex')

            plot(draw_xs, pdf);
            hold on;
            
            scatter(0,   f(vmu1(i)),'o', 'filled')
            scatter(3,   f(yup3),'o', 'filled')
            if one_and_a_half
                scatter(1.5,  f(yup15),'o', 'filled')
                scatter(-1.5,f(ydown15), 'o', 'filled')
            else
                scatter(1,  f(yup1),'o', 'filled')
                scatter(-1,f(ydown1), 'o', 'filled')
            end
            scatter(-3, f(ydown3), 'o', 'filled')
            xtickformat('%.1g\\sqrt{\\lambda}')
%             legend("marginal dist.", "v^T\mu_1", "v^T\mu_1 + 3\sigma_{v^Tx|y}", ...
%                 "v^T\mu_1 + 1.5\sigma_{v^Tx|y}", sprintf("v^T\\mu_1 - 1.5\\sqrt{\\lambda_%d}", i),...
%                 sprintf("v^T\\mu_1 - 3\\sqrt{\\lambda_%d}", i), 'NumColumns',2);
            yticks([]);
            axis tight;

            f2 = figure('units','inch','position',[0,0,10,1], 'visible','off');
%             set(gca,'TickLabelInterpreter','latex')

            plot(draw_xs, pdf);
            hold on;
            
%             scatter(0,   f(vmu1(i)),   'kx')
%             scatter(3,   f(yup3),   'k+')
%             scatter(1.5,  f(yup15),  '+', 'Color', [.7 .7 .7])
%             scatter(-1.5,f(ydown15),'p', 'Color', [.7 .7 .7])
%             scatter(1,  f(yup1),  '+', 'Color', [.7 .7 .7])
%             scatter(-1,f(ydown1),'p', 'Color', [.7 .7 .7])
%             scatter(2,  f(yup2),  '+', 'Color', [.7 .7 .7])
%             scatter(-2,f(ydown2),'p', 'Color', [.7 .7 .7])
%             scatter(-3, f(ydown3), 'kp')

            scatter(0,   f(vmu1(i)),'ok', 'filled')
            scatter(3,   f(yup3),'ok', 'filled')
            if one_and_a_half
                scatter(1.5,  f(yup15),'o', 'filled')
                scatter(-1.5, f(ydown15), 'o', 'filled')
            else
                scatter(1,  f(yup1),'o', 'filled')
                scatter(-1, f(ydown1), 'o', 'filled')
            end
            scatter(-3, f(ydown3), 'ok', 'filled')
            set(gca,'TickLabelInterpreter','latex')
%             xtickformat('%.1g\\surd\\lambda')
            xtickformat('%.1g$$\\sqrt{\\lambda}$$')

%             xticks([])
%             xticklabels({"-3\surd{\lambda}","-1.5","0", "1.5", "3"})
%             set(gca,'TickLabelInterpreter','latex')
%             legend("marginal dist.", "v^T\mu_1", "v^T\mu_1 + 3\sigma_{v^Tx|y}", ...
%                 "v^T\mu_1 + 1.5\sigma_{v^Tx|y}", sprintf("v^T\\mu_1 - 1.5\\sqrt{\\lambda_%d}", i),...
%                 sprintf("v^T\\mu_1 - 3\\sqrt{\\lambda_%d}", i), 'NumColumns',2);
            yticks([]);
            axis tight;
            if use_bigger
                exportgraphics(f2, fullfile(outputs, subfolders(subdir_i).name, subsubfolders(subsubdir_i).name, sprintf('prob_ev%d_bigger.pdf', i)), "ContentType", 'vector', "Resolution", 400)
            else
                exportgraphics(f2, fullfile(outputs, subfolders(subdir_i).name, subsubfolders(subsubdir_i).name, sprintf('prob_ev%d.pdf', i)), "ContentType", 'vector', "Resolution", 400)
            end

            close(f2);
        end
        sgtitle(fullfile(subfolders(subdir_i).name, subsubfolders(subsubdir_i).name))
%         shg
%         exportgraphics(gcf, fullfile(outputs, subfolders(subdir_i).name, subsubfolders(subsubdir_i).name, 'prob.png'), "Resolution", 400);
        if use_bigger
            exportgraphics(f1, fullfile(outputs, subfolders(subdir_i).name, sprintf('%s_prob_bigger.png', subsubfolders(subsubdir_i).name)), "Resolution", 400)
        else
            exportgraphics(f1, fullfile(outputs, subfolders(subdir_i).name, sprintf('%s_prob.png', subsubfolders(subsubdir_i).name)), "Resolution", 400)
        end
%         exportgraphics(f1, fullfile(outputs, subfolders(subdir_i).name, sprintf('%s_prob.pdf', subsubfolders(subsubdir_i).name), "ContentType", 'vector', "Resolution", 400)
%         exportgraphics(ax, fullfile(outputs, subfolders(subdir_i).name, subsubfolders(subsubdir_i).name, 'prob.emf'), "ContentType", 'vector', "Resolution", 400)
%         exportgraphics(gcf, fullfile(outputs, subfolders(subdir_i).name, subsubfolders(subsubdir_i).name, 'prob.eps'), "ContentType", 'vector', "Resolution", 400)
%         close all;
%         close(f1);
    end
end


 
%% HEAT_EQUATION_1D  1D heat equation: NumPy solver, MATLAB driver + plots
%
%   dT/dt = alpha * d2T/dx2 + Q(x,t)      on  0 <= x <= L
%
%   The numerics live in heat1d.py (NumPy) and are called through MATLAB's
%   Python interface (py.*). This script sets up the problem, evaluates the
%   initial condition and source in MATLAB, and plots the results.
%
%   Methods : 'explicit' (Forward Euler), 'implicit' (Backward Euler),
%             'cn' (Crank-Nicolson)
%   BCs     : per end, 'dirichlet' (T = value), 'neumann' (dT/dn = value,
%             outward normal; 0 = insulated) or 'robin' (convection; see heat1d.py)
%
%   Usage:
%       heat_equation_1d                         % demo with animation
%       testCase = 'verify'; heat_equation_1d    % compare to exact solution
%       method = 'explicit'; heat_equation_1d    % choose the time integrator
%
%   See validate_battery_thermal.m for validation against NASA battery data.
%   Requires Python with NumPy configured in MATLAB (check with: pyenv).

if ~exist('testCase', 'var'), testCase = 'demo'; end
if ~exist('method',   'var'), method   = 'cn';   end

% ---- Load heat1d.py from this script's folder (reload picks up edits) ----
here = fileparts(mfilename('fullpath'));
if isempty(here), here = pwd; end
if count(py.sys.path, here) == 0
    insert(py.sys.path, int32(0), here);
end
heat1d = py.importlib.reload(py.importlib.import_module('heat1d'));

switch lower(testCase)
    case 'demo'
        %% ---------------- Problem setup (edit me) ----------------
        L      = 0.5;                                % rod length [m]
        N      = 201;                                % grid nodes
        alpha  = 9.7e-5;                             % diffusivity [m^2/s] (aluminum)
        tFinal = 600;                                % simulated time [s]
        dt     = 0.5;                                % [s]; explicit uses its stable limit
        nFrames = 300;                               % snapshots to keep for plotting

        Tamb  = 20;                                  % ambient temperature [C]
        Tinit = @(x) Tamb + 80*exp(-(x - 0.15).^2 / (2*0.02^2));          % hot spot
        Q     = @(x,t) (t < 300) * 0.5 * exp(-(x - 0.35).^2 / (2*0.01^2)); % heater [K/s]

        bcTypes  = {'dirichlet', 'neumann'};         % {left, right}
        bcValues = [Tamb, 0];                        % left held at Tamb, right insulated

        yLim      = [15 105];                        % plot temperature limits [C]
        saveVideo = false;                           % write heat_equation_1d.mp4

        %% ---------------- Solve in NumPy ----------------
        x = linspace(0, L, N);
        tQ = linspace(0, tFinal, 1201);              % source sampled in time, interpolated in Python
        Qtab = zeros(numel(tQ), N);
        for k = 1:numel(tQ)
            Qtab(k, :) = Q(x, tQ(k)) + zeros(1, N);
        end

        saveEvery = max(1, round(tFinal / dt / nFrames));
        tic;
        res = heat1d.solve(L, int32(N), alpha, tFinal, ...
            method       = method, ...
            dt           = dt, ...
            bc_types     = bcTypes, ...
            bc_values    = py.numpy.array(bcValues), ...
            T_init       = py.numpy.array(Tinit(x)), ...
            source       = py.numpy.array(Qtab), ...
            source_times = py.numpy.array(tQ), ...
            save_every   = int32(saveEvery));
        x  = double(res.get('x'));
        t  = double(res.get('t'));
        T  = double(res.get('T'));                   % [nSaved x N]
        fprintf('NumPy solve: method = %s, dt = %.4g s, steps = %d, %.2f s wall\n', ...
            string(res.get('method')), double(res.get('dt')), int64(res.get('n_steps')), toc);

        %% ---------------- Plot ----------------
        iMid   = round((N + 1) / 2);
        probe  = T(:, iMid);
        energy = trapz(x, T, 2);
        animateHeat(x, t, T, probe, energy, yLim, saveVideo);

        fprintf('Done: t = %.1f s, T range [%.2f, %.2f] C, midpoint T = %.2f C\n', ...
            t(end), min(T(end, :)), max(T(end, :)), probe(end));

    case 'verify'
        %% ---------------- Verification vs. exact solution ----------------
        % Unit rod, T = 0 at both ends, T0 = sin(pi x); exact: exp(-pi^2 alpha t) sin(pi x)
        alpha  = 1;
        tFinal = 0.1;
        grids  = [11 21 41 81];                      % h = 1/10 ... 1/80
        methodList = {'explicit', 'implicit', 'cn'};

        fprintf('\n%-10s %6s %10s %12s %8s\n', 'method', 'N', 'dt', 'max error', 'ratio');
        for m = 1:numel(methodList)
            prevErr = NaN;
            for N = grids
                h = 1 / (N - 1);
                switch methodList{m}
                    case 'explicit', dtm = py.None;  % auto: 0.9 * stability limit
                    case 'implicit', dtm = h^2;      % dt ~ h^2 so time error also drops 4x
                    case 'cn',       dtm = 1e-4;
                end
                x = linspace(0, 1, N);
                res = heat1d.solve(1, int32(N), alpha, tFinal, ...
                    method = methodList{m}, dt = dtm, ...
                    bc_types = {'dirichlet', 'dirichlet'}, ...
                    bc_values = py.numpy.array([0 0]), ...
                    T_init = py.numpy.array(sin(pi*x)), ...
                    save_every = int32(1e9));        % only keep first/last
                T = double(res.get('T'));
                Texact = exp(-pi^2 * alpha * tFinal) * sin(pi*x);
                err = max(abs(T(end, :) - Texact));
                fprintf('%-10s %6d %10.2e %12.4e %8.2f\n', methodList{m}, N, ...
                    double(res.get('dt')), err, prevErr/err);
                prevErr = err;
            end
        end
        fprintf('(ratio ~4 per grid halving indicates 2nd-order convergence)\n\n');

    otherwise
        error('Unknown testCase "%s". Use ''demo'' or ''verify''.', testCase);
end


%% ======================================================================
%%                            Local functions
%% ======================================================================

function animateHeat(x, t, T, probe, energy, yLim, saveVideo)
% Animate the profile and build up the space-time map and time histories.
    fig = figure('Name', '1D Heat Equation (NumPy solver)', 'Color', 'w', ...
                 'Position', [100 100 1100 750]);

    % 1) Temperature profile
    ax1 = subplot(2, 2, [1 2]);
    plot(ax1, x, T(1, :), '--', 'Color', [0.6 0.6 0.6]); hold(ax1, 'on');
    hProf = plot(ax1, x, T(1, :), 'r', 'LineWidth', 2); hold(ax1, 'off');
    ylim(ax1, yLim); xlim(ax1, [x(1) x(end)]); grid(ax1, 'on');
    xlabel(ax1, 'x [m]'); ylabel(ax1, 'T [C]');
    legend(ax1, 't = 0', 'current', 'Location', 'best');
    hTitle = title(ax1, 'Temperature profile, t = 0.0 s');

    % 2) Space-time map (filled in as the animation runs)
    ax2 = subplot(2, 2, 3);
    C = nan(numel(x), numel(t));
    hImg = imagesc(ax2, t, x, C, 'AlphaData', ~isnan(C));
    set(ax2, 'YDir', 'normal'); clim(ax2, yLim); colormap(ax2, hot); colorbar(ax2);
    xlim(ax2, [t(1) t(end)]); xlabel(ax2, 't [s]'); ylabel(ax2, 'x [m]');
    title(ax2, 'T(x, t)');

    % 3) Midpoint probe and thermal content
    ax3 = subplot(2, 2, 4);
    yyaxis(ax3, 'left');  hProbe  = plot(ax3, t(1), probe(1), 'LineWidth', 1.5); ylabel(ax3, 'Midpoint T [C]');
    yyaxis(ax3, 'right'); hEnergy = plot(ax3, t(1), energy(1), 'LineWidth', 1.5); ylabel(ax3, '\int T dx');
    xlim(ax3, [t(1) t(end)]); grid(ax3, 'on'); xlabel(ax3, 't [s]');
    title(ax3, 'Midpoint probe and thermal content');

    vid = [];
    if saveVideo
        vid = VideoWriter('heat_equation_1d.mp4', 'MPEG-4');
        open(vid);
    end

    for k = 1:numel(t)
        if ~isvalid(fig), break; end             % user closed the window
        set(hProf, 'YData', T(k, :));
        hTitle.String = sprintf('Temperature profile, t = %.1f s', t(k));
        C(:, k) = T(k, :).';
        set(hImg, 'CData', C, 'AlphaData', ~isnan(C));
        set(hProbe,  'XData', t(1:k), 'YData', probe(1:k));
        set(hEnergy, 'XData', t(1:k), 'YData', energy(1:k));
        drawnow limitrate;
        if ~isempty(vid), writeVideo(vid, getframe(fig)); end
    end

    if ~isempty(vid), close(vid); end
end

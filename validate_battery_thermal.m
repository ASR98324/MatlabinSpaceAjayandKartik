%% VALIDATE_BATTERY_THERMAL  Validate the 1D heat equation on NASA Li-ion data
%
%   Model: radial heat conduction in an 18650 cell (cylinder, radius R)
%
%       dT/dt = alpha * (1/r) d/dr (r dT/dr) + q(t) / (rho*c)
%       dT/dr = 0                          at r = 0  (axis)
%      -k dT/dr = h_eff (T - T_amb)         at r = R  (convection, Robin)
%
%   Heat source (uniform in the cell), Q = beta * q(t) / (rho c V_cell), with
%   q(t) chosen by heatModel:
%     "voltage" (default): q = |I| (U_ocv - V), the irreversible heat. U_ocv
%                is estimated per cycle from the preceding charge curve, so
%                the heat comes from measured voltage and current.
%     "eis":     q = I^2 (Re + Rct), from the most recent impedance test.
%   beta is a fitted scale factor (beta ~ 1 means the energy balance closes).
%
%   Usage:
%       validate_battery_thermal                          % voltage-based heat
%       heatModel = "eis"; validate_battery_thermal       % EIS-based heat
%
%   Data: NASA PCoE Battery Aging set, batteries B0005, B0006, B0007, B0018
%   (2 A constant-current discharges at 24 C ambient). The thermocouple
%   measures the cell surface, which is compared with the model T(r = R).
%
%   Procedure:
%     1. Fit two parameters (h, beta) on every discharge of B0005-B0007.
%     2. Predict every discharge of B0018 with no refitting (blind validation).
%     3. Report errors and save figures to ./results.
%
%   The PDE is solved by heat1d.py (NumPy) through MATLAB's Python interface;
%   all discharge cycles are integrated together in one batched call.

if exist('heatModel', 'var'), hm = string(heatModel); else, hm = "voltage"; end
clearvars -except hm; close all;
heatModel = hm;   % "voltage": q = I (U_ocv - V)   |   "eis": q = I^2 (Re + Rct)

% ---- Paths / Python module ----
here = fileparts(mfilename('fullpath'));
if isempty(here), here = pwd; end
if count(py.sys.path, here) == 0
    insert(py.sys.path, int32(0), here);
end
heat1d = py.importlib.reload(py.importlib.import_module('heat1d'));
% Battery .mat files: data/ (may be a symlink on macOS/Linux) or battery_data/
dataDir = '';
for cand = {fullfile(here, 'data'), fullfile(here, 'battery_data')}
    if isfile(fullfile(cand{1}, 'B0005.mat')), dataDir = cand{1}; break; end
end
if isempty(dataDir)
    error(['Could not find B0005.mat. Put B0005/B0006/B0007/B0018.mat in\n  %s\n' ...
           'or\n  %s'], fullfile(here, 'data'), fullfile(here, 'battery_data'));
end
outDir  = fullfile(here, 'results', heatModel);
if ~exist(outDir, 'dir'), mkdir(outDir); end

% ---- 18650 cell geometry and thermal properties (typical literature values) ----
cellP.R   = 0.009;                       % radius [m]
cellP.Hc  = 0.065;                       % height [m]
cellP.m   = 0.045;                       % mass [kg]
cellP.c   = 1000;                        % specific heat [J/(kg K)]
cellP.k   = 0.20;                        % radial thermal conductivity [W/(m K)]
cellP.V   = pi * cellP.R^2 * cellP.Hc;   % volume [m^3]
cellP.rho = cellP.m / cellP.V;           % density [kg/m^3]
cellP.alpha = cellP.k / (cellP.rho * cellP.c);
cellP.endFactor = 1 + cellP.R / cellP.Hc;  % end caps also cool: h_eff = h * total/lateral area

% ---- Numerics ----
num.N      = 21;                         % radial nodes
num.dt     = 2;                          % time step [s]
num.method = 'cn';

trainNames = ["B0005" "B0006" "B0007"];
testNames  = "B0018";
allNames   = [trainNames testNames];

%% ---------------- Load data ----------------
cycles = [];
for name = allNames
    c = loadDischarges(fullfile(dataDir, name + ".mat"), name);
    fprintf('%s: %3d discharge cycles, R_EIS %.3f -> %.3f Ohm, capacity %.2f -> %.2f Ah\n', ...
        name, numel(c), c(1).R, c(end).R, c(1).capacity, c(end).capacity);
    cycles = [cycles, c]; %#ok<AGROW>
end
isTrain = ismember([cycles.battery], trainNames);
assert(numel(unique([cycles.Tamb])) == 1, 'Expected one ambient temperature');
Tamb = cycles(1).Tamb;

trainBatch = prepBatch(cycles(isTrain),  num.dt, heatModel);
testBatch  = prepBatch(cycles(~isTrain), num.dt, heatModel);

%% ---------------- Fit (h, beta) on B0005-B0007 ----------------
predictFn = @(theta, batch) predictSurface(heat1d, theta, batch, cellP, num, Tamb, -1);
costFn    = @(logTheta) rmseOf(predictFn(exp(logTheta), trainBatch), trainBatch);

theta0 = [10, 1];                        % initial guess: h [W/m^2K], beta [-]
fprintf('\nFitting h and beta on %d training cycles ...\n', sum(isTrain));
tic;
opts = optimset('Display', 'iter', 'TolX', 1e-3, 'TolFun', 1e-4, 'MaxFunEvals', 200);
logTheta = fminsearch(costFn, log(theta0), opts);
theta = exp(logTheta);
fprintf('Fit done in %.0f s [heat model: %s]:  h = %.2f W/(m^2 K)  (h_eff = %.2f),  beta = %.3f\n', ...
    toc, heatModel, theta(1), theta(1)*cellP.endFactor, theta(2));

%% ---------------- Predict all cycles ----------------
predTrain = predictFn(theta, trainBatch);
predTest  = predictFn(theta, testBatch);
pred = cell(1, numel(cycles));
pred(isTrain)  = predTrain;
pred(~isTrain) = predTest;

%% ---------------- Metrics ----------------
fprintf('\n%-7s %-10s %6s %9s %9s %12s %10s\n', 'Battery', 'Role', 'Cycles', 'RMSE[C]', 'MaxErr[C]', 'PeakErr[C]', 'Within1C');
cycRMSE = zeros(1, numel(cycles));
peakMeas = zeros(1, numel(cycles));
peakPred = zeros(1, numel(cycles));
for k = 1:numel(cycles)
    e = pred{k} - cycles(k).T;
    cycRMSE(k)  = sqrt(mean(e.^2));
    peakMeas(k) = max(cycles(k).T);
    peakPred(k) = max(pred{k});
end
summary = table();
for name = allNames
    sel = [cycles.battery] == name;
    e = vertcat(pred{sel}) - vertcat(cycles(sel).T);
    role = "train";
    if ismember(name, testNames), role = "VALIDATE"; end
    row = table(name, role, sum(sel), sqrt(mean(e.^2)), max(abs(e)), ...
        mean(abs(peakPred(sel) - peakMeas(sel))), mean(abs(e) <= 1), ...
        'VariableNames', {'Battery', 'Role', 'Cycles', 'RMSE_C', 'MaxErr_C', 'MeanPeakErr_C', 'FracWithin1C'});
    summary = [summary; row]; %#ok<AGROW>
    fprintf('%-7s %-10s %6d %9.3f %9.3f %12.3f %9.1f%%\n', name, role, sum(sel), ...
        row.RMSE_C, row.MaxErr_C, row.MeanPeakErr_C, 100*row.FracWithin1C);
end
writetable(summary, fullfile(outDir, 'validation_summary.csv'));
save(fullfile(outDir, 'fitted_parameters.mat'), 'theta', 'cellP', 'num', 'summary');

%% ---------------- Figures ----------------
colors = lines(numel(allNames));

% 1) B0018 example cycles: measured surface vs predicted surface and core
testIdx = find(~isTrain);
pick = testIdx(unique(round(linspace(1, numel(testIdx), 3))));
exBatch = prepBatch(cycles(pick), num.dt, heatModel);
coreEx  = predictSurface(heat1d, theta, exBatch, cellP, num, Tamb, 0);
f1 = figure('Color', 'w', 'Position', [50 50 1400 420]);
tiledlayout(f1, 1, numel(pick), 'TileSpacing', 'compact');
for j = 1:numel(pick)
    k = pick(j);
    nexttile;
    plot(cycles(k).t/60, cycles(k).T, 'k.', 'MarkerSize', 8); hold on;
    plot(cycles(k).t/60, pred{k}, 'r-', 'LineWidth', 2);
    plot(cycles(k).t/60, coreEx{j}, 'r--', 'LineWidth', 1.2); hold off;
    grid on; xlabel('time [min]'); ylabel('T [C]');
    title(sprintf('%s discharge #%d (%.2f Ah), RMSE %.2f C', cycles(k).battery, cycles(k).index, ...
        cycles(k).capacity, cycRMSE(k)));
    if j == 1, legend('measured surface', 'model surface', 'model core', 'Location', 'southeast'); end
end
sgtitle(sprintf('Blind validation on B0018 (fitted on B0005-B0007 only), heat model: %s', heatModel));
exportgraphics(f1, fullfile(outDir, 'B0018_example_cycles.png'), 'Resolution', 150);

% 2) Peak temperature vs cycle: measured vs predicted
f2 = figure('Color', 'w', 'Position', [50 50 1000 500]); hold on;
for b = 1:numel(allNames)
    sel = [cycles.battery] == allNames(b);
    idx = [cycles(sel).index];
    plot(idx, peakMeas(sel), '.', 'Color', colors(b, :), 'MarkerSize', 10, 'HandleVisibility', 'off');
    plot(idx, peakPred(sel), '-', 'Color', colors(b, :), 'LineWidth', 1.8, 'DisplayName', allNames(b));
end
hold off; grid on; xlabel('discharge cycle'); ylabel('peak surface T [C]');
title('Peak temperature per discharge: dots = measured, lines = model');
legend('Location', 'best');
exportgraphics(f2, fullfile(outDir, 'peak_temperature_vs_cycle.png'), 'Resolution', 150);

% 3) RMSE per cycle
f3 = figure('Color', 'w', 'Position', [50 50 1000 450]); hold on;
for b = 1:numel(allNames)
    sel = [cycles.battery] == allNames(b);
    lbl = allNames(b);
    if ismember(allNames(b), testNames), lbl = lbl + " (validation)"; end
    plot([cycles(sel).index], cycRMSE(sel), 'o-', 'Color', colors(b, :), 'MarkerSize', 3, 'DisplayName', lbl);
end
hold off; grid on; xlabel('discharge cycle'); ylabel('RMSE [C]');
title('Per-cycle RMSE of surface temperature');
legend('Location', 'best');
exportgraphics(f3, fullfile(outDir, 'rmse_per_cycle.png'), 'Resolution', 150);

% 4) Parity plot
f4 = figure('Color', 'w', 'Position', [50 50 600 560]);
mTr = vertcat(cycles(isTrain).T);  pTr = vertcat(pred{isTrain});
mTe = vertcat(cycles(~isTrain).T); pTe = vertcat(pred{~isTrain});
plot(mTr, pTr, '.', 'Color', [0.75 0.75 0.75], 'MarkerSize', 3); hold on;
plot(mTe, pTe, '.', 'Color', colors(end, :), 'MarkerSize', 4);
lim = [floor(min([mTr; mTe])) ceil(max([mTr; mTe]))];
plot(lim, lim, 'k-', lim, lim + 1, 'k:', lim, lim - 1, 'k:'); hold off;
axis equal; xlim(lim); ylim(lim); grid on;
xlabel('measured surface T [C]'); ylabel('model surface T [C]');
legend('train (B0005-7)', 'validation (B0018)', 'y = x', '\pm1 C', 'Location', 'northwest');
title('Model vs measurement');
exportgraphics(f4, fullfile(outDir, 'parity.png'), 'Resolution', 150);

fprintf('\nFigures and summary saved to %s\n', outDir);


%% ======================================================================
%%                            Local functions
%% ======================================================================

function cyc = loadDischarges(file, name)
% Extract discharge cycles with both heat-source estimates:
%   qEIS  = I^2 (Re + Rct), using the latest EIS test
%   qVolt = I (U_ocv - V), with U_ocv from the preceding charge (see voltageHeat)
    S = load(file);
    c = S.(char(name)).cycle;
    types = string({c.type});
    imp = find(types == "impedance");
    Rof = @(k) real(c(k).data.Re) + real(c(k).data.Rct);
    Rlast = Rof(imp(1));                 % before the first EIS, use the first one

    % Charge cycles and the charge they delivered. Some are partial top-ups
    % (or have corrupt voltage), which would ruin the U_ocv estimate.
    chg = find(types == "charge");
    Qc = arrayfun(@(k) trapz(c(k).data.Time, max(c(k).data.Current_measured, 0)) / 3600, chg);
    Vmax = arrayfun(@(k) max(c(k).data.Voltage_measured), chg);

    out = {};
    nDis = 0;
    for k = 1:numel(c)
        switch types(k)
            case "impedance"
                Rk = Rof(k);
                if isfinite(Rk) && Rk > 0, Rlast = Rk; end
            case "discharge"
                nDis = nDis + 1;
                d = c(k).data;
                [t, iu] = unique(d.Time(:));
                if numel(t) < 20 || t(end) < 600, continue; end
                % Nearest full charge (same charge as this discharge within -15/+20 %),
                % preferring the most recent one before the discharge
                full = chg(Qc > 0.85*d.Capacity & Qc < 1.2*d.Capacity & Vmax < 4.25);
                kc = full(find(full < k, 1, 'last'));
                if isempty(kc), kc = full(find(full > k, 1, 'first')); end
                if isempty(kc), continue; end
                I = d.Current_measured(iu); I = I(:);
                V = d.Voltage_measured(iu); V = V(:);
                T = d.Temperature_measured(iu); T = T(:);
                [qVolt, U] = voltageHeat(t, I, V, c(kc).data);
                out{end+1} = struct('battery', name, 'index', nDis, 't', t, ...
                    'I', I, 'V', V, 'T', T, 'U', U, 'capacity', d.Capacity, ...
                    'Tamb', c(k).ambient_temperature, 'R', Rlast, ...
                    'qEIS', I.^2 * Rlast, 'qVolt', qVolt); %#ok<AGROW>
        end
    end
    cyc = [out{:}];
end


function [q, U] = voltageHeat(t, I, V, chg)
% Irreversible heat q = |I| (U_ocv - V) for a discharge.
% U_ocv(SOC) is estimated from the preceding charge: at equal SOC, with a
% linear resistance, V_ch = U + I_ch R and V_dis = U - I_dis R, so
%   U = (I_dis V_ch + I_ch V_dis) / (I_ch + I_dis).
% SOC is normalized by coulomb counting over each half cycle (0 = discharge
% cutoff state, 1 = end of the CV charge).
    Id = abs(I);
    active = Id > 0.5;                                   % load connected
    Qd = cumtrapz(t, Id .* active);
    zd = 1 - Qd / Qd(end);

    [tc, iu] = unique(chg.Time(:));
    Ic = max(chg.Current_measured(iu), 0); Ic = Ic(:);
    Vc = chg.Voltage_measured(iu); Vc = Vc(:);
    Qc = cumtrapz(tc, Ic);
    zc = Qc / Qc(end);
    [zc, ku] = unique(zc);
    VcZ = interp1(zc, Vc(ku), zd, 'linear', 'extrap');
    IcZ = interp1(zc, Ic(ku), zd, 'linear', 'extrap');
    IcZ = max(IcZ, 0);

    U = (Id .* VcZ + IcZ .* V) ./ max(Id + IcZ, eps);
    q = Id .* (U - V);
    q(~active | ~isfinite(q)) = 0;
    q = max(q, 0);
end


function batch = prepBatch(cyc, dt, heatModel)
% Put all cycles on a common time grid for one batched PDE solve.
    tMax = max(arrayfun(@(c) c.t(end), cyc));
    tq = (0:dt:ceil(tMax/dt)*dt).';
    B = numel(cyc);
    qW = zeros(numel(tq), B);
    for b = 1:B
        qField = cyc(b).qVolt;
        if heatModel == "eis", qField = cyc(b).qEIS; end
        qW(:, b) = interp1(cyc(b).t, qField, tq, 'linear', 0);   % no heat after the log ends
    end
    batch.tq   = tq;
    batch.qW   = qW;
    batch.T0   = arrayfun(@(c) c.T(1), cyc);
    batch.tObs = {cyc.t};
    batch.TObs = {cyc.T};
end


function pred = predictSurface(heat1d, theta, batch, cellP, num, Tamb, probe)
% Solve the radial heat equation for every cycle in the batch.
% probe = -1 -> surface (r = R), 0 -> core (r = 0). Returns a cell of
% predictions at each cycle's measurement times.
    h = theta(1); beta = theta(2);
    hEff = h * cellP.endFactor;
    Q = beta * batch.qW / (cellP.rho * cellP.c * cellP.V);              % [K/s], n_t x B
    T0 = repmat(batch.T0(:), 1, num.N);                                  % B x N
    res = heat1d.solve(cellP.R, int32(num.N), cellP.alpha, batch.tq(end), ...
        method         = num.method, ...
        dt             = num.dt, ...
        geometry       = 'cylindrical', ...
        bc_types       = {'neumann', 'robin'}, ...
        bc_values      = py.numpy.array([0 Tamb]), ...
        bc_coeffs      = py.numpy.array([0 hEff/cellP.k]), ...
        T_init         = py.numpy.array(T0), ...
        source_uniform = py.numpy.array(Q), ...
        source_times   = py.numpy.array(batch.tq.'), ...
        probe_index    = int32(probe), ...
        store_field    = false);
    ts = double(res.get('t'));
    P  = double(res.get('probe'));                                       % n_saved x B
    pred = cell(1, numel(batch.tObs));
    for b = 1:numel(batch.tObs)
        pred{b} = interp1(ts(:), P(:, b), batch.tObs{b});
    end
end


function r = rmseOf(pred, batch)
    e = vertcat(pred{:}) - vertcat(batch.TObs{:});
    r = sqrt(mean(e.^2));
end

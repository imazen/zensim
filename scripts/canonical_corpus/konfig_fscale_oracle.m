function oracle_fscale(csvfile, x0file, outfile)
% KONFIGF oracle driver: reimplements main_reconstruction.m /
% reconstruct_all1.m's grouping mechanics (findgroups/splitapply/table are
% post-R2013 conveniences absent from Octave core) while calling the AUTHORS'
% UNMODIFIED utils/countvote.m, utils/getres.m, utils/compute_baselinetc.m
% and a real fmincon (optim pkg). Input CSV keeps data1's 13-column schema
% (produce it via konfig_fscale.py emit-filtered; x0 via emit-x0).
%
% Invocation used for the 2026-10-07 oracle run (image = gnuoctave/octave
% @sha256:185db7993e000d4f3f6e7bbbf7fb3f999f52e799ea52231ad8a15353381e0dcb
% + `pkg install -forge struct datatypes statistics optim`, committed locally
% as konfigf-octave:optim-11.3.0; konfig_fscale_optimoptions.m alongside):
%   docker run --rm -v /mnt/v/dataset/konfig-iqa/KonFiG-IQA:/w:ro \
%     -v <workdir>:/o konfigf-octave:optim-11.3.0 octave --no-gui --eval \
%     "addpath('<dir of this file>'); oracle_fscale('/o/data1_F_trainval.csv','/o/x0.txt','/o/oracle_out.tsv')"
  addpath('/w/utils');
  pkg load optim;
  pkg load statistics;

  fid = fopen(csvfile, 'r');
  C = textscan(fid, '%s %s %s %s %s %s %s %s %s %s %s %s %s', ...
               'Delimiter', ',', 'HeaderLines', 1, 'ReturnOnError', 0);
  fclose(fid);
  % group key = source|distortion  (all rows are BoostType 'F' already)
  keys = strcat(C{1}, '|', C{2});
  [ukeys, ~, gnum] = unique(keys);
  x0 = dlmread(x0file);
  fo = fopen(outfile, 'w');
  fprintf(fo, 'seq\tcsv_level\tvalue\n');
  for g = 1:numel(ukeys)
    idx = (gnum == g);
    data = cell(sum(idx), 13);
    for cix = 1:13
      data(:, cix) = C{cix}(idx);
    end
    % reconstruct_all1: findgroups on the 'triplet' column (13)
    [ut, ~, tg] = unique(data(:, 13));
    subvotes = cell(numel(ut), 4);
    for t = 1:numel(ut)
      subvotes(t, :) = countvote(data(tg == t, :), 'F');
    end
    items = cell2mat([subvotes(:, 1); subvotes(:, 2)]);
    unitem = unique(items);
    allres = [];
    for i = 1:size(subvotes, 1)
      it_l = subvotes{i, 1}; it_r = subvotes{i, 2};
      v_l = find(unitem == it_l); v_r = find(unitem == it_r);
      allres = [allres; v_l, v_r, subvotes{i, 3}, subvotes{i, 4}];
    end
    [rr1, rr2] = getres(allres);
    f1 = @(x) compute_baselinetc(x, rr1, rr2);
    options = optimoptions('fmincon', 'UseParallel', true, ...
                           'DiffMinChange', 0.00001);
    x = fmincon(f1, x0, [], [], [], [], [], [], [], options);
    x = sort(abs(x));
    x = fmincon(f1, x, [], [], [], [], [], [], [], options);
    x = (x - x(1)) ./ 0.6745;
    for k = 1:numel(x)
      fprintf(fo, '%s\t%d\t%.17g\n', ukeys{g}, unitem(k), x(k));
    end
    fflush(fo);
  end
  fclose(fo);
end

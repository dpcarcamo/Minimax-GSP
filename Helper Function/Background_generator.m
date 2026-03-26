currentPath = pwd;
folders = split(currentPath, '\');

% Add Helper Function to Path
newPath = join(folders(1:length(folders)-1),"\");
addpath(strcat(newPath{1}, '\Helper Function'))


n = 100;
e = 40;

s = randi(n, e, 1);
t = randi(n, e, 1);
coords = randn(n, 2);
G = graph(J);
% G = graph(true(n)); % Self-loops are possible
%G = graph(true(n), 'omitselfloops'); % Alternative without self-loops
% p = randperm(numedges(G), e);
% G = graph(G.Edges(p, :));

%% david code

n = 80;
[J, h] = RandomGSPIsing(n);
G = graph(J);
graph_plot = plot(G);

coords = [graph_plot.XData; graph_plot.YData; graph_plot.ZData]';

%
cmap = parula(n+30);
edges = table2array(G.Edges);
edges = edges(:, 1:2);

% coords = coords.*[1 1.4 1];
old_coords = coords;

figure(1), clf, hold on
scatter(coords(:, 1), coords(:, 2), 80, [cmap(1:n, :)], 'filled', 'markerfacealpha', 0.65)
scatter(old_coords(:, 1), old_coords(:, 2), 80, [cmap(1:n, :)], 'filled', 'markerfacealpha', 0.65)

for jj = [1:size(edges)]
% for jj = 1
    vec = (coords(edges(jj, 1), :) - coords(edges(jj, 2), :)).*0;
    plot(coords(edges(jj, :), 1)-[1 -1].*vec(1), coords(edges(jj, :), 2)+[1 -1].*vec(2), 'color', [cmap(floor(jj/2.5)+10, :) 0.4], 'linewidth', 1.5)
end

old_edges = edges;
for jj = [1:size(old_edges)]
      
    vec = (old_coords(old_edges(jj, 1), :) - old_coords(old_edges(jj, 2), :)).*0;
    plot(old_coords(old_edges(jj, :), 1)-[1 -1].*vec(1), old_coords(old_edges(jj, :), 2)+[1 -1].*vec(2), 'color', [cmap(floor(jj/2.5)+10, :) 0.4], 'linewidth', 1.5)
end

ylim([-6, 6])
xlim([-3, 3.5])
% daspect([1 2 1])
set(gca, 'visible', 'off')
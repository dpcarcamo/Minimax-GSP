%% Triplets for tree
clear
clc

currentPath = pwd;
folders = split(currentPath, '\');
newPath = join(folders(1:length(folders)-2),"\");

addpath(strcat(newPath{1} , '\Model Data'))




% Add Helper Function to Path
newPath = join(folders(1:length(folders)-1),"\");
addpath(strcat(newPath{1}, '\Helper Function'))
load("natimg2800_M170717_MP033_2017-08-20resp.mat")
JGSP = full(JGSP);

%% Triplets

%load("Trip.mat")
%load("Pairs.mat")


%[m, C, X, Z] = correlations_GSP_01(JGSP, hGSP);
[m, C, X, Z, Trip, Jcon, pairs] = correlations_GSP_02(JTree, hTree);

%
timeseries2 = zeros(2*num_nuerons-3,num_bins);
for i = 1:length(pairs)

    timeseries2(i,:) = (binary(:,pairs(i,1)).*binary(:,pairs(i,2)));

end

trueTrip = timeseries2*binary/num_bins  ;


binary_M = binary.';
tic
[G,D] = decimate_GSP(JTree);
toc

Data = zeros(5,100);
count = 1;

bJ = JTree ~= 0;
bJ = double(bJ);
metric = distances(graph(bJ));
metric(metric == 0) = nan;
metric(metric == Inf) = nan;





colors = zeros(length(pairs),num_nuerons);
count = 1;

for i = 1:length(pairs)

    pair = pairs(i,:);
        
    

    colors(count,:) =   (metric(pair(1),:)==1)+(metric(pair(2),:)==1);
    count = count + 1

end

ignore = zeros(size(Trip));
for i = 1:length(pairs)

    j = pairs(i,1);
    k = pairs(i,2);

    ignore(i, j) = 1;
    ignore(i, k) = 1;

end

Trip(ignore==1)=nan;
trueTrip(ignore==1)=nan;
sz = sum(sum(Trip +ignore ==0))

count = 1;

Tripmoved = Trip;
trueTripedmoved = trueTrip;

for i = 1:length(pairs)

    pair = pairs(i,:);
        
    

    Tripmoved(i,:) = Trip(i,:) - datacorr_pseudo(pair(1),pair(2)) * datamean - datacorr_pseudo(pair(2),:).* datamean(pair(1)) - datacorr_pseudo(pair(1),:).* datamean(pair(2)) + 2* datamean(pair(1))*datamean(pair(2))*datamean;
    trueTripedmoved(i,:) = trueTrip(i,:) - C(pair(1),pair(2)) * m.' - C(pair(2),:).* m(pair(1)) - C(pair(1),:).* m(pair(2)) + 2* m(pair(1))*m(pair(2))*m.';
    
    count = count + 1

end
%%
figure
plot(trueTripedmoved(colors(:)==0),Tripmoved(colors(:)==0), '.',MarkerSize=10, Color = [0 0.4470 0.7410])
hold on
plot(trueTripedmoved(colors(:)==1),Tripmoved(colors(:)==1), '.',MarkerSize=10, Color = [0.8500 0.3250 0.0980])

plot(trueTripedmoved(colors(:)==2),Tripmoved(colors(:)==2), '.',MarkerSize=10, Color = [0.9290 0.6940 0.1250])

plot([min(trueTripedmoved(:)),max(trueTripedmoved(:))], [min(trueTripedmoved(:)),max(trueTripedmoved(:))], '--',Color='k')
hold off
xlabel('[x_ix_jx_k] Data')
ylabel('[x_ix_jx_k] Model')

%%

figure

plotvariance(trueTrip(colors(:)==2),Trip(colors(:)==2),8,[0 0.4470 0.7410])
hold on
plotvariance(trueTripedmoved(colors(:)==1),Tripmoved(colors(:)==1), 6, [0.8500 0.3250 0.0980])

plotvariance(trueTripedmoved(colors(:)==0),Tripmoved(colors(:)==0), 8, [0.9290 0.6940 0.1250])
plot([min(trueTripedmoved(:)),0.03], [min(trueTripedmoved(:)),0.03], '--',Color='k')
xlim([min(trueTripedmoved(colors(:)==2)),0.03])
ylim([min(trueTripedmoved(colors(:)==2)),0.03])

hold off
xlabel('[x_ix_jx_k] Data')
ylabel('[x_ix_jx_k] Model')
axis square
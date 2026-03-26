%%
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

%% Two point model accuracy random

[m, C, X, Z] = correlations_GSP_01(JGSPrand, hGSPrand);
%

corr_coef = (datacorr - datamean.*datamean.')./sqrt((datamean-datamean.^2).*(datamean-datamean.^2).');

model_coef = X./sqrt((m-m.^2).*(m-m.^2).');
bJ = JGSPrand ~= 0;
bJ = double(bJ);
metric = distances(graph(bJ));
metric(metric == 0) = nan;
metric(metric == Inf) = nan;

%%
plot([min(corr_coef(JGSPrand ~= 0 )),max(corr_coef(JGSPrand ~= 0 ))],[min(model_coef(JGSPrand ~= 0)),max(model_coef(JGSPrand ~= 0))], '-', 'Color',[8,69,148]/256, 'LineWidth',2)
hold on
plotvariance(corr_coef(triu(metric==  2)), model_coef(triu(metric == 2)), 11 , [66,169,221]/256)
plotvariance(corr_coef(triu(metric ==  3)), model_coef(triu(metric == 3)), 5, [71,207,210]/2010)
plot(linspace(-0.1,0.9),linspace(-0.1,0.9),'--', Color='black')
hold off
xlabel('Data Corr Coef', 'FontSize',16)
ylabel('Model Corr Coef', 'FontSize',16)
legend('Top. dis. 1','','Top. dis. 2','','Top. dis. 3','Location','northwest', 'FontSize',14)
axis square
%%

%%
[m, C, X, Z, Trip, Jcon, pairs] = correlations_GSP_02(JGSPrand, hGSPrand);

[m, C, X, Z, Trip2, Jcon, pairs] = correlations_GSP_02(JGSPrand, hGSPrand);

Trip = Trip + Trip2.*(Trip == 0);

timeseries2 = zeros(2*num_nuerons-3,num_bins);
for i = 1:length(pairs)

    timeseries2(i,:) = (binary(:,pairs(i,1)).*binary(:,pairs(i,2)));

end

trueTrip = timeseries2*binary/num_bins  ;


binary_M = binary.';
tic
[G,D] = decimate_GSP(JGSPrand);
toc

Data = zeros(5,100);
count = 1;

bJ = JGSPrand ~= 0;
bJ = double(bJ);
metric = distances(graph(bJ));
metric(metric == 0) = nan;
metric(metric == Inf) = nan;





colors = zeros(2*num_nuerons-3,num_nuerons);
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
plotvariance(trueTrip(colors(:)==2),Trip(colors(:)==2),8,[0 0.4470 0.7410])
hold on
plotvariance(trueTripedmoved(colors(:)==1),Tripmoved(colors(:)==1), 6, [0.8500 0.3250 0.0980])

plotvariance(trueTripedmoved(colors(:)==0),Tripmoved(colors(:)==0), 8, [0.9290 0.6940 0.1250])
plot([min(trueTripedmoved(:)),0.03], [min(trueTripedmoved(:)),0.03], '--',Color='k')
%%

rspairsfiles = ["natimg2800_M160825_MP027_2016-12-14",
"natimg2800_M161025_MP030_2017-05-29",
"natimg2800_M170714_MP032_2017-08-07",
"natimg2800_8D_M170604_MP031_2017-07-02",
"natimg2800_8D_M170717_MP033_2017-08-22",
"natimg2800_small_M170717_MP033_2017-08-23",
"natimg32_M150824_MP019_2016-03-23",
"natimg32_M170604_MP031_2017-06-27",
"natimg32_M170714_MP032_2017-08-01",
"natimg32_M170717_MP033_2017-08-25", 
"ori32_M170604_MP031_2017-06-26",
"ori32_M170714_MP032_2017-08-02",
"ori32_M170717_MP033_2017-08-17"];


currentPath = pwd;
folders = split(currentPath, '\');
newPath = join(folders(1:length(folders)-2),"\");

listing = dir(strcat(newPath{1} , '\Stringer Data\Data'));
addpath(strcat(newPath{1} , '\Stringer Data\Data'))

% Add Helper Function to Path
newPath = join(folders(1:length(folders)-1),"\");
addpath(strcat(newPath{1}, '\Helper Function'))

data = zeros(10,2);
count = 1;
for filename = rspairsfiles.'

    disp(filename)
    disp(char(table2array(T(m,5))))
    
    ExpData = matfile(filename);
    
    
    %
    Stim = ExpData.stim;
    Response = Stim.resp;
    Spont = Stim.spont;
    Locations = ExpData.med;
    dbinfo = ExpData.db;
    mouse_name = dbinfo.mouse_name;


    

    offneuron = find(mean(Response)==0);
    Response(:,offneuron) = [];
    Locations(offneuron,:) = [];
    Spont(:,offneuron) = [];
    num_timepoints = min(size(Spont,1),size(Response,1));


    
    
    [normResponse, mu, sigma ]= zscore(Response(1:num_timepoints,:),1,1);
    [spontnormResponse, spontmu, spontsigma ]= zscore(Spont(1:num_timepoints,:),1,1);
    num_nuerons = size(Response,2);

    
    
    % Binarizing Data
    binaryR = (Response - mu)> 2*sigma;
    binaryS = (Spont - spontmu)> 2*spontsigma;
    %imshow(binary)
    
    
    binaryR = double(binaryR);
    
    % Spuedo Data
    binaryR(end+1,:) = ones(1,size(binaryR,2))/2;
    
    
    % Filter Data
    spiking_patternsR = binaryR.';
    num_binsR = size(spiking_patternsR,2);
    num_nueronsR = size(spiking_patternsR,1);
    
    datacorrR = spiking_patternsR*spiking_patternsR.'/num_binsR;
    datameanR = mean(spiking_patternsR.');
    
    datacorr_pseudoR = datacorrR;

    binaryS = double(binaryS);
    
    % Spuedo Data
    binaryS(end+1,:) = ones(1,size(binaryS,2))/2;
    
    
    % Filter Data
    spiking_patternsS = binaryS.';
    num_binsS = size(spiking_patternsS,2);
    num_nueronsS = size(spiking_patternsS,1);
    
    datacorrS = spiking_patternsS*spiking_patternsS.'/num_binsS;
    datameanS = mean(spiking_patternsS.');
    
    datacorr_pseudoS = datacorrS;
    
    topo = RandomGSPIsing(num_nueronsR) ~=0;
    
    [JGSPR, hGSPR, HR] = GSP_fit_topology(datameanR,datacorr_pseudoR, topo);
    [JGSPS, hGSPS, HS] = GSP_fit_topology(datameanS,datacorr_pseudoS, topo);

    data(count,1) = (sum(Entropy(datameanR)) - HR)/num_nueronsS;
    data(count,2) = (sum(Entropy(datameanS)) - HS)/num_nuerons;
    count = count + 1;

end


%%
figure

plot(data(1:end-3,1),data(1:end-3,2),'r.', 'MarkerSize',20)

hold on

plot(data(end-2:end,1),data(end-2:end,2), 'b.', 'MarkerSize',20)
plot(linspace(min(data,[], 'all'),max(0.038,[],'all')),linspace(min(data,[], 'all'),max(0.038,[],'all')),'--','Color','k')

xlim([min(data,[], 'all') - 0.0002,0.004])
ylim([min(data,[], 'all') - 0.0002,0.004])

xlabel('Nat Img Info per neuron')
ylabel('Spontaneous Info per neuron')
title('Info per neuron')
axis square


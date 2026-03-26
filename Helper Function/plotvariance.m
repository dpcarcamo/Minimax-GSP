function plotvariance(xdata, ydata, numBins, color)
    
    % Define the number of bins and bin edges


    % numinBins = floor(length(xdata)/numBins);
    % xdatasorted = sort(xdata);
    % binEdges = zeros(1,numBins+1);
    % 
    % binEdges(1) = min(xdatasorted);
    % for i = 2:numBins
    %     binEdges(i) = xdatasorted(numinBins*i);
    % end
    % binEdges(end) = max(xdatasorted);

    binEdges = linspace(min(xdata),max(xdata), numBins + 1);

    
    % Calculate the bin centers
    binCenters = (binEdges(1:end-1) + binEdges(2:end)) / 2;
    
    % Use histcounts to bin the data
    [counts, edges, binIdx] = histcounts(xdata, binEdges);
    
    % Initialize arrays to store the mean and variance of each bin
    binMeans = zeros(1, numBins);
    binVariances = zeros(1, numBins);
    
    % Calculate mean and variance for each bin
    for i = 1:numBins
        dataInBin = ydata(binIdx == i);
        binMeans(i) = mean(dataInBin,"omitnan");
        binVariances(i) = std(dataInBin, "omitmissing");
    end
    
    % Create a bar plot to visualize the bin means and variances
    curve1 = binMeans + binVariances;
    curve2 = binMeans - binVariances;
    x2 = [binCenters, fliplr(binCenters)];
    inBetween = [curve1, fliplr(curve2)];
    fill(x2, inBetween, color, FaceAlpha=0.1, EdgeAlpha=0);
    hold on;
    plot(binCenters, binMeans, 'color',color, 'LineWidth', 2);
    
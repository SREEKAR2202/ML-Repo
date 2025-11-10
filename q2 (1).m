% GMM Model Order Selection via Cross-Validation
% Clear workspace and close figures
clear all;
close all;
clc;

%% Configuration Parameters
sampleSizes = [10, 100, 1000];      % Dataset sizes to evaluate
numExperiments = 100;                % Number of repetitions
maxComponents = 10;                  % Maximum GMM components to test
convergenceTol = 1e-3;               % EM convergence threshold
regularization = 1e-6;               % Regularization for covariance stability
numFolds = 10;                       % K-fold cross-validation
dimensionality = 2;                  % Data dimensionality

%% Define True GMM Parameters
% Mixing coefficients
trueMixingCoeffs = [0.2, 0.3, 0.4, 0.1];

% Mean vectors for 4 components (2D) - with overlap
trueMeanVectors = [-10, 10, 10, -10;
                    10, 10, -10, -10];

% Covariance matrices (SYMMETRIC - with significant overlap between components)
trueCovMatrices = zeros(2, 2, 4);
trueCovMatrices(:,:,1) = [20, 1; 1, 3];    % Made symmetric
trueCovMatrices(:,:,2) = [7, 1; 1, 2];     % Already symmetric
trueCovMatrices(:,:,3) = [4, 1; 1, 16];    % Made symmetric
trueCovMatrices(:,:,4) = [2, 1; 1, 7];     % Already symmetric

%% Initialize Storage for Results
modelSelectionFrequency = zeros(length(sampleSizes), maxComponents);
allLogLikelihoods = cell(length(sampleSizes), 1);

%% Main Experimental Loop
fprintf('Starting GMM Model Selection Experiment...\n');
fprintf('Running %d trials for each dataset size\n\n', numExperiments);

for experiment = 1:numExperiments
    if mod(experiment, 10) == 0
        fprintf('Completed %d/%d experiments\n', experiment, numExperiments);
    end
    
    % Test each dataset size
    for sizeIndex = 1:length(sampleSizes)
        currentSize = sampleSizes(sizeIndex);
        
        % Generate synthetic data from true GMM
        syntheticData = generateGMMData(currentSize, trueMixingCoeffs, ...
                                        trueMeanVectors, trueCovMatrices);
        
        % Perform K-fold cross-validation for model selection
        [selectedOrder, avgLogLikelihoods] = crossValidateGMM(syntheticData, ...
                                             currentSize, maxComponents, numFolds, ...
                                             dimensionality, convergenceTol, regularization);
        
        % Record selection
        modelSelectionFrequency(sizeIndex, selectedOrder) = ...
            modelSelectionFrequency(sizeIndex, selectedOrder) + 1;
        
        % Store log-likelihoods
        allLogLikelihoods{sizeIndex} = [allLogLikelihoods{sizeIndex}; avgLogLikelihoods];
    end
end

%% Calculate Selection Rates
selectionRatesPercent = (modelSelectionFrequency / numExperiments) * 100;

%% Display Results
fprintf('\n========================================\n');
fprintf('EXPERIMENT RESULTS SUMMARY\n');
fprintf('========================================\n\n');

for sizeIndex = 1:length(sampleSizes)
    fprintf('Dataset Size: N = %d samples\n', sampleSizes(sizeIndex));
    fprintf('----------------------------------------\n');
    fprintf('Model Order | Selection Rate | Count\n');
    fprintf('----------------------------------------\n');
    
    for order = 1:maxComponents
        fprintf('   M = %2d   |   %6.2f%%     | %3d/%d\n', ...
                order, selectionRatesPercent(sizeIndex, order), ...
                modelSelectionFrequency(sizeIndex, order), numExperiments);
    end
    fprintf('\n');
end

%% Visualization 1: GMM Data Distribution for each sample size
figure('Position', [100, 100, 1400, 400]);
for sizeIndex = 1:length(sampleSizes)
    subplot(1, 3, sizeIndex);
    
    % Generate sample data
    sampleData = generateGMMData(sampleSizes(sizeIndex), trueMixingCoeffs, ...
                                 trueMeanVectors, trueCovMatrices);
    
    % Plot data points
    scatter(sampleData(1,:), sampleData(2,:), 50, 'o', ...
            'MarkerFaceColor', [0.2, 0.4, 0.8], ...
            'MarkerEdgeColor', 'k', ...
            'MarkerFaceAlpha', 0.6, ...
            'LineWidth', 1);
    
    xlabel('Dimension 1', 'FontSize', 12, 'FontWeight', 'bold');
    ylabel('Dimension 2', 'FontSize', 12, 'FontWeight', 'bold');
    title(sprintf('GMM Sample Distribution (N=%d)', sampleSizes(sizeIndex)), ...
          'FontSize', 13, 'FontWeight', 'bold');
    grid on;
    set(gca, 'FontSize', 11);
    axis equal;
    xlim([-30, 30]);
    ylim([-30, 30]);
end
sgtitle('True GMM Data Distribution', 'FontSize', 15, 'FontWeight', 'bold');

%% Visualization 2: Order Log-Likelihood for each dataset size
figure('Position', [100, 100, 1400, 400]);
colors = {[0.85, 0.2, 0.4], [0.2, 0.6, 0.3], [0.3, 0.4, 0.8]};

for sizeIndex = 1:length(sampleSizes)
    subplot(1, 3, sizeIndex);
    
    % Compute average log-likelihoods across all experiments
    avgLogLiks = mean(allLogLikelihoods{sizeIndex}, 1);
    
    % Plot
    plot(1:maxComponents, avgLogLiks, 'o-', ...
         'LineWidth', 2.5, ...
         'MarkerSize', 8, ...
         'MarkerFaceColor', colors{sizeIndex}, ...
         'Color', colors{sizeIndex});
    
    xlabel('Model Order (Number of Components)', 'FontSize', 12, 'FontWeight', 'bold');
    ylabel('Average Log-Likelihood', 'FontSize', 12, 'FontWeight', 'bold');
    title(sprintf('Order Log-Likelihood (N=%d)', sampleSizes(sizeIndex)), ...
          'FontSize', 13, 'FontWeight', 'bold');
    grid on;
    set(gca, 'FontSize', 11);
    xlim([0.5, maxComponents+0.5]);
    
    % Mark the most frequently selected order
    [~, mostSelected] = max(selectionRatesPercent(sizeIndex, :));
    hold on;
    plot(mostSelected, avgLogLiks(mostSelected), 'p', ...
         'MarkerSize', 15, 'MarkerFaceColor', 'yellow', ...
         'MarkerEdgeColor', 'black', 'LineWidth', 2);
    legend('Log-Likelihood', 'Most Selected', 'Location', 'southeast');
    hold off;
end
sgtitle('Model Order Selection: Log-Likelihood Comparison', 'FontSize', 15, 'FontWeight', 'bold');

%% Visualization 3: Selection Rate Bar Chart
figure('Position', [100, 100, 1000, 600]);
bar(selectionRatesPercent', 'grouped');
xlabel('Model Order (Number of Components)', 'FontSize', 13, 'FontWeight', 'bold');
ylabel('Selection Rate (%)', 'FontSize', 13, 'FontWeight', 'bold');
title('GMM Model Order Selection Rates Across Different Dataset Sizes', ...
      'FontSize', 14, 'FontWeight', 'bold');
legend(arrayfun(@(n) sprintf('N = %d', n), sampleSizes, 'UniformOutput', false), ...
       'Location', 'northeast', 'FontSize', 12);
grid on;
set(gca, 'FontSize', 11);
xlim([0.5, maxComponents+0.5]);

%% Visualization 4: Combined Log-Likelihood Comparison
figure('Position', [100, 100, 900, 600]);
hold on;
for sizeIndex = 1:length(sampleSizes)
    avgLogLiks = mean(allLogLikelihoods{sizeIndex}, 1);
    plot(1:maxComponents, avgLogLiks, 'o-', ...
         'LineWidth', 2.5, ...
         'MarkerSize', 8, ...
         'MarkerFaceColor', colors{sizeIndex}, ...
         'Color', colors{sizeIndex}, ...
         'DisplayName', sprintf('N = %d', sampleSizes(sizeIndex)));
end
hold off;

xlabel('Model Order (Number of Components)', 'FontSize', 13, 'FontWeight', 'bold');
ylabel('Average Log-Likelihood', 'FontSize', 13, 'FontWeight', 'bold');
title('Order Log-Likelihood Comparison Across Dataset Sizes', ...
      'FontSize', 14, 'FontWeight', 'bold');
legend('Location', 'southeast', 'FontSize', 12);
grid on;
set(gca, 'FontSize', 11);
xlim([0.5, maxComponents+0.5]);

%% Save Results
resultsTable = array2table(selectionRatesPercent, ...
    'VariableNames', arrayfun(@(x) sprintf('M%d', x), 1:maxComponents, 'UniformOutput', false), ...
    'RowNames', arrayfun(@(n) sprintf('N_%d', n), sampleSizes, 'UniformOutput', false));

writetable(resultsTable, 'gmm_selection_results.csv', 'WriteRowNames', true);
fprintf('\nResults saved to: gmm_selection_results.csv\n');

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% FUNCTION: Generate GMM Data
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function dataPoints = generateGMMData(numSamples, weights, means, covariances)
    % Generate samples from Gaussian Mixture Model
    dim = size(means, 1);
    numComponents = length(weights);
    
    % Component assignment via cumulative distribution
    cumWeights = [0, cumsum(weights)];
    uniformRandom = rand(1, numSamples);
    
    dataPoints = zeros(dim, numSamples);
    
    % Generate samples for each component
    for k = 1:numComponents
        indices = find(uniformRandom > cumWeights(k) & uniformRandom <= cumWeights(k+1));
        numAssigned = length(indices);
        
        if numAssigned > 0
            % Ensure covariance is symmetric and positive definite
            currentCov = covariances(:,:,k);
            currentCov = (currentCov + currentCov') / 2;  % Force symmetry
            currentCov = currentCov + 1e-6 * eye(dim);     % Ensure positive definite
            
            % Generate using standard normal and transform
            standardSamples = randn(dim, numAssigned);
            
            % Use eigenvalue decomposition for stable square root
            [V, D] = eig(currentCov);
            sqrtCov = V * sqrt(D) * V';
            
            % Transform and shift
            dataPoints(:, indices) = sqrtCov * standardSamples + repmat(means(:,k), 1, numAssigned);
        end
    end
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% FUNCTION: Evaluate Gaussian PDF
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function densities = computeGaussianDensity(points, mean, covariance)
    % Compute multivariate Gaussian probability density
    [dim, numPoints] = size(points);
    
    % Ensure symmetric covariance and regularize
    covariance = (covariance + covariance') / 2;
    covariance = covariance + 1e-8 * eye(dim);
    
    % Inverse and determinant using eigenvalue decomposition
    [V, D] = eig(covariance);
    eigVals = diag(D);
    eigVals(eigVals < 1e-10) = 1e-10;  % Prevent negative eigenvalues
    
    covInv = V * diag(1./eigVals) * V';
    covDet = prod(eigVals);
    
    % Normalization
    normFactor = (2*pi)^(-dim/2) / sqrt(covDet);
    
    % Mahalanobis distance
    centered = points - repmat(mean, 1, numPoints);
    mahalDist = sum(centered .* (covInv * centered), 1);
    
    % Density values
    densities = normFactor * exp(-0.5 * mahalDist);
    
    % Handle numerical issues
    densities(densities < 1e-300) = 1e-300;
    densities(~isfinite(densities)) = 1e-300;
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% FUNCTION: EM Algorithm for GMM
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [weights, means, covariances] = trainGMMwithEM(data, numComp, dim, tol, reg)
    % Train GMM using Expectation-Maximization
    numPoints = size(data, 2);
    
    % Initialize weights uniformly
    weights = ones(1, numComp) / numComp;
    
    % Initialize means with k-means++ style
    means = initializeMeans(data, numComp);
    
    % Initialize covariances with global covariance + regularization
    globalCov = cov(data');
    globalCov = (globalCov + globalCov') / 2;  % Force symmetry
    globalCov = globalCov + reg * eye(dim);
    covariances = repmat(globalCov, [1, 1, numComp]);
    
    % EM iteration
    maxIter = 100;
    converged = false;
    iter = 0;
    
    while ~converged && iter < maxIter
        iter = iter + 1;
        
        %% E-step: Compute responsibilities
        responsibilities = zeros(numComp, numPoints);
        for k = 1:numComp
            responsibilities(k, :) = weights(k) * ...
                computeGaussianDensity(data, means(:,k), covariances(:,:,k));
        end
        
        % Normalize
        respSum = sum(responsibilities, 1);
        respSum(respSum < 1e-300) = 1e-300;
        responsibilities = responsibilities ./ respSum;
        
        %% M-step: Update parameters
        effectiveCounts = sum(responsibilities, 2);
        
        % Update weights
        weightsNew = effectiveCounts' / numPoints;
        weightsNew(weightsNew < 1e-10) = 1e-10;
        weightsNew = weightsNew / sum(weightsNew);
        
        % Update means
        meansNew = zeros(dim, numComp);
        for k = 1:numComp
            if effectiveCounts(k) > 1e-10
                meansNew(:,k) = sum(data .* responsibilities(k,:), 2) / effectiveCounts(k);
            else
                meansNew(:,k) = means(:,k);
            end
        end
        
        % Update covariances
        covariancesNew = zeros(dim, dim, numComp);
        for k = 1:numComp
            if effectiveCounts(k) > 1e-10
                centered = data - repmat(meansNew(:,k), 1, numPoints);
                weightedCov = (centered .* responsibilities(k,:)) * centered' / effectiveCounts(k);
                
                % Force symmetry and add regularization
                weightedCov = (weightedCov + weightedCov') / 2;
                covariancesNew(:,:,k) = weightedCov + reg * eye(dim);
            else
                covariancesNew(:,:,k) = covariances(:,:,k);
            end
        end
        
        % Check convergence
        deltaWeights = sum(abs(weightsNew - weights));
        deltaMeans = sum(sum(abs(meansNew - means)));
        deltaCovs = sum(sum(sum(abs(covariancesNew - covariances))));
        
        converged = (deltaWeights + deltaMeans + deltaCovs) < tol;
        
        % Update
        weights = weightsNew;
        means = meansNew;
        covariances = covariancesNew;
    end
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% FUNCTION: Initialize Means (k-means++ style)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function means = initializeMeans(data, numComp)
    numPoints = size(data, 2);
    
    if numPoints < numComp
        % Repeat and jitter if insufficient samples
        means = data(:, mod(0:numComp-1, numPoints) + 1);
        means = means + 0.1 * randn(size(means));
    else
        % Random selection
        indices = randperm(numPoints, numComp);
        means = data(:, indices);
    end
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% FUNCTION: Cross-Validation for GMM
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [bestOrder, avgLogLikelihoods] = crossValidateGMM(data, dataSize, ...
                                          maxOrder, kFolds, dim, tol, reg)
    % Adjust folds for small datasets
    if dataSize < kFolds
        kFolds = max(2, dataSize);
    end
    
    % Create fold partitions
    foldEdges = round(linspace(0, dataSize, kFolds + 1));
    
    % Initialize storage
    avgLogLikelihoods = -Inf * ones(1, maxOrder);
    effectiveMaxOrder = min(maxOrder, max(1, floor(dataSize / 2)));
    
    % Test each model order
    for order = 1:effectiveMaxOrder
        foldScores = zeros(1, kFolds);
        validFolds = 0;
        
        % Cross-validation loop
        for fold = 1:kFolds
            % Validation indices
            valStart = foldEdges(fold) + 1;
            valEnd = foldEdges(fold + 1);
            valIndices = valStart:valEnd;
            
            % Training indices
            trainIndices = setdiff(1:dataSize, valIndices);
            
            % Skip if insufficient training data
            if length(trainIndices) < order
                continue;
            end
            
            % Split data
            trainData = data(:, trainIndices);
            valData = data(:, valIndices);
            numVal = length(valIndices);
            
            try
                % Train GMM
                [weights, means, covs] = trainGMMwithEM(trainData, order, dim, tol, reg);
                
                % Compute validation log-likelihood
                logLik = 0;
                for i = 1:numVal
                    mixtureProb = 0;
                    for k = 1:order
                        mixtureProb = mixtureProb + weights(k) * ...
                            computeGaussianDensity(valData(:,i), means(:,k), covs(:,:,k));
                    end
                    
                    if mixtureProb > 1e-300
                        logLik = logLik + log(mixtureProb);
                    else
                        logLik = logLik - 700;
                    end
                end
                
                foldScores(fold) = logLik;
                validFolds = validFolds + 1;
            catch
                foldScores(fold) = -1e10;
            end
        end
        
        % Average across valid folds
        if validFolds > 0
            avgLogLikelihoods(order) = sum(foldScores) / validFolds;
        end
    end
    
    % Select best order
    [~, bestOrder] = max(avgLogLikelihoods);
end
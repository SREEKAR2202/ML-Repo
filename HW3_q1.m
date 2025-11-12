%Expected risk minimization with neural network classifiers
clear;
close all;

%Configuration parameters
featureDim = 3;
classCount = 4;
classNames = {'Class0','Class1','Class2','Class3'};

% For min-Perror design, use 0-1 loss
costMatrix = ones(classCount,classCount) - eye(classCount);
meanScale = 2.5;
covScale = 0.2;

%Define data
dataSet.train100.N = 100;
dataSet.train500.N = 500;
dataSet.train1k.N = 1e3;
dataSet.train5k.N = 5e3;
dataSet.train10k.N = 1e4;
dataSet.test100k.N = 100e3;
setTypes = fieldnames(dataSet);

%Define Statistics
priorProb = ones(1,classCount) / classCount; %Prior

%Label data stats
meanVec.Class0 = meanScale * [1 1 0]';
randSig = covScale * rand(featureDim,featureDim);
covMat.Class0(:,:,1) = randSig * randSig' + eye(featureDim);

meanVec.Class1 = meanScale * [1 0 0]';
randSig = covScale * rand(featureDim,featureDim);
covMat.Class1(:,:,1) = randSig * randSig' + eye(featureDim);

meanVec.Class2 = meanScale * [0 1 0]';
randSig = covScale * rand(featureDim,featureDim);
covMat.Class2(:,:,1) = randSig * randSig' + eye(featureDim);

meanVec.Class3 = meanScale * [0 0 1]';
randSig = covScale * rand(featureDim,featureDim);
covMat.Class3(:,:,1) = randSig * randSig' + eye(featureDim);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Generate Data
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
for idx = 1:length(setTypes)
    dataSet.(setTypes{idx}).features = zeros(featureDim,dataSet.(setTypes{idx}).N);
    [dataSet.(setTypes{idx}).features, dataSet.(setTypes{idx}).classLabels,...
        dataSet.(setTypes{idx}).samplesPerClass, dataSet.(setTypes{idx}).empiricalPrior] = ...
        genData(dataSet.(setTypes{idx}).N, priorProb, meanVec, covMat, classNames, featureDim);
end

%Plot Training Data
figure('Position', [100, 100, 1200, 800]);
for idx = 1:length(setTypes)-1
    subplot(2,3,idx);
    plotData(dataSet.(setTypes{idx}).features, dataSet.(setTypes{idx}).classLabels, classNames);
    legend('Location', 'best', 'FontSize', 8);
    title(setTypes{idx}, 'FontSize', 12, 'FontWeight', 'bold');
    view(45, 30);
end

%Plot Test Data
figure('Position', [100, 100, 800, 600]);
plotData(dataSet.(setTypes{end}).features, dataSet.(setTypes{end}).classLabels, classNames);
legend('Location', 'best', 'FontSize', 10);
title(setTypes{end}, 'FontSize', 14, 'FontWeight', 'bold');
view(45, 30);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Determine Theoretically Optimal Classifier
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
for idx = 1:length(setTypes)
    [dataSet.(setTypes{idx}).optimal.errorProb, dataSet.(setTypes{idx}).optimal.predictions] = ...
        optClass(costMatrix, dataSet.(setTypes{idx}).features, meanVec, covMat,...
        priorProb, dataSet.(setTypes{idx}).classLabels, classNames);
    optimalErr(idx) = dataSet.(setTypes{idx}).optimal.errorProb;
    fprintf('Theoretical Optimal Error, N=%1.0f: Rate=%1.2f%%\n',...
        dataSet.(setTypes{idx}).N, 100*dataSet.(setTypes{idx}).optimal.errorProb);
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Train and Validate Data
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
maxPerceptrons = 15;
numFolds = 10;

for idx = 1:length(setTypes)-1
    %kfold validation is in this function
    [dataSet.(setTypes{idx}).network, dataSet.(setTypes{idx}).minErr,...
        dataSet.(setTypes{idx}).bestM, validationResults.(setTypes{idx}).metrics] = ...
        kfoldMLP_NN(maxPerceptrons, numFolds, dataSet.(setTypes{idx}).features, ...
        dataSet.(setTypes{idx}).classLabels, classCount);
    
    %Produce validation data from test dataset
    validationResults.(setTypes{idx}).outputs = dataSet.(setTypes{idx}).network(dataSet.test100k.features);
    [~, validationResults.(setTypes{idx}).predicted] = max(validationResults.(setTypes{idx}).outputs);
    validationResults.(setTypes{idx}).predicted = validationResults.(setTypes{idx}).predicted - 1;
    
    %Probability of Error is wrong decisions/num data points
    validationResults.(setTypes{idx}).errorProb = ...
        sum(validationResults.(setTypes{idx}).predicted ~= dataSet.test100k.classLabels) / dataSet.test100k.N;
    
    results(idx,1) = dataSet.(setTypes{idx}).N;
    results(idx,2) = validationResults.(setTypes{idx}).errorProb;
    results(idx,3) = dataSet.(setTypes{idx}).bestM;
    
    fprintf('MLP Error, N=%1.0f: Rate=%1.2f%%\n',...
        dataSet.(setTypes{idx}).N, 100*validationResults.(setTypes{idx}).errorProb);
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Results Visualization
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Extract cross validation results from structure
for idx = 1:length(setTypes)-1
    [~, bestIdx] = min(validationResults.(setTypes{idx}).metrics.meanErr);
    hiddenUnits(idx) = validationResults.(setTypes{idx}).metrics.numPerceptrons(bestIdx);
    sampleSize(idx) = dataSet.(setTypes{idx}).N;
end

%Plot number of perceptrons vs. error for the cross validation runs
figure('Position', [100, 100, 1400, 900]);
for idx = 1:length(setTypes)-1
    subplot(2,3,idx);
    stem(validationResults.(setTypes{idx}).metrics.numPerceptrons, ...
         validationResults.(setTypes{idx}).metrics.meanErr, 'LineWidth', 1.5, 'MarkerSize', 6);
    xlabel('Hidden Layer Perceptrons', 'FontSize', 11);
    ylabel('Error Probability', 'FontSize', 11);
    title(['Misclassification Rate vs. Perceptron Count for ', setTypes{idx}], ...
          'FontSize', 11, 'FontWeight', 'bold');
    grid on;
    set(gca, 'FontSize', 10);
end

%Number of perceptrons vs. size of training dataset
figure('Position', [100, 100, 900, 600]);
semilogx(sampleSize(1:end-1), hiddenUnits(1:end-1), 's-', 'LineWidth', 2.5, ...
         'MarkerSize', 10, 'MarkerFaceColor', [0.2 0.4 0.8]);
grid on;
xlabel('Training Sample Count', 'FontSize', 13, 'FontWeight', 'bold');
ylabel('Optimal Perceptron Count', 'FontSize', 13, 'FontWeight', 'bold');
ylim([0 16]);
xlim([80 12000]);
title('Best Perceptron Count vs. Training Sample Size', 'FontSize', 14, 'FontWeight', 'bold');
set(gca, 'FontSize', 11);

%Prob. of Error vs. size of training data set
figure('Position', [100, 100, 900, 600]);
semilogx(results(1:end-1,1), results(1:end-1,2), 's-', 'LineWidth', 2.5, ...
         'MarkerSize', 10, 'MarkerFaceColor', [0.8 0.2 0.2], 'Color', [0.8 0.2 0.2]);
xlim([80 12000]);
hold on;
semilogx(xlim, [optimalErr(end) optimalErr(end)], 'b--', 'LineWidth', 2.5);
legend('MLP Classifier', 'Theoretical Optimal', 'Location', 'northeast', 'FontSize', 12);
grid on;
xlabel('Training Sample Count', 'FontSize', 13, 'FontWeight', 'bold');
ylabel('Error Probability', 'FontSize', 13, 'FontWeight', 'bold');
title('Error Rate vs. Training Dataset Size', 'FontSize', 14, 'FontWeight', 'bold');
set(gca, 'FontSize', 11);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [samples, classIDs, samplesPerClass, estimatedPrior] = genData(numSamples, priors, means, covariances, labels, dimensions)
%Generates data and labels for random variable x from multiple gaussian distributions
numClasses = length(labels);
cumulativePrior = [0, cumsum(priors)];
uniformRand = rand(1, numSamples);
samples = zeros(dimensions, numSamples);
classIDs = zeros(1, numSamples);
for idx = 1:numClasses
    dataPoints = find(cumulativePrior(idx) < uniformRand & uniformRand <= cumulativePrior(idx+1));
    samplesPerClass(idx) = length(dataPoints);
    samples(:, dataPoints) = mvnrnd(means.(labels{idx}), covariances.(labels{idx}), samplesPerClass(idx))';
    classIDs(dataPoints) = idx - 1;
    estimatedPrior(idx) = samplesPerClass(idx) / numSamples;
end
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function plotData(features, classIDs, labels)
%Plots data with improved visualization
colorMap = [0.0 0.4 0.8;    % Blue
            0.8 0.2 0.2;    % Red
            0.9 0.6 0.0;    % Orange
            0.6 0.2 0.8];   % Purple

for idx = 1:length(labels)
    classIdx = classIDs == idx - 1;
    plot3(features(1,classIdx), features(2,classIdx), features(3,classIdx), '.', ...
          'Color', colorMap(idx,:), 'MarkerSize', 12, 'DisplayName', labels{idx});
    hold on;
end
grid on;
xlabel('Dimension 1', 'FontSize', 11, 'FontWeight', 'bold');
ylabel('Dimension 2', 'FontSize', 11, 'FontWeight', 'bold');
zlabel('Dimension 3', 'FontSize', 11, 'FontWeight', 'bold');
set(gca, 'FontSize', 10);
hold off;
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function likelihood = evalGaussian(features, meanVec, covMatrix)
% Evaluates the Gaussian pdf N(mu,Sigma) at each column of X
[dims, numSamples] = size(features);
invCov = inv(covMatrix);
normConst = (2*pi)^(-dims/2) * det(invCov)^(1/2);
exponent = -0.5 * sum((features - repmat(meanVec,1,numSamples)).* ...
                      (invCov * (features - repmat(meanVec,1,numSamples))), 1);
likelihood = normConst * exp(exponent);
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [minErrorProb, predictions] = optClass(costMatrix, features, means, covariances, priors, trueLabels, labels)
% Determine optimal probability of error
markers = 'ox+*';
numClasses = length(labels);
numSamples = length(features);

for idx = 1:numClasses
    conditionalProb(idx,:) = evalGaussian(features, means.(labels{idx}), covariances.(labels{idx}));
end

evidence = priors * conditionalProb;
posteriors = conditionalProb .* repmat(priors', 1, numSamples) ./ repmat(evidence, numClasses, 1);

% Expected Risk for each label (rows) for each sample (columns)
expectedRisk = costMatrix * posteriors;

% Minimum expected risk decision with 0-1 loss is the same as MAP
[~, predictions] = min(expectedRisk, [], 1);
predictions = predictions - 1;

incorrectIdx = (predictions ~= trueLabels);
minErrorProb = sum(incorrectIdx) / numSamples;

%Plot Decisions with Incorrect Results
figure('Position', [100, 100, 900, 700]);
for idx = 1:numClasses
    currentClass = predictions == idx - 1;
    
    % Correct classifications
    plot3(features(1, currentClass & ~incorrectIdx),...
        features(2, currentClass & ~incorrectIdx),...
        features(3, currentClass & ~incorrectIdx),...
        markers(idx), 'Color', [0.20 0.70 0.20], 'MarkerSize', 8, 'LineWidth', 1.5, ...
        'DisplayName', ['Category ' num2str(idx) ' Accurate']);
    hold on;
    
    % Incorrect classifications
    plot3(features(1, currentClass & incorrectIdx),...
        features(2, currentClass & incorrectIdx),...
        features(3, currentClass & incorrectIdx),...
        markers(idx), 'Color', [0.8 0.0 0.8], 'MarkerSize', 8, 'LineWidth', 1.5, ...
        'DisplayName', ['Category ' num2str(idx) ' Misclassified']);
end
xlabel('Dimension 1', 'FontSize', 12, 'FontWeight', 'bold');
ylabel('Dimension 2', 'FontSize', 12, 'FontWeight', 'bold');
zlabel('Dimension 3', 'FontSize', 12, 'FontWeight', 'bold');
grid on;
title('Feature Space with Misclassifications Highlighted', 'FontSize', 13, 'FontWeight', 'bold');
legend('Location', 'best', 'FontSize', 9);
view(45, 30);
set(gca, 'FontSize', 10);
hold off;
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [finalNetwork, finalErr, bestM, statistics] = kfoldMLP_NN(maxPerceptrons, folds, features, classIDs, numClasses)
%Performs cross validation and model selection for neural network
totalSamples = length(features);
numRandomInits = 10;

%Create output matrices from labels (one-hot encoding)
targetMatrix = zeros(numClasses, length(features));
for idx = 1:numClasses
    targetMatrix(idx,:) = (classIDs == idx - 1);
end

%Setup cross validation on training data
partitionSize = totalSamples / folds;
partitionIdx = [1:partitionSize:totalSamples length(features)];

%Perform cross validation to select number of perceptrons
for numHidden = 1:maxPerceptrons
    for foldIdx = 1:folds
        validationIdx.val = partitionIdx(foldIdx):partitionIdx(foldIdx+1);
        validationIdx.train = setdiff(1:totalSamples, validationIdx.val);
        
        %Create network with numHidden perceptrons in hidden layer
        mlpNet = patternnet(numHidden);
        mlpNet.trainParam.showWindow = false; % Suppress training window
        
        %Train using training data
        mlpNet = train(mlpNet, features(:,validationIdx.train), targetMatrix(:,validationIdx.train));
        
        %Validate with remaining data
        validationOut = mlpNet(features(:,validationIdx.val));
        [~, predictedClass] = max(validationOut);
        predictedClass = predictedClass - 1;
        foldErr(foldIdx) = sum(predictedClass ~= classIDs(validationIdx.val)) / partitionSize;
    end
    
    %Determine average probability of error for current number of perceptrons
    avgErr(numHidden) = mean(foldErr);
    statistics.numPerceptrons = 1:numHidden;
    statistics.meanErr = avgErr;
end

%Determine optimal number of perceptrons
[~, bestM] = min(avgErr);

%Train final model multiple times with different random initializations
for initIdx = 1:numRandomInits
    networkID(initIdx) = {['network' num2str(initIdx)]};
    trainedNets.(networkID{initIdx}) = patternnet(bestM);
    trainedNets.(networkID{initIdx}).trainParam.showWindow = false;
    trainedNets.(networkID{initIdx}) = train(trainedNets.(networkID{initIdx}), features, targetMatrix);
    
    finalOut = trainedNets.(networkID{initIdx})(features);
    [~, finalPred] = max(finalOut);
    finalPred = finalPred - 1;
    finalErrors(initIdx) = sum(finalPred ~= classIDs) / length(features);
end

%Select best performing network
[finalErr, bestInitIdx] = min(finalErrors);
statistics.finalErrors = finalErrors;
finalNetwork = trainedNets.(networkID{bestInitIdx});
end
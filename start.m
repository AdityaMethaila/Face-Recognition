clear all;
close all;
clc;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%% load the data
FeatureMatOD=dlmread('data/ODFeatureMat.txt');
FeatureMatHD=dlmread('data/HDFeatureMat.txt');
FeatureMat=[FeatureMatOD FeatureMatHD(:,2:end)];
clear FeatureMatHD;
clear FeatureMatOD;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%the flow of your code should look like this
Dim = size(FeatureMat,2)-1; %dimension of the feature
countfeat(Dim,2) = 0;
%%countfeat is a Mx2 matrix that keeps track of how many times a feature has been selected, where M is the dimension of the original feature space.
%%The first column of this matrix records how many times a feature has ranked within top 1% during 100 times of feature ranking.
%%The second column of this matrix records how many times a feature was selected by forward feature selection during 100 times.

%%%%%%%%%%%%%%%%%%%% test code %%%%%
%comment this out 
tmp = randperm(Dim);
topfeatures(:,1) = tmp(1:1000)';
topfeatures(:,2) = 100*rand(1000,1);
forwardselected = tmp(1:100)';
%%%%%%%%%%%%%%%%%%%%%%%%************


for i=1:100
    
    
    
    
    
    
    % randomly divide into equal test and traing sets
    [TrainMat, LabelTrain, TestMat, LabelTest]= randomDivideMulti(FeatureMat);

    % start feature ranking
    topfeatures = rankingfeat(TrainMat, LabelTrain); 
    countfeat(topfeatures(:,1),1) =  countfeat(topfeatures(:,1),1) +1;
    
    %% visualize the variance ratio of the top 1% features
    if i==1
        %% colorbar indicates the correspondance between the variance ratio
        %% of the selected feature
       plotFeat(topfeatures);
    end

    % start forward feature selection
    forwardselected = forwardselection(TrainMat, LabelTrain, topfeatures);
    countfeat(forwardselected,2) =  countfeat(forwardselected,2) +1;    
    
    % start classification
    ob=fitcdiscr(TestMat(:,forwardselected),LabelTest);
    er(1,i)=resubLoss(ob);
    
end


%% visualize the features that have ranked within top 1% most during 100 times of feature ranking
data(:,1)=[1:Dim]';
data(:,2) = countfeat(:,1);
%% colorbar indicates the number of times a feature at that location was
%% ranked within top 1%
plotFeat(data);
%% visualize the features that have been selected most during 100 times of
%% forward selection
data(:,2) = countfeat(:,2);
%% colorbar indicates the number of times a feature at that location was
%% selected by forward selection
plotFeat(data);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Note: You don't need this step for classification. This is just for the inquisitive minds who want to see how the features actually look like.
% Suppose you want to visualize 5th subject in the Test set. The following code shows how the feature of the 5'th subject would look like:
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% % uncomment to visualize the features
% FeatureMat=dlmread('data/HDFeatureMat.txt');
% k=reshape(TrainMat(5,:),[125 62]);
% imagesc(flipud([k fliplr(k)]));
% COLORBAR
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

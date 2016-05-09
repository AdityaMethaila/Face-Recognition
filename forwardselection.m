function forwardselected = forwardselection(TrainMat, LabelTrain, topfeatures)
%% input: TrainMat - a NxM matrix that contains the full list of features
%% of training data. N is the number of training samples and M is the
%% dimension of the feature. So each row of this matrix is the face
%% features of a single person.
%%        LabelTrain - a Nx1 vector of the class labels of training data
%%        topfeatures - a Kx2 matrix that contains the information of the
%% top 1% features of the highest variance ratio. K is the number of
%% selected feature (K = ceil(M*0.01)). The first column of this matrix is
%% the index of the selected features in the original feature list. So the
%% range of topfeatures(:,1) is between 1 and M. The second column of this
%% matrix is the variance ratio of the selected features.

%% output: forwardselected - a Px1 vector that contains the index of the 
%% selected features in the original feature list, where P is the number of
%% selected features. The range of forwardselected is between 1 and
M=15500;
S=[];
K = ceil(M*0.01);
best_feat=0;
Min=100;

while true
    ob =fitcnb([TrainMat(:,S) TrainMat(:,topfeatures(1,1))],LabelTrain);
    erMin=resubLoss(ob);
    best_feat=topfeatures(1,1);
    
    for i = 2:size(topfeatures,1)
        ob1=fitcnb([TrainMat(:,S) TrainMat(:,topfeatures(i,1))],LabelTrain);
        er=resubLoss(ob1);
        
        if er<erMin
            erMin=er;
            best_feat=topfeatures(i,1);
            
        end
    end
    
    if(erMin>Min)
        break;
    else
        Min=erMin;
        S=[S best_feat];
        index=find(topfeatures(:,1)==best_feat);
        topfeatures(index,:) = [];
    end
    
    l=length(S);
    if(l==155)
        break;
    end
    
    
end

forwardselected = S'

       


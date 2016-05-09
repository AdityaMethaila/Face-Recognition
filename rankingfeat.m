function topfeatures = rankingfeat(TrainMat, LabelTrain)
%% input: TrainMat - a NxM matrix that contains the full list of features
%% of training data. N is the number of training samples and M is the
%% dimension of the feature. So each row of this matrix is the face
%% features of a single person.
%%        LabelTrain - a Nx1 vector of the class labels of training data

%% output: topfeatures - a Kx2 matrix that contains the information of the
%% top 1% features of the highest variance ratio. K is the number of
%% selected feature (K = ceil(M*0.01)). The first column of this matrix is
%% the index of the selected features in the original feature list. So the
%% range of topfeatures(:,1) is between 1 and M. The second column of this
%% matrix is the variance ratio of the selected features.
M=15500;
K = ceil(M*0.01);
for i=1:18
    if(LabelTrain(i,1)==0)
        x0(i,:)= TrainMat(i,:);
    end
end
    
for i=1:35
        if(LabelTrain(i,1==1))
            x1(i,:)= TrainMat(i,:);
        end
     
end
    
for i=1:15500
    n= var(TrainMat(:,i));
    d=(var(x0(:,i)) + var(x1(:,i)))/2;
    
    vr(1,i)= n/d;
end


[vrSorted I] = sort(vr,'descend');
Imax = I(1:K);

vrmax = vrSorted(1:K);
plot(vrmax);
topfeatures=[Imax' vrmax'];


    
    
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% COMPLETE BRUTE FORCE CODE
%DATA_DIVIDED_NOF9_NOFEATURES.mat is needed
%% set global variables
global AccuracyM1_tot; global AccuracyM1_test; global AccuracyM2_tot; global AccuracyM2_test; global AccuracyM3_test; global AccuracyM3_tot;
global F1M1avg; global F1M2avg; global F1M3avg;
global Ystore; global YINDstore; global ETest;
global STOREy; global STOREtr;
global F1M1avg_tot; global F1M2avg_tot; global F1M3avg_tot;
%reset
AccuracyM1_tot ={}; AccuracyM1_test={}; AccuracyM2_tot={}; AccuracyM2_test={}; AccuracyM3_test={}; AccuracyM3_tot={};
F1M1avg={}; F1M2avg={}; F1M3avg={};
Ystore={}; YINDstore={}; ETest = [];
STOREy={}; STOREtr={};
F1M1avg_tot={}; F1M2avg_tot={}; F1M3avg_tot={};
%% 
%defining ranges,
rangeHidden = [1,2,4,8,16,32,64,128]; %NUMBEER OF NEURONS FOR EACH LAYER
rangeLayers = 2:3; % 2 or 3 hidden layers
rangeVote = 1:10; %only 2 votes
% wellArray
wellArray = {};
j = 7;
for i = 2:7
    wellArray{i-1,1} =  combnk(1:7,j);
    j = j - 1;
end
%figures
% F1=figure; F12=figure; Accuracy1=figure; Accuracy2=figure; RoCurve1=figure; RoCurve2=figure;
%% START LOOP
for z = 1:length(wellArray)
    for cz = 1:length(wellArray{z}(:,1))
        %% new x,t
        x= [] ;
        t = [] ;
        for i = wellArray{z}(cz,:)
            x = [x input_data(:,WELLS_data{2,i})];
            t = [t target_data(:,WELLS_data{2,i})];
        end
        for w = wellArray{z}(cz,:)
            for layers = rangeLayers
                for hidden = rangeHidden
                    for vote = rangeVote
                        [net] = ArchitectureSetUp(layers,hidden,x,t);
        %                 [net] = DivideindNOVaL(net,w,x,t,WELLS_data);
                        [net] = Divideind(net,w,x,t,WELLS_data,wellArray,z,cz,input_data);
                        [net] = TrainingFunctionSetUp(net);
                        [net,tr] = train(net,x,t); %training 
                        [Ystore,YINDstore,y,tind,yind,ETest] = Test(net,x,t,Ystore,YINDstore,vote,tr,ETest); %test
                    end
                    %voting results
                    [y,yind,STOREy,STOREtr,Ystore,YINDstore] = Vote(Ystore,YINDstore,tind,STOREy,STOREtr,tr,w,layers,hidden,z,cz);
        %           [y,yind,STOREy,STOREtr,Ystore,YINDstore,ETest] = VoteTheBest(ETest,Ystore,YINDstore,STOREy,STOREtr,tr,w,layers,hidden);
                    % Calculate Accuracy and F1
                    [AccuracyM1_test,AccuracyM1_tot] = Accuracy_M1(AccuracyM1_test,AccuracyM1_tot,tind,yind,hidden,tr,w,layers,z,cz);
                    [AccuracyM2_test,AccuracyM2_tot] = Accuracy_M2(AccuracyM2_test,AccuracyM2_tot,tind,yind,hidden,tr,w,layers,z,cz);
                    [AccuracyM3_test,AccuracyM3_tot] = Accuracy_M3(AccuracyM3_test,AccuracyM3_tot,tind,yind,hidden,tr,w,layers,z,cz);

                    [F1M1avg,F1M1,precision_M1,recall_M1] = F1_M1(F1M1avg,facies,w,tr,yind,tind,layers,hidden,z,cz);
                    [F1M2avg,F1M2,precision_M2,recall_M2] = F1_M2(F1M2avg,facies,w,tr,yind,tind,layers,hidden,z,cz);
                    [F1M3avg,precision_M3,recall_M3] = F1_M3(F1M3avg,w,tr,yind,tind,layers,hidden,z,cz);

                    [F1M1avg_tot,F1M1_tot,precision_M1_tot,recall_M1_tot] = F1_M1_tot(F1M1avg_tot,facies,w,tr,yind,tind,layers,hidden,z,cz);
                    [F1M2avg_tot,F1M2_tot,precision_M2_tot,recall_M2_tot] = F1_M2_tot(F1M2avg_tot,facies,w,tr,yind,tind,layers,hidden,z,cz);
                    [F1M3avg_tot,precision_M3_tot,recall_M3_tot] = F1_M3_tot(F1M3avg_tot,w,tr,yind,tind,layers,hidden,z,cz);
                end
    %             ROC(RoCurve1,RoCurve2,w,t,tr,y,layers); %needs to be here cause tr is calculated and overwritten
    %             Plotting(rangeHidden,w,layers,F1M1avg,F1M2avg,F1M3avg,AccuracyM1_test,AccuracyM2_test,AccuracyM3_test,F1,F12,Accuracy1,Accuracy2);
            end
        end     
    end
    wellArray{z}
end
%find best architecture
% Bests(rangeHidden,rangeLayers,F1M1avg,F1M2avg,F1M3avg,AccuracyM1_test,AccuracyM2_test,AccuracyM3_test)
save FINAL_SENSITIVITY_128_10tries_AVGvote.mat
%% FUNCTIONS
function [net] = ArchitectureSetUp(layers,hidden,x,t)
numInputs = 1; numLayers = layers;
biasConnect = ones(layers,1); inputConnect = [1 ;zeros(layers-1,1)]; outputConnect = [zeros(1,layers-1) 1]; 
layerConnect = zeros(layers);

for i = 2:layers
    layerConnect(i,i-1) = 1;
end 
net = network(numInputs,numLayers,biasConnect,inputConnect,layerConnect,outputConnect);
%start to configure the network
net = configure(net,x,t);

%set up the layers name
for i = 1:layers
    if i == 1
        net.layers{i,1}.name = 'input layer';
    elseif i == layers
        net.layers{i,1}.name = 'output layer';
    else 
        net.layers{i,1}.name = "hidden layer" + i ;
    end 
end 

%set up the layer dimensions
for i = 1:layers
    if i == layers
        net.layers{i,1}.dimensions = 9;
    else 
        net.layers{i,1}.dimensions = hidden;
    end 
end

%set up the layer transfer function
for i = 1:layers
    if i == layers 
        net.layers{i,1}.transferFcn = "softmax";
    else 
        net.layers{i,1}.transferFcn = "tansig";
    end 
end 

 %set up the net input function and   %set up the W and b initialization function algorithm 

for i = 1:layers
    net.layers{i,1}.netInputFcn = "netsum";
    net.layers{i,1}.initFcn= "initnw";
end 
% SET UP DATA PRE-PROCESSING FUNCTIONS
net.input.processFcns = {'removeconstantrows','mapminmax'};
% set up performance function
net.performFcn = 'crossentropy';  % Cross-Entropy
end
function [net] = DivideindNOVaL(net,w,x,t,WELLS_data)
%% ORDER DATA
training_data = [];  
training_index = [];
training_find = [];
for i = 1:length(WELLS_data)
    if i == w
        training_index = [training_index zeros(1,length(WELLS_data{2,i}))]; 
       continue
    end
    training_data =[training_data  x(:,WELLS_data{2,i})];
    training_index = [training_index ones(1,length(WELLS_data{2,i}))]; 
    training_find = find(training_index);
end


test_well_data = x(:,WELLS_data{2,w});
test_well_index = training_index <1;
test_well_find = find (test_well_index);

%test well index
testIndex = test_well_index;
%train wells indices
trainIndex = testIndex <1 ;
%%
net.divideFcn = 'divideind'; % Divide up for index
net.divideMode = 'sample';  % Divide up every sample
[trainInd,testInd]=divideind(length(t),find(trainIndex), find(testIndex));
net.divideParam.trainInd=trainInd;
net.divideParam.testInd=testInd;
end
function [net] = Divideind(net,w,x,t,WELLS_data,wellArray,z,cz,input_data)
%%
training_data = [];  
training_index = [];
training_find = [];
for i = wellArray{z}(cz,:)
    if i == w
        training_index = [training_index zeros(1,length(WELLS_data{2,i}))]; 
       continue
    end
    training_data =[training_data  input_data(:,WELLS_data{2,i})];
    training_index = [training_index ones(1,length(WELLS_data{2,i}))]; 
    training_find = find(training_index);
end
%% 

test_well_data = input_data(:,WELLS_data{2,w});
test_well_index = training_index <1;
% test_well_find = find (test_well_index);

%divide training_data in training and validation randomly 
trainIndex = [];
validationIndex = [];
testIndex = [];

trainingRatio = 0.90;
validationRatio = 0.10;

totalTrainingData=round(length(training_data)*trainingRatio);
totalValidationData=length(training_data)-totalTrainingData;

r4nd=zeros(1,length(training_data));
trainIndex = zeros(1,length(x));
for i = 1:totalTrainingData
    while true
        rv = randi(length(training_find),1); r = training_find(rv);  
        if trainIndex(1,r)== 0
            trainIndex(1,r)= true; 
            break
        end
    end
end

%training index(part used to train the model)
trainIndex;
% now you need to find the validation indeces 
validationIndex = test_well_index <1 - trainIndex; 
% test index
testIndex = test_well_index; 

%%
net.divideFcn = 'divideind'; % Divide up for index
net.divideMode = 'sample';  % Divide up every sample
[trainInd,valInd,testInd]=divideind(length(t),find(trainIndex), find(validationIndex), find(testIndex));
net.divideParam.trainInd=trainInd;
net.divideParam.valInd=valInd;
net.divideParam.testInd=testInd;
end
function [net] = TrainingFunctionSetUp(net)
%set training function
net.trainFcn = 'trainscg';  % Scaled conjugate gradient backpropagation.
%set stopping criteria
%net.trainParam.goal = 0.01;  %set performance goal to 1% 
net.TrainParam.max_fail = 8; %set early stopping criteria
% net.TrainParam.epochs = 1000;
% net.TrainParam.min_grad=0; %no very important
end
function [Ystore,YINDstore,y,tind,yind,ETest] = Test(net,x,t,Ystore,YINDstore,vote,tr,ETest)
% Test the Network
y = net(x);
tind = vec2ind(t);
yind = vec2ind(y);
YINDstore{vote,1} = yind ;
Ystore{vote,1} = y ; 
ETest(vote,1) = sum(tind(:,tr.testInd) ~= yind(:,tr.testInd))/numel(tind(:,tr.testInd));
end
function [AccuracyM1_test,AccuracyM1_tot] = Accuracy_M1(AccuracyM1_test,AccuracyM1_tot,tind,yind,hidden,tr,w,layers,z,cz)
AccuracyM1_tot{z,cz,w,layers,hidden} = 1-(sum(tind ~= yind)/numel(tind)); 
AccuracyM1_test{z,cz,w,layers,hidden} = 1-(sum(tind(:,tr.testInd) ~= yind(:,tr.testInd))/numel(tind(:,tr.testInd)));
end
function [F1M1avg,F1M1,precision_M1,recall_M1] = F1_M1(F1M1avg,facies,w,tr,yind,tind,layers,hidden,z,cz)
%initialize
TrueNegative = zeros(9,length(tind));
FalseNegative = zeros(9,length(tind));
TruePositive = zeros(9,length(tind));
FalsePositive = zeros(9,length(tind));
precision_M1 =[];
recall_M1 = [];
for j = 1:9
for i = 1:length(tr.testInd)
    %positive count
    if yind(1,tr.testInd(i)) == j 
        if yind(1,tr.testInd(i)) == tind(1,tr.testInd(i))
            TruePositive(j,i) = 1;
        else 
            FalsePositive(j,i) = 1; 
        end
    else 
        if tind(1,tr.testInd(i)) == j
            FalseNegative(j,i) = 1;
        else 
            TrueNegative(j,i) = 1;
        end
    end
end
precision_M1(1,j) = sum(TruePositive(j,:))/(sum(TruePositive(j,:)) + sum(FalsePositive(j,:))) ; % True positive/total predicted class 3 what proportion of predicted class 3 is truly class 3? 
recall_M1(1,j) = sum(TruePositive(j,:))/(sum(TruePositive(j,:)) + sum(FalseNegative(j,:))) ;  % True positive/total samples in class 3  THEY RAPRESENT THE CLASS 3 SAMPLE WHICH ARE CORRECTLY CLASSIFIED AS CLASS 3
F1M1(1,j) = 2*precision_M1(1,j)*recall_M1(1,j) / (precision_M1(1,j) + recall_M1(1,j)) ; 

if facies(w,j) == 0
    F1M1(1,j) = 0;
elseif isnan(precision_M1(1,j)) || isnan(recall_M1(1,j))
    F1M1(1,j) = 0;  
elseif isnan(F1M1(1,j))
    F1M1(1,j) = 0;
end 
end
%average F1
F1M1avg{z,cz,w,layers,hidden} = sum(facies(w,:).* F1M1(1,:)) / sum(facies(w,:)); 
end
function [AccuracyM2_test,AccuracyM2_tot] = Accuracy_M2(AccuracyM2_test,AccuracyM2_tot,tind,yind,hidden,tr,w,layers,z,cz) 
Adj = {[2]; [1,3]; [2]; [5]; [4,6]; [5,7,8]; [6,8]; [6,7,9]; [7,8]};
%test set
FAIL = 0;
TT = tind(:,tr.testInd);
YT = yind(:,tr.testInd);
for i = 1:length(TT)
    if YT(1,i) == TT(1,i) || ismember(TT(1,i),Adj{YT(1,i),1}) 
        %NADA
    else FAIL = FAIL + 1;
        
    end
end
AccuracyM2_test{z,cz,w,layers,hidden} = 1-(FAIL / length(TT));
%tot
FAIL = 0;
for i = 1:length(tind)
    if yind(1,i) == tind(1,i) || ismember(tind(1,i),Adj{yind(1,i),1}) 
        %NADA
    else FAIL = FAIL + 1;
        
    end
end
AccuracyM2_tot{z,cz,w,layers,hidden} = 1-(FAIL / length(tind));
end
function [F1M2avg,F1M2,precision_M2,recall_M2] = F1_M2(F1M2avg,facies,w,tr,yind,tind,layers,hidden,z,cz)
Adj = {[2]; [1,3]; [2]; [5]; [4,6]; [5,7,8]; [6,8]; [6,7,9]; [7,8]};
%initialize
TrueNegative = zeros(9,length(tind));
FalseNegative = zeros(9,length(tind));
TruePositive = zeros(9,length(tind));
FalsePositive = zeros(9,length(tind));
precision_M2 =[];
recall_M2 = [];
for j = 1:9
for i = 1:length(tr.testInd)
    %positive count
    if yind(1,tr.testInd(i)) == j || ismember(yind(1,tr.testInd(i)),Adj{j,1}) 
        if yind(1,tr.testInd(i)) == tind(1,tr.testInd(i)) || ismember(tind(1,tr.testInd(i)),Adj{j,1}) 
            TruePositive(j,i) = 1;
        else 
            FalsePositive(j,i) = 1; 
        end
    else 
        if tind(1,tr.testInd(i)) == j || ismember(tind(1,tr.testInd(i)),Adj{j,1}) 
            FalseNegative(j,i) = 1;
        else 
            TrueNegative(j,i) = 1;
        end
    end
end
precision_M2(1,j) = sum(TruePositive(j,:))/(sum(TruePositive(j,:)) + sum(FalsePositive(j,:))) ; % True positive/total predicted class 3 what proportion of predicted class 3 is truly class 3? 
recall_M2(1,j) = sum(TruePositive(j,:))/(sum(TruePositive(j,:)) + sum(FalseNegative(j,:))) ;  % True positive/total samples in class 3  THEY RAPRESENT THE CLASS 3 SAMPLE WHICH ARE CORRECTLY CLASSIFIED AS CLASS 3
F1M2(1,j) = 2*precision_M2(1,j)*recall_M2(1,j) / (precision_M2(1,j) + recall_M2(1,j)) ;

if facies(w,j) == 0
    F1M2(1,j) = 0;
elseif isnan(precision_M2(1,j)) || isnan(recall_M2(1,j))
    F1M2(1,j) = 0;  
elseif isnan(F1M2(1,j))
    F1M2(1,j) = 0;
end 
end
%average F1
F1M2avg{z,cz,w,layers,hidden} = sum(facies(w,:).* F1M2(1,:)) / sum(facies(w,:)); 
end
function [AccuracyM3_test,AccuracyM3_tot] = Accuracy_M3(AccuracyM3_test,AccuracyM3_tot,tind,yind,hidden,tr,w,layers,z,cz)
PayFacies = [6,7,8,9];
%test set
PayFaciesCount = 0;
FAIL = 0;
TT = tind(:,tr.testInd);
YT = yind(:,tr.testInd);
for i = 1:length(TT)
    if ismember(TT(1,i),PayFacies)
        PayFaciesCount = PayFaciesCount + 1;
        if ismember(YT(1,i),PayFacies) 
            %NADA
        else FAIL = FAIL + 1;
        end
    end
end
AccuracyM3_test{z,cz,w,layers,hidden} = 1 - (FAIL / PayFaciesCount);
%tot set
PayFaciesCount = 0;
FAIL = 0;
for i = 1:length(tind)
    if ismember(tind(1,i),PayFacies)
        PayFaciesCount = PayFaciesCount + 1;
        if ismember(yind(1,i),PayFacies) 
            %NADA
        else FAIL = FAIL + 1;
        end
    end
end
AccuracyM3_tot{z,cz,w,layers,hidden} = 1-(FAIL / PayFaciesCount);
end
function [F1M3avg,precision_M3,recall_M3] = F1_M3(F1M3avg,w,tr,yind,tind,layers,hidden,z,cz)
PayFacies = [6,7,8,9];
%initialize
TrueNegative = zeros(1,length(tind));
FalseNegative = zeros(1,length(tind));
TruePositive = zeros(1,length(tind));
FalsePositive = zeros(1,length(tind));
precision_M3 = [];
recall_M3 = [];

for i = 1:length(tr.testInd)
    %positive count
    if ismember(yind(1,tr.testInd(i)),PayFacies) 
        if ismember(tind(1,tr.testInd(i)),PayFacies) 
            TruePositive(1,i) = 1;
        else 
            FalsePositive(1,i) = 1; 
        end
    else 
        if ismember(tind(1,tr.testInd(i)),PayFacies) 
            FalseNegative(1,i) = 1;
        else 
            TrueNegative(1,i) = 1;
        end
    end
end
% for j = PayFacies
% for i = 1:length(tr.testInd)
%     %positive count
%     if yind(1,tr.testInd(i)) == j || ismember(yind(1,tr.testInd(i)),PayFacies) 
%         if yind(1,tr.testInd(i)) == tind(1,tr.testInd(i)) || ismember(tind(1,tr.testInd(i)),PayFacies) 
%             TruePositive(j,i) = 1;
%         else 
%             FalsePositive(j,i) = 1; 
%         end
%     else 
%         if tind(1,tr.testInd(i)) == j || ismember(tind(1,tr.testInd(i)),PayFacies) 
%             FalseNegative(j,i) = 1;
%         else 
%             TrueNegative(j,i) = 1;
%         end
%     end
% end
precision_M3 = sum(TruePositive(1,:))/(sum(TruePositive(1,:)) + sum(FalsePositive(1,:))) ; % True positive/total predicted class 3 what proportion of predicted class 3 is truly class 3? 
recall_M3 = sum(TruePositive(1,:))/(sum(TruePositive(1,:)) + sum(FalseNegative(1,:))) ;  % True positive/total samples in class 3  THEY RAPRESENT THE CLASS 3 SAMPLE WHICH ARE CORRECTLY CLASSIFIED AS CLASS 3
F1M3avg{z,cz,w,layers,hidden} = 2*precision_M3*recall_M3 / (precision_M3 + recall_M3) ;
end 
function [y,yind,STOREy,STOREtr,Ystore,YINDstore] = Vote(Ystore,YINDstore,tind,STOREy,STOREtr,tr,w,layers,hidden,z,cz)
%% VOTE ON THE TEST SET
YbinaryVector = zeros(9, length(tind));
for i = 1:length(Ystore)
    YbinaryVector = YbinaryVector + Ystore{i,1} ;
end
y = YbinaryVector /length(Ystore); % FINAL Y in number
yind = vec2ind(y); % FINAL Y in binary
%save tr and y
STOREy{z,cz,w,layers,hidden} = y;
STOREtr{z,cz,w,layers,hidden} = tr;
% reset 
Ystore = {};
YINDstore = {};
end
function [y,yind,STOREy,STOREtr,Ystore,YINDstore,ETest] = VoteTheBest(ETest,Ystore,YINDstore,STOREy,STOREtr,tr,w,layers,hidden)
[MIN,IND]=min(ETest);
y = Ystore{IND,1};
yind = YINDstore{IND,1};
%save tr and y
STOREy{w,layers,hidden} = y;
STOREtr{w,layers,hidden} = tr;
%reset
Ystore = {};
YINDstore = {};
ETest = [];
end
function [] = Plotting(rangeHidden,w,layers,F1M1avg,F1M2avg,F1M3avg,AccuracyM1_test,AccuracyM2_test,AccuracyM3_test,F1,F12,Accuracy1,Accuracy2)
%plot F1
if layers == 2
    figure(F1)
    subplot(2,4,w) 
    text1 = num2str(layers-1) + "H F1-M1";
    text2 = num2str(layers-1) + "H F1-M2";
    text3 = num2str(layers-1) + "H F1-M3";
    plot(rangeHidden,F1M1avg{w,layers}(rangeHidden,1),'DisplayName',text1,'Marker','o')
    hold on
    plot(rangeHidden,F1M2avg{w,layers}(rangeHidden,1),'DisplayName',text2,'Marker','o')
    hold on
    plot(rangeHidden,F1M3avg{w,layers}(rangeHidden,1),'DisplayName',text3,'Marker','o')
    ylim([0 1]);
    xlim([1 rangeHidden(end)]);
    title("Test well "+ w); 
    ax = gca; ax.XTick =rangeHidden; ax.XScale = 'log';
    legend
    hold on
else 
    figure(F12)
    subplot(2,4,w)
    xticks(subplot(2,4,w),rangeHidden)
    set(subplot(2,4,w),'XScale','log')
    text1 = num2str(layers-1) + "H F1-M1";
    text2 = num2str(layers-1) + "H F1-M2";
    text3 = num2str(layers-1) + "H F1-M3";
    plot(rangeHidden,F1M1avg{w,layers}(rangeHidden,1),'DisplayName',text1,'Marker','o')
    hold on
    plot(rangeHidden,F1M2avg{w,layers}(rangeHidden,1),'DisplayName',text2,'Marker','o')
    hold on
    plot(rangeHidden,F1M3avg{w,layers}(rangeHidden,1),'DisplayName',text3,'Marker','o')
    ylim([0 1]);
    xlim([1 rangeHidden(end)]);
    title("Test well "+ w); 
    legend
    hold on
end
%plot Accuracy
if layers == 2
    figure(Accuracy1)
    subplot(2,4,w)
    text1 = num2str(layers-1) + "H error-M1";
    text2 = num2str(layers-1) + "H error-M2";
    text3 = num2str(layers-1) + "H error-M3";
    plot(rangeHidden,AccuracyM1_test{w,layers}(rangeHidden,1),'DisplayName',text1)
    hold on
    plot(rangeHidden,AccuracyM2_test{w,layers}(rangeHidden,1),'DisplayName',text2)
    hold on
    plot(rangeHidden,AccuracyM3_test{w,layers}(rangeHidden,1),'DisplayName',text3)
    ylim([0 1]);
    xlim([1 rangeHidden(end)]);
    title("Test well "+ w); 
    legend
    hold on
else 
    figure(Accuracy2)
    subplot(2,4,w)
    text1 = num2str(layers-1) + "H error-M1";
    text2 = num2str(layers-1) + "H error-M2";
    text3 = num2str(layers-1) + "H error-M3";
    plot(rangeHidden,AccuracyM1_test{w,layers}(rangeHidden,1),'DisplayName',text1)
    hold on
    plot(rangeHidden,AccuracyM2_test{w,layers}(rangeHidden,1),'DisplayName',text2)
    hold on
    plot(rangeHidden,AccuracyM3_test{w,layers}(rangeHidden,1),'DisplayName',text3)
    ylim([0 1]);
    xlim([1 rangeHidden(end)]);
    title("Test well "+ w); 
    legend
    hold on
end
end
function [] = ROC(RoCurve1,RoCurve2,w,t,tr,y,layers)
%% roc method 2
%define colors
blue = [0 0.4470 0.7410]; 
orange = [0.8500 0.3250 0.0980];
sand = [0.9290 0.6940 0.1250];
violet = [0.4940 0.1840 0.5560];
green = [0.4660 0.6740 0.1880];
lightblue = [0.3010 0.7450 0.9330]; 
red = [0.6350 0.0780 0.1840];
yellow = [0.9 0.88 0.1];
pink = [0.86 0.355 0.74];
colors = {blue,orange,sand,violet,green,lightblue,red,yellow,pink};
if layers == 2
    figure(RoCurve1)
    subplot(2,4,w)
    [tpr,fpr,thresholds]=roc(t(:,tr.testInd),y(:,tr.testInd));
    h = plot([0 1],[0 1],'LineStyle','--','Color','k');
    h.Annotation.LegendInformation.IconDisplayStyle = 'off';
    hold on
    for i = 1:9
        AUC = trapz(fpr{1,i},tpr{1,i});
        text = num2str(layers-1) + " H " + "class" + num2str(i) + " AUC = " + num2str(AUC);
        if AUC == 0
            plot(fpr{1,i},tpr{1,i},'LineWidth',0.001,'DisplayName',text,'Color','none')
        else
            plot(fpr{1,i},tpr{1,i},'LineWidth',2,'DisplayName',text,'Color',colors{i})
        end 
        xlabel('Specificity')
        ylabel('Recall')
        title("Test well "+ w); 
        legend 
        hold on
    end
else 
    figure(RoCurve2)
    subplot(2,4,w)
    [tpr,fpr,thresholds]=roc(t(:,tr.testInd),y(:,tr.testInd));
    h = plot([0 1],[0 1],'LineStyle','--','Color','k');
    h.Annotation.LegendInformation.IconDisplayStyle = 'off';
    hold on
    for i = 1:9
        AUC = trapz(fpr{1,i},tpr{1,i});
        text = num2str(layers-1) + " H " + "class" + num2str(i) + " AUC = " + num2str(AUC);
        if AUC == 0
            plot(fpr{1,i},tpr{1,i},'LineWidth',0.001,'DisplayName',text,'Color','none')
        else
            plot(fpr{1,i},tpr{1,i},'LineWidth',2,'DisplayName',text,'Color',colors{i})
        end 
        xlabel('Specificity')
        ylabel('Recall')
        title("Test well "+ w); 
        legend 
        hold on
    end
    hold on
end
end
function [Tab] = Bests(rangeHidden,rangeLayers,F1M1avg,F1M2avg,F1M3avg,AccuracyM1_test,AccuracyM2_test,AccuracyM3_test)
AVGF1M1 = zeros(rangeHidden(end),rangeLayers(end)) ;
AVGF1M2 = zeros(rangeHidden(end),rangeLayers(end)) ;
AVGF1M3 = zeros(rangeHidden(end),rangeLayers(end)) ;
AVGAccuracyM1 = zeros(rangeHidden(end),rangeLayers(end)) ;
AVGAccuracyM2 = zeros(rangeHidden(end),rangeLayers(end)) ;
AVGAccuracyM3 = zeros(rangeHidden(end),rangeLayers(end)) ;
for j = rangeLayers
for i = 1:7
    AVGF1M1(:,j) = AVGF1M1(:,j) + F1M1avg{i,j};
    AVGF1M2(:,j) = AVGF1M2(:,j) + F1M2avg{i,j};
    AVGF1M3(:,j) = AVGF1M3(:,j) + F1M3avg{i,j};
    AVGAccuracyM1(:,j) = AVGAccuracyM1(:,j) + AccuracyM1_test{i,j};
    AVGAccuracyM2(:,j) = AVGAccuracyM2(:,j) + AccuracyM2_test{i,j};
    AVGAccuracyM3(:,j) = AVGAccuracyM3(:,j) + AccuracyM3_test{i,j};
end
end
AVGF1M1 = AVGF1M1 /7; AVGF1M2 = AVGF1M2 /7; AVGF1M3 = AVGF1M3 /7;
AVGAccuracyM1 = AVGAccuracyM1 /7; AVGAccuracyM2 = AVGAccuracyM2 /7; AVGAccuracyM3 = AVGAccuracyM3 /7;
%best for all the metrics
BEST_F1_ALLMETRICS = (AVGF1M1+AVGF1M2+AVGF1M3)/3;
BEST_ACCURACY_ALLMETRICS = (AVGAccuracyM1+AVGAccuracyM2+AVGAccuracyM3)/3;
%indices and values
[F1M1_MAX1,F1M1_MAX1Ind]= max(AVGF1M1(rangeHidden,2)); F1M1_MAX1Ind = 2^(F1M1_MAX1Ind-1);
[F1M1_MAX2,F1M1_MAX2Ind]= max(AVGF1M1(rangeHidden,3)); F1M1_MAX2Ind = 2^(F1M1_MAX2Ind-1);

[F1M2_MAX1,F1M2_MAX1Ind]= max(AVGF1M2(rangeHidden,2)); F1M2_MAX1Ind = 2^(F1M2_MAX1Ind-1);
[F1M2_MAX2,F1M2_MAX2Ind]= max(AVGF1M2(rangeHidden,3)); F1M2_MAX2Ind = 2^(F1M2_MAX2Ind-1);

[F1M3_MAX1,F1M3_MAX1Ind]= max(AVGF1M3(rangeHidden,2)); F1M3_MAX1Ind = 2^(F1M3_MAX1Ind-1);
[F1M3_MAX2,F1M3_MAX2Ind]= max(AVGF1M3(rangeHidden,3)); F1M3_MAX2Ind = 2^(F1M3_MAX2Ind-1);

[AVGAccuracyM1_MAX1,AVGAccuracyM1_MAX1Ind]= min(AVGAccuracyM1(rangeHidden,2)); AVGAccuracyM1_MAX1Ind = 2^(AVGAccuracyM1_MAX1Ind-1);
[AVGAccuracyM1_MAX2,AVGAccuracyM1_MAX2Ind]= min(AVGAccuracyM1(rangeHidden,3)); AVGAccuracyM1_MAX2Ind = 2^(AVGAccuracyM1_MAX2Ind-1);

[AVGAccuracyM2_MAX1,AVGAccuracyM2_MAX1Ind]= min(AVGAccuracyM2(rangeHidden,2)); AVGAccuracyM2_MAX1Ind = 2^(AVGAccuracyM2_MAX1Ind-1);
[AVGAccuracyM2_MAX2,AVGAccuracyM2_MAX2Ind]= min(AVGAccuracyM2(rangeHidden,3)); AVGAccuracyM2_MAX2Ind = 2^(AVGAccuracyM2_MAX2Ind-1);

[AVGAccuracyM3_MAX1,AVGAccuracyM3_MAX1Ind]= min(AVGAccuracyM3(rangeHidden,2)); AVGAccuracyM3_MAX1Ind = 2^(AVGAccuracyM3_MAX1Ind-1);
[AVGAccuracyM3_MAX2,AVGAccuracyM3_MAX2Ind]= min(AVGAccuracyM3(rangeHidden,3)); AVGAccuracyM3_MAX2Ind = 2^(AVGAccuracyM3_MAX2Ind-1);

HiddenLayers = {'1 Hidden F1';'2 Hidden F1';'1 Hidden Accuracy';'2 Hidden Accuracy'};
M1 =[F1M1_MAX1,F1M1_MAX1Ind;F1M1_MAX2,F1M1_MAX2Ind;AVGAccuracyM1_MAX1,AVGAccuracyM1_MAX1Ind;AVGAccuracyM1_MAX2,AVGAccuracyM1_MAX2Ind] ;
M2 =[F1M2_MAX1,F1M2_MAX1Ind;F1M2_MAX2,F1M2_MAX2Ind;AVGAccuracyM2_MAX1,AVGAccuracyM2_MAX1Ind;AVGAccuracyM2_MAX2,AVGAccuracyM2_MAX2Ind] ;
M3 =[F1M3_MAX1,F1M3_MAX1Ind;F1M3_MAX2,F1M3_MAX2Ind;AVGAccuracyM3_MAX1,AVGAccuracyM3_MAX1Ind;AVGAccuracyM3_MAX2,AVGAccuracyM3_MAX2Ind] ;
Tab = table(HiddenLayers,M1,M2,M3);
end
function [F1M1avg_tot,F1M1_tot,precision_M1_tot,recall_M1_tot] = F1_M1_tot(F1M1avg_tot,facies,w,tr,yind,tind,layers,hidden,z,cz)
%initialize
TrueNegative = zeros(9,length(tind));
FalseNegative = zeros(9,length(tind));
TruePositive = zeros(9,length(tind));
FalsePositive = zeros(9,length(tind));
precision_M1_tot =[];
recall_M1_tot = [];
for j = 1:9
for i = 1:length(tr.trainInd)
    %positive count
    if yind(1,tr.trainInd(i)) == j 
        if yind(1,tr.trainInd(i)) == tind(1,tr.trainInd(i))
            TruePositive(j,i) = 1;
        else 
            FalsePositive(j,i) = 1; 
        end
    else 
        if tind(1,tr.trainInd(i)) == j
            FalseNegative(j,i) = 1;
        else 
            TrueNegative(j,i) = 1;
        end
    end
end
precision_M1_tot(1,j) = sum(TruePositive(j,:))/(sum(TruePositive(j,:)) + sum(FalsePositive(j,:))) ; % True positive/total predicted class 3 what proportion of predicted class 3 is truly class 3? 
recall_M1_tot(1,j) = sum(TruePositive(j,:))/(sum(TruePositive(j,:)) + sum(FalseNegative(j,:))) ;  % True positive/total samples in class 3  THEY RAPRESENT THE CLASS 3 SAMPLE WHICH ARE CORRECTLY CLASSIFIED AS CLASS 3
F1M1_tot(1,j) = 2*precision_M1_tot(1,j)*recall_M1_tot(1,j) / (precision_M1_tot(1,j) + recall_M1_tot(1,j)) ; 

if facies(w,j) == 0
    F1M1_tot(1,j) = 0;
elseif isnan(precision_M1_tot(1,j)) || isnan(recall_M1_tot(1,j))
    F1M1_tot(1,j) = 0;  
elseif isnan(F1M1_tot(1,j))
    F1M1_tot(1,j) = 0;
end 
end
%average F1
F1M1avg_tot{z,cz,w,layers,hidden} = sum(facies(w,:).* F1M1_tot(1,:)) / sum(facies(w,:)); 
end
function [F1M2avg_tot,F1M2_tot,precision_M2_tot,recall_M2_tot] = F1_M2_tot(F1M2avg_tot,facies,w,tr,yind,tind,layers,hidden,z,cz)
Adj = {[2]; [1,3]; [2]; [5]; [4,6]; [5,7,8]; [6,8]; [6,7,9]; [7,8]};
%initialize
TrueNegative = zeros(9,length(tind));
FalseNegative = zeros(9,length(tind));
TruePositive = zeros(9,length(tind));
FalsePositive = zeros(9,length(tind));
precision_M2_tot =[];
recall_M2_tot = [];
for j = 1:9
for i = 1:length(tr.trainInd)
    %positive count
    if yind(1,tr.trainInd(i)) == j || ismember(yind(1,tr.trainInd(i)),Adj{j,1}) 
        if yind(1,tr.trainInd(i)) == tind(1,tr.trainInd(i)) || ismember(tind(1,tr.trainInd(i)),Adj{j,1}) 
            TruePositive(j,i) = 1;
        else 
            FalsePositive(j,i) = 1; 
        end
    else 
        if tind(1,tr.trainInd(i)) == j || ismember(tind(1,tr.trainInd(i)),Adj{j,1}) 
            FalseNegative(j,i) = 1;
        else 
            TrueNegative(j,i) = 1;
        end
    end
end
precision_M2_tot(1,j) = sum(TruePositive(j,:))/(sum(TruePositive(j,:)) + sum(FalsePositive(j,:))) ; % True positive/total predicted class 3 what proportion of predicted class 3 is truly class 3? 
recall_M2_tot(1,j) = sum(TruePositive(j,:))/(sum(TruePositive(j,:)) + sum(FalseNegative(j,:))) ;  % True positive/total samples in class 3  THEY RAPRESENT THE CLASS 3 SAMPLE WHICH ARE CORRECTLY CLASSIFIED AS CLASS 3
F1M2_tot(1,j) = 2*precision_M2_tot(1,j)*recall_M2_tot(1,j) / (precision_M2_tot(1,j) + recall_M2_tot(1,j)) ;

if facies(w,j) == 0
    F1M2_tot(1,j) = 0;
elseif isnan(precision_M2_tot(1,j)) || isnan(recall_M2_tot(1,j))
    F1M2_tot(1,j) = 0;  
elseif isnan(F1M2_tot(1,j))
    F1M2_tot(1,j) = 0;
end 
end
%average F1
F1M2avg_tot{z,cz,w,layers,hidden}= sum(facies(w,:).* F1M2_tot(1,:)) / sum(facies(w,:)); 
end
function [F1M3avg_tot,precision_M3_tot,recall_M3_tot] = F1_M3_tot(F1M3avg_tot,w,tr,yind,tind,layers,hidden,z,cz)
PayFacies = [6,7,8,9];
%initialize
TrueNegative = zeros(1,length(tind));
FalseNegative = zeros(1,length(tind));
TruePositive = zeros(1,length(tind));
FalsePositive = zeros(1,length(tind));
precision_M3_tot = [];
recall_M3_tot = [];

for i = 1:length(tr.trainInd)
    %positive count
    if ismember(yind(1,tr.trainInd(i)),PayFacies) 
        if ismember(tind(1,tr.trainInd(i)),PayFacies) 
            TruePositive(1,i) = 1;
        else 
            FalsePositive(1,i) = 1; 
        end
    else 
        if ismember(tind(1,tr.trainInd(i)),PayFacies) 
            FalseNegative(1,i) = 1;
        else 
            TrueNegative(1,i) = 1;
        end
    end
end

precision_M3_tot = sum(TruePositive(1,:))/(sum(TruePositive(1,:)) + sum(FalsePositive(1,:))) ; % True positive/total predicted class 3 what proportion of predicted class 3 is truly class 3? 
recall_M3_tot = sum(TruePositive(1,:))/(sum(TruePositive(1,:)) + sum(FalseNegative(1,:))) ;  % True positive/total samples in class 3  THEY RAPRESENT THE CLASS 3 SAMPLE WHICH ARE CORRECTLY CLASSIFIED AS CLASS 3
F1M3avg_tot{z,cz,w,layers,hidden}= 2*precision_M3_tot*recall_M3_tot / (precision_M3_tot + recall_M3_tot) ;
end 

pacman::p_load(readr,autoplotly,ggplot2,plotly,tidyverse,party,lubridate, caret,dplyr)


library(e1071)
library(kknn)
library(C50)
library(FactoMineR)
library(fastAdaboost)
library(adabag)
library(reshape2)

#### doParallel ####
# Required
library(doParallel)

# Find how many cores are on your machine
detectCores() # Result = Typically 4 to 6

# Create Cluster with desired number of cores. Don't use them all! Your computer is running other processes. 
cl <- makeCluster(6)

# Register Cluster
registerDoParallel(cl)

# Confirm how many cores are now "assigned" to R and RStudio
getDoParWorkers() # Result 2 

# Stop Cluster. After performing your tasks, stop your cluster. 
stopCluster(cl)


#### 1. Loading Data ####
Iphone_smallmatrix <- read.csv("Data/Raw/iphone_smallmatrix_labeled_8d.csv")
galaxy_smallmatrix <- read.csv("Data/Raw/galaxy_smallmatrix_labeled_8d.csv")

LargeMatrix <- read.csv("Data/Raw/concatenated_factors.csv")

#### 2. Explore the Data ####

# to check all levels

library(dplyr)

Iphone_smallmatrix %>% 
  sapply(levels)

# view the variables or attributes of the matrices, and its dimension
dim(Iphone_smallmatrix)
dim(galaxy_smallmatrix)
dim(LargeMatrix)

names(Iphone_smallmatrix)
names(galaxy_smallmatrix)
names(LargeMatrix)

summary(Iphone_smallmatrix$iphonesentiment)
summary(galaxy_smallmatrix$galaxysentiment)
summary(LargeMatrix$ios)

summary(Iphone_smallmatrix$ios) #counts mentions of iOS on a webpage
summary(Iphone_smallmatrix$iphonecampos) # counts positive sentiment mentions of the iphone camera
summary(galaxy_smallmatrix$galaxysentiment) # there is nothing 

# distribution of the sentiments
plot_ly(Iphone_smallmatrix, x= ~Iphone_smallmatrix$iphonesentiment, type='histogram')
plot_ly(galaxy_smallmatrix, x= ~galaxy_smallmatrix$galaxysentiment, type='histogram')

plot(Iphone_smallmatrix$iphonesentiment, ylab = "Sentiment", main = "Iphone Sentiment")
plot(galaxy_smallmatrix$galaxysentiment, ylab = "Sentiment", main = "Galaxy Sentiment")


hist(Iphone_smallmatrix$iphonesentiment, xlim = c(0,5), ylim = c(0,2000), breaks = 100, 
     xlab = "Iphone sentiment", main="Histogram of Iphone Sentiment")





# exclude the instances with no information

Iphone_smallmatrix$Mean_row<-rowMeans(Iphone_smallmatrix, na.rm = TRUE)
summary(Iphone_smallmatrix$rowMeans)
class
nuevoIphone<-filter(Iphone_smallmatrix, Mean_row !=0)
nuevoIphone$Mean_row<-NULL





#### Preprocessing and feature selection ####

#### Examine Correlation - Correlation matrix ####
require(corrplot)
cor(Iphone_smallmatrix)
corrplot(cor(Iphone_smallmatrix), order = "hclust")

corrplot(cor(galaxy_smallmatrix), order = "hclust")

# control a wide variety of global options.
options(max.print=1000000)



# After identifying features for removal, create a new data set if needed.
# If there are no highly correlated features with the dependant, move on to Feature Variance. 

# create a new data set and remove features highly correlated with the dependant 
# iphoneCOR <- iphoneDF
# iphoneCOR$featureToRemove <- NULL

#### Examine Feature Variance ####

# The distribution of values within a feature is related to how much information that feature holds
# in the data set. Features with no variance can be said to hold little to no information.
# Features that have very little, or "near zero variance", may or may not have useful information. 
# To explore feature variance we can use nearZeroVar() from the caret package. 

#nearZeroVar() with saveMetrics = TRUE returns an object containing a table including:
# frequency ratio, percentage unique, zero variance and near zero variance 

nzvMetrics <- nearZeroVar(Iphone_smallmatrix, saveMetrics = TRUE)
nzvMetrics
OnlyZeroVar  <- nzvMetrics %>%  filter(zeroVar==TRUE)


# Let’s use nearZeroVar() again to create an index of near zero variance features. 
# The index will allow us to quickly remove features.

# nearZeroVar() with saveMetrics = FALSE returns an vector 
nzv <- nearZeroVar(Iphone_smallmatrix, saveMetrics = FALSE) 
nzv

# After identifying features for removal, create a new data set.
# create a new data set and remove near zero variance features
iphoneNZV <- Iphone_smallmatrix[,-nzv]
str(iphoneNZV)

#### Recursive Feature Elimination ####

# Let's sample the data before using RFE
set.seed(123)
iphoneSample <- Iphone_smallmatrix[sample(1:nrow(Iphone_smallmatrix), 1000, replace=FALSE),]

# Set up rfeControl with randomforest, repeated cross validation and no updates
ctrl <- rfeControl(functions = rfFuncs, 
                   method = "repeatedcv",
                   repeats = 5,
                   verbose = FALSE)

# Use rfe and omit the response variable (attribute 59 iphonesentiment) 
rfeResults <- rfe(iphoneSample[,1:58], 
                  iphoneSample$iphonesentiment, 
                  sizes=(1:58), 
                  rfeControl=ctrl)

# Get results
rfeResults

saveRDS(rfeResults, file= "rfeResults.rds")
rfeResults <- readRDS("Models/rfeResults.rds", refhook = NULL)

# Plot results
plot(rfeResults, type=c("g", "o"))

# with 18 features RMSE is the lowest - 1.324,
# while, using all 58 featurs will give us RMSE=1.329.
# The difference is not very high, but it will impact computational cost for running model

# Create dataset contianing only important feature accordint to the rfeResults
importance <- varImp(rfeResults, scale=TRUE)
importance['feature'] <- row.names(importance)
importance <- importance[,c(2,1)]
rownames(importance) <- NULL

# Plot important features
ggplot(data=importance, aes(x=reorder(feature, -Overall), y=Overall, fill=feature)) +
geom_bar(stat="identity")+ theme_classic()+theme(axis.text.x = element_text(angle=60, hjust=1))

# After identifying features for removal, create a new data set and add the dependant variable.  
# create new data set with rfe recommended features
iphoneRFE <- Iphone_smallmatrix[,predictors(rfeResults)]

# add the dependent variable to iphoneRFE
iphoneRFE$iphonesentiment <- Iphone_smallmatrix$iphonesentiment

# review outcome
str(iphoneRFE)


#### Repeat the same steps for galaxy_smallmatrix ####

# Recursive Feature Elimination #

# Let's sample the data before using RFE
set.seed(123)
galaxySample <- galaxy_smallmatrix[sample(1:nrow(galaxy_smallmatrix), 1000, replace=FALSE),]

# Set up rfeControl with randomforest, repeated cross validation and no updates
ctrl_galaxy <- rfeControl(functions = rfFuncs, 
                   method = "repeatedcv",
                   repeats = 5,
                   verbose = FALSE)

# Use rfe and omit the response variable (attribute 59 iphonesentiment) 
rfeResults_galaxy <- rfe(galaxySample[,1:58], 
                         galaxySample$galaxysentiment, 
                  sizes=(1:58), 
                  rfeControl=ctrl)

# Get results
rfeResults_galaxy
saveRDS(rfeResults_galaxy, file= "rfeResults_galaxy.rds")

# Plot results
plot(rfeResults_galaxy, type=c("g", "o"))

# with 13 features RMSE is the lowest - 1.244,
# while, using all 58 featurs will give us RMSE=1.252.


# Create dataset contianing only important feature accordint to the rfeResults
importance_galaxy <- varImp(rfeResults_galaxy, scale=TRUE)
importance_galaxy['feature'] <- row.names(importance_galaxy)
importance_galaxy <- importance_galaxy[,c(2,1)]
rownames(importance_galaxy) <- NULL

# Plot important features
ggplot(data=importance_galaxy, aes(x=reorder(feature, -Overall), y=Overall, fill=feature)) +
  geom_bar(stat="identity")+ theme_classic()+theme(axis.text.x = element_text(angle=60, hjust=1))


rfeResults_galaxy
# After identifying features for removal, create a new data set and add the dependant variable.  
# create new data set with rfe recommended features
galaxyRFE <- galaxy_smallmatrix[,predictors(rfeResults_galaxy)]

# add the dependent variable to iphoneRFE
galaxyRFE$galaxysentiment <- galaxy_smallmatrix$galaxysentiment

# review outcome
str(galaxyRFE)

#### Model development and evaluation #### 

#### ********** Down Sampling with function downSample  (combined_1)  **********  ####
# Lets downSample  a data set so that all classes have the same frequency as the minority class.

Iphone_smallmatrix_down <- downSample(Iphone_smallmatrix, Iphone_smallmatrix$iphonesentiment, list = FALSE, yname = "Class")
# distribution of the sentiments after downsmapling
plot_ly(Iphone_smallmatrix_down, x= ~Iphone_smallmatrix_down$Class, type='histogram')


# Data partition for training & testing sets #
Iphone_smallmatrix_down$iphonesentiment <- as.factor(Iphone_smallmatrix_down$iphonesentiment)

set.seed(122)
inTraining <- createDataPartition(Iphone_smallmatrix_down$iphonesentiment, p = .7, list = FALSE)
trainSet <- Iphone_smallmatrix_down[inTraining,]
testSet <- Iphone_smallmatrix_down[-inTraining,]

fitControl <- trainControl(method = "cv", number=3, verboseIter = T, returnResamp = "all",savePredictions = TRUE)
#Testing models with loop----

combined <- c()
# models <- c("kknn", "rf","C5.0","svmRadial","gbm","pcaNNet")

models <- c("kknn", "rf","C5.0","svmRadial","gbm","pcaNNet")
t_0 <- proc.time()
for(i in models) {
  
  Fit_1 <- train(iphonesentiment~.,
               data= trainSet,
               method = i, 
               trControl = fitControl,
               tuneLength = 5, 
               preProc = "zv",
              na.action = na.omit)
  
    pred <- predict(Fit_1,testSet)
    res <- postResample(pred,testSet$iphonesentiment)
    combined <- cbind(combined, res) 
}

t_1 <- proc.time()
time_knn <- t_1-t_0
print(time_knn/60)

colnames(combined) <- models
combined
combined_1 <- combined
combined_1 <- as.data.frame(combined_1)
combined_1$dataset <- "downsample func"

saveRDS(Fit_1, file= "Predicting model for downsampled dataset.rds")

# stored performance metrics
colnames(combined) <- models
combined_1 <- combined
combined_1 <- as.data.frame(combined_1)

# stored predicted values
colnames(combined_pred) <- models
combined_pred_1 <-combined_pred 
combined_pred_1 <-as.data.frame(combined_pred_1)

#### SVM downsample manual -1  ####

model_svm <- svm(iphonesentiment~., data = trainSet, na.action =na.omit,scale = FALSE)
pred_svm_1 <- predict(model_svm,testSet)
svm_1 <- postResample(pred_svm_1,testSet$iphonesentiment)

# Adding results of svm funcation to the pool of the previous models
combined_1$svm <- svm_1
combined_1 <- combined_1[,c(ncol(combined_1), 1:(ncol(combined_1)-1))]
combined_1$dataset <- "downsample func"
combined_pred_1$svm<- pred_svm_1

# Converting values from integer to factor and adjusting levels of values according to testSet
for(i in 1:ncol(combined_pred_1)) {
  combined_pred_1[,i] <- as.factor(combined_pred_1[,i])
  levels(combined_pred_1[,i]) <- c("0", "1", "2","3", "4", "5")
  
}

# Print confusion matrix for all the models
for(i in 1:ncol(combined_pred_1)) {
  cmRF <- confusionMatrix(combined_pred_1[,i], testSet$iphonesentiment)
  print(cmRF$overall)
  print(colnames(combined_pred_1)[i])
  print(cmRF)
}

# get accuracy and cappa for testing set
check1 <- data.frame (Error="", value="",Model="",dataset="")
check1[1:3] <- factor(check1[1:3])  
 
for(i in 1:ncol(combined_pred_1)) {
  cmRF <- confusionMatrix(combined_pred_1[,i], testSet$iphonesentiment)
  
  check<- cmRF$overall
  check <- as.data.frame(check)
  check['Error'] <- row.names(check)
  colnames(check)[1]<-"value"
  check$value<- round(check$value, digits = 2)
  check <- check[1:2,]
  check <- check[,2:1]
  check$Model <- colnames(combined_pred_1)[i]
  check$dataset <- "downsample func"
  
  check1<- rbind(check1,check)
}

check1 <- check1[-1,]
print(check1)

ggplot(check1, aes(x=Model, y=value,fill=Model))+
  geom_col()+ggtitle("Error metrics")+theme(plot.title = element_text(hjust = 0.5))+
  facet_wrap(Error ~ ., scales="fixed")

# check<- cmRF$overall
# check <- as.data.frame(check)
# check['Error'] <- row.names(check)
# colnames(check)[1]<-"value"
# check <- check[1:2,]
# check <- check[,2:1]
# check$model <- "svm"

#### ********** Manual downsampling (combined_2) ********** ####

Iphone_smallmatrix_down2 <- Iphone_smallmatrix %>% group_by(iphonesentiment) %>% sample_n(390)
Iphone_smallmatrix_down2$iphonesentiment <- as.factor(Iphone_smallmatrix_down2$iphonesentiment)

set.seed(122)
inTraining <- createDataPartition(Iphone_smallmatrix_down2$iphonesentiment, p = .7, list = FALSE)
trainSet <- Iphone_smallmatrix_down2[inTraining,]
testSet <- Iphone_smallmatrix_down2[-inTraining,]

fitControl <- trainControl(method = "cv", number=3, verboseIter = T, returnResamp = "all",savePredictions = TRUE)
#Testing models with loop----

# distribution of the sentiments after manual downsmapling
plot_ly(Iphone_smallmatrix_down2, x= ~Iphone_smallmatrix_down2$iphonesentiment, type='histogram')

combined <- c()
combined_pred <- c()
models <- c("kknn", "rf","C5.0","svmRadial","gbm","pcaNNet")
t_0 <- proc.time()
for(i in models) {
  
  Fit_2 <- train(iphonesentiment~.,
               data= trainSet,
               method = i, 
               trControl = fitControl,
               tuneLength = 5, 
               preProc = "zv",
               na.action = na.omit)
  
  pred <- predict(Fit_2,testSet)
  res <- postResample(pred,testSet$iphonesentiment)
  combined_pred <- cbind(combined_pred, pred) 
  combined <- cbind(combined, res) 
}

t_1 <- proc.time()
time_knn <- t_1-t_0
print(time_knn/60)

saveRDS(Fit_2, file= "Predicting model for manually downsampled dataset.rds")

# stored performance metrics
colnames(combined) <- models
combined_2 <- combined
combined_2 <- as.data.frame(combined_2)

# stored predicted values
colnames(combined_pred) <- models
combined_pred_2 <-combined_pred 
combined_pred_2 <-as.data.frame(combined_pred_2)

#### SVM downsample manual -2 ####

model_svm <- svm(iphonesentiment~., data = trainSet, na.action =na.omit,scale = FALSE)

pred_svm_2 <- predict(model_svm,testSet)
svm_2 <- postResample(pred_svm_2,testSet$iphonesentiment)

# Adding results of svm funcation to the pool of the previous models
combined_2$svm <- svm_2
combined_2 <- combined_2[,c(ncol(combined_2), 1:(ncol(combined_2)-1))]
combined_2$dataset <- "downsample manual"
combined_pred_2$svm<- pred_svm_2

# Converting values from integer to factor and adjusting levels of values according to testSet
for(i in 1:ncol(combined_pred_2)) {
  combined_pred_2[,i] <- as.factor(combined_pred_2[,i])
  levels(combined_pred_2[,i]) <- c("0", "1", "2","3", "4", "5")
  
}

# Print confusion matrix for all the models
for(i in 1:ncol(combined_pred_2)) {
  cmRF <- confusionMatrix(combined_pred_2[,i], testSet$iphonesentiment)
  print(colnames(combined_pred_2)[i])
  print(cmRF)
}

# get accuracy and cappa for testing set
check2 <- data.frame (Error="", value="",Model="",dataset="")
check2[1:3] <- factor(check2[1:3])  

for(i in 1:ncol(combined_pred_2)) {
  cmRF <- confusionMatrix(combined_pred_2[,i], testSet$iphonesentiment)
  
  check<- cmRF$overall
  check <- as.data.frame(check)
  check['Error'] <- row.names(check)
  colnames(check)[1]<-"value"
  check$value<- round(check$value, digits = 2)
  check <- check[1:2,]
  check <- check[,2:1]
  check$Model <- colnames(combined_pred_1)[i]
  check$dataset <- "downsample manual"
  
  check2<- rbind(check2,check)
}

check2 <- check2[-1,]
print(check2)


####  ********** TRAINING MODELS FOR DATASET WITH Recursive Feature Eliminated (RFE) (combined_3) ####

#### Data partition for training & testing sets ####
iphoneRFE$iphonesentiment <- as.factor(iphoneRFE$iphonesentiment)
plot_ly(iphoneRFE, x= ~iphoneRFE$iphonesentiment, type='histogram')

iphoneRFE_down_m <- iphoneRFE %>% group_by(iphonesentiment) %>% sample_n(390)

set.seed(122)

inTraining <- createDataPartition(iphoneRFE_down_m$iphonesentiment, p = .7, list = FALSE)
trainSet <- iphoneRFE_down_m[inTraining,]
testSet <- iphoneRFE_down_m[-inTraining,]

fitControl <- trainControl(method = "cv", number=3, verboseIter = T, returnResamp = "all",savePredictions = TRUE)

#Testing models with loop----

combined <- c()
combined_pred <- c()
models <- c("kknn", "rf","C5.0","svmRadial","gbm","pcaNNet")
t_0 <- proc.time()
for(i in models) {
  
  Fit_3<- train(iphonesentiment~.,
              data= trainSet,
              method = i, 
              trControl = fitControl,
              tuneLength = 5, 
              preProc = "zv",
              na.action = na.omit)
  
  pred <- predict(Fit_3,testSet)
  res <- postResample(pred,testSet$iphonesentiment)
  combined_pred <- cbind(combined_pred, pred) 
  combined <- cbind(combined, res) 
}

t_1 <- proc.time()
time_knn <- t_1-t_0
print(time_knn/60)

saveRDS(Fit_3, file= "Predicting model for RFE & manually downsampled dataset.rds")

# stored performance metrics
colnames(combined) <- models
combined_3 <- combined
combined_3 <- as.data.frame(combined_3)

# stored predicted values
colnames(combined_pred) <- models
combined_pred_3 <-combined_pred 
combined_pred_3 <-as.data.frame(combined_pred_3)


#### SVM downsample manual - 3 ####

model_svm <- svm(iphonesentiment~., data = trainSet, na.action =na.omit,scale = FALSE)
summary (model_svm)

pred_svm_3 <- predict(model_svm,testSet)
svm_3 <- postResample(pred_svm_3,testSet$iphonesentiment)

# Adding results of svm funcation to the pool of the previous models
combined_3$svm <- svm_3
combined_3 <- combined_3[,c(ncol(combined_3), 1:(ncol(combined_3)-1))]
combined_3$dataset <- "RFE & downsample manual"
combined_pred_3$svm<- pred_svm_3

# Converting values from integer to factor and adjusting levels of values according to testSet
for(i in 1:ncol(combined_pred_3)) {
  combined_pred_3[,i] <- as.factor(combined_pred_3[,i])
  levels(combined_pred_3[,i]) <- c("0", "1", "2","3", "4", "5")
  
}

# Print confusion matrix for all the models
for(i in 1:ncol(combined_pred_3)) {
  cmRF <- confusionMatrix(combined_pred_3[,i], testSet$iphonesentiment)
  print(colnames(combined_pred_3)[i])
  print(cmRF$table)
}

# get accuracy and cappa for testing set
check3 <- data.frame (Error="", value="",Model="",dataset="")
check3[1:3] <- factor(check3[1:3])  

for(i in 1:ncol(combined_pred_1)) {
  cmRF <- confusionMatrix(combined_pred_3[,i], testSet$iphonesentiment)
  
  check<- cmRF$overall
  check <- as.data.frame(check)
  check['Error'] <- row.names(check)
  colnames(check)[1]<-"value"
  check$value<- round(check$value, digits = 2)
  check <- check[1:2,]
  check <- check[,2:1]
  check$Model <- colnames(combined_pred_1)[i]
  check$dataset <- "RFE & downsample manual"
  
  check3<- rbind(check3,check)
}

check3 <- check3[-1,]
print(check3)

####  Melt function to store the errors ####
combined_all <-rbind(combined_1,combined_2,combined_3) 
combined_all['Error'] <- row.names(combined_all)
combined_all

combined_melt <- melt(combined_all)
colnames(combined_melt)[1]<-"dataset"
colnames(combined_melt)[3]<-"Model"

ggplot(combined_melt, aes(x=Model, y=value,fill=Model))+
  geom_col()+ggtitle("Error metrics")+theme(plot.title = element_text(hjust = 0.5))+
  facet_wrap(Error ~ dataset, scales="fixed")

check_all <-rbind(check1,check2,check3)

ggplot(check_all, aes(x=Model, y=value,fill=Model))+
  geom_col()+ggtitle("Error metrics for testing set")+theme(plot.title = element_text(hjust = 0.5))+
  facet_wrap(Error ~ dataset, scales="fixed")

#### Engineering the Dependant variable ####

# Perhaps combining some of these levels will help increase accuracy and kappa. 
# The dplyr package () function can help us with this. 

# create a new dataset that will be used for recoding sentiment
Iphone_smallmatrix_RC <- Iphone_smallmatrix

# recode sentiment to combine factor levels 0 & 1 and 4 & 5
Iphone_smallmatrix_RC <- recode(Iphone_smallmatrix$iphonesentiment, '0' = 1, '1' = 1, '2' = 2, '3' = 3, '4' = 4, '5' = 4) 

# inspect results
summary(Iphone_smallmatrix_RC)
str(Iphone_smallmatrix_RC)
# make iphonesentiment a factor
Iphone_smallmatrix_RC$iphonesentiment <- as.factor(Iphone_smallmatrix_RC$iphonesentiment)


# Now model using your best learner. 
# Did accuracy and kappa increase? Did you review the confusion matrix? 


#### Principal Component Analysis ####
# Principal Component Analysis (PCA) is a form of feature engineering that removes all of your features
# and replaces them with mathematical representations of their variance.

# data = training and testing from Iphone_smallmatrix (no feature selection) 
# create object containing centered, scaled PCA components from training set
# excluded the dependent variable and set threshold to .95

# Data partition for training & testing sets #
Iphone_smallmatrix$iphonesentiment <- as.factor(Iphone_smallmatrix$iphonesentiment)

set.seed(122)
inTraining <- createDataPartition(Iphone_smallmatrix_down$iphonesentiment, p = .7, list = FALSE)
trainSet <- Iphone_smallmatrix_down[inTraining,]
testSet <- Iphone_smallmatrix_down[-inTraining,]
preprocessParams <- preProcess(trainSet[,-59], method=c("center", "scale", "pca"), thresh = 0.95)
print(preprocessParams)

# Examine the output. How many components were needed to capture 95% of the variance?
# If you lower the variance threshold does the number of components stay the same?
  
# We now need to apply the PCA model, create training/testing and add the dependant variable.

# use predict to apply pca parameters, create training, exclude dependant
train.pca <- predict(preprocessParams, trainSet[,-59])

# add the dependent to training
train.pca$iphonesentiment <- trainSet$iphonesentiment

# use predict to apply pca parameters, create testing, exclude dependant
test.pca <- predict(preprocessParams, testSet[,-59])

# add the dependent to training
test.pca$iphonesentiment <- testSet$iphonesentiment

# inspect results
str(train.pca)
str(test.pca)


fitControl <- trainControl(method = "cv", number=3, verboseIter = T, returnResamp = "all",savePredictions = TRUE)
#Testing models with loop----

combined <- c()
# models <- c("kknn", "rf","C5.0","svmRadial","gbm","pcaNNet")

models <- c("kknn", "rf","C5.0","svmRadial","gbm","pcaNNet")
t_0 <- proc.time()
for(i in models) {
  
  Fit_4 <- train(iphonesentiment~.,
                 data= train.pca,
                 method = i, 
                 trControl = fitControl,
                 tuneLength = 5, 
                 preProc = "zv",
                 na.action = na.omit)
  
  pred <- predict(Fit_4,test.pca)
  res <- postResample(pred,test.pca$iphonesentiment)
  combined <- cbind(combined, res) 
}

t_1 <- proc.time()
time_knn <- t_1-t_0
print(time_knn/60)

colnames(combined) <- models
combined
combined_4 <- combined
combined_4 <- as.data.frame(combined_4)
combined_4$dataset <- "pca"

saveRDS(Fit_4, file= "Predicting model pca dataset.rds")

# stored performance metrics
colnames(combined) <- models
combined_4 <- combined
combined_4 <- as.data.frame(combined_4)

# stored predicted values
colnames(combined_pred) <- models
combined_pred_4 <-combined_pred 
combined_pred_4 <-as.data.frame(combined_pred_4)

#### SVM downsample manual -4  ####

model_svm <- svm(iphonesentiment~., data = train.pca, na.action =na.omit,scale = FALSE)
pred_svm_4 <- predict(model_svm,test.pca)
svm_4 <- postResample(pred_svm_4,test.pca$iphonesentiment)

# Adding results of svm funcation to the pool of the previous models
combined_4$svm <- svm_4
combined_4 <- combined_1[,c(ncol(combined_4), 1:(ncol(combined_4)-1))]
combined_4$dataset <- "downsample manual"
combined_pred_1$svm<- pred_svm_4

# Converting values from integer to factor and adjusting levels of values according to test.pca
for(i in 1:ncol(combined_pred_4)) {
  combined_pred_1[,i] <- as.factor(combined_pred_4[,i])
  levels(combined_pred_4[,i]) <- c("0", "1", "2","3", "4", "5")
  
}

# Print confusion matrix for all the models
for(i in 1:ncol(combined_pred_4)) {
  cmRF <- confusionMatrix(combined_pred_4[,i], test.pca$iphonesentiment)
  print(cmRF$overall)
  print(colnames(combined_pred_4)[i])
  print(cmRF)
}


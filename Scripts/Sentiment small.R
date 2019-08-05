pacman::p_load(readr,autoplotly,ggplot2,plotly,tidyverse,party,lubridate, caret,dplyr)

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
plot_ly(galaxy_smallmatrix, x= ~Iphone_smallmatrix$iphonesentiment, type='histogram')
plot_ly(Iphone_smallmatrix, x= ~galaxy_smallmatrix$galaxysentiment, type='histogram')

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

nzvMetrics <- ?(Iphone_smallmatrix, saveMetrics = TRUE)
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

#### Data partition for training & testing sets ####
Iphone_smallmatrix$iphonesentiment <- as.factor(Iphone_smallmatrix$iphonesentiment)

set.seed(122)

# Clean_data_strat_all_B<- select(Clean_data_strat,   -LATITUDE, -LONGITUDE)

inTraining <- createDataPartition(Iphone_smallmatrix$iphonesentiment, p = .7, list = FALSE)
trainSet <- Iphone_smallmatrix[inTraining,]
testSet <- Iphone_smallmatrix[-inTraining,]



fitControl <- trainControl(method = "cv", number=2, verboseIter = T, sampling = "up")
#Testing models with loop----

library(e1071)
library(kknn)
library(C50)
library(FactoMineR)

combined <- c()
# models <- c("C5.0", "kknn", "rf", "svm", "kknn")

models <- c("kknn", "rf","C5.0","gbm","adaboost")
t_0 <- proc.time()
for(i in models) {
  
  Fit <- train(iphonesentiment~.,
               data= trainSet,
               method = i, 
               trControl = fitControl,
               tuneLength = 8, 
               preProcess = c("center","scale"),
               na.action = na.omit)
  
  
  pred <- predict(Fit,testSet)
  
  res <- postResample(pred,testSet$iphonesentiment)
  
  combined <- cbind(combined, res) 
  
}
t_1 <- proc.time()
time_knn <- t_1-t_0
print(time_knn/60)

colnames(combined) <- models

combined
#### SVM  ####


model_svm <- svm(iphonesentiment~., data = trainSet)
summary (model_svm)

pred <- predict(model_svm,testSet)
confusionMatrix(pred,testSet$iphonesentiment)
postResample(pred,testSet$iphonesentiment)



# :::::::::::::::::::::
for(i in models) {
  
  if (i == "kknn") {
    Fit_KNNN<- train(iphonesentiment~.,
                 data= trainSet,
                 method = i, 
                 trControl = fitControl,
                 tuneLength = 8, 
                 na.action = na.omit)
  }
  
  if (i == "kknn") {
    Fit_KNNN1<- train(iphonesentiment~.,
                     data= trainSet,
                     method = i, 
                     trControl = fitControl,
                     tuneLength = 8, 
                     na.action = na.omit)
  }
  
  
  pred <- predict(Fit,testSet)
  
  res <- postResample(pred,testSet$iphonesentiment)
  
  combined <- cbind(combined, res) 
  
}

# :::::::::::::::::::::


#Melt function to store the errors
require(reshape2)
compare_model_melt <- melt(combined, varnames = c("metric", "model"))
compare_model_melt <- as_data_frame(compare_model_melt)
compare_model_melt

#Plot errors with ggplot----

ggplot(compare_model_melt, aes(x=model, y=value, fill=model))+
  geom_col()+
  facet_grid(metric~., scales="free")

str(compare_model_melt)

#Run model RF----
Fit_RF_iphone <- train(iphonesentiment~.,
                data= trainSet,
                method = "rf", 
                trControl = fitControl,
                tuneLength = 4, 
                preProcess = c("center","scale"),
                na.action = na.omit
)

varImp(Fit_RF_iphone)
postResample(Fit_RF_iphone)
pred_RF_iphone <- predict(Fit_RF_iphone,testSet)

testSet$predict_iphonesentiment_RF<-pred_RF_iphone 
testSet$error_RF <- testSet$iphonesentiment - testSet$predict_iphonesentiment_RF
testSet$error_abs_RF <- abs(testSet$iphonesentiment - testSet$predict_iphonesentiment_RF)


ggplot(testSet, aes(x=iphonesentiment, y=error_abs_RF, col=iphonesentiment)) + 
geom_point() +geom_smooth() 


#### Engineering the Dependant variable ####

# Perhaps combining some of these levels will help increase accuracy and kappa. 
# The dplyr package recode() function can help us with this. 

# create a new dataset that will be used for recoding sentiment
Iphone_smallmatrix_RC <- Iphone_smallmatrix
# recode sentiment to combine factor levels 0 & 1 and 4 & 5
Iphone_smallmatrix_RC <- recode(Iphone_smallmatrix$iphonesentiment, '0' = 1, '1' = 1, '2' = 2, '3' = 3, '4' = 4, '5' = 4) 
# inspect results
summary(Iphone_smallmatrix_RC)
str(Iphone_smallmatrix_RC)
# make iphonesentiment a factor
Iphone_smallmatrix_RC$iphonesentiment <- as.factor(Iphone_smallmatrix$iphonesentiment)


# Now model using your best learner. 
# Did accuracy and kappa increase? Did you review the confusion matrix? 


# Principal Component Analysis 
# Principal Component Analysis (PCA) is a form of feature engineering that removes all of your features
# and replaces them with mathematical representations of their variance.

# data = training and testing from Iphone_smallmatrix (no feature selection) 
# create object containing centered, scaled PCA components from training set
# excluded the dependent variable and set threshold to .95

preprocessParams <- preProcess(trainSet[,-59], method=c("center", "scale", "pca"), thresh = 0.95)
print(preprocessParams)

# Examine the output. How many components were needed to capture 95% of the variance?
# If you lower the variance threshold does the number of components stay the same?
  
# We now need to apply the PCA model, create training/testing and add the dependant variable.

# use predict to apply pca parameters, create training, exclude dependant
train.pca <- predict(preprocessParams, training[,-59])

# add the dependent to training
train.pca$iphonesentiment <- training$iphonesentiment

# use predict to apply pca parameters, create testing, exclude dependant
test.pca <- predict(preprocessParams, testing[,-59])

# add the dependent to training
test.pca$iphonesentiment <- testing$iphonesentiment

# inspect results
str(train.pca)
str(test.pca)

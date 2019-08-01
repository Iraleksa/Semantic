pacman::p_load(readr,autoplotly,ggplot2,plotly,tidyverse,party,lubridate, caret,dplyr)

#### 1. Loading Data ####
Iphone_smallmatrix <- read.csv("Data/Raw/iphone_smallmatrix_labeled_8d.csv")
galaxy_smallmatrix <- read.csv("Data/Raw/galaxy_smallmatrix_labeled_8d.csv")

LargeMatrix <- read.csv("Data/Raw/concatenated_factors.csv")

#### 2. Explore the Data ####
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

#### Examine Feature Variance ####

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


# After identifying features for removal, create a new data set and add the dependant variable.  

# create new data set with rfe recommended features
iphoneRFE <- Iphone_smallmatrix[,predictors(rfeResults)]

# add the dependent variable to iphoneRFE
iphoneRFE$iphonesentiment <- Iphone_smallmatrix$iphonesentiment

# review outcome
str(iphoneRFE)



#### Preprocessing and feature selection ####

# Examine Correlation - Correlation matrix
require(corrplot)
cor(Iphone_smallmatrix)
corrplot(cor(Iphone_smallmatrix), order = "hclust")

corrplot(cor(galaxy_smallmatrix), order = "hclust")

# control a wide variety of global options.
options(max.print=1000000)



# After identifying features for removal, create a new data set if needed. If there are no highly correlated features with the dependant, move on to Feature Variance. 

# create a new data set and remove features highly correlated with the dependant 
# iphoneCOR <- iphoneDF
# iphoneCOR$featureToRemove <- NULL

# Examine Feature Variance

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

#### doParallel ####
# Required
library(doParallel)

# Find how many cores are on your machine
detectCores() # Result = Typically 4 to 6

# Create Cluster with desired number of cores. Don't use them all! Your computer is running other processes. 
cl <- makeCluster(4)

# Register Cluster
registerDoParallel(cl)

# Confirm how many cores are now "assigned" to R and RStudio
getDoParWorkers() # Result 2 

# Stop Cluster. After performing your tasks, stop your cluster. 
stopCluster(cl)





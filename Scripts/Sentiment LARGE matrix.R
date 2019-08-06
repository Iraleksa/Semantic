pacman::p_load(readr,autoplotly,ggplot2,plotly,tidyverse,party,lubridate, caret,dplyr,esquisse)


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
Kosta <- read.csv("Data/Created/concatenated_factors_1-200_kosta.csv")
# Toni <- read.csv("Data/Created/")
Irina <- read.csv("Data/Created/concatenated_factors_irina.csv")
# David <- read.csv("Data/Created/")
Precious <- read.csv("Data/Created/concatenated_factors_precious.csv")
Ivan <- read.csv("Data/Created/IVAN_LargeMatrix_1001_1200.csv",header = TRUE, sep = ";")

LargeMatrix <- rbind(Kosta,Irina,Precious,Ivan)

# esquisse:: esquisser()
summary(LargeMatrix)

Fit_4 <- readRDS("Models/Winner/Predicting model pca dataset.rds", refhook = NULL)
pred <- predict(Fit_4,LargeMatrix)

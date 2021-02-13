#Case study
#-------------------------------------------------------------
#Supplementary to the paper "Mapping the groundwater level and soil moisture of a montane peat bog using UAV monitoring and machine learning"  
#Authors: Theodora Lendzioch, Jakub Langhammer, Lukas Vlcek, and Robert Minarik
#Date: Feb 14, 2021"
#-------------------------------------------------------------
#load packages
library(corrplot)
library(PerformanceAnalytics)
#-------------------------------------------------------------
#The use of forward feature selection (FFS) and the leave-location-out (LLO) spatial cross-validation (CV) is time consuming using 
#many input variables. 
#It is helpful to reduce the number of predictor variables to a suitable level, but it is not necessary. 

#set file directory
setwd("~/input_data")
#input file 
pred<-read.csv("data.csv", header = TRUE)#create a folder with all csv files

#script example how to reduce irrelevant data due to high correlation, NANs, and near-zero
#function to wrap original corrplot::cor.mtest for keeping names and structure
#significance test, which produces p-values and confidence intervals for each pair of input feature

#take the input file of all training data derived either from raster stack extraction and/or field data sampling
trDF <- pred
# check names
names(pred)
# and drop IDs which are not necessary
drops <- c("Tube","Easting", "Northing", "Z", "Area", "Date", "Veg","GWL", "SM" )
trDF<-trDF[ , !(names(trDF) %in% drops)]
# then check on complete cases
clean_trDF<-trDF[complete.cases(trDF), ]

# have a look on the results
# now you may check on Zero- and Near-Zero-Variance Predictors
nearZeroVar(clean_trDF, saveMetrics= TRUE)

# removing descriptors with absolute correlations above 0.75
corclean_trDF <-  cor(clean_trDF)
highCorr <- sum(abs(corclean_trDF[upper.tri(corclean_trDF)]) > .999)
summary(corclean_trDF[upper.tri(corclean_trDF)])

highlyCorDescr <- findCorrelation(corclean_trDF, cutoff = .75)
filtered_trainDF <- as.data.frame(corclean_trDF[,-highlyCorDescr])

#check correlations
descrCor2 <- cor(filtered_trainDF)
summary(descrCor2[upper.tri(descrCor2)])

# now perform the significance test 
p.mat <- cor.mtest(filtered_trainDF,na.action="na.omit")

#plot it with different styles
corrplot(descrCor2, type="upper", order="hclust", p.mat = p.mat, sig.level =
           0.05,tl.col="black")
corrplot(descrCor2, method="circle")
corrplot(descrCor2, add=TRUE, type="lower", method="number",order="AOE",
         diag=FALSE, tl.pos="n", cl.pos="n")

names(trDF)
names(filtered_trainDF)

#we have reduced the predictor inputs, complete cases, and highly correlated data

#CAST model adapted from Meyer and Pebesma et al.,2020.
#------------------------------------------------------------
#load packages
library(caret)
library(CAST)
library(parallel)
library(doParallel)
library(raster)
library(tidyverse)
library(sp)
#------------------------------------------------------------
#set file directory
setwd("~/filtered_input_data")

#input file
pred <- read.csv("filtered_data", header = TRUE)#create a folder with filtered data  

#this is used as the LLO parameter
indices <- CreateSpacetimeFolds(pred,spacevar = "Tube.ID", k=5) #provide index argument and index defines which data points are used for the model training

pred_1 <- pred[,9:18] #choose predictor variables 
explained <- pred[,7:8] #choose response variable
Table_GWL <- cbind(explained[,2], pred_1)
Table_GWL <- Table_GWL[complete.cases(Table_GWL), ] 

#example model without FFS using all variables (without preselection)
#all_GWL<-caret::train(Table_GWL[,2:35], Table_GWL[,1]) #train model without forward feature selection (FFS)
#saveRDS(all_GWL, paste0("~/file-path/output file", sub(".csv","", ".RDS"))) # make a new folder for output

#run FFS model with LLO CV
perfect_GWL <- CAST::ffs(Table_GWL[,2:9], Table_GWL[,1], #train model with forward feature selection (FFS)
                       method="rf",
                       importance=TRUE,
                       trControl=trainControl(method="cv", savePredictions = TRUE, index = indices$index, number = 5))
saveRDS(perfect_GWL, paste0("~/file-path/output file", sub(".csv","", ".RDS")))

#load raster stack that contains spatial data of all predictor variables 
#the trained data set will be applied on this data set
r1 <- stack("D:/PeatbogR/ALL/Final_Rasters/LP/November_2018_LP.grd")

#check number of raster layers
nlayers(r1)
#check names
names(r1)

#use the subset of raster layers which were filtered 
newr1 <- subset(r1, c('VVI','ERGBVE','NDVI','PlnCurv', 'ProfCurv','Temperature', 'TRI4','TWI','VRM4','WEI'))
names(newr1)

#example model prediction without FFS and without preselection 
#prediction_GWL_all <- predict(r1, all_GWL)
#saveRDS(prediction_GWL_ALL, paste0("~/file-path/output file", sub(".csv","", ".RDS")))#create new folder for output


#FFS model prediction for the study area 
prediction_GWL <- predict(newr1, perfect_GWL)
saveRDS(prediction_GWL, paste0("~/file-path/output file", sub(".csv","", ".RDS")))#create new folder for output

#calculating the AOA might be time-consuming
#run in parallel
cl <- makeCluster(4)
registerDoParallel(cl)
registerDoSEQ()
memory.limit(9999999999)# needed for the big raster stacks (e.g., Upper peat bog)

#calculate the AOA on the trained model 
AOA_GWL <- aoa(newr1,perfect_GWL)
saveRDS(AOA_GWL, paste0("~/file-path/output file", sub(".csv","", ".RDS"))) 

#calculating the AOA without FFS
#AOA_random_GWL <- aoa(r1,all_GWL)
#saveRDS(AOA_random_GWL, paste0("~/file-path/output file", sub(".csv","", ".RDS")))#create new folder for output

stopCluster(cl)

#plotting results
#plot model prediction 
spplot(prediction_GWL)

#plot predictions for the AOA
spplot(prediction_GWL,main="prediction for the AOA \n(spatial CV error applied)") + 
  spplot(AOA_GWL$AOA,col.regions=c("grey","transparent"))







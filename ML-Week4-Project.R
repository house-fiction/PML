### Prepare environment
.libPaths("D:/R/win-library/3.3");
.libPaths()

##  Load necessary Packages for the project
library(data.table)
library(dplyr)
library(xtable)
library(ggplot2)
library(knitr)
library(reshape2)
library(gridExtra)

# caret-Package (short for _C_lassification _A_nd _RE_gression _T_raining)
library(caret)

# Breiman and Cutler's Random Forests for Classification and Regression
library(randomForest)


# Set working directory on local computer 
setwd("D:/01 - DATA SCIENCE/20 - J. Hopkins University (Coursera)/08 - Practical Machine Learning/R-WorkDir")

# Checking for and creating directories
if (!file.exists("data")) {
	dir.create("data")
}
### 
# Load Weight Lifting Exercise Data sets
# Training data
download.file("https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv", destfile = "./data/pml-training.csv")
# Test data
download.file("https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv", destfile = "./data/pml-testing.csv")

## Read and inspecting the raw training data set
raw_train_dat <- read.csv("./data/pml-training.csv", header = TRUE)
raw_test_dat  <- read.csv("./data/pml-testing.csv", header = TRUE)

str(raw_train_dat)
summary(raw_train_dat)

str(raw_test_dat)
summary(raw_train_dat)


# Findings: the raw training dataset contained 19622 rows of data, with 160 variables, 
# the testing data set 20 rows of data, with 160 variables . 
# Many variables contained largely missing data (usually with only one row of data)

# Data preparation steps
# Sanitize the data by removing excel division error strings `#DIV/0!` and replacing them with `NA` values.
# and converting empty strings to 'NA' values.

train_dat <- read.csv("./data/pml-training.csv", header = TRUE, na.strings = c("NA", "#DIV/0!",""))
test_dat <- read.csv("./data/pml-testing.csv", header = TRUE, na.strings = c("NA", "#DIV/0!",""))

summary(train_dat)

# Six young health participants were asked to perform one set of 10 repetitions of the 
# Unilateral Dumbbell Biceps Curl in five different fashions: 
# Class A: exactly according to the specification, 
# Class B: throwing the elbows to the front, 
# Class C: lifting the dumbbell only halfway, 
# Class D: lowering the dumbbell only halfway and 
# Class E: throwing the hips to the front 
# 
# Class A corresponds to the specified execution of the exercise, while the other 4 classes 
# correspond to common mistakes.
# Read more: http://groupware.les.inf.puc-rio.br/har#ixzz4VqNkhHg9

# Visualize how participants did their exercises
theme_set(theme_classic(base_size = 16))
qplot(x = classe, y = cvtd_timestamp, data = train_dat, color = user_name, size = I(3))


## Feature selection 
str(test_dat)

# With a clean data set, the next task was to explore the data and determine what is likely useful information.
# An important goal of any model is to generalize well with unseen data. Therefore
# any features that contained NA values have should be removed and as well as columns that appeared to be 
# entirely metadata.

# Belt, arm, dumbbell, and forearm variables that do not have any missing values in the test dataset 
# will be chosen as predictor candidates

is_missing <- sapply(test_dat, function (x) any(is.na(x) | x == ""))

is_predictor <- !is_missing & grepl("belt|[^(fore)]arm|dumbbell|forearm", names(is_missing))
pred_candidates <- names(is_missing)[is_predictor]
pred_candidates

# [1] "roll_belt"            "pitch_belt"           "yaw_belt"             "total_accel_belt"    
# [5] "gyros_belt_x"         "gyros_belt_y"         "gyros_belt_z"         "accel_belt_x"        
# [9] "accel_belt_y"         "accel_belt_z"         "magnet_belt_x"        "magnet_belt_y"       
#[13] "magnet_belt_z"        "roll_arm"             "pitch_arm"            "yaw_arm"             
#[17] "total_accel_arm"      "gyros_arm_x"          "gyros_arm_y"          "gyros_arm_z"         
#[21] "accel_arm_x"          "accel_arm_y"          "accel_arm_z"          "magnet_arm_x"        
#[25] "magnet_arm_y"         "magnet_arm_z"         "roll_dumbbell"        "pitch_dumbbell"      
#[29] "yaw_dumbbell"         "total_accel_dumbbell" "gyros_dumbbell_x"     "gyros_dumbbell_y"    
#[33] "gyros_dumbbell_z"     "accel_dumbbell_x"     "accel_dumbbell_y"     "accel_dumbbell_z"    
#[37] "magnet_dumbbell_x"    "magnet_dumbbell_y"    "magnet_dumbbell_z"    "roll_forearm"        
#[41] "pitch_forearm"        "yaw_forearm"          "total_accel_forearm"  "gyros_forearm_x"     
#[45] "gyros_forearm_y"      "gyros_forearm_z"      "accel_forearm_x"      "accel_forearm_y"     
#[49] "accel_forearm_z"      "magnet_forearm_x"     "magnet_forearm_y"     "magnet_forearm_z"


# Subsetting the training dataset to include only the predictor candidates and the outcome variable = classe.

train_variables <- c("classe", pred_candidates)
train_dat <- train_dat[, train_variables]

dim(train_dat)
str(train_dat)
summary(train_dat$classe)
#    A    B    C    D    E 
# 5580 3797 3422 3216 3607


# Visualize which variables have high correlations
cor_matrix <- cor(train_dat[sapply(train_dat, is.numeric)])
c <- melt(cor_matrix)

qplot(x = Var1, y = Var2, data = c, fill = value, geom = "tile") +
	scale_fill_gradient2(limits = c(-1,1))  +
	theme(axis.text.x = element_text(angle = -90, vjust = 0.5, hjust = 0))
	

##### Cross Validation 
# While the original Load Weight Lifting Exercise Data sets are split into training and testing already,
# the small testing data set has to be used for the purpose of the validation of the assignment only. 
# For cross validation the original training data set (train_data) will be splitted into 60%  for model 
# training (model_train_dat) and the remaining 40% for model testing (model_test_dat).

seed <- as.numeric(as.Date("2014-10-26"))
set.seed(seed)

inTrain <- createDataPartition(train_dat$classe, p = 0.6, list = FALSE)
model_train_dat <- train_dat[inTrain,]
model_test_dat <- train_dat[-inTrain,]

# Check for near zero variance.
nzv <- nearZeroVar(model_train_dat, saveMetrics=TRUE)
if (any(nzv$nzv)) nzv else message("No variables with near zero variance")

# Optional: verify manually
#                     freqRatio percentUnique zeroVar   nzv
#classe                1.469065    0.04245924   FALSE FALSE
#roll_belt             1.164384    8.87398098   FALSE FALSE
#pitch_belt            1.017241   13.68036685   FALSE FALSE
#yaw_belt              1.052117   14.47010870   FALSE FALSE
#total_accel_belt      1.058799    0.24626359   FALSE FALSE
#gyros_belt_x          1.026284    1.01052989   FALSE FALSE
#gyros_belt_y          1.127098    0.56046196   FALSE FALSE
#gyros_belt_z          1.003676    1.39266304   FALSE FALSE
#accel_belt_x          1.078059    1.27377717   FALSE FALSE
#accel_belt_y          1.152198    1.12092391   FALSE FALSE
#accel_belt_z          1.003717    2.42866848   FALSE FALSE
#magnet_belt_x         1.114679    2.53057065   FALSE FALSE
#magnet_belt_y         1.080201    2.37771739   FALSE FALSE
#magnet_belt_z         1.035088    3.57506793   FALSE FALSE
#roll_arm             56.243243   19.27649457   FALSE FALSE
#pitch_arm            83.240000   22.35054348   FALSE FALSE
#yaw_arm              35.271186   21.31453804   FALSE FALSE
#total_accel_arm       1.014625    0.56046196   FALSE FALSE
#gyros_arm_x           1.065292    5.29891304   FALSE FALSE
#gyros_arm_y           1.488449    3.05706522   FALSE FALSE
#gyros_arm_z           1.137705    1.94463315   FALSE FALSE
#accel_arm_x           1.020202    6.41134511   FALSE FALSE
#accel_arm_y           1.023622    4.40726902   FALSE FALSE
#accel_arm_z           1.051282    6.47927989   FALSE FALSE
#magnet_arm_x          1.180000   11.13281250   FALSE FALSE
#magnet_arm_y          1.081633    7.23505435   FALSE FALSE
#magnet_arm_z          1.044776   10.57235054   FALSE FALSE
#roll_dumbbell         1.125000   87.87364130   FALSE FALSE
#pitch_dumbbell        2.283951   85.68274457   FALSE FALSE
#yaw_dumbbell          1.173913   87.16032609   FALSE FALSE
#total_accel_dumbbell  1.095415    0.35665761   FALSE FALSE
#gyros_dumbbell_x      1.016086    1.97010870   FALSE FALSE
#gyros_dumbbell_y      1.317647    2.22486413   FALSE FALSE
#gyros_dumbbell_z      1.102564    1.63892663   FALSE FALSE
#accel_dumbbell_x      1.030151    3.35427989   FALSE FALSE
#accel_dumbbell_y      1.154412    3.78736413   FALSE FALSE
#accel_dumbbell_z      1.110345    3.35427989   FALSE FALSE
#magnet_dumbbell_x     1.046729    8.75509511   FALSE FALSE
#magnet_dumbbell_y     1.282828    6.89538043   FALSE FALSE
#magnet_dumbbell_z     1.068966    5.57914402   FALSE FALSE
#roll_forearm         12.433862   14.87771739   FALSE FALSE
#pitch_forearm        55.952381   21.10224185   FALSE FALSE
#yaw_forearm          15.352941   14.23233696   FALSE FALSE
#total_accel_forearm   1.120448    0.56046196   FALSE FALSE
#gyros_forearm_x       1.105442    2.33525815   FALSE FALSE
#gyros_forearm_y       1.008772    5.99524457   FALSE FALSE
#gyros_forearm_z       1.164912    2.41168478   FALSE FALSE
#accel_forearm_x       1.320000    6.57269022   FALSE FALSE
#accel_forearm_y       1.053571    8.22010870   FALSE FALSE
#accel_forearm_z       1.244186    4.62805707   FALSE FALSE
#magnet_forearm_x      1.122449   12.08389946   FALSE FALSE
#magnet_forearm_y      1.127273   15.30230978   FALSE FALSE
#magnet_forearm_z      1.000000   13.27275815   FALSE FALSE


#### Training of the prediction model (Method = Random Forest)

# Using random forest, the out of sample error should be small. The error will be estimated using 
# the 40% probing sample. 

training_model <- train(classe ~ ., data = model_train_dat, method = "rf")
training_model

# Random Forest 
#
# 11776 samples
#    52 predictor
#     5 classes: 'A', 'B', 'C', 'D', 'E' 
#
# No pre-processing
# Resampling: Bootstrapped (25 reps) 
# Summary of sample sizes: 11776, 11776, 11776, 11776, 11776, 11776, ... 
# Resampling results across tuning parameters:
#
#  mtry  Accuracy   Kappa    
#   2    0.9857255  0.9819443
#  27    0.9872909  0.9839259
#  52    0.9773708  0.9713754
#
# Accuracy was used to select the optimal model using  the largest value.
# The final value used for the model was mtry = 27. 

pred <- predict(training_model, model_train_dat)
confusionMatrix(pred, model_train_dat$classe)


# Confusion Matrix and Statistics
# 
#          Reference
# Prediction    A    B    C    D    E
#          A 3348    0    0    0    0
#          B    0 2279    0    0    0
#          C    0    0 2054    0    0
#          D    0    0    0 1930    0
#          E    0    0    0    0 2165
# 
# Overall Statistics
#                                      
#                Accuracy : 1          
#                  95% CI : (0.9997, 1)
#     No Information Rate : 0.2843     
#     P-Value [Acc > NIR] : < 2.2e-16  
#                                      
#                   Kappa : 1          
#  Mcnemar's Test P-Value : NA         
# 
# Statistics by Class:
# 
#                      Class: A Class: B Class: C Class: D Class: E
# Sensitivity            1.0000   1.0000   1.0000   1.0000   1.0000
# Specificity            1.0000   1.0000   1.0000   1.0000   1.0000
# Pos Pred Value         1.0000   1.0000   1.0000   1.0000   1.0000
# Neg Pred Value         1.0000   1.0000   1.0000   1.0000   1.0000
# Prevalence             0.2843   0.1935   0.1744   0.1639   0.1838
# Detection Rate         0.2843   0.1935   0.1744   0.1639   0.1838
# Detection Prevalence   0.2843   0.1935   0.1744   0.1639   0.1838
# Balanced Accuracy      1.0000   1.0000   1.0000   1.0000   1.0000


### Evaluation of the model on the testing dataset
pred <- predict(training_model, model_test_dat)
confusionMatrix(pred, model_test_dat$classe)

# Confusion Matrix and Statistics
# 
#           Reference
# Prediction    A    B    C    D    E
#          A 2230   16    0    0    0
#          B    1 1498    6    2    2
#          C    0    4 1353   18    7
#          D    0    0    9 1265    7
#          E    1    0    0    1 1426
# 
# Overall Statistics
#                                           
#                Accuracy : 0.9906          
#                  95% CI : (0.9882, 0.9926)
#     No Information Rate : 0.2845          
#     P-Value [Acc > NIR] : < 2.2e-16       
#                                           
#                   Kappa : 0.9881          
#  Mcnemar's Test P-Value : NA              
# 
# Statistics by Class:
# 
#                      Class: A Class: B Class: C Class: D Class: E
# Sensitivity            0.9991   0.9868   0.9890   0.9837   0.9889
# Specificity            0.9971   0.9983   0.9955   0.9976   0.9997
# Pos Pred Value         0.9929   0.9927   0.9790   0.9875   0.9986
# Neg Pred Value         0.9996   0.9968   0.9977   0.9968   0.9975
# Prevalence             0.2845   0.1935   0.1744   0.1639   0.1838
# Detection Rate         0.2842   0.1909   0.1724   0.1612   0.1817
# Detection Prevalence   0.2863   0.1923   0.1761   0.1633   0.1820
# Balanced Accuracy      0.9981   0.9925   0.9923   0.9906   0.9943



### Displaying the final model
varImp(training_model)

# rf variable importance
#
#  only 20 most important variables shown (out of 52)
#
#                      Overall
# roll_belt            100.000
# pitch_forearm         58.459
# yaw_belt              51.520
# pitch_belt            44.831
# roll_forearm          43.671
# magnet_dumbbell_z     43.387
# magnet_dumbbell_y     42.577
# accel_dumbbell_y      22.396
# roll_dumbbell         17.507
# magnet_dumbbell_x     17.279
# accel_forearm_x       16.119
# magnet_belt_z         14.983
# accel_belt_z          14.192
# magnet_forearm_z      13.991
# accel_dumbbell_z      13.660
# total_accel_dumbbell  12.906
# magnet_belt_y         12.230
# yaw_arm               11.507
# gyros_belt_z          11.233
# magnet_belt_x          9.674


training_model$finalModel
# Call:
#  randomForest(x = x, y = y, mtry = param$mtry) 
#                Type of random forest: classification
#                      Number of trees: 500
# No. of variables tried at each split: 27
# 
#         OOB estimate of  error rate: 0.79%
# Confusion matrix:
#      A    B    C    D    E class.error
# A 3340    7    1    0    0 0.002389486
# B   15 2254    9    1    0 0.010969724
# C    0   12 2033    9    0 0.010223953
# D    1    0   25 1901    3 0.015025907
# E    0    1    2    7 2155 0.004618938


# Finding: The estimated error rate is less than 1%.

# Save training model object for later.
save(training_model, file="training_model.RData")

### Predicting on the original test data (test_dat)

# Load the training model.
load(file="training_model.RData", verbose=TRUE)

# Get predictions and evaluate.
pred <- predict(training_model, model_train_dat)


pred_test <- predict(training_model, test_dat[, pred_candidates])
pred_test

pred_result <- cbind(pred_test, test_dat)
subset(pred_result, select=names(pred_result)[grep("belt|[^(fore)]arm|dumbbell|forearm", names(pred_result), invert=TRUE)])

# pred_test  X user_name raw_timestamp_part_1 raw_timestamp_part_2   cvtd_timestamp new_window num_window problem_id
# 1       B  1     pedro           1323095002               868349 05/12/2011 14:23         no         74          1
# 2       A  2    jeremy           1322673067               778725 30/11/2011 17:11         no        431          2
# 3       B  3    jeremy           1322673075               342967 30/11/2011 17:11         no        439          3
# 4       A  4    adelmo           1322832789               560311 02/12/2011 13:33         no        194          4
# 5       A  5    eurico           1322489635               814776 28/11/2011 14:13         no        235          5
# 6       E  6    jeremy           1322673149               510661 30/11/2011 17:12         no        504          6
# 7       D  7    jeremy           1322673128               766645 30/11/2011 17:12         no        485          7
# 8       B  8    jeremy           1322673076                54671 30/11/2011 17:11         no        440          8
# 9       A  9  carlitos           1323084240               916313 05/12/2011 11:24         no        323          9
# 10      A 10   charles           1322837822               384285 02/12/2011 14:57         no        664         10
# 11      B 11  carlitos           1323084277                36553 05/12/2011 11:24         no        859         11
# 12      C 12    jeremy           1322673101               442731 30/11/2011 17:11         no        461         12
# 13      B 13    eurico           1322489661               298656 28/11/2011 14:14         no        257         13
# 14      A 14    jeremy           1322673043               178652 30/11/2011 17:10         no        408         14
# 15      E 15    jeremy           1322673156               550750 30/11/2011 17:12         no        779         15
# 16      E 16    eurico           1322489713               706637 28/11/2011 14:15         no        302         16
# 17      A 17     pedro           1323094971               920315 05/12/2011 14:22         no         48         17
# 18      B 18  carlitos           1323084285               176314 05/12/2011 11:24         no        361         18
# 19      B 19     pedro           1323094999               828379 05/12/2011 14:23         no         72         19
# 20      B 20    eurico           1322489658               106658 28/11/2011 14:14         no        255         20




### Submission to Coursera
# Write submission files to Assignment_files

# Checking for and creating directories
if (!file.exists("Assignment_files")) {
	dir.create("Assignment_files")
}

write_files = function(x){
  n = length(x)
  path <- "./Assignment_files"
  for(i in 1:n){
    filename = paste0("problem_id_",i,".txt")
    write.table(x[i],file=file.path(path, filename),quote=FALSE,row.names=FALSE,col.names=FALSE)
  }
}

write_files(pred_test)

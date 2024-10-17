# Hotel Reservation Cancellation Prediction

data1 <- read.csv('Hotel Reservations.csv', sep = ',',na.strings = "?", header = TRUE)


# Libraries -----------------------------------------------------------------

library(corrgram)
library(patchwork)
library(reshape2)
library(ROCR)
library(MASS)
library(tidyverse)
library(ggplot2)
library(dplyr)
library(plyr)
library(VIM)
library(mice)
library(rpart)
library(rpart.plot)
library(caret)
library(randomForest)
library(Boruta)
library(car)
library(factorMerger)
library(parsnip)
library(caretEnsemble)
library(caTools)
library(funModeling)
library(DALEX)
library(breakDown)

set.seed(1)
# Feature Engeneering -----------------------------------------------------------

data1$type_of_meal_plan <- as.factor(data1$type_of_meal_plan) # 4 lvls
data1$required_car_parking_space <- as.factor(data1$required_car_parking_space) #0 no, 1 yes
data1$room_type_reserved <- as.factor(data1$room_type_reserved) # 7 lvls
data1$market_segment_type <- as.factor(data1$market_segment_type) # 5 lvls
data1$repeated_guest <- as.factor(data1$repeated_guest) #0 no, 1 yes
data1$booking_status <- as.factor(data1$booking_status) #TARGET 2 lvls
data1$no_of_nights <- data1$no_of_week_nights+data1$no_of_weekend_nights

data1$arrival<-paste0(as.character(data1$arrival_year),"-", 
                      as.character(data1$arrival_month), "-",
                      as.character(data1$arrival_date))
data1$arrival <- as.Date(data1$arrival, format = "%Y-%m-%d")

data1$arrival_weekday <-  weekdays(data1$arrival)
data1$arrival_weekday <- as.factor(data1$arrival_weekday)
data1$arrival_month <- as.factor(data1$arrival_month)

## the date conversion ruins 37 obs making them useless
data1 <- data1 %>% drop_na()

status=df_status(data1, print_results = F)
status

## remove ID (unique)
data1$Booking_ID <- NULL

# Starting plots --------------------------------------------------

## Simple plots
plot_gg = function(column){
  ggplot(data = data1, mapping = aes(x = {{column}}, fill = booking_status)) +
    geom_bar(position = 'dodge') +
    scale_fill_manual('Legenda', values = c("lightblue4", "mediumseagreen"))
}

plot_gg(arrival_weekday) + 
  ggtitle("Cancellazioni per giorno della settimana") 

plot_gg(arrival_month) + 
  ggtitle("Cancellazioni per mese") 

plot_gg(market_segment_type) + 
  ggtitle("Cancellazioni per market segment type") 

## More detailed plots
ggplot(data1, aes(x=arrival_weekday, fill=booking_status)) +
  geom_bar(position="fill") +
  scale_fill_manual(values=c("lightblue4", "mediumseagreen")) +
  labs(x="Day of Week", y="Percentage of Bookings", fill="Status") +
  theme_classic()

ggplot(data1, aes(x=arrival_month, fill=booking_status)) +
  geom_bar(position="fill") +
  scale_fill_manual(values=c("lightblue4", "mediumseagreen")) +
  labs(x="Day of Week", y="Percentage of Bookings", fill="Status") +
  theme_classic()

# Subsets -----------------------------------------------------------------

## We take 30% of the dataset, of that percentage we take:
# - 10% for scoring
#   - 30% validation
#   - 70/ train

SbsIndex <- createDataPartition(y = data1$booking_status, p = .30, list = FALSE)
subset <-  data1[SbsIndex, ]

SIndex<- createDataPartition(y = subset$booking_status, p = .10, list = FALSE)
score <- subset[SIndex, ]
Temp <- subset[-SIndex, ]

VIndex <- createDataPartition(y= Temp$booking_status, p = .30, list= FALSE)
validation <- Temp[VIndex, ]
train <- Temp[-VIndex, ]

2937+6847+1088

## The subset are stratified
prop.table(table(data1$booking_status))
prop.table(table(train$booking_status))
prop.table(table(validation$booking_status))
prop.table(table(score$booking_status))


# Model Selection ---------------------------------------------------------

cvCtrl = trainControl(method = "cv", number=10, search="grid", classProbs = TRUE)

## Tree
rpartTuneCvA = train(booking_status ~ ., data = Temp, method = "rpart",   #Temp=train+val
                     tuneLength = 10,
                     trControl = cvCtrl)

rpartTuneCvA
getTrainPerf(rpartTuneCvA)

plot(varImp(object=rpartTuneCvA),main="train tuned - Variable Importance")
plot(rpartTuneCvA)

vi_t = as.data.frame(rpartTuneCvA$finalModel$variable.importance)
tenutaTree = row.names(vi_t)

# To keep:
tenutaTree
plot(varImp(rpartTuneCvA))

## Random Forest 
rfTune = train(booking_status ~ ., data = Temp, method = "rf",
               tuneLength = 10,
               trControl = cvCtrl)

rfTune
getTrainPerf(rfTune)

plot(varImp(object=rfTune),main="train tuned - Variable Importance")
plot(rfTune)

vi_rf = data.frame(varImp(rfTune)[1])
vi_rf$var = row.names(vi_rf)
head(vi_rf)
tenuteRandomForest = vi_rf[,2]

# To keep:
tenuteRandomForest

## Boruta
boruta.train = Boruta( booking_status~., data = subset, doTrace = 1)
plot(boruta.train, xlab = "features", xaxt = "n", ylab="MDI")

print(boruta.train)           
boruta.metrics = attStats(boruta.train)
table(boruta.metrics$decision)

# To keep:
kb  = subset(boruta.metrics, decision == "Confirmed")
tenuteBoruta = rownames(kb)
tenuteBoruta


## Let's keep the boruta ones
paste(tenuteBoruta, collapse = ',') 
selected = c("no_of_adults","no_of_children","no_of_weekend_nights","no_of_week_nights",
             "type_of_meal_plan","required_car_parking_space","room_type_reserved",
             "lead_time","arrival_year","arrival_month","arrival_date","market_segment_type",
             "repeated_guest","no_of_previous_bookings_not_canceled","avg_price_per_room",
             "no_of_special_requests","no_of_nights","arrival","arrival_weekday", "booking_status")

train_selected = train[,selected]
validation_selected = validation[,selected]
# ### MODELS ### --------------------------------------------------
# GLM -------------------------------------------------------
cvCtrl = trainControl(method = "boot", number=10, search="grid", classProbs = TRUE)
glm = train(booking_status ~ ., data = train_selected,  method = "glm",   trControl = cvCtrl,
            preProcess=c("corr", "nzv"))
glm
confusionMatrix(glm)

glmpred = predict(glm, validation_selected)
glmpred_p = predict(glm, validation_selected, type = c("prob"))

confusionMatrix(glmpred, validation_selected$booking_status)

# LASSO -------------------------------------------------------------------

cvCtrl = trainControl(method = "boot", number=10, searc="grid", 
                      summaryFunction = twoClassSummary, 
                      classProbs = TRUE)
lasso = train(booking_status ~., data=train,
              method = "glmnet", tuneLength = 10,
              preProcess = c("corr", "nzv"),   
              metric="ROC",
              trControl = cvCtrl)
lasso
confusionMatrix(lasso)

lassoPred_p = predict(lasso, validation, type = c("prob"))
lassoPred = predict(lasso, validation)

confusionMatrix(lassoPred, validation$booking_status)

# PLS ---------------------------------------------------------------------

cvCtrl = trainControl(method = "boot", number=10, searc="grid", 
                      summaryFunction = twoClassSummary, 
                      classProbs = TRUE)
pls = train(booking_status ~., data=train,
            method = "pls", tuneLength = 10,
            preProcess = c("center"),
            metric="ROC",
            trControl = cvCtrl)

pls
confusionMatrix(pls)

plsPred_p = predict(pls, validation, type = c("prob"))
plsPred = predict(pls, validation)

confusionMatrix(plsPred, validation$booking_status)

# KNN ---------------------------------------------------------------
cvCtrl = trainControl(method = "boot", number=10, searc="grid", 
                      summaryFunction = twoClassSummary, 
                      classProbs = TRUE)
knn = train(booking_status ~., data=train_selected,
            method = "knn", tuneLength = 10,
            preProcess = c("center", "scale", "corr", "nzv"),
            metric="ROC",
            trControl = cvCtrl)

knn
confusionMatrix(knn)

KNNPred_p = predict(knn, validation_selected, type = c("prob"))
KNNPred = predict(knn, validation_selected)

confusionMatrix(KNNPred, validation_selected$booking_status)

# Let's try another vesion with sens as metric

cvCtrl = trainControl(method = "boot", number=10, searc="grid", 
                      summaryFunction = twoClassSummary, 
                      classProbs = TRUE)
knn_sens = train(booking_status ~., data=train_selected,
            method = "knn", tuneLength = 10,
            preProcess = c("center", "scale", "corr", "nzv"),
            metric="Sens",
            trControl = cvCtrl)

knn_sens
confusionMatrix(knn_sens)

KNNPred_p_sens = predict(knn_sens, validation_selected, type = c("prob"))
KNNPred_sens = predict(knn_sens, validation_selected)

confusionMatrix(KNNPred_sens, validation_selected$booking_status)

# NAIVE -------------------------------------------------------------------

cvCtrl = trainControl(method = "boot", number=10, searc="grid", 
                      summaryFunction = twoClassSummary, 
                      classProbs = TRUE)
naive = train(booking_status ~., data=train_selected,
              method = "naive_bayes", tuneLength = 10,
              preProcess = c("corr", "nzv"), 
              metric="ROC",
              trControl = cvCtrl)

naive
confusionMatrix(naive)

naivePred_p = predict(naive, validation_selected, type = c("prob"))
naivePred = predict(naive, validation_selected)

confusionMatrix(naivePred, validation_selected$booking_status)

# TREE --------------------------------------------------------------------

## Using rpart__________________________________________________

tree_rpart = rpart(booking_status ~ ., data = train, method = "class", cp = 0, minsplit = 1)
tree_rpart$cptable

tree_pruned = prune(tree_rpart, cp=  
                      tree_rpart$cptable[which.min(tree_rpart$cptable[,"xerror"]),"CP"])
rpart.plot(tree_pruned, type = 4, extra = 1) #carino ma non particolarmente interpretabile

treePred_pruned_p = predict(tree_pruned, validation, type = c("prob"))
treePred_pruned = predict(tree_pruned, validation, type = c("class"))

confusionMatrix(treePred_pruned, validation$booking_status)

## Using caret -just for reference________________

cvCtrl = trainControl(method = "cv", number=10, searc="grid", 
                      summaryFunction = twoClassSummary, 
                      classProbs = TRUE)

tree = train(booking_status ~., data=train,
              method = "rpart", tuneLength = 10,
              metric="ROC",
              trControl = cvCtrl)

tree
confusionMatrix(tree)
treePred_p = predict(tree, validation, type = c("prob"))
treePred = predict(tree, validation)

confusionMatrix(treePred, validation$booking_status)

# STACKING ----------------------------------------------------------------

cvCtrl = trainControl(method = "boot", number=10, searc="grid", 
                      summaryFunction = twoClassSummary, 
                      classProbs = TRUE)

model_list = caretList(
  booking_status ~., data = train,
  trControl = cvCtrl,
  methodList = c("glm", "rpart", "knn",  "glmnet", "pls", "naive_bayes", "gbm", "rf", "nnet", "treebag")
)

## GLM as metamodel:
glm_ensemble = caretStack(
  model_list,
  method="glm",
  metric="ROC",
  trControl = cvCtrl
)

model_preds = lapply(model_list, predict, newdata = validation, type="prob")
model_preds2 = model_preds
model_preds$ensemble = predict(glm_ensemble, newdata = validation, type="prob")
model_preds2$ensemble = predict(glm_ensemble, newdata = validation)
CF = coef(glm_ensemble$ens_model$finalModel)[-1] #coefficienti
colAUC(model_preds$ensemble, validation$booking_status)
confusionMatrix(model_preds2$ensemble, validation$booking_status)


## GBM as metamodel:
gbm_ensemble = caretStack(
  model_list,
  method="gbm",
  metric="ROC",
  trControl = cvCtrl
)

model_preds3 = model_preds
model_preds4 = model_preds
model_preds3$ensemble = predict(gbm_ensemble, newdata=validation, type="prob")
model_preds4$ensemble = predict(gbm_ensemble, newdata=validation)
colAUC(model_preds3$ensemble, validation$booking_status)
confusionMatrix(model_preds4$ensemble, validation$booking_status)

# BAGGING -----------------------------------------------------------------

cvCtrl = trainControl(method = "boot", number=10, searc="grid", 
                      summaryFunction = twoClassSummary, 
                      classProbs = TRUE)

bagging = train(booking_status ~., data=train,
                method = "treebag", ntree = 250,
                trControl = cvCtrl)

bagging
confusionMatrix(bagging)

baggingPred_p = predict(bagging, validation, type = c("prob"))
baggingPred = predict(bagging, validation)

confusionMatrix(baggingPred, validation$booking_status) 

## Not the highest accuracy but ends up being the best

# GRADIENT BOOSTING -------------------------------------------------------

cvCtrl = trainControl(method = "boot", number=10, searc="grid", 
                      summaryFunction = twoClassSummary, 
                      classProbs = TRUE)

gbm_tune = expand.grid(
  n.trees = 500,
  interaction.depth = 4,
  shrinkage = 0.1,
  n.minobsinnode = 10
)

gb = train(booking_status ~., data=train,
           method = "gbm", tuneLength = 10,
           metric="ROC", tuneGrid = gbm_tune,
           trControl = cvCtrl)

gb
confusionMatrix(gb)

gbPred_p = predict(gb, validation, type = c("prob"))
gbPred = predict(gb, validation)

confusionMatrix(gbPred, validation$booking_status)

# RANDOM FOREST -----------------------------------------------------------

cvCtrl = trainControl(method = "boot", number=10, searc="grid", 
                      summaryFunction = twoClassSummary, 
                      classProbs = TRUE)

rf = train(booking_status ~., data=train,
           method = "rf", tuneLength = 10,
           metric="ROC",
           trControl = cvCtrl,
           verbose = FALSE)

rf
confusionMatrix(rf)

rfPred_p = predict(rf, validation, type = c("prob"))
rfPred = predict(rf, validation)

confusionMatrix(rfPred, validation$booking_status)

varImp(rf)
plot(varImp(rf))

## As in KNN, second run with sens as metric

cvCtrl = trainControl(method = "boot", number=10, searc="grid", 
                      summaryFunction = twoClassSummary, 
                      classProbs = TRUE)

rf_sens = train(booking_status ~., data=train,
           method = "rf", tuneLength = 10,
           metric="Sens",
           trControl = cvCtrl,
           verbose = FALSE)

rf_sens
confusionMatrix(rf_sens)

rfPred_p_sens = predict(rf_sens, validation, type = c("prob"))
rfPred_sens = predict(rf_sens, validation)

confusionMatrix(rfPred_sens, validation$booking_status)

varImp(rf_sens)
plot(varImp(rf_sens))

# NN ------------------------------------------------------

cvCtrl = trainControl(method = "boot", number=10, searc="grid", 
                      summaryFunction = twoClassSummary, 
                      classProbs = TRUE)

nn = train(booking_status ~., data=train_selected,
           method = "nnet",
           preProcess = "range", 
           tuneLength = 5, metric="ROC", trControl=cvCtrl, trace = TRUE,
           maxit = 100)

plot(nn)
print(nn)
getTrainPerf(nn)

confusionMatrix(nn)

nnPred_p = predict(nn, validation_selected, type = c("prob"))
nnPred = predict(nn, validation_selected)

confusionMatrix(nnPred, validation_selected$booking_status)

# ROC ---------------------------------------------------------------------

y = validation$booking_status
y = ifelse(y == "Canceled", 1, 0)

## Let's take the 8 best

#glm
glmpredR = prediction(glmpred_p[,1], y)
roc_glm = performance(glmpredR, measure = "tpr", x.measure = "fpr")

#lasso
lassoPredR = prediction(lassoPred_p[,1], y)
roc_lasso = performance(lassoPredR, measure = "tpr", x.measure = "fpr")

#nn 
nnPredR = prediction(nnPred_p[,1], y)
roc_nn = performance(nnPredR, measure = "tpr", x.measure = "fpr")

#tree
treePredR = prediction(treePred_pruned_p[,1], y)
roc_tree = performance(treePredR, measure = "tpr", x.measure = "fpr")

#gb
gbPredR = prediction(gbPred_p[,1], y)
roc_gb = performance(gbPredR, measure = "tpr", x.measure = "fpr")

#bagging
baggingPredR = prediction(baggingPred_p[,1], y)
roc_bagging = performance(baggingPredR, measure = "tpr", x.measure = "fpr")

#glm stack
glm_sPredR = prediction(model_preds$ensemble, y)
roc_glm_s = performance(glm_sPredR, measure = "tpr", x.measure = "fpr")

#rf
rfPredR = prediction(rfPred_p[,1], y)
roc_rf = performance(rfPredR, measure = "tpr", x.measure = "fpr")

####

plot(roc_glm, col = "dodgerblue", lwd = 2) #
par(new = TRUE)
plot(roc_lasso, col = "gray", lwd = 2) #
par(new = TRUE)
plot(roc_nn, col = "purple", lwd = 2) #
par(new = TRUE)
plot(roc_tree, col = "cyan", lwd = 2) #
par(new = TRUE)
plot(roc_gb, col = "darkorange", lwd = 2) #
par(new = TRUE)
plot(roc_bagging, col = "pink", lwd = 2) #
par(new = TRUE)
plot(roc_glm_s, col = "red", lwd = 2) #
par(new = TRUE)
plot(roc_rf, col = "green", lwd = 2) #


legend("bottomright", legend=c("glm", "lasso", "nn", "tree", "gb", "bagging", "stacking glm", "rf"),
       col=c("dodgerblue", "gray", "purple", "cyan", "darkorange", "pink", "red", "green"),
       lty = 1, cex = 0.7, text.font=4, y.intersp=0.5, x.intersp=0.1, lwd = 3)

# LIFT --------------------------------------------------------------------

copy = train
copy$glm = predict(glm, copy, type = c("prob"))[,1]
gain_lift(data = copy, score='glm', target='booking_status')

copy = train
copy$lasso = predict(lasso, copy, type = c("prob"))[,1]
gain_lift(data = copy, score='lasso', target='booking_status')

copy = train
copy$nn = predict(nn, copy, type = c("prob"))[,1]
gain_lift(data = copy, score='nn', target='booking_status')

copy = train
copy$tree = predict(tree, copy, type = c("prob"))[,1]
gain_lift(data = copy, score='tree', target='booking_status')

copy = train
copy$gb = predict(gb, copy, type = c("prob"))[,1]
gain_lift(data = copy, score='gb', target='booking_status')

copy = train
copy$bagging = predict(bagging, copy, type = c("prob"))[,1]
gain_lift(data = copy, score='bagging', target='booking_status')

copy = train
copy$glm_s = predict(glm_ensemble, copy, type = c("prob"))
gain_lift(data = copy, score='glm_s', target='booking_status')

copy = train
copy$rf = predict(rf, copy, type = c("prob"))[,1]
gain_lift(data = copy, score='rf', target='booking_status')


# Threshold -----------------------------------------------------------

acc.perf = performance(baggingPredR, measure = "acc")
spec.perf = performance(baggingPredR, measure = "spec")
sens.perf = performance(baggingPredR, measure = "sens")
soglia <- 0.35  # threshold with a good trade-off

plot(acc.perf, col = "red", lwd = 2, ylim = c(0,1), main='Bagging', ylab='Metrics')
par(new = TRUE)
plot(spec.perf, col = "dodgerblue", lwd = 2, ylim = c(0,1), ylab=' ')
par(new = TRUE)
plot(sens.perf, col = "springgreen2", lwd = 2, ylim = c(0,1), ylab=' ')
abline(v = soglia, col = "gray", lwd = 2, lty=2)

legend("bottomright", legend=c("Accuracy", "Specificity", "Sensitivity", "Threshold"),
       col=c("red", "dodgerblue", "springgreen2", "gray"), 
       lty = c(1,1,1,2), cex = 0.7, text.font=4, y.intersp=0.5, x.intersp=0.1, lwd = c(3,3,3,1))


baggingPred_p = predict(bagging, validation, type = c("prob"))
baggingPred_T = as.factor(if_else(baggingPred_p[,2] > soglia, "Not_Canceled", "Canceled"))
confusionMatrix(baggingPred_T, validation$booking_status)
acc[acc$cutoff == soglia,]

# Bagging analysis ---------------------------------------------------------------

copy_2 = train
copy_2$y = ifelse(copy_2$booking_status == "Not_Canceled",1,0)
predict.fun = function(model, x) predict(model, x, type = "prob")[,2]

explainer_bagging = explain(bagging, data = copy_2, y=copy_2$y, predict.function = predict.fun )
vd_bagging = variable_importance(explainer_bagging, type = "raw")
plot(vd_bagging)

#Partial dependency lead
sv_Lead = single_variable(explainer_bagging, variable = "lead_time",  type = "partial")
plot(sv_Lead)
#Partial dependency no_req
sv_req = single_variable(explainer_bagging, variable = "no_of_special_requests",  type = "partial")
plot(sv_req)
#Partial dependency avg_price
sv_price = single_variable(explainer_bagging, variable = "avg_price_per_room",  type = "partial")
plot(sv_price)

# SCORE -------------------------------------------------------------------

scored = predict(bagging, score, type = c("prob"))
score$Status = as.factor(if_else(scored[,2] > soglia, "Not_Canceled", "Canceled"))

score$booking_status <- NULL
prop.table(table(score$Status))

# Samples' breakdown ------------------------------------------------------

random_obs = score[450, ] # not canceled
explain = broken(bagging, random_obs, data = score, predict.function = predict.fun)
plot(explain) + ggtitle("Random obs classified by bagging model breakdown")

random_obs2 = score[600, ] # not canceled with lower confidence
explain2 = broken(bagging, random_obs2, data = score, predict.function = predict.fun)
plot(explain2) + ggtitle("random obs classified by bagging model breakdown")

random_obs3 = score[701, ] # canceled 
explain3 = broken(bagging, random_obs3, data = score, predict.function = predict.fun)
plot(explain3) + ggtitle("Random obs classified by bagging model breakdown")

summary(train)

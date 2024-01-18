
### GOAL ###

#It is your job to predict if a passenger survived the sinking of the Titanic or not.
#For each in the test set, you must predict a 0 or 1 value for the variable.

##############

### Metric ###
#Your score is the percentage of passengers you correctly predict. This is known as accuracy

##############

# Data Import #

library(readr)
library(corrplot)
library(ggplot2)
library(dplyr) # data man
library(dlookr)
library(e1071)
library(MASS) # SVM
library(naivebayes) # naive Bayes
library(rpart) # Decision Tree
library(rpart.plot)
library(caret)
library(randomForest)

train <- read_csv("C:/Users/mtey4/Desktop/Titanic/train.csv")
test <- read_csv("C:/Users/mtey4/Desktop/Titanic/test.csv")
gender_submission <- read_csv("C:/Users/mtey4/Desktop/Titanic/gender_submission.csv")

# Train Split Done #

# Data clean #

train <- train %>% dplyr::select(-c(Name))
test <- test %>% dplyr::select(-c(Name))

summary(train)
sum(is.na(train)) # 866
sum(is.na(train$Cabin)) # 687

Cabin_Data <- train[is.na(train$Cabin) == FALSE,] # Cabin analysis

sum(duplicated(train)) # 0 No duplicated -> No repeating rows

colSums(is.na(train))
age_missing <- table(ifelse(is.na(train$Age) == TRUE, "Missing","Not Missing")) # 177 
prop.table(age_missing)

male <- train[train$Sex == "male",5]
tmale <- test[test$Sex == "male",5]
female <- train[train$Sex == "female",5]
tfemale <- test[test$Sex == "female",5]
mean_male <- mean(male$Age,na.rm = TRUE)
mean_malet <- mean(tmale$Age,na.rm = TRUE)
mean_female <- mean(female$Age,na.rm = TRUE)
mean_femalet <- mean(tfemale$Age,na.rm = TRUE)

train$Age <- ifelse(is.na(train$Age) == TRUE & train$Sex == "male",mean_male,train$Age)
train$Age <- ifelse(is.na(train$Age) == TRUE & train$Sex == "female",mean_female,train$Age)
test$Age <- ifelse(is.na(test$Age) == TRUE & test$Sex == "male",mean_malet,test$Age)
test$Age <- ifelse(is.na(test$Age) == TRUE & test$Sex == "female",mean_femalet,test$Age)

sum(is.na(train$Age))

class(train$Survived)
train$Survived <- as.factor(train$Survived)


# EDA #

describe(train)
table(train$Survived)
correlation <- cor(train[,6:7]) # Somewhat +0.41 Correlated, possibly redundant info. 
correlation_fare_age <- cor(train[,c(5,9)])
corrplot(correlation_fare_age,method = "number") # 0.09 Practically Independent
corrplot(correlation,method = "number")
ggplot(train, mapping = aes(x = as.factor(Survived), y = Fare, fill = as.factor(Pclass))) + geom_bar(stat = "identity")
ggplot(train, mapping = aes(x = Sex, y = Survived)) + geom_bar(stat = "identity", fill = "red") + theme_bw()
ggplot(train, mapping = aes(x = Age)) + geom_histogram(fill = "red") + geom_vline(aes(xintercept = mean(Age)),size = 2) + theme_bw()
ggplot(train, mapping = aes(x = Fare, fill = as.factor(Survived))) + geom_histogram() + theme_bw()
ggplot(train, mapping = aes(x = Pclass, y = Survived)) + geom_bar(stat = "identity",fill = "red") + theme_bw()
ggplot(train, mapping = aes(x = Age, y = Fare, color = as.factor(Survived))) + geom_point() + theme_bw()
ggplot(train, mapping = aes(x = Embarked, y = Fare)) + geom_boxplot(fill = "red")
ggplot(train,mapping = aes(x = Embarked, y = Fare, fill = as.factor(Survived))) + geom_bar(stat = "identity")

# Choose Model #

### Logistic Regression Model
logtrain <- train %>% dplyr::select(Survived,Pclass,Embarked,Parch,Sex,SibSp,Age)
logtest <- test %>% dplyr::select(Pclass,Embarked,Parch,Sex,SibSp,Age)
fullmodel <- glm(Survived ~ ., data = logtrain, family = "binomial")
stepmodel <- fullmodel %>% stepAIC(trace = F)
stepmodel # Parsimony

### Naive Bayes
Bayes_model <- naive_bayes(as.factor(Survived) ~ ., data = logtrain)
summary(Bayes_model)

### SVM 
svm_data <- logtrain[complete.cases(logtrain),]
SVM_model <- svm(formula = as.factor(Survived) ~ ., data = logtrain, kernal = "radial") # Based on the correlation matrix, kernal != linear -> nonlinear, so radial

### Decision Tree
dec_model <- rpart::rpart(formula = as.factor(Survived) ~ .,data = logtrain, method = "class")
rpart.plot::rpart.plot(dec_model)

### Random Forest
rf_model <- randomForest(formula = as.factor(Survived) ~ ., data = logtrain,na.action = na.exclude)



# Evaluating Model #

predict_foo <- function(model,test,class_type){
  if(class_type == "logistic"){
    result <- predict(model, test, type = "response")
    my_predict_result <- ifelse(result > 0.5,1,0)
  }else{
    my_predict_result <- predict(model,test,type = "class")
  }
  
  if(length(as.data.frame(my_predict_result) != 1))
  
  result_class_algo <- table(gender_submission$Survived,my_predict_result)
  accuracy <- (result_class_algo[1] + result_class_algo[4]) / sum(result_class_algo[1:4])
  
  
  return(list(result_class_algo, accuracy))
}


predict_foo(stepmodel,logtest,"logistic") # Logistic Confusion Matrix

predict_foo(Bayes_model,logtest,"Bayes") # Bayes Confusion Matrix

predict_foo(dec_model,logtest,"DEC")

predict_foo(rf_model,logtest,"RF")
plot(rf_model)
varImpPlot(rf_model)








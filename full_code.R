library(readxl)
library(caret)
library(corrplot)
library(dplyr)
library(ggplot2)
options(warn = -1)

house_train <- read.csv("home_data-train .csv", 
                        header = FALSE,
                        stringsAsFactors = FALSE)[, -c(1:2)]
house_test <- data.frame(read_excel("HomePrices-Test.xlsx", 
                         sheet = "Sheet1"))[, -c(1:2)]
names(house_train) <- names(house_test)

str(house_train)

ggplot(house_train) + geom_density(aes(x = price))
ggplot(house_train) + geom_density(aes(x = log(price)))

cor_mat <- cor(house_train) %>% round(digits = 2)
corrplot(cor_mat, method = 'number',
         tl.srt = 45, tl.col = 'black')
         
unnecessary <- c('sqft_lot', 'condition', 'yr_built', 'yr_renovated',
'zipcode', 'long', 'lat', 'sqft_lot15', 'sqft_basement')

train <- house_train[, -which(names(house_train) %in% unnecessary)]
test <- house_test[, -which(names(house_test) %in% unnecessary)]

rmse <- function(y, x){
  sqrt(mean((y - x)^2))
}

rsq <- function(y, x) { 
  1 - sum((y - x) ^ 2) / sum((y - mean(y)) ^ 2) 
}

control <- trainControl(method = "cv", number = 10)
# train the model
model <- train(log(price) ~ ., data = train, method = "lm", 
               trControl = control, verbose = FALSE) 
               
model
summary(model)

vimp <- varImp(model)
vimp
plot(vimp)

train$predicted <- predict(model, newdata = train)
cor(log(train$price), train$predicted)
rmse(log(train$price), train$predicted)
rsq(log(train$price), train$predicted)

test$predicted <- predict(model, newdata = test)
cor(log(test$price) ,test$predicted)
rmse(log(test$price), test$predicted)
rsq(log(test$price), test$predicted)

ggplot(data = test, aes(x = predicted, y = log(price))) +
  geom_point(alpha = 0.2, color = "blue") +
  geom_smooth(aes(x = predicted,
                  y = log(price)), color="blue") +
  geom_line(aes(x = log(price),
                y = log(price)), color = "black", 
            linetype = 2, size = 1) 
            
ggplot(data = test, aes(x = predicted,
                      y = predicted - log(price))) +
  geom_point(alpha = 0.2, color = "blue") +
  geom_smooth(aes(x = predicted,
                  y = predicted - log(price)),
              color="black")
              
              

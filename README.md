---
title: "HousePrice_Prediction_using_R"
output: github_document
---

```{r setup, include=FALSE}
knitr::opts_chunk$set(echo = TRUE)
```
```{r libraries, include=FALSE}
library(readxl)
library(caret)
library(corrplot)
library(dplyr)
library(ggplot2)
options(warn = -1)
```

## Fitting a linear model using R's caret package to predict house prices

we first load the train data and the test data without the "id" and "date" columns
(the data is already splitted into two datasets "train" and "test" with different extensions)

```{r import}
library(readxl)
house_train <- read.csv("home_data-train .csv", 
                        header = FALSE,
                        stringsAsFactors = FALSE)[, -c(1:2)]
house_test <- data.frame(read_excel("HomePrices-Test.xlsx", 
                         sheet = "Sheet1"))[, -c(1:2)]
names(house_train) <- names(house_test)
```

## exploring the data 

```{r str}
str(house_train)
```

## plotting the price variable
```{r density1}
library(ggplot2)
ggplot(house_train) + geom_density(aes(x = price))
```

it seems that the price variable distribution is skewed and this indicates that we have to get log(price)

```{r density2}
ggplot(house_train) + geom_density(aes(x = log(price)))
```

we then create a correlation map to know which variables affect the price of a house
```{r corr}
library(corrplot)
library(dplyr)
cor_mat <- cor(house_train) %>% round(digits = 2)
corrplot(cor_mat, method = 'number',
         tl.srt = 45, tl.col = 'black')
```

it seems that some variables have weak correlation with the price, so we remove them
```{r remove}
unnecessary <- c('sqft_lot', 'condition', 'yr_built', 'yr_renovated',
'zipcode', 'long', 'lat', 'sqft_lot15', 'sqft_basement')

train <- house_train[, -which(names(house_train) %in% unnecessary)]
test <- house_test[, -which(names(house_test) %in% unnecessary)]
```

then before fitting the model we define functions to extract RSquared and RMSE values from our model.
RMSE = root-mean-square error "the lower the better"
RSQ = RSquared "the greater the better"
```{r RMSE}
rmse <- function(y, x){
  sqrt(mean((y - x)^2))
}

rsq <- function(y, x) { 
  1 - sum((y - x) ^ 2) / sum((y - mean(y)) ^ 2) 
}
```

## MODEL FITTING
```{r model}
library(caret)
control <- trainControl(method = "cv", number = 10)
# train the model
model <- train(log(price) ~ ., data = train, method = "lm", 
               trControl = control, verbose = FALSE) 
```

## exploring model summary 
```{r modsum}
model
summary(model)
```

## important variables in our model
```{r varimp}
vimp <- varImp(model)
vimp
plot(vimp)
```

surprisingly number of bedrooms has no effect at all on the price, meanwhile the grade is the most important variable in the model.

## Generalizing the model and getting predictions 
we first fit the model over the training data to know how well the model fits the data from which it has been trained.
```{r trained}
train$predicted <- predict(model, newdata = train)
cor(log(train$price), train$predicted)
rmse(log(train$price), train$predicted)
rsq(log(train$price), train$predicted)
```

the model seems to fit the training data well;
let's see how well it can be generalized over the test data that it hasn't seen yet.

```{r predict}
test$predicted <- predict(model, newdata = test)
cor(log(test$price) ,test$predicted)
rmse(log(test$price), test$predicted)
rsq(log(test$price), test$predicted)
```

the model has even better results over the test data than what it had with the training one. this tells that we have a pretty good model to use over new data

## visualizing the predicted vs actual data
to get a sense of how well and close our model fits the data we plot the actual price values as a function of the predicted values with a straight line indicating the ideal relationship it should have and a smoothed one indicating how it actually fits the data.
```{r pva}
ggplot(data = test, aes(x = predicted, y = log(price))) +
  geom_point(alpha = 0.2, color = "blue") +
  geom_smooth(aes(x = predicted,
                  y = log(price)), color="blue") +
  geom_line(aes(x = log(price),
                y = log(price)), color = "black", 
            linetype = 2, size = 1) 
```

our model isn't so far from the ideal one. this is good.

# it also good to look at the residuals.
the residual is the difference between the observed value of the dependent variable "log(price)" and the predicted value "predicted"
if the points in a residual plot are randomly dispersed around the horizontal axis, a linear regression model is appropriate for the data; otherwise, a non-linear model is more appropriate.

```{r resid}
ggplot(data = test, aes(x = predicted,
                      y = predicted - log(price))) +
  geom_point(alpha = 0.2, color = "blue") +
  geom_smooth(aes(x = predicted,
                  y = predicted - log(price)),
              color="black")
```

the residuals are pretty randomly dispersed so this means that our model is so fine.


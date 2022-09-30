# Predicting Hospitalization Rate due to Covid-19/ ISEN-613


library(readxl)
data<-read_excel("trainData.xlsx", na="NA")
data
df<-data.frame(data)
head(df)
dim(df)
class(df)

# Finding NA values in the dataset
is.na(df)
sum(is.na(df))

# my_df$x1[is.na(my_df$x1)] <- mean(my_df$x1, na.rm = TRUE)- only for trial
library(dplyr) 
library(tidyr)


# Checking the columns which has NA values 
# Return the column names containing missing observations
list_na <- colnames(df)[ apply(df, 2, anyNA) ]
list_na


#Using the library data.table and shift() to create lags
library(data.table)
df.lag<-shift(df[,44:71],1:6, give.names=TRUE) # Creates a list of lag vectors for columns 44 to 71 and 
# 6 columns for for each influx pincode 
class(df.lag)
length(df.lag) # Returns the length of the list


df1<-as.data.frame(df.lag) # converts the list to dataframe 

dim(df1)
names(df1) # Returns the variable names of the dataframe of lag 

# Combining the two dataframes ( original dataframe and lag dataframe)
df2<-data.frame(cbind(df,df1))
dim(df2)
View(df2) # can view the dataset which contains the lag as well as original dataset. 

# Treating the NA values of the new dataframe 

sum(is.na(df2))

# Replacing the NA with the mean of each column 

for(i in 1:ncol(df2)){
  df2[is.na(df2[,i]), i] <- mean(df2[,i], na.rm = TRUE)
}
is.na(df2)
sum(is.na(df2))

# Building MACHINE LEARNING MODELS on the dataset

#1. Simple Linear Regression
names(df2[1:5])

lm.fit<-lm(Hospitalizations~., data=df2)
summary(lm.fit)
df3<-df2[,-1]
names(df3[1:3])
corrc<-cor(df3)


# Ridge Regression and Lasso

#(a) Fitting Ridge Regression 

install.packages("glmnet")
library(glmnet)

# glmnet() by default standardizes the variables so that they are on same scale. 

y<-df3$Hospitalizations
x<-as.matrix(df3[-1])
dim(x)

grid<-10^seq(10,-2,length=100) # Defining the lambda from 10^10 to 10^-2
ridge.mod<-glmnet(x,y,alpha=0, lambda=grid)
dim(coef(ridge.mod))
names(ridge.mod) # Gives the output parameteres of the ridge model 
summary(ridge.mod)

ridge.mod$lambda[50] # returns the value of Lambda
coef(ridge.mod)[,50]
sqrt(sum(coef(ridge.mod)[-1,50]^2))

# Predicting the ridge coefficients for new values of lambda 
predict(ridge.mod, s=50, type="coefficients")[1:20,]


# Splitting the sample into training and test datasets 
set.seed(1)
train<-sample(1:nrow(x), nrow(x)*0.7)
test<-(-train)
y.test<-y[test]

# Fitting ridge regression model on training set and finding MSE on the test set

ridge.mod1<-glmnet(x[train,],y[train], alpha=0, lambda=grid, thresh=1e-12)

# Predicting on the test dataset when lambda=0~ equivalent to simple linear regression 
ridge.pred<-predict(ridge.mod1, s=0, newx=x[test,], exact=T, x=x[train,], y=y[train])
mean((ridge.pred-y.test)^2)  # Gives the mean squared error equal to 2551.751 for lambda=0

# Predicting on the test dataset when lambda=4 
ridge.pred<-predict(ridge.mod1, s=4, newx=x[test,])
mean((ridge.pred-y.test)^2) # # Gives the mean squared error equal to 603.7388 for lambda=4


# Instead of choosing different tuning parameters lambda, we use Cross Validation to choose the lambda

set.seed(1)
cv.out<-cv.glmnet(x[train,], y[train], alpha=0) # cv.glmnet performs 10 cross valdiation by default 
plot(cv.out)
bestlam<-cv.out$lambda.min
bestlam
# The best lambda value that results in lowest cross valdiation error is 114

# Finding the test MSE associated with lambda=114
ridge.pred<-predict(ridge.mod1, s=bestlam, newx=x[test,])
mean((ridge.pred-y.test)^2) # Gives mean squared error to 400.6362 for lambda= 114 (best lambda value)

# Refitting ridge regression on the entire dataset using best lambda value 
out<-glmnet(x,y, alpha=0)
predict(out, type= "coefficients", s=bestlam)[1:20,] # Gives the 20 coefficients 



# Fitting Lasso Regression 

# Using glmnet(), alpha=1


set.seed(1)
lasso.mod<-glmnet(x[train,], y[train], alpha=1, lambda=grid)
plot(lasso.mod, main= " ") # From the plot, it is evident that depnding on tuning of tuning parameter. some of coefficients 
# can become equal to zero. 

# Performing cross validation and computaion of associated test error 
set.seed(1)
cv.out1<-cv.glmnet(x[train,], y[train], alpha=1)
plot(cv.out1)
names(cv.out) 


# Finding the minimum value of lambda

bestlam1<-cv.out1$lambda.min
bestlam1 # Returns the best lambda value=0.76823
lasso.pred<-predict(lasso.mod, s=bestlam1, newx = x[test,]) # Predicting the value on the test dataset
mean((lasso.pred-y.test)^2) # returns the test MSE =472.728 for the bestlambda value 
#which is close to ridge regression and much less than NULL model and of least squares 


out<-glmnet(x,y, alpha=1, lambda=grid)
lasso.coef<-predict(out, type="coefficients", s=bestlam1)[1:20,] # Reduces many variabkles to zero 
lasso.coef

# Finding the non zero coefficients 

lasso.coef[lasso.coef !=0] # Returns only 10 variables, rest all the variables are made equal to zero. 













  
  








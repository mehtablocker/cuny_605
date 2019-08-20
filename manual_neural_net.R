### Load data
library(dplyr)
adult <- read.table('https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data', 
                    sep = ',', fill = F, strip.white = T)
colnames(adult) <- c('age', 'workclass', 'fnlwgt', 'education', 
                     'education_num', 'marital_status', 'occupation', 'relationship', 'race', 'sex', 
                     'capital_gain', 'capital_loss', 'hours_per_week', 'native_country', 'income')
### Convert target from factor to binary
adult <- adult %>% mutate(target = as.integer(adult$income)-1)

### Define derivative of activation function
tanh_deriv <- function(x){
  1 - tanh(x)^2
}

### Set up feature matrix, target vector, initial random weights and bias
X <- adult %>% select(age, fnlwgt, education_num, hours_per_week) %>% as.matrix()
maxs <- apply(X, 2, max)
mins <- apply(X, 2, min)
X <- scale(X, center=mins, scale=maxs-mins)   #normalize the feature matrix
y <- adult$target
w <- rep(0.5, ncol(X))
b <- 0.5

### Define learning rate
learning_rate <- 0.001

### Training loop
trials <- 100000
mse_vec <- vector()

for (i in 1:trials){
  ri <- sample(1:nrow(X), 1)
  point <- X[ri,]
  
  z <- as.numeric( c(1, point) %*% cbind(c(b, w)) )   #this is just point[0]*b + point[1]*w[1] + point[2]*w[2]... in matrix form
  pred <- tanh(z)
  
  target <- y[ri]
  cost <- (pred - target)^2
  
  #Derivatives of inside functions
  dcost_pred <- 2 * (pred - target)   #derivative of cost with respect to prediction.  just the power rule.
  dpred_dz <- tanh_deriv(z)   #deriv of pred with respect to z
  dz_dw <- point   #deriv of z with respect to weights is just the point vector
  dz_db <- 1   #deriv of z with respect to b is just 1
  
  #Chain rule for derivatives
  dcost_dw <- dcost_pred * dpred_dz * dz_dw
  dcost_db <- dcost_pred * dpred_dz * dz_db
  
  #Update weights based on the partial derivs and the learning rate
  w <- w - learning_rate * dcost_dw
  b <- b - learning_rate * dcost_db
  
  #Keep track of MSE
  if (i %% 1000==0){
    full_predictions <- cbind(1, X) %*% cbind(c(b, w))
    mse <- mean((full_predictions - y)^2)
    mse_vec <- c(mse_vec, mse)
    
    message(i)}
}

### Error should be decreasing throughout the training process
plot(1:length(mse_vec), mse_vec, type="l")

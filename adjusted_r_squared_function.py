""" This function calculates the adjusted R squared value for a pre-defined regression model, given the array of real values for the target variable
(real_y), the array of predicted target values generated using the above-mentioned model (pred_y), the number of observations (n) and the number
of predictors/features used to train the regression model"""

def adj_r_sq(real_y, pred_y, n, p):
    # calculate the means of the Y variable
    Y_mean = np.mean(real_y)
    
    # calculate the deviations from the mean of each data point Y variable
    Y_mean_dev = [el-Y_mean for el in real_y]

    # calculate the deviations from the predicted value of each data point Y variable
    Y_error = [el-el_pred for el, el_pred in zip(real_y, pred_y)]

    # calculate R squared
    SSres = np.sum(np.array(Y_error)**2)
    SStot = np.sum(np.array(Y_mean_dev)**2)
    r_sq = 1 - (SSres/SStot)

    return 1 - (((1-r_sq)*(n-1))/(n-p-1))
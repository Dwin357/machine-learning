function p = predict(theta, X)
%PREDICT Predict whether the label is 0 or 1 using learned logistic 
%regression parameters theta
%   p = PREDICT(theta, X) computes the predictions for X using a 
%   threshold at 0.5 (i.e., if sigmoid(theta'*x) >= 0.5, predict 1)

m = size(X, 1); % Number of training examples

% You need to return the following variables correctly
p = zeros(m, 1);

% ====================== YOUR CODE HERE ======================
% Instructions: Complete the following code to make predictions using
%               your learned logistic regression parameters. 
%               You should set p to a vector of 0's and 1's
%

% ===============  from cost function =========
predicted_values = X * theta;
% returns a vector (m x 1)
curve_fitted_predictions = sigmoid(predicted_values);
% this is just a scalar operation, output is still (m x 1)
% ==============================================

best_guess = curve_fitted_predictions;

p = round(best_guess);
% round functions as a scalar operation, >=.5 == 1 && <.5 == 0





% =========================================================================


end

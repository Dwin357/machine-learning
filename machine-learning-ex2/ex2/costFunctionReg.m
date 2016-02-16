function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta

% ============  copied from previous exercise =============

predicted_values = X * theta;
curve_fitted_predictions = sigmoid(predicted_values);
cost_false_neg = -(log(curve_fitted_predictions)' * y);
cost_false_pos = -(log(1-curve_fitted_predictions)' * (1-y));
total_cost = cost_false_neg + cost_false_pos;

J_first_term = (1/m) * total_cost; 
% previously J


errorsVector = curve_fitted_predictions - y;
errorSumWeightedByInput = (errorsVector' * X)';
alpha = 1;
grad_one = (alpha / m) * errorSumWeightedByInput; 
% previously grad

% ==================  excluding theta(1) from normalization =====

theta_clone = theta;
% (theta x 1)

theta_clone(1) = 0;
% the first term is replaced with 0
% this prevents the first term from being included in the normalization


% ===================== J(theta)  ==============================

sum_of_squared_thetas = theta_clone' * theta;
% first term multiplies by zero and so doesn't add anything
% all other terms are squared then added together

J_second_term = ( lambda / (2 * m) ) * sum_of_squared_thetas;
% per formula, result is multiplied by lambda / (2m)
% lambda seems to be like alpha in that it is an arbitray value
% dividing by m scales weight of term, since it is sumed over all ex
% 1/2 again seems arbitray & is probably an artifact of derivation

J = J_first_term + J_second_term;

% =====================  gradient descent ===================

  grad_two = (lambda/m) * theta_clone;
  % lambda and m are both scalar, so this is scalar multiplication
  % theta_clone(1) == 0, before and after this multiplication
  % this adds the modifier +(theta(j)*lambda/m) for j >= 1

  grad = grad_one + grad_two;

% =============================================================

end

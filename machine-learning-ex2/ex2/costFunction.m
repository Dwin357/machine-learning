function [J, grad] = costFunction(theta, X, y)
%COSTFUNCTION Compute cost and gradient for logistic regression
%   J = COSTFUNCTION(theta, X, y) computes the cost of using theta as the
%   parameter for logistic regression and the gradient of the cost
%   w.r.t. to the parameters.

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
%
% Note: grad should have the same dimensions as theta

predicted_values = X * theta;
% returns a vector (m x 1)

curve_fitted_predictions = sigmoid(predicted_values);
% this is just a scalar operation, output is still (m x 1)

cost_false_neg = -(log(curve_fitted_predictions)' * y);
% log fits the sigmoid curve to a log curve (why middle fitting?)
% both y & fitted curve are (m x 1) vectors, need to transpose one in order to be able to multiply them together
% if y^i == 0, that position will cancel out
% ie, if equ is neg when supposed to be neg, that is "cost free"
% if y^i == 1, (ie false negitive) the cost will be counted and aggergated for all training examples

cost_false_pos = -(log(1-curve_fitted_predictions)' * (1-y));
% fits sigmoid curve to a log curve (again why middle fitting?)
% (1-curve) && (1-y) are both scalar xforms, no chng to vector size
% (1-curve) captures the negitive prob, ie predicts the neg
% again transpose log so it can be multiplied by y
% if y^i == 1, that position will cancel out
% ie, if equ is pos when supposed to be pos, that is "cost free"
% if y^i == 0, ie false pos, the cost is counted and added to training examples

total_cost = cost_false_neg + cost_false_pos;

J = (1/m) * total_cost;
% total_cost is the sum of costs across all training examples
% in order to get mean cost, need to divide by num of examples

% ==================== gradient descent  =======================
% from ex1 HW
% hypothesisVector = X * theta;
% errorsVector = hypothesisVector - y;
% errorSumWeightedByInput = (errorsVector' * X)';
% thetaAdjustment = (alpha / m) * errorSumWeightedByInput;
% theta = theta - thetaAdjustment;


errorsVector = curve_fitted_predictions - y;
% hypothesis vector is now the sigmond fitted curve
% this also answers above Q of the double fitting, ->sig ->log
% sigmond is needed for this, log is needed for cost

errorSumWeightedByInput = (errorsVector' * X)';
% so errors vector is an (m x 1) 
% errors' is a (1 x m)
% X is an (m x theta)
% for each feature, this goes down the list of examples
% and multiplies the feature-coefficient by total error for the ex
% this product is accumulated for each feature
% transpose on the end just to change (1 x theta) to (theta x 1)

alpha = 1;
% alpha arbitraily set to 1 b/c nothing else given
% this works -b/c "*1", but the stuff in fminunc makes it sound like I shouldn't be using this...

grad = (alpha / m) * errorSumWeightedByInput; 
% b/c errors Sum is total accross all training examples,
% needs to be averaged across total number of examples







% =============================================================

end

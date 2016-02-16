function J = computeCost(X, y, theta)
%COMPUTECOST Compute cost for linear regression
%   J = COMPUTECOST(X, y, theta) computes the cost of using theta as the
%   parameter for linear regression to fit the data points in X and y

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta
%               You should set J to the cost.

%objective, implement j := θj − α/m * m.E.i=1 (hθ(x^(i)) − y^(i))*xj^(i)
% so I am supposed to create a hypothesis vetcor which will have all the different different coefficients with the first term being 1.

% theta is transposed from a vector of coefficients to 

hypothesisVector = X * theta;
sumOfSquaredErrors = sum( (hypothesisVector - y) .^ 2 );
J = 1/(2*m) * sumOfSquaredErrors;


% =========================================================================

end

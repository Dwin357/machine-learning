function g = sigmoid(z)
%SIGMOID Compute sigmoid functoon
%   J = SIGMOID(z) computes the sigmoid of z.

% You need to return the following variables correctly 
g = zeros(size(z));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the sigmoid of each value of z (z can be a matrix,
%               vector or scalar).

% exp is to e^z what .* is to multiplication
% so having 1 / (1 + e^(-z)) apply accross all elements is
%% note, b/c (1 / matrix) is scalar, ./ is not strictly needed

g = 1 ./ (1+ exp(-z));


% =============================================================

end

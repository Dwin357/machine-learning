function [theta, J_history] = gradientDescent(X, y, theta, alpha, num_iters)
%GRADIENTDESCENT Performs gradient descent to learn theta
%   theta = GRADIENTDESENT(X, y, theta, alpha, num_iters) updates theta by 
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y); % number of training examples
J_history = zeros(num_iters, 1);

for iter = 1:num_iters

    % ====================== YOUR CODE HERE ======================
    % Instructions: Perform a single gradient step on the parameter vector
    %               theta. 
    %
    % Hint: While debugging, it can be useful to print out the values
    %       of the cost function (computeCost) and gradient here.
    %

    % so the idea is to: 1) for each theta
    % 2) sum for all cases the error of (predicted - actual)*input    
    % 3) update theta -= (alpha*sum / m)
    % 4) do the above as vector math so all theta are updated

    hypothesisVector = X * theta;
    errorsVector = hypothesisVector - y;
    errorSumWeightedByInput = (errorsVector' * X)';
    thetaAdjustment = (alpha / m) * errorSumWeightedByInput;
    theta = theta - thetaAdjustment;

    % ============================================================

    % Save the cost J in every iteration    
    J_history(iter) = computeCost(X, y, theta);

end

end

function [J, grad] = linearRegCostFunction(X, y, theta, lambda)
%LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear 
%regression with multiple variables
%   [J, grad] = LINEARREGCOSTFUNCTION(X, y, theta, lambda) computes the 
%   cost of using theta as the parameter for linear regression to fit the 
%   data points in X and y. Returns the cost in J and the gradient in grad

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost and gradient of regularized linear 
%               regression for a particular choice of theta.
%
%               You should set J to the cost and grad to the gradient.
%

theta1=theta(2:size(theta));
theta2=[0;theta1];

H = (X * theta ) ; % H representing the H(theta) value 
S = sum(( H - y ) .^ 2 ); % sum of each column
J = S / ( 2 * m );  % computation of J(theta) 




J=J+(lambda/(2*m))*(theta2'*theta2);

grad=(1/m)*X'*(X*theta-y)+(lambda/m)*theta2;











% =========================================================================

grad = grad(:);

end

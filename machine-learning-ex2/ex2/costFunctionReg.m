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

%a = -y(1) * log(sigmoid(X(1) * theta(1)) - ((1 - y(1)) * log(sigmoid(X(1) * theta(1))));  
%for i = 2:m
%a += -y(i) * log(sigmoid(X(i) * theta(i)) - ((1 - y(i)) * log(sigmoid(X(i) * theta(i)))) + lamda / 
   
%end
%j = a;
[g,h] = size(X);
k   = zeros (g,h -1);
k(1:end,1:end) = X (1:end,2:end);



[g,h] = size(theta);
t   = zeros (g-1,h);
t(1:end,1:end) = theta (2:end,1:end);

%size(k)
%size(y)

%size(theta)
%size(t)
a = sigmoid(k * t);
c = sigmoid(X * theta);
%size(a)

J = -1 /m * sum ( y' * log(c) + (1-y)' * log(1-c))   + (lambda/ (2 * m)) * sum((t).^2);

G =1/m .* (sum(k' * (c - y) ) + t * lambda);
grad = grad = (X' * (c - y)  ) / m;
temp = theta;
temp(1) = 0;
grad = grad + temp .* (lambda /m) ;
grad = grad(:);



% =============================================================

end

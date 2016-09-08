% Compute the distance profile on a given time series with the query
% Modify by Chin-Chia Michael Yeh 05/28/2016
% Original from http://www.cs.unm.edu/~mueen/FastestSimilaritySearch.html
%
% dist = distanceProfile(data, query)
% Output:
%     dist: distance profile (vector)
% Input:
%     data: the time series (vector)
%     query: the query (vector)
%

function dist = distanceProfile(x,y)
%% x is the data, y is the query
if length(x) == size(x, 2)
   x = x'; 
end
if length(y) == size(y, 2)
   y = y'; 
end

%% prepaire data
n = length(x);
meany = mean(y);
sigmay = std(y,1);
m = length(y);
x(n+1:2*n) = 0;
y = y(end:-1:1);                                %Reverse the query
y(m+1:2*n) = 0;

%% The main trick of getting dot products in O(n log n) time
X = fft(x);
Y = fft(y);
Z = X.*Y;
z = ifft(Z);

%% compute x stats -- O(n)
cum_sumx = cumsum(x);
cum_sumx2 =  cumsum(x.^2);
sumx2 = cum_sumx2(m:n)-[0;cum_sumx2(1:n-m)];
sumx = cum_sumx(m:n)-[0;cum_sumx(1:n-m)];
meanx = sumx./m;
sigmax2 = (sumx2./m)-(meanx.^2);
sigmax = sqrt(sigmax2);

%% computing the distances -- O(n) time
dist = 2*m*(1-(z(m:n)-m*meanx*meany)./(m*sigmax*sigmay));
dist = sqrt(dist);
dist = real(dist);
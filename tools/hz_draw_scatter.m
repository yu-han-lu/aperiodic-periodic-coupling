function hz_draw_scatter(x, y)

x1 = x;
y1 = y;
X1 = [ones(length(y1), 1), x1];
Y1 = y1;
[b1,~,~,~,~] = regress(Y1, X1);
z1 = b1(1) + b1(2) * x1;

hold on;
scatter(x1, y1, 50 , [0.4660 0.6740 0.1880], 'filled');
plot(x1, z1, 'r');
% axis square

end

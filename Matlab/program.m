% Read the data
data = xlsread("G-3 Dataset.xlsx");

% Splitting the data into 80% train and 20% test
split_ratio = 0.8;
split_idx = size(data, 1) * split_ratio;
train_data = data(1:split_idx, :);
test_data = data(split_idx + 1:size(data,1), :);

% Define inputs and outputs for training
train_inp = train_data(:, 7:9).';
train_out = train_data(:, 1:6).';

% Feed-Forward Network
net = newff(train_inp, train_out, [32], {'tansig'}, 'trainbfg');
tStart = tic;
ff_net = train(net, train_inp, train_out);
ff_time = toc(tStart);

% Radial Basis Network
tStart = tic;
rb_net = newrb(train_inp,train_out,50,0.001,200,50);
rb_time = toc(tStart);

% Generelized Regression Network
tStart = tic;
gr_net = newgrnn(train_inp, train_out);
gr_time = toc(tStart);

% Simulation and evaluation on train data for all networks
ff_train_out = sim(ff_net, train_inp);
ff_train_mse = mse(ff_train_out, train_out);
ff_train_R2 = CalcR2(ff_train_out,train_out);

rb_train_out = sim(rb_net,train_inp);
rb_train_mse = mse(rb_train_out, train_out);
rb_train_R2 = CalcR2(rb_train_out,train_out);

gr_train_out = sim(gr_net,train_inp);
gr_train_mse = mse(gr_train_out, train_out);
gr_train_R2 = CalcR2(gr_train_out,train_out);

fprintf("\nTRAINING\nFF MSE: %f\nRB MSE: %f\nGR MSE: %f\n\nFF R2: %f\nRB R2: %f\nGR R2: %f\n\nFF Time: %f\nRB Time: %f\nGR Time: %f\n\n\n",...
    ff_train_mse,rb_train_mse,gr_train_mse,ff_train_R2,rb_train_R2,gr_train_R2,ff_time,rb_time,gr_time)

% Compare the training results with real values
figure
hold on
plot(ff_train_out(:,1:2),'.','MarkerSize',24,DisplayName="FF Prediction")
plot(rb_train_out(:,1:2),'^','LineWidth',2,'MarkerSize',6,DisplayName="RB Prediction")
plot(gr_train_out(:,1:2),'square','LineWidth',2,'MarkerSize',6,DisplayName="GR Prediction")
plot(train_out(:,1:2),'x','MarkerSize',12,DisplayName="Real")
title("Train Simulation vs Real Data")
xlabel("Joint Number")
ylabel("Joint Value")
lgd = legend('Location','northwest');
lgd.NumColumns = 2;

% Define inputs and outputs for test
test_inp = test_data(:, 7:9).';
test_out = test_data(:, 1:6).';

% Simulation and evaluation on test data for all networks
ff_test_out = sim(ff_net, test_inp);
ff_test_mse = mse(ff_test_out,test_out);
ff_test_R2 = CalcR2(ff_test_out,test_out);

rb_test_out = sim(rb_net, test_inp);
rb_test_mse = mse(rb_test_out ,test_out);
rb_test_R2 = CalcR2(rb_test_out ,test_out);

gr_test_out = sim(gr_net, test_inp);
gr_test_mse = mse(gr_test_out,test_out);
gr_test_R2 = CalcR2(gr_test_out,test_out);

fprintf("TESTING\nFF MSE: %f\nRB MSE: %f\nGR MSE: %f\n\nFF R2: %f\nRB R2: %f\nGR R2: %f\n\n",ff_test_mse,rb_test_mse,gr_test_mse,ff_test_R2,rb_test_R2,gr_test_R2)

% Compare the test results with real values
figure
hold on
plot(ff_test_out(:,1:2),'.','MarkerSize',24,DisplayName="FF Prediction")
plot(rb_test_out(:,1:2),'^','LineWidth',2,'MarkerSize',6,DisplayName="RB Prediction")
plot(gr_test_out(:,1:2),'square','LineWidth',2,'MarkerSize',6,DisplayName="GR Prediction")
plot(test_out(:,1:2),'x','MarkerSize',12,DisplayName="Real")
title("Test Simulation vs Real Data")
xlabel("Joint Number")
ylabel("Joint Value")
lgd = legend('Location','northwest');
lgd.NumColumns = 2;


function r2 = CalcR2(yhat,y)
% Sum of squared residuals
SSR = sum((yhat - y).^2);
% Total sum of squares
TSS = sum(((y - mean(y)).^2));
% R squared
r2 = 1 - SSR/TSS;
end

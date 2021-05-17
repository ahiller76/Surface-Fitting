% copyright (c) Dr. Ying Wang, Kennesaw State University
clear all;
close all;
clc; 

%declare inputs
X = linspace(-8,8, 50); 
X2 = linspace(-8,8, 50);
[Xm,Ym] = meshgrid(X,X2);
R = sin(sqrt(Xm.^2+Ym.^2))./sqrt(Xm.^2+Ym.^2);

% normalize the training samples
[X_n,ps] = mapminmax(X, 0, 1); 
[X2_n,ls] = mapminmax(X2, 0, 1);
[R_n,ts] = mapminmax(R, 0, 1);

N=size(X);   % determine the number of training samples: N
N=N(2);

% Initialization
H=80;        %hidden layers
x=zeros(1,2);  
r=zeros(1,1);  
z=zeros(H,1);  
y=zeros(1,1);  
w=zeros(2,H);        
v=zeros(H,1);        
delta_w=zeros(2,H);  
delta_v=zeros(H,1);  

% define the learning rate
eta=0.08;      

%initialize w using a random number within (-0.01, 0.01) 
for i=1:2       
   for h=1:H
       w(i,h)=(0.01-(-0.01))*rand()+(-0.01); 
   end
end
%initialize v using a random number within (-0.01, 0.01) 
for h=1:H        
   for k=1:1
       v(h,k)=(0.01-(-0.01))*rand()+(-0.01); 
   end
end

%% %BP algorithm
% training the neural network for 20,000 times
count=0;
M=3000;  % select the total training steps
for j=1:M

%take all training samples into the BP algorithm
    err=0;   
    for t=1:N
        c=(N-1)*rand()+1; % generate a random number between 1 and N
        c=round(c);
        x=transpose([X_n(:,c),X2_n(:,c)]); % randomly select a sample based on c
        r=R_n(:,c);

        %calculate the outputs of the hidden layer
        for h=1:H
            w_h=w(:,h);
            z(h)=1/(1+exp(-(w_h'*x))); % the sigmoid fuction here
        end
        
        %calculate the output of the output layer
        for k=1:1
            y(k)=v(:,k)'*z;
            err=err+abs(r(k)-y(k));  % calculate the error
        end
        
        for k=1:1
             delta_v(:,k)=eta*(r(k)-y(k))*z;
        end

        for h=1:H
            sum=0;
            for k=1:1
                sum=sum+(r(k)-y(k))*v(h,k);
            end
            delta_w(:,h)=eta*sum*z(h)*(1-z(h))*x;
        end
        
        v=v+delta_v;
        w=w+delta_w;
        
        % save the history of v(10,1) and w(1,15) 
        count=count+1;
        v_history(count)=v(5,1);
        w_history(count)=w(1,4);
    end
    err_history(j)=err/(N*1.0); % save the history of error.
    disp(j)
end

% show the training result
plot(v_history);
legend ('v(5,1)');
figure;
plot(w_history);
legend('w(1,4)');
figure;
plot(err_history);
legend('err');

%%% test the network
% generate the test samples 
X_t=-8:0.2:8;   
X2_t=-8:0.2:8;
% normalize the test samples
X_n = [mapminmax('apply',X_t,ps); mapminmax('apply',X2_t,ps)]; 
Z=1./(1+exp(-(w'*X_n))); 
% calculate the network ouputs for the test input 
Y_n=v'*Z;            
% reverse the normalization
Y= mapminmax('reverse',Y_n,ts);  
% the expected output
Y2= sin(sqrt(Xm.^2+Ym.^2))./sqrt(Xm.^2+Ym.^2);

% graph the expected output
figure
mesh(Y2);
title('Expected Output')

% graph the trained output
figure
mesh(Y);
title('Trained Output H=70')


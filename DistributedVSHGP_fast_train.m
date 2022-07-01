function [models,vshgp_model,t_train,evidence] = DistributedVSHGP_fast_train(x, y, options)
% Distributed VSHGP for large scale training data
% Inputs:
%        x - an n x d matrix representing n training points
%        y - an n x 1 vector representing the responses of n points
%        options - configurations for distributed vshgp
%              .m - a 2 x 1 vector representing the numbers of inducing variables for f and g of each expert
%              .PseudoType - type of initial inducing points including 'random', 'kmeans'
%              .InitialKernelParas - initilize kernel paras by standard GP or mannually, 'GP', 'User'
%              .flag - mode of optimizaing hyperparameters
%                     = 1: optimize variational + f&g kernel + inducing paras
%                     = 2: optimize variational + f&g kernel paras
%                     = 3: optimize variational + g kernel paras
%                     = 4: optimize variational paras
%              .M - number of experts
%              .numIter - optimization setting
%                         negative: number of function evaluations; positive: number of line searches
%              .partitionCriterion - type of data partition;
%              .criterion - aggregation criterion, including 'PoE', 'GPoE', 'BCM' and 'RBCM'
% Outputs:
%        models - a model structure wherein model{i} contains the configurations for the ith expert
%        vshgp_model - a model structure wherein vshgp_model{i} contains the configurations for vshgp
%        t_train - training time
%
% H.T. Liu 2018/02/04 (htliu@ntu.edu.sg)
%
% Changkai Zhou adapted the part of 'multi-stage inference' for the EM algorithm of greedy selecting inducing points.
% The early stop criterion is on line 138.

% Extract configurations
flag = options.flag;
numIter = options.numIter;
M = options.M;
partitionCriterion = options.partitionCriterion;

% Normalize training data to N(0,1) if needed
[n,d] = size(x);
x_train_mean = zeros(1,d) ; x_train_std  = ones(1,d) ;
y_train_mean = 0 ;          y_train_std  = 1 ;

if strcmp(options.Xnorm,'Y'); x_train_mean = mean(x); x_train_std  = std(x); end
x_train = (x-repmat(x_train_mean,n,1)) ./ repmat(x_train_std,n,1) ;    
if strcmp(options.Ynorm,'Y'); y_train_mean = mean(y); y_train_std  = std(y); end
y_train = (y-y_train_mean)/y_train_std ;

% Partition training data into M subsets
[x_trains,y_trains,~,~] = partitionData(x_train, y_train, x, y, M, partitionCriterion);

% Extract hyperparameters
% variational and inducing paras vary over experts, while kernel paras share across experts
paras_VP = []; paras_inducing = [];
opts.m = options.m; opts.PseudoType = options.PseudoType; opts.InitialKernelParas = options.InitialKernelParas;
if strcmp(opts.InitialKernelParas,'GP')
    opts.n_SoD = options.n_SoD;
    opts.GPnumIter = options.GPnumIter;
    %logKernelParas = InitialKernelParasByGP(x_train, y_train, opts);
end
vshgp_model=cell(1,M);
logKernelParas=cell(1,M);
parfor i = 1:M %
    optss{i}=opts;
    if strcmp(opts.InitialKernelParas,'User') 
        vshgp_model{i} = vshgpCreate(x_trains{i}, y_trains{i}, optss{i}); 
        vshgp_model{i}.greedy=[1,1];
    elseif strcmp(opts.InitialKernelParas,'GP')
        logKernelParas{i} = InitialKernelParasByGP(x_trains{i}, y_trains{i}, optss{i});
        vshgp_model{i} = vshgpCreate(x_trains{i}, y_trains{i}, opts, logKernelParas{i});
        [m]=Pointnumber(vshgp_model{i}.GPf.logtheta,x_trains{i});
        [u]=Pointnumber(vshgp_model{i}.GPg.logtheta,x_trains{i});
        optss{i}.m=[m,u];
        vshgp_model{i} = vshgpCreate(x_trains{i}, y_trains{i}, optss{i}, logKernelParas{i});
        vshgp_model{i}.greedy=[1,1];
    end
end
for i = 1:M
    paras_inducing = [paras_inducing;...
                      reshape(vshgp_model{i}.Pseudo.Xm, vshgp_model{i}.m(1)*vshgp_model{i}.Pseudo.nParams,1); 
                      reshape(vshgp_model{i}.Pseudo.Xu, vshgp_model{i}.m(2)*vshgp_model{i}.Pseudo.nParams,1)];
    paras_VP = [paras_VP;vshgp_model{i}.Variation.loglambda]; % extract variational paras
end

% total initial paras
paras = [paras_VP; vshgp_model{1}.GPf.logtheta; vshgp_model{1}.GPg.logtheta; vshgp_model{1}.GPg.mu0; paras_inducing];
% reinitialize variational paras & kernel paras if needed
paras_reint_VP = [paras_VP]; 
paras_reint_GPf = [vshgp_model{1}.GPf.logtheta];
paras_reint_GPg = [vshgp_model{1}.GPg.logtheta; vshgp_model{1}.GPg.mu0];

% Train model
% indics of variational paras for each expert. start/end
for i = 1:M 
    if i == 1
        st{i} = 1; en{i} = length(y_trains{i});
    else
        st{i} = en{i-1} + 1; en{i} = en{i-1} + length(y_trains{i});
    end
end

% optimize paras via gradient descent algorithm
t1 = clock;
num_stage = length(flag);
if num_stage == 1 % one-shot inference
    [paras_opt,evidence] = minimize_vshgp(paras, @vshgp_factorise_fast, numIter, vshgp_model, flag, x_trains, y_trains, st, en, n);   
    evidence = evidence(end);
else % multi-stage inference
    evidence=[]; 
    for i = 1 : num_stage
        fprintf('This is %d th iteration \n', i)
        if strcmp(options.GradientNoise,'Y')
            [paras_opt,evide] = minimize_vshgp_GradientNoise(paras, @vshgp_factorise_fast, numIter, vshgp_model, flag(i), x_trains, y_trains, st, en, n);; 
        else
            [paras_opt,evide] = minimize_vshgp(paras, @vshgp_factorise_fast, numIter, vshgp_model, flag(i), x_trains, y_trains, st, en, n);
        end
        evide=evide(end);
        evidence=[evidence,evide]; % evidence of each iteration of adding inducing points
        
        % if the number of inducing points >= data points of all experts, then break
        countforinducingpoints=0;
        for k=1:M
            countforinducingpoints = countforinducingpoints + (vshgp_model{k}.m(1)>= vshgp_model{k}.n & vshgp_model{k}.m(2)>= vshgp_model{k}.n);
        end
        if countforinducingpoints == M
            break
        end               
        % Judge whether to stop early
        if strcmp(options.ES_expert,'Y');
            sumgreedy=0;
            for l=1:M
                sumgreedy=sum(vshgp_model{l}.greedy)+sumgreedy;
            end
            if sumgreedy==0
                break % early stopping of each expert, if all experts no not need update by trace then early stop
            end
        end
        if options.ES_ELBO <= i && options.ES_ELBO > 1
            evidence_fomer = evidence(end-options.ES_ELBO+1); 
            evidence_latter = min(evidence(end-options.ES_ELBO+1:end)); % the smallest
            if evidence_fomer - evidence_latter < 10^(floor(log10(abs(evidence_latter)))-3)*sum(options.ith)/2*(options.ES_ELBO-1) %criterion
                break % early stopping of ELBO, if formmer ELBO - latter ELBO < a value then early stop
            end
        end
        if i == num_stage
        else
            % Update models
            paras_out = paras_opt;
            for j = 1:M
                m = vshgp_model{j}.m(1); u = vshgp_model{j}.m(2);
                % training data for ith expert
                model.X_norm = x_trains{j} ;
                model.Y_norm = y_trains{j} ;
                % hyper-paras for ith expert
                model.hyp_VP = paras_out(st{j}:en{j});           % variational paras for ith expert
                model.hyp_kernel = paras_out(n+1:n+2*d+3);       % shared kernel paras
                model.hyp_indcuing = paras_out(n+2*d+4:n+2*d+3+(m+u)*d);% inducing paras for ith expert
                paras_out(n+2*d+4:n+2*d+3+(m+u)*d)=[];
                model.hyp = [model.hyp_VP;model.hyp_kernel;model.hyp_indcuing];
                model.Xm = reshape(model.hyp_indcuing(1:m*d), m, d);
                model.Xu = reshape(model.hyp_indcuing(1+m*d:end), u, d);
                % partition info
                model.M = M ;
                % normalization factors
                model.x_train_mean = x_train_mean; model.x_train_std  = x_train_std;
                model.y_train_mean = y_train_mean; model.y_train_std  = y_train_std;
                % model.greedy
                model.ES_expert = options.ES_expert;                
                model.greedy = vshgp_model{j}.greedy;
                models4greedy{j} = model ;
            end
            % E step
            [vshgp_model] = greedy_update(models4greedy,options.ith,flag); 
            % M step
            paras_inducing =[];
            paras_VP =[];
            for k = 1:M
                paras_inducing = [paras_inducing;...
                    reshape(vshgp_model{k}.Pseudo.Xm, vshgp_model{k}.m(1)*vshgp_model{k}.Pseudo.nParams,1);
                    reshape(vshgp_model{k}.Pseudo.Xu, vshgp_model{k}.m(2)*vshgp_model{k}.Pseudo.nParams,1)];
                if sum(strcmp(options.reint, 'VP')) ~= 1
                    paras_VP = [paras_VP;vshgp_model{k}.Variation.loglambda]; % extract variational paras
                end
            end
            if sum(strcmp(options.reint, 'VP')) == 1
                paras = paras_reint_VP;
            else
                paras = paras_VP;
            end
            if sum(strcmp(options.reint, 'GPf')) == 1
                paras = [paras; paras_reint_GPf];
            else
                paras = [paras; vshgp_model{1}.GPf.logtheta];
            end
            if sum(strcmp(options.reint, 'GPg')) == 1
                paras = [paras; paras_reint_GPg];
            else
                paras = [paras;  vshgp_model{1}.GPg.logtheta; vshgp_model{1}.GPg.mu0];
            end            
            paras = [paras; paras_inducing];
        end
    end;
end
t2 = clock ;
t_train = etime(t2,t1) ;
    
% Export models
%m = options.m(1); u = options.m(2);
for i = 1:M 
    m = vshgp_model{i}.m(1); u = vshgp_model{i}.m(2);
    % training data for ith expert
    model.X_norm = x_trains{i} ;                     
    model.Y_norm = y_trains{i} ;
    % hyper-paras for ith expert                                    
    model.hyp_VP = paras_opt(st{i}:en{i});           % variational paras for ith expert
    model.hyp_kernel = paras_opt(n+1:n+2*d+3);       % shared kernel paras
    model.hyp_indcuing = paras_opt(n+2*d+4:n+2*d+3+(m+u)*d);% inducing paras for ith expert
    paras_opt(n+2*d+4:n+2*d+3+(m+u)*d)=[];
    model.hyp = [model.hyp_VP;model.hyp_kernel;model.hyp_indcuing];
    model.Xm = reshape(model.hyp_indcuing(1:m*d), m, d); 
    model.Xu = reshape(model.hyp_indcuing(1+m*d:end), u, d); 
    % partition info
    model.M = M ;
    % normalization factors
    model.x_train_mean = x_train_mean; model.x_train_std  = x_train_std;
    model.y_train_mean = y_train_mean; model.y_train_std  = y_train_std;
    %model.greedy
    model.greedy=vshgp_model{i}.greedy;
    models{i} = model ;
end


end % end main function




%--------------------------------------------------------
function [xs,ys,Xs,Ys] = partitionData(x,y,X,Y,M,partitionCriterion)
% Assign training points to M subsets
% x, y - normalized training data
% X, Y - original training data
% M    - number of subsets

[n,d] = size(x) ;
if M > n
    warning('The partition number M exceeds the number of training points.');
end

if M > 1
    switch partitionCriterion
        case 'pureKmeans' % M subsets
            opts = statset('Display','off');
            [idx,C] = kmeans(x,M,'MaxIter',500,'Options',opts);
    
            for i = 1:M
                xs{i} = x(idx==i,:) ; ys{i} = y(idx==i,:) ;
                Xs{i} = X(idx==i,:) ; Ys{i} = Y(idx==i,:) ;
            end
        otherwise
            error('No such partition criterion.') ;
    end 
elseif M == 1
    xs{1} = x; ys{1} = y;
    Xs{1} = X; Ys{1} = Y;
end   

end 

function [nVB,dnVB] = vshgp_factorise_fast(paras, vshgp_model, flag, xs, ys, st, en, n)
% Factorised NLML of vshgp
% -logp(Y|X,theta) = -\sum_{k=1}^M logp_k(Y^k|X^K,theta)

M = length(xs); d = size(xs{1},2);
%m = vshgp_model{1}.m(1); u = vshgp_model{1}.m(2);

nVB = zeros(M,1);

% assign paras for each expert
for i = 1:M
    m = vshgp_model{i}.m(1); u = vshgp_model{i}.m(2);
    paras0{i} = [paras(st{i}:en{i});...                          % variational paras for ith expert
                paras(n+1:n+2*d+3);...                           % shared kernel paras
                paras(n+2*d+4:n+2*d+3+(m+u)*d)];                 % inducing paras for ith expert
    paras(n+2*d+4:n+2*d+3+(m+u)*d)=[];        
end

parfor i = 1:M %
    [nVB_i,dnVB_i] = vshgp_fast(paras0{i}, vshgp_model{i}, flag);    % compute negative variational lower bound based on xs{i}
 
    nVB(i) = nVB_i; % ELBO
   
    ni = length(ys{i});
    dnVB_VPs{i} = dnVB_i(1:ni);                % extract updated variational paras for ith expert
    dnVB_kernels{i} = dnVB_i(ni+1:ni+2*d+3);   % extract updated kernel paras
    dnVB_inducings{i} = dnVB_i(ni+2*d+4:end);  % extract updated inducing paras for ith expert   
end

% reshape paras
dnVB_VP = [];                     % variational parameters
dnVB_kernel = zeros(2*(d+1)+1,M); % kernel parameters
dnVB_inducing = [];               % inducing parameters
for i = 1:M
    dnVB_VP = [dnVB_VP;dnVB_VPs{i}];
    dnVB_kernel(:,i) = dnVB_kernels{i};
    dnVB_inducing = [dnVB_inducing;dnVB_inducings{i}];
end   
nVB = sum(nVB);                   % combination of ELBO from M experts
dnVB_kernel = sum(dnVB_kernel,2); % combination of the gradients of kernel paras from M experts                  
dnVB = [dnVB_VP;dnVB_kernel;dnVB_inducing]; % combination of the gradients of variational, kernel and inducing paras from M experts
        %these are gradients, using the gradient noise may improve the results?
end


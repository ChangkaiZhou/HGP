function [updated_models] = greedy_update(models,ith,flag) 
%Greedy inducing points selection based on traces

    ith1 = ith(1); % delta m for GPf
    ith2 = ith(2); % delta u for GPg
parfor i = 1 : models{1}.M %
    model = models{i};
    [n,d] = size(models{i}.X_norm); % n training data
    [m1,~] = size(models{i}.Xm); % m1 inducing points of GPf
    [u1,~] = size(models{i}.Xu); % u1 inducing points of GPf
    if sum(model.greedy) == 0 || (m1>=n && u1>=n)% wrap the paras directly without adding inducing points
        updated_models{i} = wrapbox(model);
    else
        greedy_f = []; greedy_g = [];
        % repeat part when calculating trace(K-Q)
        if ith1 > 0 && model.greedy(1) == 1 && m1<n
            diagKnn_f = covSEard_jitter_vshgp(model.hyp_kernel(1:d+1), model.X_norm, 'diag');  % n x 1
            Kmm_f = covSEard_jitter_vshgp(model.hyp_kernel(1:d+1), model.Xm); % m x m
            Knm_f = covSEard_jitter_vshgp(model.hyp_kernel(1:d+1), model.X_norm, model.Xm); % n x m
        end
        if ith2 > 0 && model.greedy(2) == 1 && u1<n
            diagKnn_g = covSEard_jitter_vshgp(model.hyp_kernel(d+2:2*d+2), model.X_norm, 'diag'); % n x 1
            Kuu_g = covSEard_jitter_vshgp(model.hyp_kernel(d+2:2*d+2), model.Xu); % u x u
            Knu_g = covSEard_jitter_vshgp(model.hyp_kernel(d+2:2*d+2), model.X_norm, model.Xu); % n x u
        end        
        
        for j = 1 : n % finding the max trace increment of adding each training data 
                      % it is equal to minimize the trace(K-Q)
                      % of course sampling of full data can save time
            if ith1 > 0 && model.greedy(1) == 1 && m1<n
                Km1_f = covSEard_jitter_vshgp(model.hyp_kernel(1:d+1), model.Xm, model.X_norm(j,:));
                K11_f = covSEard_jitter_vshgp(model.hyp_kernel(1:d+1), model.X_norm(j,:));
                Km1m1_f = [Kmm_f,Km1_f;Km1_f',K11_f];
                Kn1_f = covSEard_jitter_vshgp(model.hyp_kernel(1:d+1), model.X_norm, model.X_norm(j,:));
                Knm1_f = [Knm_f,Kn1_f]; % n x m+1
                Lm1m1_f = chol(Km1m1_f)';      % m+1 x m+1, a lower triangular matrix
                invLm1m1_f = Lm1m1_f\eye(length(model.Xm(:,1))+1);         % m+1 x m+1, (L_m1m1^f)^{-1}                            ---- O(m^3)
                Qnn_f_half = Knm1_f*invLm1m1_f';            % n x m+1, K_nm1*L_m1m1_f^{-T}                     ---- O(n m^2)  !!!expensive!!!
                diagQnn_f = diagAB(Qnn_f_half,Qnn_f_half');    % n x 1
                greedy_f(j)=sum(diagKnn_f-diagQnn_f); % trace(K-Q)                
            end
            if ith2 > 0 && model.greedy(2) == 1 && u1<n
                Ku1_g = covSEard_jitter_vshgp(model.hyp_kernel(d+2:2*d+2), model.Xu, model.X_norm(j,:));
                K11_g = covSEard_jitter_vshgp(model.hyp_kernel(d+2:2*d+2), model.X_norm(j,:));
                Ku1u1_g = [Kuu_g,Ku1_g;Ku1_g',K11_g];
                Kn1_g = covSEard_jitter_vshgp(model.hyp_kernel(d+2:2*d+2), model.X_norm, model.X_norm(j,:));
                Knu1_g = [Knu_g,Kn1_g]; 
                Lu1u1_g = chol(Ku1u1_g)';                 
                invLu1u1_g = Lu1u1_g\eye(length(model.Xu(:,1))+1);                                              
                Qnn_g_half = Knu1_g*invLu1u1_g';          
                diagQnn_g = diagAB(Qnn_g_half,Qnn_g_half');    
                greedy_g(j)=sum(diagKnn_g-diagQnn_g);             
            end
        end
        if ith1 > 0 && model.greedy(1) == 1 && m1<n
            [~,index_f] = sort(greedy_f); %,'descend'
            model.Xm = [model.Xm;model.X_norm(index_f(1:min(ith1,n)),:)];
        end
        if ith2 > 0 && model.greedy(2) == 1 && u1<n
            [~,index_g] = sort(greedy_g); %,'descend'
            model.Xu = [model.Xu;model.X_norm(index_g(1:min(ith2,n)),:)];
        end
        % update the inducing points number once
        model_latter = wrapbox(model);
        model_latter.greedy = model.greedy;
        % early stopping of each expert by trace, if the trace is small
        % enough, then we do not need update points.
        if strcmp(model.ES_expert,'Y');
            if model.greedy(1) == 1 && m1<n
                if min(greedy_f) < sum(diagKnn_f)*0.01  % the trace criterion of early stopping of each expert
                    model_latter.greedy(1)=0;
                end
            end
            if model.greedy(2) == 1 && u1<n
                if min(greedy_g) < sum(diagKnn_g)*0.01
                    model_latter.greedy(2)=0;
                end
            end
        end
        updated_models{i} = model_latter;       
    end
end
end

function [modelbox]= wrapbox(model)
    [n,d]=size(model.X_norm);
    model.GPf.logtheta=model.hyp_kernel(1:d+1);
    model.GPf.nParams=d+1;
    model.GPg.logtheta=model.hyp_kernel(d+2:2*d+2);
    model.GPg.mu0=model.hyp_kernel(2*d+3);
    model.GPg.nParams=d+2;
    m=length(model.Xm(:,1)); u=length(model.Xu(:,1));
    model.m=[m,u];
    model.Variation.nParams=n;
    %model.Variation.loglambda=model.hyp_VP;            %reinitialization variational paras
    model.Variation.loglambda=log(0.5)*ones(model.Variation.nParams,1);  %do not reint variational paras 
    model.Pseudo.nParams=d;
    model.Pseudo.Xm=model.Xm;
    model.Pseudo.Xu=model.Xu;
    model.X=model.X_norm;
    model.y=model.Y_norm;
    model.jitter=1e-6;
    model.n=n;
    model.D=d;
    paras_inducing = [reshape(model.Pseudo.Xm, model.m(1)*model.Pseudo.nParams,1); 
                      reshape(model.Pseudo.Xu, model.m(2)*model.Pseudo.nParams,1)];
    model.paras=[model.hyp_VP;model.hyp_kernel;paras_inducing];
    modelbox=model;
end



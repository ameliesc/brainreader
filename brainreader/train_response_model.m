function response_model = train_response_model(response_model, feature, response)

if strcmp(response_model.name, 'kernel_ridge')
    
    [response_model.Beta_hat,response_model.Sigma_hat] = kernel_ridge(response_model.fold_number, solve_df_for_lambda(response_model.lambda_number, feature), feature, response);
    
elseif strcmp(response_model.name, 'elastic_net')
    
    for index = size(response, 2) : -1 : 1
        
        response_model.CVerr{index} = cvglmnet(feature', response(:, index), [], struct('alpha', 0.5, 'nlambda', response_model.lambda_number), [], response_model.fold_number, [], true);
        
    end
    
elseif strcmp(response_model.name, 'Lasso')
    
    for index = size(response, 2) : -1 : 1
        
        response_model.CVerr{index} = cvglmnet(feature', response(:, index), [], struct('alpha', 1, 'nlambda', response_model.lambda_number), [], response_model.fold_number, [], true);
        
    end
    
end

end


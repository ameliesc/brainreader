function response_hat = simulate_response_model(response_model, feature)

if strcmp(response_model.name, 'kernel_ridge')
    
    response_hat = cat(1, ones(1, size(feature, 2)), feature)' * response_model.Beta_hat;
    
elseif strcmp(response_model.name, 'elastic_net') || strcmp(response_model.name, 'Lasso')
    
    for index = length(response_model.CVerr) : -1 : 1
        
        response_hat(:, index) = cvglmnetPredict(response_model.CVerr{index}, feature');
        
    end
    
end

end


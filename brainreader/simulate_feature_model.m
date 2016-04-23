function feature = simulate_feature_model(feature_model, stimulus)

stimulus_size = size(stimulus);

if strcmp(feature_model.name, 'identity');
    
    feature = reshape(stimulus, stimulus_size(1) * stimulus_size(2), stimulus_size(3));
    
elseif strcmp(feature_model.name, 'PCA') || strcmp(feature_model.name, 'ICA')
    
    for index = stimulus_size(3) : -1 : 1
        
        feature(:, index) = subsref(feature_model.static_nonlinearity(feature_model.W * im2col(stimulus(:, :, index), feature_model.feature_size([1 2]), 'distinct')), substruct('()', {':'}));
        
    end
    
elseif strcmp(feature_model.name, 'TICA')
    
    for index = stimulus_size(3) : -1 : 1
        
        feature(:, index) = subsref(feature_model.static_nonlinearity(feature_model.H * (feature_model.W * im2col(stimulus(:, :, index), feature_model.feature_size([1 2]), 'distinct')) .^ 2), substruct('()', {':'}));
        
    end
    
end

end


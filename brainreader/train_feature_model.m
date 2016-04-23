function feature_model = train_feature_model(feature_model, stimulus)

if strcmp(feature_model.name, 'identity');
    
    return;
    
end

stimulus_size = size(stimulus);

for index = feature_model.patch_number : -1 : 1
    
    patch(:, index) = subsref(stimulus((1 : feature_model.feature_size(1)) + randi(stimulus_size(1) - feature_model.feature_size(1)), (1 : feature_model.feature_size(2)) + randi(stimulus_size(2)  - feature_model.feature_size(2)), randi(stimulus_size(3))), substruct('()', {':'}));
    
end

if strcmp(feature_model.name, 'PCA')
    
    feature_model.W = subsref(pca(patch'), substruct('()', {':', 1 : feature_model.feature_size(3) * feature_model.feature_size(4)}))';
    
elseif strcmp(feature_model.name, 'ICA')
    
    [~, feature_model.W] = fastica(patch, 'approach', 'symm', 'g', 'tanh', 'lastEig', feature_model.feature_size(3) * feature_model.feature_size(4));
    
elseif strcmp(feature_model.name, 'TICA')
    
    [feature_model.H, feature_model.W] = TICA([feature_model.feature_size([3 4]) feature_model.neighborhood_size feature_model.step_number], patch);
    
end

end


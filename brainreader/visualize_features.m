function visualize_features(feature_model)

if strcmp(feature_model.name, 'identity');
    
    disp('Features of the identity model cannot be visualized.');
    
    return;
    
end

if strcmp(feature_model.name, 'PCA') || strcmp(feature_model.name, 'ICA') || strcmp(feature_model.name, 'TICA')
    
    A                  = pinv(feature_model.W);
    A                  = bsxfun(@minus, A, min(A));
    A                  = bsxfun(@rdivide, A, max(A));
    visualization_size = feature_model.feature_size([1 2]) .* feature_model.feature_size([3 4]);
    
    imagesc(col2im(A, feature_model.feature_size([1 2]), visualization_size, 'distinct')); daspect([visualization_size 1]); axis off; title('features');
    
end

end


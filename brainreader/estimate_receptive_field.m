function receptive_field = estimate_receptive_field(feature_model, field_of_view, response_model)
%ESTIMATE_RECEPTIVE_FIELD Summary of this function goes here
%   Detailed explanation goes here

point_stimulus = zeros([field_of_view field_of_view]);

for index_1 = 1 : field_of_view(1)
    
    for index_2 = 1 : field_of_view(2)
        
        point_stimulus(index_1, index_2, index_1, index_2) = 1;
        
    end
    
end

point_stimulus  = reshape(point_stimulus, [field_of_view field_of_view(1) * field_of_view(2)]);
point_feature   = simulate_feature_model(feature_model, point_stimulus);
point_response  = simulate_response_model(response_model, point_feature);
point_response  = bsxfun(@minus, point_response, min(point_response));
point_response  = bsxfun(@rdivide, point_response, max(point_response));
point_response  = reshape(point_response, field_of_view(1), field_of_view(2), []);
m               = size(point_response, 3);
receptive_field = cell(1, m);

[X, Y] = meshgrid(1 : field_of_view(1), 1 : field_of_view(2));

for index_3 = 1 : m
    
    receptive_field{index_3} = autoGaussianSurf(X, Y, point_response(:, :, index_3), struct('errorbars', 'none', 'iso', true, 'tilted', false, 'positive', true));
    
end

end


function visualize_images(stimuli,reconstructions,delay)

    if nargin<3, delay=0.5; end

    sz = size(stimuli);
    
    figure;
    for index = 1 : sz(3)
        
        x = stimuli(:, :, index);
        y = reconstructions(:,:,index);
        
        subplot(1,3,1); 
        imagesc(x); 
        daspect([sz([1 2]) 1]); 
        axis off; 
        colormap(gray); 
        title('stimulus');
        
        subplot(1,3,2); 
        imagesc(y); 
        daspect([sz([1 2]) 1]); 
        axis off; 
        colormap(gray); 
        title('reconstruction');

        subplot(1,3,3); 
        scatter(x(:),y(:),'.');
        xlim([min(stimuli(:)) max(stimuli(:))]);
        ylim([min(reconstructions(:)) max(reconstructions(:))]);
        axis square;
        xlabel('stimulus');
        ylabel('reconstruction');

        drawnow;
        
        pause(delay);
        
    end

end


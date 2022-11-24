function [tensor, wtensor, abs_img] = calcTensor(gray_img, tensorbg, size_grad)

    if(~exist('tensorab', 'var'))
        beta_at = 10;
        gamma_at = 0.75;
    else
        beta_at = tensorbg(1);
        gamma_at = tensorbg(2);
    end
    
    gray_img = im2double(gray_img);
    
    [M, N] = size(gray_img);

    if size_grad == 2
        gray_img = imfilter(gray_img, fspecial('gaussian',[3 3], 0.5));
        csvwrite("test_imgs/gray_img.csv", gray_img);
        G_x = [-1 1;
               -1 1];
    elseif size_grad == 3
        G_x = [1 0 -1; 2 0 -2; 1 0 -1];
    elseif size_grad == 5
        G = fspecial('gaussian',[5 5], 1);
        G_x = gradient(G);
    else
        error('tensor gradient size must be 2/3/5');
    end
    
    G_y = G_x';

    min_n_length = 1e-8;
    min_tensor_val = 1e-8;

    grad_x = imfilter(gray_img, G_x, 'replicate');
    csvwrite("test_imgs/gray_x.csv", grad_x),
    
    grad_y = imfilter(gray_img, G_y, 'replicate');
    csvwrite("test_imgs/grad_y.csv", grad_y);

    abs_img = sqrt(grad_x.^2 + grad_y.^2);
    csvwrite("test_imgs/abs_img.csv", abs_img);

    n = [grad_x(:)'; grad_y(:)'];
    csvwrite("test_imgs/n.csv", n);
    %disp(n)

    norm_n = (sqrt(sum(n.^2)));

    n_normed = n./repmat(norm_n, [2,1]);

    n_normed(1,norm_n < min_n_length) = 1;
    %disp(n_normed)
    n_normed(2,norm_n < min_n_length) = 0;
    %disp(n_normed)
    
    nT_normed = [n_normed(2,:); -n_normed(1,:)];

    wtensor = max(min_tensor_val, exp(-beta_at*abs_img.^gamma_at));

    atensor = reshape(wtensor(:)'.*n_normed(1,:).^2 + nT_normed(1,:).^2, M, N);
    ctensor = reshape(wtensor(:)'.*n_normed(1,:).*n_normed(2,:) + nT_normed(1,:).*nT_normed(2,:), M, N);
    btensor = reshape(wtensor(:)'.*n_normed(2,:).^2 + nT_normed(2,:).^2, M, N);

    tensor{1} = atensor;
    tensor{2} = btensor;
    tensor{3} = ctensor;

end

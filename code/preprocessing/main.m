%% SLT PROJECT - PREPROCESSING STEP
%Team members: Sjoerd van Bekhoven, Torres Garcia Moises, Antonio Orvieto

%The following code will do a huge compression of the data, each "star"
%will be reduced to a vector in R^3 corresponding to the maximum diffusion
%direction. to do that we fit an ellipsoid to the star, using the the
%Ellipsoid fit library one can download at the following link

%http://www.mathworks.com/matlabcentral/fileexchange/24693-ellipsoid-fit

%the smallest axis of the ellipsoid will give us the maximum diffusion
%direction

close all;

%% DATA LOADING
disp('Loading data..');
addpath('/');
addpath('preprocessing/nifti/')
addpath('preprocessing/ellipsoid_fit/')
data = load_nii('data/diff_data.nii.gz'); 
bvecs = load('data/bvecs');
disp('Finished loading data.');
pause(1);

%% ELLIPSOID FIT OF THE BVECS
%drawStar(bvecs, data.img(100,60,100,:));
%data point are the rows of the folling matrix

%maximum diffusion direction
embeddings = zeros(210,210,130,3);

%fractional anisotropy
FA = zeros(210,210,130);

for i=1:210
    for j=1:210
        for k=1:130
            d=squeeze(data.img(i,j,k,:));
            x = times(d,bvecs(:,1));
            y = times(d,bvecs(:,2));
            z = times(d,bvecs(:,3));
            [ center, radii, evecs, v, chi2 ] = ellipsoid_fit( [x y z ], '' );
            [m,in] = min(radii);
            embeddings(i,j,k,:)=m*evecs(in);
            FA(i,j,k)=sqrt((1/radii(1)-1/radii(2))^2+(1/radii(1)-1/radii(3))^2+(1/radii(2)-1/radii(3))^2)/(sqrt(2*((1/radii(2))^2+(1/radii(3))^2+(1/radii(1))^2)));
            fprintf('Computed ellipsoid at(%d,%d,%d), FA=%f, ax1:%f, ax2=%f, ax3=%f, min=%f \n',i,j,k,FA(i,j,k),radii(1),radii(2),radii(3),m);
        end
    end
end

%% WRITING EMBEDDINGS INTO FILES
disp('Writing costs to File..');
emb = fopen('embeddings','wt');

for i=1:210
    for j=1:210
        for k=1:130
            fprintf(emb,'%3.7f\t\t',embeddings(i,j,k,:));
            fprintf(emb,'\n');
        end
    end
end

fclose(emb);

%% WRITING ISOTROPY DATA INTO FILES
disp('Writing isotropies to file..');
emb = fopen('FA','wt');

for i=1:210
    for j=1:210
        for k=1:130
            fprintf(emb,'%g\t',FA(i,j,k,:));
            fprintf(emb,'\n');
        end
    end
end

fclose(emb);


%% SEE ACTIVE AREAS OF THE BRAIN (THRESHOLD AT K=0.31)
% image = zeros(130,210,3);
% 
% for z=1:130
%     for y=1:210
%         if(FA(60,y,z)>0.34)
%             image(131-z,y,1)=1; 
%         end
%     end
% end
% 
% figure, imshow(image)


%% DRAW THE STAR!
%drawStar(bvecs, data.img(3,90,50,:));

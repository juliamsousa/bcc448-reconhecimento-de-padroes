%----------------------- DEFINICAO DE PARAMETROS --------------------------
% define o caminho das imagens do banco de dados e armazena as imagens do banco de dados no programa
data_base_path = "..\Folds_Dataset_Final";
data_base = imageDatastore(data_base_path, 'IncludeSubfolders',true, 'LabelSource','foldernames');
% particiona o banco de dados em 5 partes, a serem permutadas pelo k-fold
% [testing_data_base, training_data_base] = splitEachLabel(data_base, 0.40, 0.60);
% [testing_data_base, training_data_base] = splitEachLabel(data_base, 0.30, 0.70);
[testing_data_base, training_data_base] = splitEachLabel(data_base, 0.20, 0.80);
%--------------------------------------------------------------------------

%---------------------------- TREINAMENTO ---------------------------------
% lê uma unica imagem do banco de dados
img = readimage(data_base, 1);
% define o tamanho da celula de hog features
cell_size = [16, 16];
% extrai as features da primeira imagem do banco
[hogfv, hogvis] = extractHOGFeatures(img, 'CellSize', cell_size);
% encontra o tamanho do vetor de features
hog_feature_size = length(hogfv);

% extrai o numero de imagens que estao no banco de dados
total_train_images = numel(training_data_base.Files);
% disp("A quantidade total de imagens de treino é: " + total_train_images);

% banco de dados de features, dado por uma matriz, armazena todas as features do banco de testes
training_features = zeros(total_train_images, hog_feature_size, 'single');

% percorre todas as imagens do banco de dados e extrai as HOG features
for i = 1:total_train_images
    image_train = readimage(training_data_base, i);
    training_features(i, :) = extractHOGFeatures(image_train, 'CellSize', cell_size);
end

% as labels são criadas usando o tipo categorical
training_labels = training_data_base.Labels;
%--------------------------------------------------------------------------

%-------------------------------- TESTE -----------------------------------
% uso do SVM  baseado em ECOC utilizando 'One vs One'
classifier = fitcecoc(training_features, training_labels); % Training

% testando as imagens
total_test_images = numel(testing_data_base.Files);
test_features = zeros(total_test_images, hog_feature_size, 'single');

for j = 1:total_test_images
    image_test = readimage(testing_data_base, j);
    test_features(j, :) = extractHOGFeatures(image_test, 'CellSize',cell_size);
end

% as labels são criadas usando o tipo categorical
test_labels = testing_data_base.Labels;
predcited_labels = predict(classifier, test_features);
%--------------------------------------------------------------------------

%------------------------------ RESULTADOS --------------------------------
accuracy = (sum(predcited_labels == test_labels)/numel (test_labels)) * 100;
disp("A acurácia do programa foi de: " + accuracy);
plotconfusion(test_labels, predcited_labels);
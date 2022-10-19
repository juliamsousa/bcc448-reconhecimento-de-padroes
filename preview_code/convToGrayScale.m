function n = convToGrayScale()
  ext = '.png';
  endereco = '.\P6\G7\';
  arquivos = dir([endereco '*' ext]);
  quant_img = length(arquivos);
  for j=1:quant_img
    img = imread([endereco, arquivos(j,1).name]);
    dir = '.\P4\G7\';
    nimg = rgb2gray(img);
    nome = num2str(j,'%d');
    s = strcat(nome,'.png');
    s1 = strcat(dir,s);
    imwrite(nimg,s1);
  end
  n=0;
end
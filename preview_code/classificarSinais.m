
%  RECOGNITION OF DYNAMIC SIGNS IN A SIGN LANGUAGE
%  Alunos: Julia Miranda e Myllene Ferreira


function percentual_acertoHOG = classificarSinais(percentual_teste)
  if nargin < 1
      percentual_teste = 0.3;%Percentual recebe um valor padrão caso não seja fornecido
  end
  
  %----- Declaração e inicialização prévia de algumas das variáveis ----- 
  contadorSinais=1;
  quant_sinais = zeros(1,7);
  diretorio_raiz = 'C:\Users\julia\Documents\UFOP\8_Semestre\PDI\TP\TrabalhoFinal\BancodeImagens\';%Diretório do banco de imagens
  ext = '.png';
  etiquetas = zeros(397,1);
  percentual_acertoHOG = zeros(7,1);
  
  %---------------- VARIÁVEIS REFERENTES AO DESCRITOR HOG ---------------- 
  descritor_HOG = zeros(397,51984);
  %-----------------------------------------------------------------------
  
  %------------- LOOPS PARA PERCORRER AS IMAGENS E AS PASTAS -------------
  for i=1:7
    num_gesto = num2str(i,'%d');
    for j=1:9
      num_pessoa = num2str(j,'%d');
      endereco = strcat(diretorio_raiz,'P',num_pessoa,'\','G',num_gesto,'\');
      arquivos_sinais = dir([endereco '*' ext]);
      quant_img = length(arquivos_sinais);
      quant_sinais(1,i) = quant_img + quant_sinais(1,i);
      
      for k = 1 : quant_img
        img = imread ([endereco, arquivos_sinais(k,1).name]);
        imgPadrao = imresize(img,[317 317],'bilinear');
        caract = extractHOGFeatures(imgPadrao);%Extrai o descritor HOG das imagens
        descritor_HOG(contadorSinais,:) = caract;
        contadorSinais = contadorSinais + 1;
      end
    end
  end
  
  %PREENCHE O VETOR DE ETIQUETAS DE ACORDO COM O CONHECIMENTO PRÉVIO
  for i=1:397
      if(i<=57)
          etiquetas(i) = 1;%Rótulo numérico para o sinal da letra 'A'
      end
      if(i>57 && i<=113)
          etiquetas(i) = 2;%Rótulo numérico para o sinal da letra 'D'
      end
      if(i>113 && i<=170)
          etiquetas(i) = 3;%Rótulo numérico para o sinal da letra 'I'
      end
      if(i>170 && i<=226)
          etiquetas(i) = 4;%Rótulo numérico para o sinal da letra 'L'
      end
      if(i>226 && i<=282)
          etiquetas(i) = 5;%Rótulo numérico para o sinal da letra 'V'
      end
      if(i>282 && i<=339)
          etiquetas(i) = 6;%Rótulo numérico para o sinal da letra 'W'
      end
      if(i>339 && i<=397)
          etiquetas(i) = 7;%Rótulo numérico para o sinal da letra 'Y'
      end
  end
  
  %---------------- OBTEM OS DADOS DE TREINO E DE TESTE ------------------
  descritor_HOG = normaliza(descritor_HOG);
  [treinoHOG, testeHOG] = crossvalind('HoldOut', etiquetas, percentual_teste);
  desc_treinoHOG = descritor_HOG(treinoHOG,:);
  desc_testeHOG = descritor_HOG(testeHOG,:);
  etiqueta_treinoHOG = etiquetas(treinoHOG,:);
  etiqueta_testeHOG = etiquetas(testeHOG,:);
  %-----------------------------------------------------------------------

  %---------- TREINAMENTO E CLASSIFICAÇÃO PARA O DESCRITOR HOG -----------
  parametrosSVM = templateSVM('KernelFunction','polynomial','KernelScale','auto','Standardize',1);
  modelo = fitcecoc(desc_treinoHOG, etiqueta_treinoHOG,'Learners',parametrosSVM,'Coding','onevsall');
  classificador = predict(modelo, desc_testeHOG);
  matConf = confusionmat(etiqueta_testeHOG,classificador)
  %-----------------------------------------------------------------------
  
  %------------------ CALCULANDO E EXIBINDO RESULTADOS ------------------- 
  for i = 1 : 7
    acertoHOG = etiqueta_testeHOG == i & etiqueta_testeHOG == classificador; 
    percentual_acertoHOG(i) = sum(acertoHOG)/sum(etiqueta_testeHOG == i);
  end
  percentual_acertoHOG
  mean(percentual_acertoHOG)
  %-----------------------------------------------------------------------
  
end
  
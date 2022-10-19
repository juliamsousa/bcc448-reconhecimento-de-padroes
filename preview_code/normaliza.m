function M = normaliza(M)
  Mmax = max(M);
  M = M./repmat(Mmax,size(M,1),1);
end
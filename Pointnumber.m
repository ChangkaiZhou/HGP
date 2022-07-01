function [m] = Pointnumber(parameter,X)
KernelMatric=covSEard_jitter_vshgp(parameter,X);
Keig=eig(KernelMatric);
sumeig=sum(Keig);
m=0;contribution=0;contri=Keig/sumeig;
while(contribution<0.95)
    m=m+1;
    contribution=contribution+contri(m);
end

end
function [Pv,obj]=CFSMO(fea,Rv,alpha,beta,gamma,n,reducedim)
[~,v]=size(fea);
Pv=cell(1,v);
Sv=cell(1,v);
Lv=cell(1,v);
Kv=cell(1,v);
Dv=cell(1,v);
d=zeros(v,1);
MaxIter=30;

for num = 1:v
    d(num)=size(fea{num},2);
    Dv{num}=eye(d(num));
    Pv{num}=randn(d(num),reducedim);
    Kv{num}=fea{num}'*fea{num};
end
H=randn(n,reducedim);

for iter=1:MaxIter
    %%update S
    for num=1:v
        PX=fea{num}*Pv{num};
        R=Rv{num};
        parfor ij=1:n
            G=veccomp(ij,n,PX);
            S(:,ij) = EProjSimplex_new(R(:,ij)-G(:,ij)/(4*gamma));
        end
        Sv{num}=(S+S')/2;
        Lv{num}=diag(sum(Sv{num}))-Sv{num};
    end
    
    %%update P
    for num=1:v     
        Pv{num}=(Kv{num}+alpha*Dv{num}+beta*fea{num}'*Lv{num}*fea{num})\(fea{num}'*H);
        Pi=sqrt(sum(Pv{num}.*Pv{num},2)+eps);
        diagonal=0.5./Pi;
        Dv{num}=diag(diagonal);
    end

    %%update H
    sumPX=0;
    for num=1:v
        sumPX=sumPX+fea{num}*Pv{num};
    end
    H=1/v*sumPX;

    %%obj
    objsum=0;
    for num=1:v
        objsum=objsum+norm(fea{num}*Pv{num}-H,'fro')^2+alpha*trace(Pv{num}'*Dv{num}*Pv{num})+beta*trace(Pv{num}'*fea{num}'*Lv{num}*fea{num}*Pv{num})+beta*gamma*norm(Sv{num}-Rv{num},'fro')^2;       
    end
    obj(iter)=objsum;
end

end
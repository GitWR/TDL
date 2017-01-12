function [sub_t , q_value] = compute_sub(data)
  num_t=length(data);%统计训练图像集个数
  sub_t=cell(1,num_t);
  %pca_radio=0.999999;%0.9654;%0.9665;%所保存的特征信息量
for i=1:num_t
  sample_t=zscore(data{i});%取出每一个图像集,并中心化操作
% sample_t=(data{i}-ones(400,1)*min(data{i}))*diag(1./(max(data{i})-min(data{i}))); %数据极差规范化
% mean_train=mean(sample_train,2);%求均值,按杭求均值
% sample_train=sample_train-repmat(mean_train,1,size(sample_train,2));%中心化操作
  cov_t=sample_t*sample_t';
%   cov_t=cov_t+trace(cov_t)*(1e-4)*eye(size(sample_t,1));%添加扰动，防止出现奇异
  [U,V,D]=svd(cov_t);%奇异值分解
%   [e_vector,e_value]=eig(cov_train);%特征分解
%   e_value_unsort=diag(e_value);
%   [e_value_sort,e_index]=sort(e_value_unsort,'descend');%从大到小排序
%   sum_e_value=sum(e_value_sort);
%   sum_e_value=sum_e_value*pca_radio;
%   sum_now=0;
%   num_e_value=size(e_value_sort,1);
%   for q_value=1:num_e_value
%       sum_now=sum_now+e_value_sort(q_value);
%         if (sum_now>=sum_e_value) 
%             break;
%         end
%   end
%   if (q_value > 50 && q_value-50 < 5)
%        q_value=50;
%   elseif (q_value-50 >= 5)
%        q_value=q_value;
%   end
%   e_vector=fliplr(e_vector);%按照特征值的顺序，将特征向量降序排列
%   sub_train{i}=e_vector(:,1:q_value);%训练图像集的子空间
%     V_diag=diag(V);%放到一个列向量里面
%     sum_V_diag=sum(V_diag);%求和
%     sum_V_diag=sum_V_diag*pca_radio;%目标信息量
%     sum_now=0;
%     num_e=size(V_diag,1);%特征值个数
%     for q_value=1:num_e
%         sum_now=sum_now+V_diag(q_value);
%         if (sum_now >= sum_V_diag)
%             break;
%         end
%     end
%     if (q_value > 40 && q_value-40 < 3)
%         q_value=40;
%     elseif (q_value-40 >= 3)
%         q_value=q_value;
%     end
    q_value=10;
    sub_t{i}=U(:,1:q_value);%取出目标维数下的数据
    %sub_t{i}=sub_t{i}+trace(cov_t)*(1e-10)*eye(size(cov_t,1),q_value);
   %从上面计算来看，线性子空间维数q=40，而论文中的D=400，这两个参数仅针对ETH-80数据集
end
end


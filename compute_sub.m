function [sub_t , q_value] = compute_sub(data)
  num_t=length(data);%ͳ��ѵ��ͼ�񼯸���
  sub_t=cell(1,num_t);
  %pca_radio=0.999999;%0.9654;%0.9665;%�������������Ϣ��
for i=1:num_t
  sample_t=zscore(data{i});%ȡ��ÿһ��ͼ��,�����Ļ�����
% sample_t=(data{i}-ones(400,1)*min(data{i}))*diag(1./(max(data{i})-min(data{i}))); %���ݼ���淶��
% mean_train=mean(sample_train,2);%���ֵ,�������ֵ
% sample_train=sample_train-repmat(mean_train,1,size(sample_train,2));%���Ļ�����
  cov_t=sample_t*sample_t';
%   cov_t=cov_t+trace(cov_t)*(1e-4)*eye(size(sample_t,1));%����Ŷ�����ֹ��������
  [U,V,D]=svd(cov_t);%����ֵ�ֽ�
%   [e_vector,e_value]=eig(cov_train);%�����ֽ�
%   e_value_unsort=diag(e_value);
%   [e_value_sort,e_index]=sort(e_value_unsort,'descend');%�Ӵ�С����
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
%   e_vector=fliplr(e_vector);%��������ֵ��˳�򣬽�����������������
%   sub_train{i}=e_vector(:,1:q_value);%ѵ��ͼ�񼯵��ӿռ�
%     V_diag=diag(V);%�ŵ�һ������������
%     sum_V_diag=sum(V_diag);%���
%     sum_V_diag=sum_V_diag*pca_radio;%Ŀ����Ϣ��
%     sum_now=0;
%     num_e=size(V_diag,1);%����ֵ����
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
    sub_t{i}=U(:,1:q_value);%ȡ��Ŀ��ά���µ�����
    %sub_t{i}=sub_t{i}+trace(cov_t)*(1e-10)*eye(size(cov_t,1),q_value);
   %��������������������ӿռ�ά��q=40���������е�D=400�����������������ETH-80���ݼ�
end
end


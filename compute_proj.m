function data_low = compute_proj( sub_data , T ,q )
   len=length(sub_data);%统计训练样本维数
   data_low=cell(1,len);
   %q=size(sub_data{1},2);%线性子空间的维数，也即论文中的q
   q_dim=q;
   for i=1:len
       single_sample=sub_data{i};%取出每一幅训练图像集
       proj_sample=T'*single_sample;%论文中的W’*Y_change
       [Q_p , R_p]=qr(proj_sample,0);
       temp=Q_p(:,1:q_dim);
       data_low{i}=temp*temp'; %论文中的公式（2）也即投影后的低维流形
   end
   
end


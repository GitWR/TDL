function data_low = compute_proj( sub_data , T ,q )
   len=length(sub_data);%ͳ��ѵ������ά��
   data_low=cell(1,len);
   %q=size(sub_data{1},2);%�����ӿռ��ά����Ҳ�������е�q
   q_dim=q;
   for i=1:len
       single_sample=sub_data{i};%ȡ��ÿһ��ѵ��ͼ��
       proj_sample=T'*single_sample;%�����е�W��*Y_change
       [Q_p , R_p]=qr(proj_sample,0);
       temp=Q_p(:,1:q_dim);
       data_low{i}=temp*temp'; %�����еĹ�ʽ��2��Ҳ��ͶӰ��ĵ�ά����
   end
   
end


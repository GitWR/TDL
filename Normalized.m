function Y_change = Normalized( Y , T )
%�ò��ֵ�Ŀ������Y'������Y
  len=length(Y);
  Y_change=cell(1,len);%���ڴ洢�淶���������
  for i=1:len
      Y_sample=Y{i};%ȡ��ÿһ��ѵ��ͼ��
      transform=T'*Y_sample;%�����е�W'*Y
      [Q , R]=qr(transform,0);%qr�ֽ�
      for j=1:size(Q,2)
          if (Q(:,j)'*transform(:,j)<0)
              R(j,:)=-R(j,:); %Ϊ�˱�֤�������õ���Y'��������
          end
      end
      Y_sample_change=Y_sample*inv(R);%�����е�Y��
      Y_change{i}=Y_sample_change;
  end
end


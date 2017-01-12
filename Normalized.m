function Y_change = Normalized( Y , T )
%该部分的目的是用Y'来代替Y
  len=length(Y);
  Y_change=cell(1,len);%用于存储规范化后的数据
  for i=1:len
      Y_sample=Y{i};%取出每一个训练图像集
      transform=T'*Y_sample;%论文中的W'*Y
      [Q , R]=qr(transform,0);%qr分解
      for j=1:size(Q,2)
          if (Q(:,j)'*transform(:,j)<0)
              R(j,:)=-R(j,:); %为了保证后面计算得到的Y'的正定性
          end
      end
      Y_sample_change=Y_sample*inv(R);%论文中的Y’
      Y_change{i}=Y_sample_change;
  end
end


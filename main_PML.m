clc
clear ;
%step1��import the toolbox of manopt
addpath(genpath(pwd));
%importlocal_manopt;%���������Ϲ����ݶȷ��İ�
% load ImgData_HE_itera
load ImgData_HE_Honda
% load Youtube_New_1-40
%sub_dim=?;�����ͨ����������Ϣ�����õ�
Train_labels=zeros(1,40);
Test_labels=zeros(1,32);
%step2��Ϊѵ��������ӱ�ǩ
l=5;
k=l;
a=linspace(1,8,8);%����40��ѽ
i1=1;
while(k<=40)
    while(i1<=8)
        for i=1:8
            i_train=l*(i-1)+1;
            Train_labels(i_train:k)=a(i1);
            k=k+5;
            i1=i1+1;
        end
    end
end

%step3:Ϊ����������ӱ�ǩ
l1=4;
k1=l1; 
a1=linspace(1,8,8);%����40��ѽ
i2=1;
while(k1<=32)
    while(i2<=8)
        for i=1:8
            i_test=l1*(i-1)+1;
            Test_labels(i_test:k1)=a(i2);
            k1=k1+4;
            i2=i2+1;
        end
    end
end
d=200:20:220;%Ϊ��׼ȷȷ�������ӿռ��ά��q������ͨ�����ֶ������ķ�ʽȥ��̽
num_d=length(d);%���ж��ٸ�
all_accuracy=cell(1,num_d);%������е�������µľ���
accuracy_matrix=zeros(1,10);
accuracy_average_final=zeros(1,num_d);%���ÿ����ͬ��d�µ�ƽ�����Ծ���
for i_d=1:num_d
  d_value=d(i_d);
  %fprintf(1,'��%d��Ŀ��ά��ʱ\n',i_d);
 for iter=1:10
%step4������ѵ���Լ������������ӿռ�
    Data_train=All_ImgData_HE_train{iter};%ȡ��ÿһ��ѵ������
    Data_test=All_ImgData_HE_test{iter};%ȡ��ÿһ���������
    %����ETH-80���ݼ�������ѵ���Լ����Ե�ͼ�񼯸�����ͬ�������ͬһ��compute_sub��������
    t_star_train1=cputime;%��ʱ��
    [sub_train , q1]=compute_sub(Data_train);
    t_train1= cputime - t_star_train1;%��������ʱ��q
    clear cputime;
    t_star_test1=cputime;%��ʱ��;
    [sub_test , q2]=compute_sub(Data_test);
    t_test1= cputime - t_star_test1;
%step5:PML-����ͶӰ����W
    t_star_train2=cputime;%��ʱ��
    T=compute_PML(sub_train,Train_labels,d_value);  
    t_train2=cputime-t_star_train2;
    clear cputime;
    t_star_train3=cputime;%��ʱ��
    %fprintf('det_log: %d',J);
%step6:����ͶӰ��õ��ĵ�γ����
    low_Y_train=compute_proj(sub_train,T,q1);%ѵ������ͶӰ��ĵ�γ������
    t_train3=cputime-t_star_train3;
    clear cputime;
    t_star_test2=cputime;%��ʱ��;
    low_Y_test=compute_proj(sub_test,T,q2);%��������ͶӰ��ĵ�γ������
%step7:PML�ϼ򵥵ķ���
    dist=zeros(size(Train_labels,2),size(Test_labels,2));%���ڴ�ž���ľ���
 for i_dist=1:size(Train_labels,2)
        Y_train=low_Y_train{i_dist};%һ��ѵ��ͼ��
     for j_dist=1:size(Test_labels,2)
        Y_test=low_Y_test{j_dist};%һ������ͼ��
        Y_dist=Y_train-Y_test;
        dist(i_dist,j_dist)=norm(Y_dist(:),'fro');%�������
     end
 end
    test_num=size(Test_labels,2);%����������
    [dist_sort,index] = sort(dist,1,'ascend');%�Ѿ��밴����������
    %right_num=length(find((Test_labels'-Train_labels'(index(1,:)))==0));
    right_num = length(find((Test_labels'-Train_labels(index(1,:))')==0)); %ͳ�Ƴ���ȷ����Ĳ�����������
    accuracy=right_num/test_num;%����
    t_test2=cputime-t_star_test2;
    clear cputime;
    accuracy_matrix(iter)=accuracy*100;
    fprintf(1,'��%d�ε���׼ȷʶ�����������Ϊ��%d %d\n',iter,right_num );
    fprintf(1,'��%d�ε����ľ���Ϊ: %d %d\n', iter ,accuracy*100);
 end
    accuracy_average_final(i_d)=sum(accuracy_matrix)/iter;
    fprintf(1,'��%d��Ŀ��ά��ʱ��ƽ�����Ծ���Ϊ: %d %d\n',i_d,accuracy_average_final(i_d));
    all_accuracy{i_d}=accuracy_matrix;
end
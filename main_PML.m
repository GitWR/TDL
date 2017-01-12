clc
clear ;
%step1：import the toolbox of manopt
addpath(genpath(pwd));
%importlocal_manopt;%导入流形上共轭梯度法的包
% load ImgData_HE_itera
load ImgData_HE_Honda
% load Youtube_New_1-40
%sub_dim=?;最好是通过保留的信息量来得到
Train_labels=zeros(1,40);
Test_labels=zeros(1,32);
%step2：为训练样本添加标签
l=5;
k=l;
a=linspace(1,8,8);%共有40类呀
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

%step3:为测试样本添加标签
l1=4;
k1=l1; 
a1=linspace(1,8,8);%共有40类呀
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
d=200:20:220;%为了准确确定线性子空间的维数q，我们通过这种定步长的方式去试探
num_d=length(d);%共有多少个
all_accuracy=cell(1,num_d);%存放所有单个情况下的精度
accuracy_matrix=zeros(1,10);
accuracy_average_final=zeros(1,num_d);%存放每个不同的d下的平均测试精度
for i_d=1:num_d
  d_value=d(i_d);
  %fprintf(1,'第%d个目标维数时\n',i_d);
 for iter=1:10
%step4：构建训练以及测试样本的子空间
    Data_train=All_ImgData_HE_train{iter};%取出每一组训练数据
    Data_test=All_ImgData_HE_test{iter};%取出每一组测试数据
    %由于ETH-80数据集中用于训练以及测试的图像集个数相同，因此用同一个compute_sub函数即可
    t_star_train1=cputime;%计时用
    [sub_train , q1]=compute_sub(Data_train);
    t_train1= cputime - t_star_train1;%机器运行时间q
    clear cputime;
    t_star_test1=cputime;%计时用;
    [sub_test , q2]=compute_sub(Data_test);
    t_test1= cputime - t_star_test1;
%step5:PML-计算投影矩阵W
    t_star_train2=cputime;%计时用
    T=compute_PML(sub_train,Train_labels,d_value);  
    t_train2=cputime-t_star_train2;
    clear cputime;
    t_star_train3=cputime;%计时用
    %fprintf('det_log: %d',J);
%step6:计算投影后得到的低纬流形
    low_Y_train=compute_proj(sub_train,T,q1);%训练样本投影后的低纬度流形
    t_train3=cputime-t_star_train3;
    clear cputime;
    t_star_test2=cputime;%计时用;
    low_Y_test=compute_proj(sub_test,T,q2);%测试样本投影后的低纬度流形
%step7:PML上简单的分类
    dist=zeros(size(Train_labels,2),size(Test_labels,2));%用于存放距离的矩阵
 for i_dist=1:size(Train_labels,2)
        Y_train=low_Y_train{i_dist};%一个训练图像集
     for j_dist=1:size(Test_labels,2)
        Y_test=low_Y_test{j_dist};%一个测试图像集
        Y_dist=Y_train-Y_test;
        dist(i_dist,j_dist)=norm(Y_dist(:),'fro');%计算距离
     end
 end
    test_num=size(Test_labels,2);%测试样本数
    [dist_sort,index] = sort(dist,1,'ascend');%把距离按列升序排列
    %right_num=length(find((Test_labels'-Train_labels'(index(1,:)))==0));
    right_num = length(find((Test_labels'-Train_labels(index(1,:))')==0)); %统计出正确分类的测试样本数。
    accuracy=right_num/test_num;%精度
    t_test2=cputime-t_star_test2;
    clear cputime;
    accuracy_matrix(iter)=accuracy*100;
    fprintf(1,'第%d次迭代准确识别的样本个数为：%d %d\n',iter,right_num );
    fprintf(1,'第%d次迭代的精度为: %d %d\n', iter ,accuracy*100);
 end
    accuracy_average_final(i_d)=sum(accuracy_matrix)/iter;
    fprintf(1,'第%d个目标维数时，平均测试精度为: %d %d\n',i_d,accuracy_average_final(i_d));
    all_accuracy{i_d}=accuracy_matrix;
end
# -*- coding: utf-8 -*-
"""
Created on Mon May 11 21:00:26 2020
@author: DELL
mnist数字识别，单隐层神经网络。
"""
 
 
# 1、载入数据
# MNIST 数据集可在 http://yann.lecun.com/exdb/mnist/ 获取
import tensorflow as tf
import tensorflow.examples.tutorials.mnist.input_data as input_data
mnist = input_data.read_data_sets("MNST_data/",one_hot=True)
# 该站点上有四个文件：
# train-images-idx3-ubyte.gz：训练集图像（9912422字节）
# train-labels-idx1-ubyte.gz：训练集标签（28881 字节）
# t10k-images-idx3-ubyte.gz：测试集图像（1648877字节）
# t10k-labels-idx1-ubyte.gz：测试集标签（4542字节）
 
# 2、构建模型
# 定义待输入数据的占位符
x = tf.placeholder(tf.float32, [None, 784], name ="X")  # mnist中每张图片共有28*28=784个像素点
y = tf.placeholder(tf.float32,[None, 10], name="Y")     # 0-9一共10个数字->10个类别
 
# 构建隐藏层
H1_NN = 256
 
W1 = tf.Variable(tf.random.normal([784,H1_NN])) # 正态分布随机数
b1 = tf.Variable(tf.zeros([H1_NN]))
 
Y1 = tf.nn.relu(tf.matmul(x,W1)+b1)
 
# 构建输出层
W2 = tf.Variable(tf.random.normal([H1_NN,10]))
b2 = tf.Variable(tf.zeros([10]))
 
forward = tf.matmul(Y1,W2) + b2    # 定义前向计算
pred = tf.nn.softmax(forward)   # 结果分类
 
 
# 3、训练模型
# 设置训练参数
train_epochs = 30                                       # 训练轮数
batch_size = 50                                        # 单次训练样本数（批次大小）
total_batch = int(mnist.train.num_examples/batch_size)  # 一轮训练有多少批次
display_step = 2                                        # 显示粒度
learning_rate = 0.01                                    # 学习率
 
# 定义损失函数
# loss_function = tf.reduce_mean(-tf.reduce_sum(y*tf.log(pred),reduction_indices=1)) # 交叉熵
loss_function = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=y,logits=forward)) # tensorflow提供带softmax的交叉熵函数
                                                                                                   # 用于避免因为log(0)值为NaN造成的数据不稳定
 
# 选择优化器
# optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss_function) # 梯度下降优化器
optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate).minimize(loss_function)
 
# 定义准确率
correct_prediction = tf.equal(tf.argmax(pred,1),tf.argmax(y,1))     # 检查预测类别与实际类别的匹配情况
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))  # 将布尔值转化为浮点数，并计算平均值
 
# 记录训练开始时间
from time import time
startTime=time()
 
sess = tf.compat.v1.Session()                         # 声明会话
init = tf.compat.v1.global_variables_initializer()    # 变量初始化
sess.run(init)
 
# 开始训练
for epoch in range(train_epochs):
    for batch in range(total_batch):
        xs,ys = mnist.train.next_batch(batch_size)  # 读取批次数据
        sess.run(optimizer,feed_dict={x:xs,y:ys})   # 执行批次训练
    
    # total_batch个批次训练完成后，使用验证数据计算误差与准确率；验证集没有分批
    loss,acc = sess.run([loss_function,accuracy],
                        feed_dict={x:mnist.validation.images,
                                   y:mnist.validation.labels})
    
    # 打印训练过程中的详细信息
    if (epoch+1) % display_step ==0:
        print("Train Epoch:",'%02d' %(epoch+1), "Loss=","{:.9f}".format(loss),\
              "Accuracy=","{:.4f}".format(acc))
print("Train Finished!")
 
# 显示运行总时间
duration = time()-startTime
print("Train Finished takes:","{:.2f}".format(duration))
 
 
# 4、评估模型
# 完成训练后，在验证集上评估模型的准确率
accu_validation = sess.run(accuracy,feed_dict={x:mnist.validation.images,y:mnist.validation.labels})
print("Test Accuracy:",accu_validation)
# Test Accuracy: 0.9058
 
# 完成训练后，在测试集上评估模型的准确率
accu_test = sess.run(accuracy,feed_dict={x:mnist.test.images,y:mnist.test.labels})
print("Test Accuracy:",accu_test)
# Test Accuracy: 0.9056
 
 
# 5、模型应用与可视化
# 在建立模型并进行训练后，若认为准确率可以接受，则可以使用此模型进行预测
# 由于pred预测结果是One-hot编码格式，所以需要转换为0-9数字
prediction_result = sess.run(tf.argmax(pred,1),
                             feed_dict={x:mnist.test.images})
# 查看预测结果中的前10项
prediction_result[0:10]     # array([7, 2, 1, 0, 4, 1, 4, 9, 6, 9], dtype=int64)
# 定义可视化函数
import matplotlib.pyplot as plt
import numpy as np
def plot_images_labels_prediction(images,       # 图像列表
                                  labels,       # 标签列表
                                  prediction,   # 预测值列表
                                  index,        # 从第index个开始显示
                                  num=10):      # 缺省显示10幅
    fig = plt.gcf()                     # 获取当前图表，get current figure
    fig.set_size_inches(10,12)          # 1英寸 = 2.54cm
    if num>25:
        num = 25                        # 最多显示25个子图
    for i in range(0,num):
        ax = plt.subplot(5,5,i+1)       # 获取当前要处理的子图
        ax.imshow(np.reshape(images[index],(28,28)),cmap='binary')   # 显示第index个图像
        title = "label=" + str(np.argmax(labels[index]))             # 构建该图上要显示的title信息
        if len(prediction)>0:
            title += ", predict=" + str(prediction[index])
        ax.set_title(title,fontsize=10)
        ax.set_xticks([])               # 不显示坐标轴
        ax.set_yticks([])
        index+=1
    plt.show()
 
plot_images_labels_prediction(mnist.test.images,mnist.test.labels,prediction_result,0,10)
 
plot_images_labels_prediction(mnist.test.images,mnist.test.labels,[],0,5)
 
 
# 6、找出预测错误
compare_lists = prediction_result==np.argmax(mnist.test.labels,1)
print(compare_lists)
 
err_lists =[i for i in range(len(compare_lists)) if compare_lists[i]==False]
print(err_lists, len(err_lists))
 
# 定义一个输出错误分类的函数
import numpy as np
def print_predict_errs(labels,      # 标签列表
                       prediction): # 预测值列表
    count = 0
    compare_lists = (prediction==np.argmax(labels,1))
    err_lists =[i for i in range(len(compare_lists)) if compare_lists[i]==False]
    for x in err_lists:
        print("index="+str(x)+
              "标签值=",np.argmax(labels[x]),
              "预测值=",prediction[x])
        count+=1
    print("总计："+str(count))
 
print_predict_errs(labels=mnist.test.labels,
                   prediction=prediction_result)
 
plot_images_labels_prediction(mnist.test.images,mnist.test.labels,prediction_result,9745,20)
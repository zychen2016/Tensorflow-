import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="3"
#coding:utf-8
#酸奶成本为1元，酸奶利润为9元
#预测少了损失大，故不要预测少，故生成的模型预测多一些
#0导入模块，生成数据集
import
import tensorflow as tf
import numpy as np
BATCH_SIZE=8
SEED=23455
COST=1
PROFIT=9
#产生伪随机数
rdm=np.random.RandomState(SEED)
X=rdm.rand(32,2)
Y_=[[x1+x2+(rdm.rand()/10.0-0.05)]for (x1,x2) in X]#X为32行两列的矩阵，x1、x2为每一行的第一个第二个数
#定义神经网络的输入、参数和输出，定义前向传播过程。
x=tf.placeholder(tf.float32,shape=(None,2))#占位符，shape中的None参数表示第一维数目不限定，这样比较方便
y_=tf.placeholder(tf.float32,shape=(None,1))
w1=tf.Variable(tf.random_normal([2,1],stddev=1,seed=1))#产生随机分布，两行一列的格式
y=tf.matmul(x,w1)
#2定义损失函数及反向传播方法
#定义损失函数使得预测少了的损失大，于是模型应该偏向多的方向预测。
#loss_mse=tf.reduce_mean(tf.square(y_ -y))
#train_step=tf.train.GradientDescentOptimizer(0.001).minimize(loss_mse)
loss=tf.reduce_sum(tf.where(tf.greater(y,y_),(y-y_)*COST,(y_-y)*PROFIT))
train_step=tf.train.GradientDescentOptimizer(0.001).minimize(loss)
#3生成会话，训练STEPS论
with tf.Session() as sess:
    init_op=tf.global_variables_initializer()##所有参数一起初始化
    sess.run(init_op)#运行会话才能产生初值
    STEPS=20000
    for i in range(STEPS):
        start=(i*BATCH_SIZE)%32
        end=(i*BATCH_SIZE)%32+BATCH_SIZE
        #print("After %d steps,start is: %d"%(i,start))
        #print("After %d steps,end is: %d"%(i,end))
        sess.run(train_step,feed_dict={x:X[start:end],y_:Y_[start:end]})
        if i%500==0:#每500步后输出优化的参数值
            print("After %d training steps,w1 is : "%(i))
            print(sess.run(w1),"\n")
    print("Final w1 is: \n",sess.run(w1))

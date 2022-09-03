import tensorflow
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import numpy as np
import target_functions as tgf
import matplotlib.pyplot as pltpythom



class NN(object):
    #
    def __init__(self, learning_rate, num_of_training, input_size, hidden_n, parameters):
        self.learning_rate = learning_rate
        self.num_of_training = num_of_training

        # 设置好占位符
        self.X = tf.placeholder(tf.float32, shape=[None, input_size])
        self.input_size = input_size
        self.sess = tf.Session()


        '''
        需要注意的地方是，初始化权重的时候还需要再注意！！！！！！！
        '''

        # 这些是变量，因为会在训练过程中发生变化
        #

        #
        weight1 = []
        for i in range(hidden_n):
            weight1.append(parameters[i*input_size:(i+1)*input_size])
        weight2 = [parameters[hidden_n*input_size:]]
        weight1 = tf.convert_to_tensor(np.array(weight1).T)
        weight2 = tf.convert_to_tensor(np.array(weight2).T)

        weight1 = tf.cast(weight1, dtype=tf.float32)
        weight2 = tf.cast(weight2, dtype=tf.float32)
        self.weight1 = tf.Variable(tf.random_normal([self.input_size, hidden_n], stddev = 0.1, seed = 1))

        self.weight2 = tf.Variable(tf.random_normal([hidden_n, 1], stddev=0.1, seed=1))

        self.weight1.assign(weight1)
        self.weight2.assign(weight2)

        self.bias1 = tf.Variable(tf.constant(0.1), [hidden_n])
        self.bias2 = tf.Variable(tf.constant(0.1), [1])

        #计算图
        self.hidden_opt = tf.matmul(self.X, self.weight1) + self.bias1 # 输入层到隐藏层正向传播
        self.hidden_opt = tf.nn.sigmoid(self.hidden_opt)  # 激活函数，用于计算节点输出值

        self.final_opt = tf.matmul(self.hidden_opt, self.weight2) + self.bias2 # 隐藏层到输出层正向传播
        self.final_opt = tf.nn.relu(self.final_opt)
        # 初始化变量
        self.init = tf.global_variables_initializer()



    def predict(self, X):
        self.sess.run(self.init)
        predValue = self.sess.run(self.final_opt, feed_dict = {self.X: X})
        return predValue














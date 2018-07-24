import tensorflow as tf

W = tf.Variable(initial_value=tf.random_normal(shape=(1, 4), mean=100, stddev=0.35), name="W")



W = tf.Variable(tf.random_normal(shape=(1, 4), stddev=0.35), name="W")
# 使用W的值初始化新变量w_replica
w_replica = tf.Variable(W.initialized_value(), name="w_replica")
W_twice = tf.Variable(W.initialized_value() * 2, name="w_twice")



# 初始化全局变量
weights = tf.Variable(tf.random_normal(shape=(1, 4), stddev=0.35), name="weights")
biases = tf.Variable(tf.zeros([4]), name="biases")
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    print(sess.run([weights, biases]))



# 初始化部分变量
weights = tf.Variable(tf.random_normal(shape=(1, 4), stddev=0.35), name="weights")
biases = tf.Variable(tf.zeros([4]), name="biases")
with tf.Session() as sess:
    sess.run(tf.variables_initializer([weights]))
    print(sess.run(weights))



# 更新模型参数
W = tf.Variable(0.0, name="W")
double = tf.multiply(2.0, W)
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(4):
        sess.run(tf.assign_add(W, 1.0))
        print('W=%s, double=%s' % (sess.run(W), sess.run(double)))

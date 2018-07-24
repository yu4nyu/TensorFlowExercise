import tensorflow as tf

# 使用Saver存储变量
W = tf.Variable(0.0, name='W')
double = tf.multiply(2.0, W)
saver = tf.train.Saver({'weights':W})
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(4):
        sess.run(tf.assign_add(W, 1.0))
        saver.save(sess, '/tmp/summary/test.ckpt')

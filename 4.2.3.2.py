import tensorflow as tf

# 恢复模型参数
W = tf.Variable(0.0, name='weights')
#W = tf.Variable(weights.initialized_value(), name='W')
double = tf.multiply(2.0, W)
saver = tf.train.Saver()
with tf.Session() as sess:
    saver.restore(sess, '/tmp/summary/test.ckpt')
    print('restored:W=%s' % sess.run(W))
    for i in range(4):
        sess.run(tf.assign_add(W, 1.0))
        print('W=%s, double=%s' % (sess.run(W), sess.run(double)))

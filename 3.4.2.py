import tensorflow as tf

a = tf.constant(5.0)
b = tf.constant(6.0)
c = a * b
sess = tf.InteractiveSession()
print(c.eval())
sess.close()

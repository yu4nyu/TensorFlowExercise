import tensorflow as tf
import numpy as np

with tf.name_scope("PlaceHolderExample"):
    x = tf.placeholder(tf.float32, shape=(2, 2), name="x")
    y = tf.matmul(x, x, name="matmul")

with tf.Session() as sess:
    # print(sess.run(y)) # 直接执行会报错，因为没有为x填充数据
    rand_array = np.random.rand(2, 2)
    print(sess.run(y, feed_dict={x: rand_array}))

print()



x = tf.sparse_placeholder(tf.float32)
y = tf.sparse_reduce_sum(x)

with tf.Session() as sess:
    indices = np.array([[3, 2, 0], [4, 5, 1]], dtype=np.int64)
    values = np.array([1.0, 2.0], dtype=np.float32)
    shape = np.array([7, 9, 2], dtype=np.int64)
    print(sess.run(y, feed_dict={x: tf.SparseTensorValue(indices, values, shape)}))
    print(sess.run(y, feed_dict={x: (indices, values, shape)}))
    sp = tf.SparseTensor(indices=indices, values=values, dense_shape=shape)
    sp_value = sp.eval()
    print(sess.run(y, feed_dict={x: sp_value}))

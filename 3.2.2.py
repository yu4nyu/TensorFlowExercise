import tensorflow as tf

sp = tf.SparseTensor(indices=[[0, 2], [1, 3]], values=[1, 2], dense_shape=[3, 4])

with tf.Session() as sess:
    print(sp.eval())

print()



x = tf.SparseTensor(indices=[[0,0], [0,2], [1,1]], values=[1,1,1], dense_shape=[2,3])
reduce_x = [tf.sparse_reduce_sum(x),
            tf.sparse_reduce_sum(x, axis=1),
            tf.sparse_reduce_sum(x, axis=1, keep_dims=True),
            tf.sparse_reduce_sum(x, axis=[0, 1])]
with tf.Session() as sess:
    print(sess.run(reduce_x))

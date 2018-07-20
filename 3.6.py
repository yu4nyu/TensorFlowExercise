import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

# super parameters
learning_rate = 0.01
max_train_steps = 1000

# data
train_X = np.array([[3.3],[4.4],[5.5],[6.71],[6.93],[4.168],[9.779],[6.182],[7.59],[2.167],[7.042],[10.791],[5.313],[7.997],[5.654],[9.27],[3.1]], dtype=np.float32)
train_Y = np.array([[1.7], [2.76],[2.09],[3.19],[1.694],[1.573],[3.366],[2.596],[2.53],[1.221],[2.827],[3.465],[1.65],[2.904],[2.42],[2.94],[1.3]], dtype=np.float32)
total_samples = train_X.shape[0]

# model
X = tf.placeholder(tf.float32, [None, 1]) # None means it takes any number of input, 1 means 1 dimension.
W = tf.Variable(tf.random_normal([1, 1]), name="weight")
b = tf.Variable(tf.zeros([1]), name="bias")
Y = tf.matmul(X, W) + b

# loss function
Y_ = tf.placeholder(tf.float32, [None, 1])
loss = tf.reduce_sum(tf.pow(Y-Y_, 2)/(total_samples))

# optimizer
optimizer = tf.train.GradientDescentOptimizer(learning_rate)

# one step operation
train_op = optimizer.minimize(loss)

# training
log_step = 10
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    print("Start training:")
    for step in range(max_train_steps):
        sess.run(train_op, feed_dict={X: train_X, Y_: train_Y})
        # print the log for each log_step steps
        if step % log_step == 0:
            c = sess.run(loss, feed_dict={X: train_X, Y_: train_Y})
            print("Step:%d, loss==%.4f, W==%.4f, b==%.4f" % (step, c, sess.run(W), sess.run(b)))
    final_loss = sess.run(loss, feed_dict={X: train_X, Y_: train_Y})
    weight, bias = sess.run([W, b])
    print("Step:%d, loss==%.4f, W==%.4f, b==%.4f" % (max_train_steps, final_loss, sess.run(W), sess.run(b)))
    print("Linear Regression Model: Y==%.4f*X+%.4f" % (weight, bias))

# plot
plt.plot(train_X, train_Y, 'ro', label="Training data")
plt.plot(train_X, weight * train_X + bias, label='Fitted line')
plt.legend()
plt.show()

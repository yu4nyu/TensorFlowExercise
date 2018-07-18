import tensorflow as tf

a = tf.constant(1.0)
b = tf.constant(2.0)
c = tf.add(a, b)
print([a, b, c])

with tf.Session() as sess:
    print(c.eval())
    print(sess.run([a, b, c]))



a = tf.constant([1, 1])
b = tf.constant([2, 2])
c = tf.add(a, b)
with tf.Session() as sess:
    print("a[0]=%s, a[1]=%s" % (a[0].eval(), a[1].eval()))
    print("c.name=%s" % c.name)
    print("c.value=%s" % c.eval())
    print("c.shape=%s" % c.shape)
    print("a.consumers=%s" % a.consumers())
    print("b.consumers=%s" % b.consumers())
    print("[c.op]:\n%s" % c.op)

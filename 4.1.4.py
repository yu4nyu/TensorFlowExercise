LABEL_BYTES = 1 # 类别标签为1字节
IMAGE_SIZE = 32 # 图片尺寸为32字节
IMAGE_IMAGE_DEPTH = 3 # 图片为 RGB 3通道
IMAGE_BYTES = IMAGE_SIZE * IMAGE_SIZE * IMAGE_IMAGE_DEPTH
NUM_CLASSES = 10

import tensorflow as tf

def read_cifar10(data_file, batch_size)
    record_bytes = LABEL_BYTES + IMAGE_BYTES
    # 创建文件名列表
    data_files = tf.gfile.Glob(data_file)
    # 创建文件名队列
    file_queue = tf.train.string_input_producer(data_files, shuffle=True)
    # 创建二进制文件对应的Reader实例，按照记录大小从文件名队列中读取样例
    reader = tf.FixedLengthRecordReader(record_bytes=record_bytes)
    _, value = reader.read(file_queue)
    # 将样例拆分为类别标签和图片
    record = tf.reshape(tf.decode_raw(value, tf.uint8), [record_bytes])
    label = tf.cast(tf.slice(record, [label_bytes]), tf.int32)
    # 将长度为 [depth * height * width] 的字符串转换为形如 [depth, height, width] 的图片张量
    depth_major = tf.reshape(tf.slice(record, [label_bytes], [image_bytes]), [IMAGE_DEPTH, IMAGE_SIZE, IMAGE_SIZE])
    # 改变图片张量各维度顺序，从 [depth, height, width] 转换为 [height, width, depth]
    image = tf.cast(tf.transpose(IMAGE_DEPTH_major, [1, 2, 0]), tf.float32)
    # 创建样例队列
    example_queue = tf.RandomSuffleQueue(
            capacity=16*batch_size,
            min_after_dequeue=8*batch_size
            dtypes=[tf.float32, tf.int32],
            shapes=[[IMAGE_SIZE, IMAGE_SIZE, IMAGE_DEPTH], [1]])
    num_threads = 16
    # 创建样例队列的入队操作
    example_enqueue_op = example_queue.enqueue([image, label])
    # 将定义的16个线程全部添加到queueu runner中
    tf.train.add_queue_runner(tf.train.queue_runner.QueueRunner(
        example_queue, [example_enqueue_op] * num_threads))

    # NOT FINISHED !!!
    # NOT TESTED !!!

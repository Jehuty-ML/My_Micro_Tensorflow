
import os
import numpy as np
import tensorflow as tf
"""
1、输入占位符的机制改进。 32个占位符  ---->  1个占位符。
2、使用one-hot 作为输入获得嵌入的输出 ---->  使用嵌入矩阵查找表。
3、预测： 取top 1 的值  ----> 取top 5的值。
         做藏头诗预测   的改进。
"""

class Tensor(object):
    def __init__(self, number_time_steps, num_units, vocab_size, embedding_size, learning_rate):
        """
        构建模型图
        :param number_time_steps:   时间步  实际是32
        :param num_units:           隐藏层节点数量（隐藏层神经元的个数）
        :param vocab_size:          词表的大小。（设置了8000）
        :param embedding_size:      词向量的长度  （一般设置 200--800）
        :param learning_rate:       学习率
        """
        self.learning_rate = learning_rate
        self.keep_prob = tf.placeholder(dtype=tf.float32, name='keep_prob')

        with tf.variable_scope('Net', initializer=tf.random_normal_initializer(stddev=0.1)):
            batch_size = tf.placeholder(dtype=tf.int32, shape=[])
            # 1、定义rnn隐藏层(2层隐藏层)
            cell = tf.nn.rnn_cell.MultiRNNCell(cells=[
                tf.nn.rnn_cell.BasicLSTMCell(num_units=num_units),
                tf.nn.rnn_cell.GRUCell(num_units=num_units)]
            )
            # 对细胞核添加 丢弃
            cell = tf.nn.rnn_cell.DropoutWrapper(cell=cell, output_keep_prob=self.keep_prob)
            # 2、获取取得初始化的状态信息
            state = cell.zero_state(batch_size=batch_size, dtype=tf.float32)

            # 3、对每个时刻进行遍历，进行数据的输入操作。
            self.rnn_outputs = []
            # 输入数据的占位符
            self.inputs = tf.placeholder(tf.int32, shape=[None, number_time_steps])
            inputs = tf.split(self.inputs, num_or_size_splits=number_time_steps, axis=1)

            # 创建 inputs---embedding 层的嵌入矩阵。
            with tf.variable_scope('embeding', reuse=tf.AUTO_REUSE):
                # fixme 因为该矩阵是权重共享（在不同时刻 这个权重是重用的）
                embeding_weights = tf.get_variable('w', shape=[vocab_size, embedding_size])

            for time in range(number_time_steps):
                with tf.variable_scope('rnn_{}'.format(time)):
                    # a、定义当前时刻的输入, 并传入嵌入层，获取嵌入层的输出。
                    temp_input_x = tf.nn.embedding_lookup(
                        params=embeding_weights, ids=inputs[time]
                    )
                    # temp_input_x shape = (batch_size, 1, 300)

                    temp_input_x = tf.squeeze(temp_input_x, axis=1)
                    # shape = [batch_size, embedding_size]

                    # b、调用rnn 细胞核的call方法，获取rnn隐藏层输出和新的状态值
                    rnn_output, state = cell(temp_input_x, state)

                    # d、将当前时刻的隐藏层输出保存
                    self.rnn_outputs.append(rnn_output)

            # 构建最终输出。
            losses = []
            self.y_predicts = []  # 预测的真实值列表
            self.y_preds = []  # 预测的概率值列表
            self.targets = tf.placeholder(tf.int32, shape=[None, number_time_steps])
            targets = tf.split(self.targets, num_or_size_splits=number_time_steps, axis=1)
            # targets [[batch_size, 1], [batch_size, 1], [batch_size, 1], ....]

            # 创建 输出层的矩阵。
            with tf.variable_scope('fc_', reuse=tf.AUTO_REUSE):
                # fixme 初始化rnn_output --> logitd的参数（在不同时刻 这个权重是共享的）
                w = tf.get_variable('w', shape=[num_units, vocab_size])
                b = tf.get_variable('b', shape=[vocab_size])

            for time in range(number_time_steps):
                with tf.variable_scope('FC_{}'.format(time)):
                    # a、获取对应时刻的 rnn隐藏层的输出
                    r_output = self.rnn_outputs[time]
                    logits = tf.add(tf.matmul(r_output, w), b)

                    # b、获取真实的预测值。
                    y_predict = tf.argmax(logits, axis=1)
                    self.y_predicts.append(y_predict)

                    y_pred = tf.nn.softmax(logits)
                    self.y_preds.append(y_pred)

                    # d、计算当前时刻的损失
                    tmp_loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
                        logits=logits,
                        labels=tf.reshape(targets[time], shape=[-1])
                    ))
                    losses.append(tmp_loss)

            self.loss = tf.reduce_mean(losses)

        # 构建模型优化器
        with tf.variable_scope('train'):
            optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
            self.train_op = optimizer.minimize(self.loss)
        self.batch_size = batch_size


def chinese_to_index(text):
    rezult = []
    bs = text.encode('gb2312')  # 转换成了字节bytes格式
    bs_list = [b for b in bs]
    # print(bs_list)
    num = len(bs_list)
    i = 0
    while i < num:
        b = bs[i]
        # 如果取值小于160，表示是单个的字符（这里很特殊）
        if b <=160:
            rezult.append(b)
        else:
            # 计算区码（大于160的先计算区码）

            block = b - 160
            if block >=16:
                # 因为10--15区为空，不需要考虑
                block -= 6

            # 计算在当前区有多少汉字。（计算位置码）
            block -= 1
            i += 1
            b2 = bs[i]

            # 基于区码+位置码 计算出1个数字（这个数字就是 tokenize）
            rezult.append(block*94 + b2)
        i += 1
    return rezult


def index_to_chinese(index_list):
    """
    功能：{整数：单词}
    :param index_list:
    :return:
    """
    result = ''
    for index in index_list:
        if index <= 160:
            result += chr(index)
        else:
            index = index - 161
            block = int(index / 94) +1
            if block >= 10:
                block += 6
            block += 160
            # 位置码
            location = int(index % 94) + 161
            result += str(bytes([block, location]), encoding='gb2312')
    return result


def read_poems(path='../qts_7X4.txt'):
    """
    读入唐诗数据
    :param path:
    :return:
    """
    rezult = []
    error = 0
    with open(path, mode='r', encoding='utf-8') as reader:
        for line in reader:
            # 对每行数据进行处理，去除掉前后空格。
            line = line.strip()
            length = len(line)
            try:
                if length == 32:
                    index = chinese_to_index(line)
                    rezult.append(index)
                else:
                    error += 1
            except:
                error += 1
    print('成功获取诗歌:{} - 错误:{}'.format(len(rezult), error))
    return rezult


def fetch_samples(path='../qts_7X4.txt'):
    """
    基于原始数据构建 X和Y
    :param path:
    :return:
    """
    total_samples = 0
    X = read_poems(path)
    Y = []
    for xi in X:
        total_samples +=1
        # 使用前一个字预测下一个字。
        yi = xi[1:]
        yi.append(10)  # 目标yi最后会少1个字符，用10代替。
        # 添加到Y的列表当中。
        Y.append(yi)
    # 将X 和 Y 转换成为 numpy 数组
    X = np.asarray(X)
    Y = np.asarray(Y)
    print(X.shape, Y.shape)
    return total_samples, X, Y


def create_dir_with_no_exists(path):
    if not os.path.exists(path):
        os.makedirs(path)
        print('成功创建文件路径:{}'.format(path))


def train(train_data_path, checkpoint_dir, max_epochs=100, batch_size=64, num_units=128):
    """
    模型训练
    :param train_data_path:
    :param checkpoint_dir:
    :param max_epochs:
    :param batch_size:
    :param num_units:
    :return:
    """
    create_dir_with_no_exists(checkpoint_dir)

    with tf.Graph().as_default():
        # 一、构建模型网络
        tensor = Tensor(
            number_time_steps=32, num_units=num_units, vocab_size=8000,
            embedding_size=300, learning_rate=0.001
        )

        # 二、执行模型图
        with tf.Session() as sess:
            # 1、创建持久化对象
            saver = tf.train.Saver(max_to_keep=1)

            # 2、模型恢复或者模型参数初始化
            ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
            if ckpt and ckpt.model_checkpoint_path:
                saver.restore(sess, ckpt.model_checkpoint_path)
                print('恢复模型继续训练!')
            else:
                sess.run(tf.global_variables_initializer())
                print('没有持久化模型，从头开始训练！')

            # 3、迭代数据的获取
            total_samples, X, Y = fetch_samples()
            total_batch = total_samples // batch_size
            times = 0
            random_index = np.random.permutation(total_samples)

            for epoch in range(1, max_epochs+1):
                # 4、获取当前批次对应的训练数据
                start_idx = times * batch_size
                end_idx = start_idx + batch_size
                idx = random_index[start_idx: end_idx]
                train_x = X[idx]
                train_y = Y[idx]

                # 5、构建喂入数据的对象（字典）
                feed_dict = {
                    tensor.batch_size: batch_size,
                    tensor.inputs: train_x,
                    tensor.targets: train_y,
                    tensor.keep_prob: 0.7
                }

                _, train_loss = sess.run([tensor.train_op, tensor.loss], feed_dict)
                # 打印模型损失
                if epoch % 2 == 0:
                    print('Step:{} - Train Loss:{:.5f}'.format(epoch, train_loss))

                # 模型持久化
                if epoch % 200 == 0:
                    save_path_file = os.path.join(checkpoint_dir, 'models.ckpt')
                    saver.save(sess, save_path=save_path_file, global_step=epoch)

                # 更新样本顺序
                times += 1
                if times == total_batch:
                    times = 0
                    random_index = np.random.permutation(total_samples)


def pick_top_n(preds, vocab_size, top_n=5):
    """
    随机从前n个概率最大的值中选取一个值作为预测值。
    :param preds:        预测的概率
    :param vocab_size:   单词表的大小
    :param top_n:        选取前n个概率最大的
    :return:
    """
    p = np.squeeze(preds)
    p[np.argsort(p)[:-top_n]] = 0
    p = p / sum(p)
    c = np.random.choice(vocab_size, 1, p=p)[0]
    return c


def predict(first, checkpoint_dir, num_units=256, vocab_size=8000):
    """
    基于第一个汉字，生成一首唐诗
    :param first:
    :param checkpoint_dir:
    :param num_units:
    :return:
    """
    if len(first) == 1:
        rezult = []
        with tf.Graph().as_default():
            # 一、构建模型网络图
            tensor = Tensor(
                number_time_steps=32, num_units=num_units, vocab_size=vocab_size, embedding_size=300, learning_rate=0.001
            )

            # 二、加载模型进行预测。
            with tf.Session() as sess:
                saver = tf.train.Saver()
                # a、恢复模型
                ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
                if ckpt and ckpt.model_checkpoint_path:
                    saver.restore(sess, ckpt.model_checkpoint_path)
                    print('恢复模型进行预测!')
                else:
                    raise Exception('加载失败....')

                # b. 做一个预测
                first_idx = chinese_to_index(first)[0]
                input_list = np.zeros(shape=[1, 32])  # 因为我们模型中定义的占位符是32个时刻，所以先将后面31个输入设置为0，后面预测出来值后逐个替换掉。
                input_list[:, 0] = first_idx

                rezult.append(first_idx)

                # c、构建输入的字典。
                feed_dict = {tensor.batch_size: 1,
                             tensor.inputs: input_list,
                             tensor.keep_prob: 1.0
                             }

                # d、循环31次获取相应的预测值。
                for time in range(1, 32):
                    # 获取time-1时刻的预测值。
                    pred = sess.run(tensor.y_preds[time-1], feed_dict)

                    c = pick_top_n(pred, vocab_size=vocab_size, top_n=5)
                    rezult.append(c)
                    # 设置time时刻对应的输入
                    input_list[:, time] = c
                print('预测值为:{}'.format(rezult))
                print('预测值的唐诗为:\n{}'.format(index_to_chinese(rezult)))
    else:
        # 做一首藏头诗
        rezult = []
        with tf.Graph().as_default():
            # 一、构建模型网络图
            tensor = Tensor(
                number_time_steps=32, num_units=num_units, vocab_size=vocab_size, embedding_size=300,
                learning_rate=0.001
            )

            # 二、加载模型进行预测。
            with tf.Session() as sess:
                saver = tf.train.Saver()
                # a、恢复模型
                ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
                if ckpt and ckpt.model_checkpoint_path:
                    saver.restore(sess, ckpt.model_checkpoint_path)
                    print('恢复模型进行预测!')
                else:
                    raise Exception('加载失败....')

                # b. 做一个预测
                first_idx = chinese_to_index(first)
                input_list = np.zeros(shape=[1, 32], dtype=int)  # 因为我们模型中定义的占位符是32个时刻，所以先将后面31个输入设置为0，后面预测出来值后逐个替换掉。
                for idx, word in zip([0, 8, 16, 24], first_idx):
                    input_list[:, idx] = word

                rezult.append(first_idx[0])

                # c、构建输入的字典。
                feed_dict = {tensor.batch_size: 1,
                             tensor.inputs: input_list,
                             tensor.keep_prob: 1.0
                             }

                # d、循环31次获取相应的预测值。
                for time in range(1, 32):
                    # fixme 如果是以下几个时刻，就直接用输入的藏头字。
                    if time in [8, 16, 24]:
                        rezult.append(input_list[:, time][0])
                        continue
                    else:
                        # 获取time-1时刻的预测值。
                        pred = sess.run(tensor.y_preds[time - 1], feed_dict)

                        c = pick_top_n(pred, vocab_size=vocab_size, top_n=5)
                        rezult.append(c)
                        # 设置time时刻对应的输入
                        input_list[:, time] = c
                print('预测值为:{}'.format(rezult))
                print('预测值的唐诗为:\n{}'.format(index_to_chinese(rezult)))


if __name__ == '__main__':
    op = 1
    train_data_path = '../qts_7X4.txt'
    checkpoint_dir = './models/model_embed06'
    if op == 0:
        train(train_data_path, checkpoint_dir, max_epochs=100, batch_size=64, num_units=256)
    else:
        # 预测。
        # predict('天', checkpoint_dir, num_units=256)
        # 天上梦魂何杳杳.宫中消息太沈沈.君恩不似黄金井.一处团圆万丈深.
        predict('清明', checkpoint_dir, num_units=256)
        # predict('我爱珍珍珍珍爱我', checkpoint_dir, num_units=256)


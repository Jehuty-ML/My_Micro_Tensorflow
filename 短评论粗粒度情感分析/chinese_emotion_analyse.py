import numpy as np
import matplotlib.pyplot as plt
import re
import jieba
import os
from sklearn.model_selection import train_test_split
from gensim.models import KeyedVectors
from keras.models import Sequential
from keras.layers import Dense, GRU, Embedding, LSTM, Bidirectional
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.optimizers import Adam, RMSprop
from keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard, ReduceLROnPlateau
import time
import warnings
warnings.filterwarnings("ignore")

class Config:
    def __init__(self):
        self.model_path = 'chinese_word_vectors\sgns.zhihu.bigram'
        self.pos_text = os.listdir('./pos')
        self.neg_text = os.listdir('./neg')
        self.num_words = 50000  # 1、减少模型大小，只使用该预训练包中的前50000个词语。
        self.embedding_dims = 300
        self.path_chekpoint = 'sentiment_checkpoint.keras'



class Sample:
    def __init__(self, config: Config):
        self.config = config
        start_time = time.time()
        self.cn_model = KeyedVectors.load_word2vec_format(self.config.model_path, binary=False)
        end_time = time.time()
        print('加载预训练词向量，使用了:{}/秒'.format((end_time - start_time)))

        self.train_text_origin = self.read_data()
        print('train_text_origin[:10]:', self.train_text_origin[:10])
        self.train_tokenize = self.tokenize(self.train_text_origin)
        print('train_tokenize[:10]', self.train_tokenize[:10])

        self.max_tokens = self.prepocess_data(self.train_tokenize)
        self.show_reverse_token(self.train_tokenize[0], self.train_text_origin[0])

        self.embedding_matrix = self.embed_matrix()
        self.X_train, self.X_test, self.y_train, self.y_test = self.pad_and_truncate(self.train_tokenize, self.max_tokens)


    def show_embedding(self):
        # 这个词向量的长度为300.
        embedding_dims = self.cn_model['山东大学'].shape[0]
        print('词向量的长度为:{}'.format(embedding_dims))
        print('**' * 50)
        print(self.cn_model['小学'])

    # def show_similarity(w1, w2):
    #     print('调包:', cn_model.similar(w1, w2))
    #     cos_ab = np.dot(cn_model[w1]/np.linalg.norm(cn_model[w1]),
    #                     cn_model[w2]/np.linalg.norm(cn_model[w2]))
    #     print('自己实现的:', cos_ab)

    def cos_similarity(self):
        x = self.cn_model.similarity('橘子', '橙子')
        print(x)

        y = np.dot(self.cn_model['橘子'] / np.linalg.norm(self.cn_model['橘子']),
                   self.cn_model['橙子'] / np.linalg.norm(self.cn_model['橙子']))
        print(y)

        z = self.cn_model.most_similar(positive='大学', topn=10)
        print(z)

        test_word = '老师 会计师 程序员 律师 医生 老人'
        test_word_result = self.cn_model.doesnt_match(test_word.split())
        print('在' + test_word + '之中，不是该类的词是：\n{}'.format(test_word_result))

        k = self.cn_model.most_similar(positive=['女人', '出轨'], negative=['男人'], topn=3)
        print(k)

    def read_data(self):
        pos_text = self.config.pos_text
        neg_text = self.config.neg_text
        print('总样本数：', len(pos_text) + len(neg_text))

        self.train_text_origin = []
        for i in range(len(pos_text)):
            with open('./pos/' + pos_text[i], 'r', errors='ignore') as f:
                text = f.read().strip()
                self.train_text_origin.append(text)
                f.close()

        for i in range(len(neg_text)):
            with open('./neg/' + neg_text[i], 'r', errors='ignore') as f:
                text = f.read().strip()
                self.train_text_origin.append(text)
                f.close()

        return self.train_text_origin

    def tokenize(self, train_text_origin):
        self.train_tokenize = []
        for text in train_text_origin:
            text = re.sub("[\s+\.\!\/_,$%^*(+\"\']+|[+——！，。？、~@#￥%……&*（）]+", "", text)
            cut = jieba.cut(text)
            cut_list = [i for i in cut]
            for i, word in enumerate(cut_list):
                try:
                    cut_list[i] = self.cn_model.vocab[word].index
                except KeyError:
                    cut_list[i] = 0
            self.train_tokenize.append(cut_list)
        return self.train_tokenize  # [[idx?, idx?...],[idx?, idx?...],[idx?, idx?...]...]

    def prepocess_data(self, train_tokenize):
        '''
        每段文本的长度需要一致，对长度不足的进行填充，超长的进行裁剪.
        这个函数用于判断阶段长度多少比较合理
        :param train_tokenize:
        :return:self.max_tokens
        '''
        #train_tokenize:  [[idx?, idx?...],[idx?, idx?...],[idx?, idx?...]...]
        num_tokens_len = np.array([len(token) for token in train_tokenize])
        print(np.mean(num_tokens_len), np.max(num_tokens_len))

        # plt.hist(np.log(num_tokens_len), bins=100)
        # plt.xlim((0, 10))
        # plt.ylabel('nums of tokens')  #对应句子长度的数量
        # plt.xlabel('length of tokens')  #句子长度（有多少个token)
        # plt.title('distribution of tokens')
        # plt.show()
        self.max_tokens = int(np.mean(num_tokens_len) + 2 * np.std(num_tokens_len))  #长度为均值加2倍标准差
        print('max_tokens:', self.max_tokens)
        print('max_tokens小于文本的百分比', np.sum(num_tokens_len < self.max_tokens) / len(num_tokens_len))  #这个长度小于文本长度的百分比，约0.9565
        return self.max_tokens

    def reverse_tokens(self, tokens):
        # 把token从index变回word， 即index2word
        self.text = ''
        for i in tokens:
            if i != 0:
                self.text += self.cn_model.index2word[i]
            else:
                self.text += ' '
        return self.text

    def show_reverse_token(self, train_tokens, train_text_origin):
        reverse_word = self.reverse_tokens(train_tokens)
        print('处理后第一个文本为：', reverse_word)
        print('原始文本为：', train_text_origin)

    def embed_matrix(self):
        embedding_dims = self.config.embedding_dims
        num_words = self.config.num_words
        # 初始化一个自己的嵌入矩阵（后面会用在模型中，之后keras中）
        embedding_m = np.zeros((num_words, embedding_dims))
        for i in range(num_words):
            embedding_m[i, :] = self.cn_model[self.cn_model.index2word[i]]
        self.embedding_matrix = embedding_m.astype('float32')

        # 检查index是否 一一对应。
        print('正确的列数', np.sum(self.cn_model[self.cn_model.index2word[222]] == embedding_m[222]))
        print('embedding_m.shape', self.embedding_matrix.shape)

        return self.embedding_matrix

    def pad_and_truncate(self, train_tokens, max_tokens):
        '''
        对文本进行填充截断
        :param train_tokens:
        :param max_tokens:
        :param num_words:
        :return:
        '''
        num_words = self.config.num_words
        train_pad = pad_sequences(train_tokens, maxlen=max_tokens, padding='pre', truncating='pre')
        # 索引超出5万的词语，用0代替
        train_pad[train_pad >= num_words] = 0
        print('train_pad[33]:', train_pad[33])

        train_targets = np.concatenate([np.ones(2000), np.zeros(2000)])
        print('train_targets.shape:', train_targets.shape, 'train_pad.shape:', train_pad.shape)

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(train_pad, train_targets, test_size=0.1, random_state=12)
        # 查看训练样本。
        print('其中一个训练样本：', self.reverse_tokens(self.X_train[35]))
        print('对应类别是：', self.y_train[35])

        return self.X_train, self.X_test, self.y_train, self.y_test

class Tensors:
    def __init__(self, config:Config):
        self.config = config
        self.model = Sequential()
        self.model.add(Embedding(self.config.num_words, self.config.embedding_dims, weights=[sample.embedding_matrix],
                            input_length=sample.max_tokens, trainable=False))
        self.model.add(Bidirectional(LSTM(64, return_sequences=False)))  # 后面接全连接，不是返回序列
        self.model.add(Dense(1, activation='sigmoid'))

        self.optimizer = Adam(lr=1e-3)
        self.model.compile(loss='binary_crossentropy', optimizer=self.optimizer, metrics=['accuracy'])
        print(self.model.summary())
        path_chekpoint = config.path_chekpoint

        self.checkpoint = ModelCheckpoint(filepath=path_chekpoint, monitor='val_loss', verbose=1,
                                     save_weights_only=True, save_best_only=True)

        try:
            self.model.load_weights(path_chekpoint)
            print('成功恢复模型')
        except Exception as e:
            print(e)

        self.early_stopping = EarlyStopping(monitor='val_loss', patience=3, verbose=1)

        self.lr_reduction = ReduceLROnPlateau(monitor='val_loss', factor=0.1, min_lr=1e-5, patience=0, verbose=1)

        self.callbacks = [self.early_stopping, self.checkpoint, self.lr_reduction]


class App:
    def __init__(self, config: Config):
        self.config = config
        self.tensor = Tensors(config)


    def train(self, embedding_m, max_tokens, X_train, y_train, X_test, y_test):
        model = self.tensor.model
        model.fit(X_train, y_train, validation_split=0.1, nb_epoch=10, batch_size=128, callbacks=self.tensor.callbacks)
        result = model.evaluate(X_test, y_test)
        print('accracy:{}'.format(result[1]))

    def test(self, embedding_m, max_tokens, X_test, y_test):
        model = self.tensor.model
        result = model.evaluate(X_test, y_test)
        print('accracy:{}'.format(result[1]))

        test_list = [
            '酒店设施不是新的，服务态度很不好',
            '酒店卫生条件非常不好',
            '床铺非常舒适',
            '我觉得还好吧，就是有点吵'
            '我觉得一般吧'
        ]

        for text in test_list:
            self.predict_sentiment(text, model)


    def predict_sentiment(self, text, model):
        '''

        :param text:  需要预测的文本
        :param model:  训练的模型
        :return:
        '''
        text = re.sub("[\s+\.\!\/_,$%^*(+\"\']+|[+——！，。？、~@#￥%……&*（）]+", "", text)
        cut = jieba.cut(text)
        cut_list = [i for i in cut]
        for i, word in enumerate(cut_list):
            try:
                cut_list[i] = sample.cn_model.vocab[word].index
            except KeyError:
                cut_list[i] = 0

        tokens_pad = pad_sequences([cut_list], maxlen=sample.max_tokens, padding='pre', truncating='pre')

        result = model.predict(x=tokens_pad)
        coef = result[0][0]

        if coef >= 0.5:
            print('这是一个正面评价，预测概率为：{:.4f}'.format(coef))
        else:
            print('这是一个负面评价，预测概率为：{:.4f}'.format(coef))


if __name__ == '__main__':
    config = Config()
    sample = Sample(config)
    app = App(config)

    # app.train(sample.embedding_matrix, sample.max_tokens, sample.X_train, sample.y_train, sample.X_test, sample.y_test)
    app.test(sample.embedding_matrix, sample.max_tokens, sample.X_test, sample.y_test)




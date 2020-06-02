import tensorflow as tf
import numpy as np
from data_process import get_input, Token
import json

vocab_size = 4996
s_length = 128
H = 8
dim = 256
batch_size = 32
hidden = 512
blocks = 6
num_epochs = 20
dropout = 0.5
label_class = 3
lr = 0.0005

train_data = json.load(open('./data_trans/train_data_me.json', encoding='utf-8'))
dev_data = json.load(open('./data_trans/dev_data_me.json', encoding='utf-8'))

class data_loader():
    def __init__(self):
        self.input_x, self.input_ner = get_input(train_data)
        self.input_x = self.input_x.astype(np.int32)
        self.input_ner = self.input_ner.astype(np.int32)
        self.num_train = self.input_x.shape[0]
        self.db_train = tf.data.Dataset.from_tensor_slices((self.input_x, self.input_ner))
        self.db_train = self.db_train.shuffle(self.num_train).batch(batch_size, drop_remainder=True)

    def get_batch(self, batch_s):
        indics = np.random.randint(0, self.num_train, batch_s)
        return self.input_x[indics], self.input_ner[indics]

'''
Mask
X [batch_size, q_length]
tf.math.equal(x,0), if x == 0 return True else False
tf.cast(x) if x == True return 1 else 0
expand_dim [batch_size, 1, 1, s_length]
'''
class Mask(tf.keras.Model):
    def __init__(self):
        super(Mask, self).__init__()
    def call(self, inputs):
        mask = tf.cast(tf.math.equal(inputs, 0), dtype=tf.float32)
        mask = mask[:, np.newaxis, np.newaxis, :]
        return mask

'''
position_encoding
PE(pos,2i) = sin(pos/10000**(2i/d_model))
PE(pos,2i+1) = cos(pos/10000**(2i/d_model))
'''
class position_encoding(tf.keras.Model):
    def __init__(self):
        super(position_encoding, self).__init__()

    def call(self, sequence_length, embedding_dim, batchsize):
        #sequence_length = int(np.array(sequence_length))
        positionIndex = tf.tile(tf.expand_dims(tf.range(sequence_length), 0), [batchsize, 1])
        position_embedding = np.array([[p/10000**((i - i % 2)/embedding_dim) for i in range(embedding_dim)] for p in range(sequence_length)])
        position_embedding[:, 0::2] = np.sin(position_embedding[:, 0::2])
        position_embedding[:, 1::2] = np.cos(position_embedding[:, 1::2])

        position_embedding = tf.cast(position_embedding, dtype=tf.float32)
        p_embeded = tf.nn.embedding_lookup(position_embedding, positionIndex)
        return p_embeded

'''
Scale_Dot_Product_Attention
Q,K,V
Q*K.transpose/sqrt(d_model)
mask * (-1e9) ;  softmax(-1e9) = 0
weight = softmax(Q*K.transpose/sqrt(d_model))
attention = weight * V
'''
class Scale_Dot_Product_Attention(tf.keras.Model):
    def __init__(self):
        super(Scale_Dot_Product_Attention, self).__init__()
        self.softmax = tf.keras.layers.Softmax()

    def call(self, Q, K, V, scale, mask):
        x = tf.matmul(Q, K, transpose_b=True)
        x = x / tf.math.sqrt(float(scale))
        mask = mask * (-1e9)
        x = x + mask
        #掩码的token乘以-1e9(表示负无穷),这样softmax之后就为0
        x = self.softmax(x)
        x = tf.matmul(x, V)
        return x

'''
Multi_H_Attention
X
Q = X * W_q
K = X * W_k
V = X * W_v
Q, K, V = reshape(Q, K, V, n_heads)  shape=[batch_size, n_heads, s_length, d_model/n_heads]
scale_dot_product_attention(Q, K, V)
concat n_heads
fully connection
dropout
shortcut connection
layernorm
'''
class Multi_H_Attention(tf.keras.Model):
    def __init__(self, n_heads, dropout=0.0):
        super(Multi_H_Attention, self).__init__()
        assert dim % n_heads == 0
        self.H = n_heads
        self.H_dim = dim // n_heads
        self.L_Q = tf.keras.layers.Dense(dim)
        self.L_K = tf.keras.layers.Dense(dim)
        self.L_V = tf.keras.layers.Dense(dim)
        self.attention = Scale_Dot_Product_Attention()
        self.dropout = tf.keras.layers.Dropout(dropout)
        self.attention_dense = tf.keras.layers.Dense(dim)
        self.layer_norm = tf.keras.layers.LayerNormalization(axis=-1)

    def call(self, X, mask):
        scale = X.get_shape().as_list()[-1]
        nbatchs = X.get_shape().as_list()[0]
        Q = self.L_Q(X)
        K = self.L_Q(X)
        V = self.L_Q(X)

        #1.[batch_size, s_length, H, h_dim]
        #2.[batch_size, H, s_length, h_dim]
        Q = tf.transpose(tf.reshape(Q, (nbatchs, -1, self.H, self.H_dim)), [0, 2, 1, 3])
        K = tf.transpose(tf.reshape(K, (nbatchs, -1, self.H, self.H_dim)), [0, 2, 1, 3])
        V = tf.transpose(tf.reshape(V, (nbatchs, -1, self.H, self.H_dim)), [0, 2, 1, 3])
        s_attention = self.attention(Q, K, V, scale, mask)
        #[batch_size, s_length, H, h_dim]
        s_attention = tf.transpose(s_attention, [0, 2, 1, 3])
        c_attention = tf.reshape(s_attention, (nbatchs, -1, dim))
        attention = self.attention_dense(c_attention)
        out = self.dropout(attention)
        out = X + out
        out = self.layer_norm(out)
        return out

'''
Position_wise_Feed_Forward
fully connection
activation function relu = max(0,x)
fully connection
dropout
shortcut
layernorm
'''
class Position_wise_Feed_Forward(tf.keras.Model):
    def __init__(self, hidden, dim, dropout=0.0):
        super(Position_wise_Feed_Forward, self).__init__()
        self.hidden = hidden
        self.keep_p = dropout
        self.dim = dim
        self.fc1 = tf.keras.layers.Dense(hidden, activation="relu")
        self.fc2 = tf.keras.layers.Dense(dim)
        self.drouout = tf.keras.layers.Dropout(self.keep_p)
        self.layer_norm = tf.keras.layers.LayerNormalization(axis=-1)

    def call(self, inputs):
        #[None, 128, 128]
        x = tf.transpose(inputs, [0, 1, 2])
        x = self.fc2(self.fc1(x))
        x = self.drouout(tf.transpose(x, [0, 1, 2]))
        x = inputs + x
        x = self.layer_norm(x)
        return x


'''
NER
'''
class Transformer_Ner(tf.keras.Model):
    def __init__(self, _blocks, sequence_length, embedding_dim):
        super(Transformer_Ner, self).__init__()
        self.blocks = _blocks
        self.length = sequence_length
        self.dim = embedding_dim
        self.mask = Mask()
        self.word_embedding = tf.keras.layers.Embedding(vocab_size, self.dim)
        self.position_encoder = position_encoding()
        self.multi_h_attention = Multi_H_Attention(H, dropout)
        self.position_feed_forward = Position_wise_Feed_Forward(hidden, self.dim, dropout)
        self.dropout = tf.keras.layers.Dropout(dropout)
        self.dense = tf.keras.layers.Dense(label_class)

    def call(self, inputs):
        nbatchs = inputs.get_shape().as_list()[0]
        x = self.word_embedding(inputs)
        p = self.position_encoder(self.length, self.dim, nbatchs)
        x = x + p
        mask = self.mask(inputs)
        for _ in range(self.blocks):
            x = self.multi_h_attention(x, mask)
            x = self.position_feed_forward(x)
        output = x
        output = self.dense(output)
        output = self.dropout(output)
        output = tf.nn.softmax(output)
        return output


def loss_function(ner, input_ner):
    ner_one_hot = tf.one_hot(input_ner, depth=3, dtype=tf.float32)
    loss_ner = tf.keras.losses.categorical_crossentropy(y_true=ner_one_hot, y_pred=ner)
    loss = tf.reduce_sum(loss_ner)
    return loss

'''
Extra
'''
class Extra_result(object):
    def __init__(self, text):
        self.text = text[0:128]
    def call(self):
        token = np.zeros(128)
        text2id = Token(self.text)
        token[0:len(text2id)] = text2id
        Model_ner = model_Ner
        ner = Model_ner(np.array([token], dtype=np.int32))
        subjects = self.extra_sujects(ner)
        print(subjects)
        return subjects

    def extra_sujects(self, ner_label):
        ner = ner_label[0]
        ner = tf.round(ner)
        ner = [tf.argmax(ner[k]) for k in range(ner.shape[0])]
        ner = list(np.array(ner))
        ner.append(0)  # 防止最后一位不为0
        text_list = [key for key in self.text]
        subject = []
        for i, k in enumerate(text_list):
            if int(ner[i]) == 0 or int(ner[i]) == 2:
                continue
            elif int(ner[i]) == 1:
                ner_back = [int(j) for j in ner[i + 1:]]
                if 1 in ner_back and 0 in ner_back:
                    indics_1 = ner_back.index(1) + i
                    indics_0 = ner_back.index(0) + i
                    subject.append(''.join(text_list[i: min(indics_0, indics_1) + 1]))
                elif 1 not in ner_back:
                    indics = ner_back.index(0) + i
                    subject.append(''.join(text_list[i:indics + 1]))
        return subject


class Evaluate(object):
    def __init__(self):
        pass
    def reset(self,spo_list):
        xx = []
        for key in spo_list:
            xx.append(key[0])
            xx.append(key[2])
        return xx
    def evaluate(self, data):
        A, B, C = 1e-10, 1e-10, 1e-10
        for d in data[0:10]:
            extra_items = Extra_result(d['text'])
            R = set(extra_items.call())
            T = set(self.reset(d['spo_list']))
            A += len(R & T)
            B += len(R)
            C += len(T)
        return 2 * A / (B + C), A / B, A / C

'''
Training
'''
evaluate = Evaluate()
data_loader = data_loader()
model_Ner = Transformer_Ner(blocks, s_length, dim)
optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
for epoch in range(num_epochs):
    print('Epoch:', epoch + 1)
    best = 0.0
    num_batchs = int(data_loader.num_train / batch_size) + 1
    for batch_index in range(num_batchs):
        input_x, input_ner = data_loader.get_batch(batch_size)

        with tf.GradientTape() as tape:
            y_pred = model_Ner(input_x) #预测ner
            loss = loss_function(y_pred, input_ner)
            if (batch_index+1) % 100 == 0:
                print("batch %d: loss %f" % (batch_index+1, loss.numpy()))

        variables = model_Ner.variables
        grads = tape.gradient(loss, variables)
        optimizer.apply_gradients(grads_and_vars=zip(grads, variables))
    F, P, R = evaluate.evaluate(dev_data)
    print(F, P, F)

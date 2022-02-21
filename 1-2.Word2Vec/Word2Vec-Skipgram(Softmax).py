# %%
# code by Tae Hwan Jung @graykode
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

def random_batch():
    random_inputs = []
    random_labels = []
    random_index = np.random.choice(range(len(skip_grams)), batch_size, replace=False)

    for i in random_index:
        random_inputs.append(np.eye(voc_size)[skip_grams[i][0]])  # target 从生成的单位矩阵中抽出一行作为一个target word，此处取了4行和1行
        random_labels.append(skip_grams[i][1])  # context word 上述target word对应的context word

    return random_inputs, random_labels

# Model
class Word2Vec(nn.Module):
    def __init__(self):
        super(Word2Vec, self).__init__()
        # W and WT is not Traspose relationship
        self.W = nn.Linear(voc_size, embedding_size, bias=False) # voc_size > embedding_size Weight 定义一个线性变换操作
        self.WT = nn.Linear(embedding_size, voc_size, bias=False) # embedding_size > voc_size Weight 定义一个线性变换操作

    def forward(self, X): # 前向传播过程，注：不含softmax层
        # X : [batch_size, voc_size]
        hidden_layer = self.W(X) # hidden_layer : [batch_size, embedding_size] # 对输入层作线性变换得到隐藏层
        output_layer = self.WT(hidden_layer) # output_layer : [batch_size, voc_size] # 对隐藏层作线性变换得到输出层
        return output_layer

if __name__ == '__main__':
    batch_size = 2 # mini-batch size
    embedding_size = 2 # embedding size

    sentences = ["apple banana fruit", "banana orange fruit", "orange banana fruit",
                 "dog cat animal", "cat monkey animal", "monkey dog animal"]

    word_sequence = " ".join(sentences).split()
    word_list = " ".join(sentences).split()
    word_list = list(set(word_list))
    word_dict = {w: i for i, w in enumerate(word_list)}
    voc_size = len(word_list)

    # Make skip gram of one size window 左右各一格范围内作为context word
    skip_grams = []
    for i in range(1, len(word_sequence) - 1):
        target = word_dict[word_sequence[i]]
        context = [word_dict[word_sequence[i - 1]], word_dict[word_sequence[i + 1]]]
        for w in context:
            skip_grams.append([target, w]) # skip_grams形如2维数组，每个元素是一个1维数组，代表一个target-context对，数组中第一个元素是target，第二个元素是其中一个context

    model = Word2Vec()

    criterion = nn.CrossEntropyLoss() # 交叉熵
    optimizer = optim.Adam(model.parameters(), lr=0.001) # 定义优化器

    # Training
    for epoch in range(5000):
        input_batch, target_batch = random_batch()
        input_batch = torch.Tensor(input_batch)
        target_batch = torch.LongTensor(target_batch)

        optimizer.zero_grad()
        output = model(input_batch) # 会自动调用前向传播函数，得到输出

        # output : [batch_size, voc_size], target_batch : [batch_size] (LongTensor, not one-hot)
        loss = criterion(output, target_batch) # 计算交叉熵损失函数
        if (epoch + 1) % 1000 == 0:
            print('Epoch:', '%04d' % (epoch + 1), 'cost =', '{:.6f}'.format(loss))

        loss.backward() # 损失函数反向传播
        optimizer.step() # 更新参数

    for i, label in enumerate(word_list):
        W, WT = model.parameters() # W, WT是Parameter类对象
        x, y = W[0][i].item(), W[1][i].item()
        plt.scatter(x, y)
        plt.annotate(label, xy=(x, y), xytext=(5, 2), textcoords='offset points', ha='right', va='bottom')
    plt.show()

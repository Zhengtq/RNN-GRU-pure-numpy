import numpy as np
import random
from structure import Model


time_step = 5
feature_dim = 1025
hidden_dim = 128
batch_size = 100

np.random.seed(5)

def load_all_train_feature(all_rnn_fea_txt):


    file = open(all_rnn_fea_txt, 'r')


    all_neg_fea_easy = []
    all_neg_fea_hard = []
    all_pos_fea_easy = []
    all_pos_fea_hard = []
    for ind, item in enumerate(file):

        try:
            item = item.strip()
            img_label = float(item.split('####')[-1])
            img_fea = item.split('####')[0].split()
            img_fea = [float(x) for x in img_fea]
        except:
            continue

        if img_label == 0:
            if img_fea[-1] < 100:
                all_neg_fea_easy.append(img_fea)
            else:
                all_neg_fea_hard.append(img_fea)
        if img_label == 1:
            if img_fea[-1] > -100:
                all_pos_fea_easy.append(img_fea)
            else:
                all_pos_fea_hard.append(img_fea)

    file.close()


    print(len(all_pos_fea_easy))
    print(len(all_pos_fea_hard))
    print(len(all_neg_fea_easy))
    print(len(all_neg_fea_hard))

    return all_pos_fea_easy, all_pos_fea_hard, all_neg_fea_easy, all_neg_fea_hard



def generate_neg_sample(all_neg_fea_easy, all_neg_fea_hard):

    ran_timestep = random.randint(1,30)
    neg_part = random.sample(all_neg_fea_easy, ran_timestep)
    random.shuffle(neg_part)
    neg_part = np.array(neg_part)
    return neg_part

def generate_pos_sample(all_pos_fea_easy, all_pos_fea_hard):

    ran_timestep = random.randint(1,30)
    pos_part = random.sample(all_pos_fea_easy, ran_timestep)
    random.shuffle(pos_part)
    pos_part = np.array(pos_part)
    return pos_part



def main():
    time_step = 5
    all_rnn_fea_txt = './rnn_fea/all_combine_fea.txt'
    all_pos_fea_easy, all_pos_fea_hard, all_neg_fea_easy, all_neg_fea_hard = load_all_train_feature(all_rnn_fea_txt)
    rnn = Model(feature_dim, hidden_dim, time_step= 5)

    lr = 0.1
    for i in range(1000000):
        if i % 2==0:
            fea = generate_neg_sample(all_neg_fea_easy, all_neg_fea_hard)
        #    fea = np.random.rand(time_step, feature_dim)/1000 + -1
            label = np.array(0)
        else:
            fea = generate_pos_sample(all_pos_fea_easy, all_pos_fea_hard)
    #        fea = np.random.rand(time_step, feature_dim)/1000 + 1
            label = np.array(1)


        if i % 200 == 0:
            lr = lr / 2
        time_step = len(fea)
        loss = rnn.train(fea, label, time_step, learning_rate = lr)

        print('step: %d, loss: %.2f'%(i, loss))
        if i !=0 and i % 200 == 0:
            rnn.save_wt()



main()






















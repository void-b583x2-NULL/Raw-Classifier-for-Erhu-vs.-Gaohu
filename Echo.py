import librosa.display
import librosa
import random
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

N_Mels = 128


def sample_audio(filename: str, items=10):
    '''
    Return a list containing sampled spectrum segments with given amount. Tool: librosa
    :param filename: From where the result is sampled.
    :param items: The given amount.
    :return: A list of samples.
    '''
    y, sr = librosa.load(filename, sr=None)
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=1024, hop_length=512, n_mels=N_Mels)  # Mel spectrogram
    # Plot
    # # convert to log scale
    # logmelspec = librosa.power_to_db(S)
    # # plot mel spectrogram
    # plt.figure()
    # librosa.display.specshow(logmelspec, sr=sr, x_axis='time', y_axis='mel')
    # plt.title('Spectrogram')
    # plt.show()

    ST = S.T  # for sampling
    # print(ST)
    # 1. Fetch a max reference
    maxref = 0
    maxidx = 0
    for i in range(len(ST)):
        cur_vecnorm = np.linalg.norm(ST[i], ord=1)  # std: l1-norm
        if (maxref < cur_vecnorm):
            maxref = cur_vecnorm
            maxidx = i
    # 2. Choose samples
    ret = ST[maxidx].reshape((1, -1))
    items -= 1
    maxref *= 0.219
    while items > 0:
        i = random.randint(0, len(ST) - 1)
        if (np.linalg.norm(ST[i], ord=1) >= maxref):
            ret = np.concatenate((ret, ST[i].reshape((1, -1))), axis=0)
            items -= 1
    # print(ret)
    return ret  # shape in (items,features=128 by default)


unit_samples = 12
train_enable = True  # Control training

audio_echoed_trainers = [
    'gh-0-g1.wav', 'gh-0-a1.wav', 'gh-0-b1.wav', 'gh-0-c2.wav', 'gh-0-d2.wav',
    'gh-1-d2.wav', 'gh-1-e2.wav', 'gh-1-f2.wav', 'gh-1-g2.wav', 'gh-1-a2.wav', 'gh-1-b2.wav',  # 0-10
    'eh-0-d1.wav', 'eh-0-e1.wav', 'eh-0-f1.wav', 'eh-0-g1.wav', 'eh-0-a1.wav', 'eh-0-b1.wav', 'eh-0-c2.wav',
    'eh-0-d2.wav',
    'eh-1-a1.wav', 'eh-1-b1.wav', 'eh-1-c2.wav', 'eh-1-d2.wav', 'eh-1-e2.wav', 'eh-1-f2.wav', 'eh-1-g2.wav',
    'eh-1-a2.wav', 'eh-1-b2.wav']  # 11-27

# Network
Yielded_DataNet = nn.Sequential(
    nn.Linear(N_Mels, 219),
    nn.Sigmoid(),
    nn.Linear(219, 152),
    nn.Sigmoid(),
    nn.Linear(152, 2),
    nn.Softmax(dim=1)
)

# preperation
optimizer = torch.optim.SGD(Yielded_DataNet.parameters(), lr=0.05)
loss_func = torch.nn.CrossEntropyLoss()

if train_enable:  # Train
    # Prepare data

    data_sample = sample_audio('data_train/' + audio_echoed_trainers[0], items=unit_samples)
    # print('0-th sample fetched!')

    for i in range(1, len(audio_echoed_trainers)):
        filename = audio_echoed_trainers[i]
        data_sample = np.concatenate((data_sample, sample_audio('data_train/' + filename, items=unit_samples)), axis=0)
        # print(len(data_sample))
        # print(len(data_sample[0]))
        # print(f'{i}-th sample fetched!')
        # print(data_sample)

    data_label = np.concatenate((np.zeros(11 * unit_samples), np.ones(17 * unit_samples)))

    x_t = torch.from_numpy(data_sample)
    y_t = torch.from_numpy(data_label)

    print('Data Fetched!')

    # train
    num_epoch = 12200
    for epoch in range(num_epoch):
        y_p = Yielded_DataNet(x_t)
        loss = loss_func(y_p, y_t.long())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if epoch % 1000 == 0:
            print('epoch: {}, loss: {}'.format(epoch, loss.data.item()))

    # print("Labels: \n", torch.max(y_p, dim=1)[1])
    torch.save(Yielded_DataNet.state_dict(), 'echoed_params.pth')

else:  # Load params
    Yielded_DataNet.load_state_dict(torch.load('echoed_params.pth'))
    Yielded_DataNet.eval()


# Test
def Tester(filename: str):
    tester = sample_audio(filename, items=unit_samples)
    echo_ans = Yielded_DataNet(torch.from_numpy(tester)).mean(axis=0)
    print(echo_ans)  # Tensor
    ans = 'GaoHu' if echo_ans[0] > 0.5 else 'Erhu'
    print(ans)


audio_echoed_testers = ['gh-sample1.wav', 'gh-sample2.wav',
                        'eh-sample1.wav', 'eh-sample2.wav', 'eh-sample3.wav', 'eh-sample4.wav']

for files in audio_echoed_testers:
    Tester('data_test/' + files)

# Test whatever you want(Maybe?)

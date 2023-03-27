import matplotlib
#matplotlib.use('Agg')
import matplotlib.pyplot as plt
#plt.switch_backend('agg')
import torch
import numpy as np

def plot_fig(record, name, path_summary):
    durations_t = torch.FloatTensor(np.asarray(record))

    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(20, 15))
    ax.grid(True)
    ax.set_ylabel('Reward')
    ax.set_xlabel('Episode')
    plt.axhline(0, color='black')
    plt.axvline(0, color='black')
    # plt.yticks(np.arange(-200, 10, 10.0))

    #ax.plot(durations_t.numpy())
    if len(durations_t) >= 100:
        means = durations_t.unfold(0, 100, 1).mean(1).view(-1)
        means = torch.cat((torch.zeros(99), means))
        ax.plot(means.numpy())

    plt.draw()
    # plt.ylim([-200,10])

    plt.show()
    fig.savefig('{}/{}.png'.format(path_summary, name))
    plt.close(fig)

if __name__ == '__main__':
    with open('../experiments/2023-03-22_Boxworld_box_std_seed4001/summary/reward_total.txt', 'r') as fp:
        lines = [float(line) for line in fp]

        plot_fig(lines, 'reward_total', '../experiments/2023-03-22_Boxworld_box_std_seed4001/summary')
        print('worked')
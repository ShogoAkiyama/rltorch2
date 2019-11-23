from __future__ import print_function
import time
import numpy as np
import argparse
import torch
import matplotlib.animation as manimation
from scipy.ndimage.filters import gaussian_filter
# from scipy.misc import imresize
import cv2

from model import QNetwork
from env import make_pytorch_env

import warnings ; warnings.filterwarnings('ignore') # mute warnings, live dangerously

import matplotlib.pyplot as plt
import matplotlib as mpl ; mpl.use("Agg")

def blur_func(I, mask):
    return I * (1 - mask) + gaussian_filter(I, sigma=3)* mask

device = 'cuda' if torch.cuda.is_available() else 'cpu'

mask = np.zeros((84*84, 84, 84))

for i in range(84):
    for j in range(84):
        y, x = np.ogrid[-i:84 - i, -j:84 - j]
        circle = np.zeros([84, 84])
        circle[x * x + y * y <= 1] = 1
        circle = gaussian_filter(circle, sigma=5)
        mask[i*84+j] = circle / circle.max()


# 時刻ixのスコアを入れる
def score_frame(model, image, d, n_frames, mode='actor'):
    # ix: 時刻の値
    # r: radius of blur
    # d: density of scores (if d==1, then get a score for every pixel...
    #    if d==2 then every other, which is 25% of total pixels for a 2D image)
    # assert mode in ['actor', 'critic'], 'mode must be either "actor" or "critic"'

    # ネットワークの出力を得る
    state = torch.FloatTensor(image).to(device).unsqueeze(0)
    with torch.no_grad():
        L = model(state)

    # 各ピクセルにマスクする
    masked_image = blur_func(image, mask[:, np.newaxis, :, :])

    state = torch.FloatTensor(masked_image).to(device).view(-1, 4, 84, 84)
    with torch.no_grad():
        l = model(state)

    n_act = 6
    l = l.view(-1, 84*84, n_act)

    # スコアを記憶する配列
    scores = np.zeros((int(84/d)+1, int(84/d)+1))   # saliency scores S(t,i,j)

    for i in range(0, 84, d):
        for j in range(0, 84, d):
            # d=5としてその部分を描画する
            arr = (L-l[:, i*84+j]).pow(2).sum().mul_(.5).item()
            scores[int(i/d), int(j/d)] = arr

    # 正規化
    pmax = scores.max()
    scores = cv2.resize(scores, (84, 84), interpolation=cv2.INTER_LINEAR).astype(np.float32)
    # scores = imresize(scores, size=[84, 84], interp='bilinear').astype(np.float32)

    return pmax * scores / scores.max()


def saliency_on_atari_frame(saliency, atari, n_frames, fudge_factor):
    # sometimes saliency maps are a bit clearer if you blur them
    # slightly...sigma adjusts the radius of that blur

    S = saliency.copy()
    pmax = S.max()

    S -= S.min()
    S = fudge_factor * pmax * S / S.max()

    # # atariの元画像
    I = (atari[-1, :, :].copy()*255).astype('uint16')
    # # attentionを上書きする
    S = S.astype('uint16')
    I += S
    I = I.clip(1, 255).astype('uint8')
    return I


def make_movie(env_name, checkpoint='*.tar', num_frames=20, first_frame=0, resolution=75, \
               save_dir='./movies/', density=5, radius=5, prefix='default', overfit_mode=False):
    # log dirの名前を入れる
    load_dir = env_name.lower()
    # メタデータ
    meta = {}
    # if env_name == "Pong-v0":
    meta['critic_ff'] = 600
    meta['actor_ff'] = 500
    # 環境を作成
    # env = gym.make(env_name)
    env = make_pytorch_env(env_name)

    # actor crtic ネットワークをロードする
    n_state = env.observation_space.shape[0]
    n_act = env.action_space.n
    model = QNetwork(n_state, n_act)
    model.to(device)

    model.load()

    # movieのdirを取ってくる(default-100-PongNoFrameSkip)
    movie_title = "{}-{}-{}.mp4".format(prefix, num_frames, env_name.lower())
    print('\tmaking movie "{}" using checkpoint at {}{}'.format(movie_title, load_dir, checkpoint))
    max_ep_len = first_frame + num_frames + 1
    torch.manual_seed(0)

    # プレイしてログをえる(logitsはActorの値)
    history = rollout(model, env, max_ep_len=max_ep_len)
    print()

    # 保存用の作成
    start = time.time()

    total_frames = len(history['ins'])

    FFMpegWriter = manimation.writers['ffmpeg']
    metadata = dict(title=movie_title, artist='greydanus', comment='atari-saliency-video')
    writer = FFMpegWriter(fps=8, metadata=metadata)
    f = plt.figure(figsize=[6, 6 * 1.3], dpi=resolution)

    # 画像
    seq_image = np.array(history['ins'][first_frame:first_frame + num_frames])

    with writer.saving(f, save_dir + movie_title, resolution):
        for i in range(num_frames):
            print('i: ', i)
            frame = seq_image[i].copy()
            actor_saliency = score_frame(model, seq_image[i].copy(), density, num_frames, mode='actor')
            frame = saliency_on_atari_frame(actor_saliency, frame, num_frames, fudge_factor=meta['actor_ff'])

            # 描画する
            plt.imshow(frame)
            plt.gray()
            plt.title(env_name.lower(), fontsize=15)
            plt.show()
            writer.grab_frame()
            f.clear()

            tstr = time.strftime("%Hh %Mm %Ss", time.gmtime(time.time() - start))
            print('\ttime: {} | progress: {:.1f}%'.format(tstr, 100 * i / min(num_frames, total_frames)), end='\r')

    print('\nfinished.')


def rollout(model, env, max_ep_len=3e3, render=False):
    history = {'ins': [], 'logits': [], 'values': [], 'outs': [], 'hx': [], 'cx': []}

    state = env.reset()
    # state = torch.FloatTensor(state).to(device)  # get first state
    episode_length, epr, eploss, done = 0, 0, 0, False  # bookkeeping

    while not done and episode_length <= max_ep_len:
        episode_length += 1
        # 出力: value, actor, 隠れ層
        state = torch.FloatTensor(state).to(device).unsqueeze(0)
        logit = model(state)

        # actionを決定
        action = logit.max(1)[1].item()
        next_state, reward, done, expert_policy = env.step(action)
        if render:
            env.render()
        state = next_state
        epr += reward

        # save info!
        history['ins'].append(np.array(state))
        # history['logits'].append(logit.data.numpy()[0])
        # history['outs'].append(prob.data.numpy()[0])
        print('\tstep # {}, reward {:.0f}'.format(episode_length, epr), end='\r')

    return history


# user might also want to access make_movie function from some other script
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=None)
    parser.add_argument('-e', '--env', default='PongNoFrameskip-v4', type=str, help='gym environment')
    parser.add_argument('-d', '--density', default=5, type=int, help='density of grid of gaussian blurs')
    parser.add_argument('-r', '--radius', default=5, type=int, help='radius of gaussian blur')
    parser.add_argument('-f', '--num_frames', default=100, type=int, help='number of frames in movie')
    parser.add_argument('-i', '--first_frame', default=150, type=int, help='index of first frame')
    parser.add_argument('-dpi', '--resolution', default=75, type=int, help='resolution (dpi)')
    parser.add_argument('-s', '--save_dir', default='./movies/', type=str,
                        help='dir to save agent logs and checkpoints')
    parser.add_argument('-p', '--prefix', default='default', type=str, help='prefix to help make video name unique')
    parser.add_argument('-c', '--checkpoint', default='*.tar', type=str,
                        help='checkpoint name (in case there is more than one')
    parser.add_argument('-o', '--overfit_mode', default=False, type=bool,
                        help='analyze an overfit environment (see paper)')
    args = parser.parse_args()

    make_movie(args.env, args.checkpoint, args.num_frames, args.first_frame, args.resolution,
               args.save_dir, args.density, args.radius, args.prefix, args.overfit_mode)

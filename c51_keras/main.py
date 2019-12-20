#!/usr/bin/env python
from __future__ import print_function

import skimage as skimage
from skimage import transform, color
import numpy as np


import tensorflow.python.keras.backend as K

# from vizdoom import *
import gym
import tensorflow as tf

from networks import Networks
from actor import C51Agent


def preprocessImg(img, size):
    img = np.rollaxis(img, 0, 3)  # It becomes (640, 480, 3)
    img = skimage.transform.resize(img, size)
    img = skimage.color.rgb2gray(img)

    return img

if __name__ == "__main__":

    # Avoid Tensorflow eats up GPU memory
    config = tf.compat.v1.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.compat.v1.Session(config=config)
    K.set_session(sess)

    # game = DoomGame()
    # game.load_config("./scenarios/defend_the_center.cfg")
    # game.set_sound_enabled(True)
    # game.set_screen_resolution(ScreenResolution.RES_640X480)
    # game.set_window_visible(False)
    # game.init()
    #
    # game.new_episode()
    # game_state = game.get_state()
    # misc = game_state.game_variables  # [KILLCOUNT, AMMO, HEALTH]
    # prev_misc = misc
    #
    # action_size = game.get_available_buttons_size()

    env = gym.make('CartPole-v0')
    action_size = env.action_space.n

    img_rows, img_cols = 64, 64
    # Convert image into Black and white
    img_channels = 4  # We stack 4 frames

    # C51
    num_atoms = 51

    state_size = (img_rows, img_cols, img_channels)
    agent = C51Agent(state_size, action_size, num_atoms)

    agent.model = Networks.value_distribution_network(state_size, num_atoms, action_size, agent.learning_rate)
    agent.target_model = Networks.value_distribution_network(state_size, num_atoms, action_size, agent.learning_rate)

    # x_t = game_state.screen_buffer  # 480 x 640
    # x_t = preprocessImg(x_t, size=(img_rows, img_cols))
    # s_t = np.stack(([x_t] * 4), axis=2)  # It becomes 64x64x4
    # s_t = np.expand_dims(s_t, axis=0)  # 1x64x64x4
    s_t = env.reset()

    # is_terminated = game.is_episode_finished()
    done = False

    # Start training
    epsilon = agent.initial_epsilon
    GAME = 0
    t = 0
    max_life = 0  # Maximum episode life (Proxy for agent performance)
    life = 0

    # Buffer to compute rolling statistics
    life_buffer, ammo_buffer, kills_buffer = [], [], []

    while not done:

        loss = 0
        r_t = 0
        a_t = np.zeros([action_size])

        # Epsilon Greedy
        action_idx = agent.get_action(s_t)
        a_t[action_idx] = 1

        a_t = a_t.astype(int)
        # game.set_action(a_t.tolist())
        # skiprate = agent.frame_per_action
        # game.advance_action(skiprate)

        # game_state = game.get_state()  # Observe again after we take the action
        # is_terminated = game.is_episode_finished()

        # 報酬
        # r_t = game.get_last_reward()  # each frame we get reward of 0.1, so 4 frames will be 0.4

        s_t1, r_t, done, _ = env.step(a_t.argmax())

        if done:
            if (life > max_life):
                max_life = life
            GAME += 1
            # life_buffer.append(life)
            # ammo_buffer.append(misc[1])
            # kills_buffer.append(misc[0])
            # print("Episode Finish ", misc)
            # game.new_episode()
            # game_state = game.get_state()
            # misc = game_state.game_variables
            # x_t1 = game_state.screen_buffer

        # # 状態
        # x_t1 = game_state.screen_buffer
        # misc = game_state.game_variables
        #
        # # 画像を正規化
        # x_t1 = preprocessImg(x_t1, size=(img_rows, img_cols))
        # x_t1 = np.reshape(x_t1, (1, img_rows, img_cols, 1))
        # s_t1 = np.append(x_t1, s_t[:, :, :, :3], axis=3)

        # r_t = agent.shape_reward(r_t, misc, prev_misc, t)

        if done:
            life = 0
        else:
            life += 1

        # update the cache
        # prev_misc = misc

        # save the sample <s, a, r, s'>
        agent.replay_memory(s_t, action_idx, r_t, s_t1, done, t)

        # Do the training
        if t > agent.observe and t % agent.timestep_per_train == 0:
            loss = agent.train_replay()

        s_t = s_t1
        t += 1

        # save progress every 10000 iterations
        if t % 10000 == 0:
            print("Now we save model")
            agent.model.save_weights("models/c51_ddqn.h5", overwrite=True)

        # print info
        state = ""
        if t <= agent.observe:
            state = "observe"
        elif t > agent.observe and t <= agent.observe + agent.explore:
            state = "explore"
        else:
            state = "train"

        if done:
            print("TIME", t, "/ GAME", GAME, "/ STATE", state, \
                  "/ EPSILON", agent.epsilon, "/ ACTION", action_idx, "/ REWARD", r_t, \
                  "/ LIFE", max_life, "/ LOSS", loss)

            # Save Agent's Performance Statistics
            if GAME % agent.stats_window_size == 0 and t > agent.observe:
                print("Update Rolling Statistics")
                agent.mavg_score.append(np.mean(np.array(life_buffer)))
                agent.var_score.append(np.var(np.array(life_buffer)))
                agent.mavg_ammo_left.append(np.mean(np.array(ammo_buffer)))
                agent.mavg_kill_counts.append(np.mean(np.array(kills_buffer)))

                # Reset rolling stats buffer
                life_buffer, ammo_buffer, kills_buffer = [], [], []

                # Write Rolling Statistics to file
                with open("statistics/c51_ddqn_stats.txt", "w") as stats_file:
                    stats_file.write('Game: ' + str(GAME) + '\n')
                    stats_file.write('Max Score: ' + str(max_life) + '\n')
                    stats_file.write('mavg_score: ' + str(agent.mavg_score) + '\n')
                    stats_file.write('var_score: ' + str(agent.var_score) + '\n')
                    stats_file.write('mavg_ammo_left: ' + str(agent.mavg_ammo_left) + '\n')
                    stats_file.write('mavg_kill_counts: ' + str(agent.mavg_kill_counts) + '\n')

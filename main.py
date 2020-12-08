import gym
import numpy as np
from gym import wrappers  # gymの画像保存
import tensorflow as tf
from tensorflow import keras
from collections import deque
from collections import namedtuple
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
import random

train_repeat_times = 2000
Exp = namedtuple("Experience", ["s", "a", "r", "n_s", "d"])
# state, act ,reward , next state, done
#  observe, rew, done, info = self.env.step(action)

class Agent:
    actions = [0, 1, 2]
    gamma = 0.99
    eps = 1.0
    eps_decay = 0.05
    eps_min = 0.01
    iteration_times = 201  # game is 201steps.
    buffer_get_size = 32  # bufferの大きさが32になったらtrain.
    exp = deque(maxlen=20000)
    env = gym.make("MountainCar-v0")
    state = env.reset()
    episode_times = 0  # 現在のepisode数

    def __init__(self):
        self.train_network = self.create_network()  # 毎回更新する.(fixed target Q-network)
        self.target_network = self.create_network()  # 1episodeごとに更新する.
        print(self.target_network.inputs)
        self.target_network.set_weights(self.train_network.get_weights())
        # self.env = wrappers.Monitor(self.env, "/home/emile/Videos/", video_callable=(lambda ep: ep % 50 == 0))

    def create_network(self):
        hide_size = 16
        random.seed(0)
        model = Sequential()
        print(self.env.observation_space)
        model.add(Dense(hide_size, activation='relu', input_shape=self.env.observation_space.shape))
        model.add(Dense(hide_size*2, activation='relu'))
        model.add(Dense(self.env.action_space.n, activation='linear'))
        model.compile(loss='mse', optimizer=Adam(lr=0.001))
        model.summary()
        return model

    def eps_change(self):
        self.eps = max(self.eps-self.eps_decay, self.eps_min)

    def get_best_action(self, state):
        return np.argmax(self.predict2(state)[0])

    def predict2(self, states):  # 評価値を算出.
        stat = np.vstack([[states]])
        return self.train_network.predict(stat)

    def get_policy_action(self, state):  # epsilon-greedy
        tmp = random.uniform(0.0, 1.0)
        if tmp > self.eps:
            return self.get_best_action(state)
        else:  # random
            return np.random.randint(0, self.env.action_space.n)

    def update2(self):
        if len(self.exp) < self.buffer_get_size:
            return
        experiences = random.sample(self.exp, self.buffer_get_size)
        #self.exp.clear() <- ここを外したら上手く動き始めた...
        states = np.vstack([e.s for e in experiences])
        state_n = np.vstack([e.n_s for e in experiences])
        est = self.train_network.predict(states)
        fut = self.target_network.predict(state_n)  # Q_{train}(s,a)更新用
        for i, e in enumerate(experiences):
            rew = e.r
            if not e.d:
                rew += max(fut[i]) * self.gamma
            est[i][e.a] = rew
        self.train_network.fit(states, est, epochs=1, verbose=0)

    def play_step(self):
        action = self.get_policy_action(self.state)
        new_state, rew, done, info = self.env.step(action)  # 新しい状態を取得
        if new_state[0] >= 0.4:
            rew += 0.5
        elif new_state[0] >= 0.5:
            rew += 10.0
        e = Exp([self.state], action, rew, [new_state], done)
        self.exp.append(e)
        self.state = new_state
        return done, rew

    def play_episode(self, flag):
        self.state = self.env.reset()  # envを初期化
        ac_score = 0.0

        for tim in range(self.iteration_times):
            if flag:  # Trueなら映像を表示.
                self.env.render()
            done, reward = self.play_step()
            ac_score += reward
            self.update2()
            if done:
                break

        self.target_network.set_weights(self.train_network.get_weights())  # 両方のnetworkの同期.
        self.episode_times += 1
        if tim < 198:
            print("Success!.")
        print("eps is {},episode_times is {},length is {}times, score is {}".format(self.eps, self.episode_times, tim, ac_score))
        self.eps_change()

    def learn2(self):
        for i in range(train_repeat_times):
            flag = False
            if (i % 20) == 0:
                flag = True
            self.play_episode(flag)
        print("end.")


player = Agent()
player.learn2()

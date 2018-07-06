from agent_dir.agent import Agent
import os
import sys
import random
import numpy as np
from itertools import count

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd
import torch.optim as optim
import math

def sample_n_unique(sampling_f, n):
    res = []
    while len(res) < n:
        candidate = sampling_f()
        if candidate not in res:
            res.append(candidate)
    return res


class ReplayBuffer(object):
    def __init__(self, size, frame_history_len):
        self.size = size
        self.frame_history_len = frame_history_len
        
        self.next_idx      = 0
        self.num_in_buffer = 0
        
        self.obs    = None
        self.action = None
        self.reward = None
        self.done   = None
    
    def can_sample(self, batch_size):   # return True or False
        return batch_size + 1 <= self.num_in_buffer
    
    def _encode_sample(self, idxes):
        obs_batch      = np.concatenate([self._encode_observation(idx)[np.newaxis, :] for idx in idxes], 0)
        act_batch      = self.action[idxes]
        rew_batch      = self.reward[idxes]
        next_obs_batch = np.concatenate([self._encode_observation(idx + 1)[np.newaxis, :] for idx in idxes], 0)
        done_mask      = np.array([1.0 if self.done[idx] else 0.0 for idx in idxes], dtype=np.float32) # 將True, False轉為1, 0
        
        return obs_batch, act_batch, rew_batch, next_obs_batch, done_mask
    
    def sample(self, batch_size):
        assert self.can_sample(batch_size)
        idxes = sample_n_unique(lambda: random.randint(0, self.num_in_buffer - 2), batch_size)
        return self._encode_sample(idxes)
    
    def encode_recent_observation(self):
        assert self.num_in_buffer > 0
        return self._encode_observation((self.next_idx - 1) % self.size)
    
    def _encode_observation(self, idx):
        end_idx   = idx + 1
        start_idx = end_idx - self.frame_history_len
        
        if len(self.obs.shape) == 2: return self.obs[end_idx-1]
        
        if start_idx < 0 and self.num_in_buffer != self.size: start_idx = 0
        
        for idx in range(start_idx, end_idx - 1):
            if self.done[idx % self.size]:
                start_idx = idx + 1
        
        missing_context = self.frame_history_len - (end_idx - start_idx)
        if start_idx < 0 or missing_context > 0:
            frames = [np.zeros_like(self.obs[0]) for _ in range(missing_context)]
            
            for idx in range(start_idx, end_idx):
                frames.append(self.obs[idx % self.size])
            return np.concatenate(frames, 0)
        else:
            
            img_h, img_w = self.obs.shape[2], self.obs.shape[3]
            return self.obs[start_idx:end_idx].reshape(-1, img_h, img_w)
    
    def store_frame(self, frame):
        if len(frame.shape) > 1:
            frame = frame.transpose(2, 0, 1)
            
        if self.obs is None:
            self.obs    = np.empty([self.size] + list(frame.shape), dtype=np.uint8)
            self.action = np.empty([self.size],                     dtype=np.int32)
            self.reward = np.empty([self.size],                     dtype=np.float32)
            self.done   = np.empty([self.size],                     dtype=np.bool)
            
        self.obs[self.next_idx] = frame
        
        ret = self.next_idx
        self.next_idx = (self.next_idx + 1) % self.size
        self.num_in_buffer = min(self.size, self.num_in_buffer + 1)
        
        return ret
    
    def store_effect(self, idx, action, reward, done):
        self.action[idx] = action
        self.reward[idx] = reward
        self.done[idx]   = done

class DQN(nn.Module):
    def __init__(self):
        super(DQN, self).__init__()
        self.conv1 = nn.Conv2d( 4, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        self.fc4 = nn.Linear(7 * 7 * 64, 512)
        self.fc5 = nn.Linear(512, 3)
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.fc4(x.view(x.size(0), -1)))
        return self.fc5(x)
# detect GPU
USE_CUDA = torch.cuda.is_available()
dtype = torch.cuda.FloatTensor if USE_CUDA else torch.FloatTensor

class Variable(autograd.Variable):
    def __init__(self, data, *args, **kwargs):
        if USE_CUDA:
            data = data.cuda()
        super(Variable, self).__init__(data, *args, **kwargs)

class LinearSchedule(object):
    def __init__(self, schedule_timesteps, final_p, initial_p=1.0):
        self.schedule_timesteps = schedule_timesteps
        self.final_p            = final_p
        self.initial_p          = initial_p
        
    def value(self, t):
        fraction = min(float(t) / self.schedule_timesteps, 1.0)
        return self.initial_p + fraction * (self.final_p - self.initial_p)

class Agent_DQN(Agent):
    def __init__(self, env, args):
        """
        Initialize every things you need here.
        For example: building your model
        """

        super(Agent_DQN,self).__init__(env)

        self.Q = DQN().type(dtype)
        self.testCounter = 0

        if args.test_dqn:
            #you can load your model here
            print('loading trained model')
            self.Q.load_state_dict(torch.load('./DQN.pkl'))
        ##################
        # YOUR CODE HERE #
        ##################


    def init_game_setting(self):
        """

        Testing function will call this function at the begining of new game
        Put anything you want to initialize if necessary

        """
        ##################
        # YOUR CODE HERE #
        ##################
        pass


    def train(self):
        """
        Implement your training algorithm here
        """
        ##################
        # YOUR CODE HERE #
        ##################
    
        exploration = LinearSchedule(1000000, 0.1)
        replay_buffer_size=50000
        batch_size=32
        gamma=0.99
        learning_starts=5000
        learning_freq=4
        frame_history_len=1
        target_update_freq=1000
        
        input_arg = 4
        num_actions = 3
        
        # Construct an epilson greedy policy with given exploration schedule
        def select_epilson_greedy_action(model, obs, t):
            sample = random.random()
            eps_threshold = exploration.value(t)
            if sample > eps_threshold:
                obs = torch.from_numpy(obs).type(dtype).unsqueeze(0)
                return model(Variable(obs, volatile=True)).data.max(1)[1].cpu()[0]
            else:
                return torch.IntTensor([[random.randrange(num_actions)]])[0][0]
            
        
        Q = DQN().type(dtype)
        target_Q = DQN().type(dtype)
        
        optimizer = optim.RMSprop(Q.parameters(), lr=0.00025)
        replay_buffer = ReplayBuffer(replay_buffer_size, frame_history_len)
        loss_function = nn.MSELoss()
        
        ### RUN ENV
        num_param_updates = 0
        mean_episode_reward = -float('nan')
        best_mean_episode_reward = -float('inf')
        last_obs = self.env.reset()
        LOG_EVERY_N_STEPS = 10000
        episode = 0
        episodeReward = 0

        for t in count():            
            
            last_idx = replay_buffer.store_frame(last_obs)
            recent_observations = replay_buffer.encode_recent_observation()
            
            if t > learning_starts:
                Q.eval()
                action = select_epilson_greedy_action(Q, recent_observations, t)
                Q.train()
            else:
                action = random.randrange(num_actions)
                
            obs, reward, done, _ = self.env.step(action + 1)
            episodeReward += reward
            replay_buffer.store_effect(last_idx, action, reward, done)
            
            if done:
                obs = self.env.reset()
                episode += 1
                episodeReward = 0
            last_obs = obs
            
            if (t > learning_starts and t % learning_freq == 0 and replay_buffer.can_sample(batch_size)):
                obs_batch, act_batch, rew_batch, next_obs_batch, done_mask = replay_buffer.sample(batch_size)
                
                obs_batch = Variable(torch.from_numpy(obs_batch).type(dtype))
                act_batch = Variable(torch.from_numpy(act_batch).long())
                rew_batch = Variable(torch.from_numpy(rew_batch))
                next_obs_batch = Variable(torch.from_numpy(next_obs_batch).type(dtype))
                not_done_mask = Variable(torch.from_numpy(1 - done_mask)).type(dtype) # 如果下一個state是episode中的最後一個，則done_mask = 1
                
                if USE_CUDA:
                    act_batch = act_batch.cuda()
                    rew_batch = rew_batch.cuda()
                    
                current_Q_values = Q(obs_batch).gather(1, act_batch.unsqueeze(1))
                
                next_max_q = target_Q(next_obs_batch).detach().max(1)[0].unsqueeze(1)
                next_Q_values = not_done_mask.unsqueeze(1) * next_max_q

                target_Q_values = rew_batch.unsqueeze(1) + (gamma * next_Q_values)

                loss = loss_function(current_Q_values, target_Q_values)
                optimizer.zero_grad()
                loss.backward()
                
                optimizer.step()
                num_param_updates += 1
                
                if num_param_updates % target_update_freq == 0:
                    target_Q.load_state_dict(Q.state_dict())
            
            if t % LOG_EVERY_N_STEPS == 0 and t > learning_starts:
                print("Timestep %d" % (t,))
                print("exploration %f" % exploration.value(t))
                sys.stdout.flush()
                
                torch.save(Q.state_dict(), 'DQN.pkl')
            

    def make_action(self, observation, test=True):
        """
        Return predicted action of your agent

        Input:
            observation: np.array
                stack 4 last preprocessed frames, shape: (84, 84, 4)

        Return:
            action: int
                the predicted action from trained model
        """
        ##################
        # YOUR CODE HERE #
        ##################
        if USE_CUDA:
            action = self.Q(Variable(torch.FloatTensor(observation.transpose(2, 0, 1)).unsqueeze(0)).cuda()).max(-1)[1].data[0]
        else:
            action = self.Q(Variable(torch.FloatTensor(observation.transpose(2, 0, 1)).unsqueeze(0))).max(-1)[1].data[0]
        action += 1
        self.testCounter += 1
        if self.testCounter % 100 == 0:
            action = 0
        return action
import scipy, torch, time
import numpy as np
import matplotlib.pyplot as plt
from agent_dir.agent import Agent
from torch.autograd import Variable
import cv2

def prepro(data,image_size=(80,80)):
    """
    Call this function to preprocess RGB image to grayscale image if necessary
    This preprocessing code is from
        https://github.com/hiwonjoon/tf-a3c-gpu/blob/master/async_agent.py
    
    Input: 
    RGB image: np.array
        RGB screen of game, shape: (210, 160, 3)
    Default return: np.array 
        Grayscale image, shape: (80, 80)
    
    """
    image = 0.2126 * data[:, :, 0] + 0.7152 * data[:, :, 1] + 0.0722 * data[:, :, 2] # grayscale
    image = image[30:,:] # remove scoreboard
    image = cv2.resize(image, dsize=image_size, interpolation=cv2.INTER_CUBIC) # resize to (80, 80)
    image = np.expand_dims(image, axis=0) # (1, 80, 80)
    # print('IMAGE SHAPE', image.shape)
    return image

class Policy(torch.nn.Module):
    def __init__(self):
        super(Policy, self).__init__()

        self.conv1 = torch.nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3) # (batch_size, 16, 208, 158)
        self.conv2 = torch.nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3) # (batch_size, 32, 206, 156)
        self.fc1 = torch.nn.Linear(32 * 76 * 76, 64)
        self.fc2 = torch.nn.Linear(64, 32)
        self.fc3 = torch.nn.Linear(32, 6) # 6 actions to choose from
        # TODO: decrease number of actions?
        # known actions: 2(up) , 3(down)
        
        self.saved_log_probs = []
        self.rewards = []
        
    def forward(self, x): # x: np.array (1, 80, 80)
        x = x.reshape(-1, 1, 80, 80) # (batch_size, channels, ...)
        x = Variable(torch.Tensor(x))
        if torch.cuda.is_available():
            x = x.cuda()
        x = self.conv1(x)
        x = self.conv2(x)
        x = torch.nn.functional.selu(x)
        x = x.view(-1, 32*76*76)
        x = self.fc1(x) # TODO: add batch norm?
        x = torch.nn.functional.selu(x)
        x = self.fc2(x)
        x = torch.nn.functional.selu(x)
        x = self.fc3(x)
        action_probs = torch.nn.functional.softmax(x, dim=1)
        return action_probs # (batch_size, 6)

    def reset(self):
        self.saved_log_probs = []
        self.rewards = []

class Agent_PG(Agent):
    def __init__(self, env, args):
        """
        Initialize every things you need here.
        For example: building your model
        """

        super(Agent_PG,self).__init__(env)

        if args.test_pg:
            #you can load your model here
            print('loading trained model')

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


    def train(self, render): # NOTE: supposing batch_size=1
        """
        Implement your training algorithm here
        """
        ##################
        # YOUR CODE HERE #
        ##################
        from tensorboardX import SummaryWriter
        writer = SummaryWriter()

        policy = Policy()
        policy.train()
        if torch.cuda.is_available():
            print('Using CUDA')
            policy = policy.cuda()
        else:
            print('Not using CUDA')
        optimizer = torch.optim.Adam(policy.parameters())
        for i_episode in range(100000): # 21 points/reward per episode, ~6000 episodes to reach baseline?
            observation = self.env.reset()
            optimizer.zero_grad()
            policy.reset()
            a = time.time()
            for t in range(10000): # usually done in around 1000 steps
                if render:
                    self.env.env.render()

                observation = prepro(observation) # (1, 80, 80)
                action_probs = policy(observation) # (batch_size, 6)
                action_sampler = torch.distributions.Categorical(action_probs[0])
                # action = int(torch.max(policy_output, dim=1)[1].data) # torch.max returns (max val, argmax)
                action = action_sampler.sample()
                policy.saved_log_probs.append(action_sampler.log_prob(action))

                observation, reward, done, info = self.env.step(int(action))
                policy.rewards.append(reward)

                if torch.cuda.is_available():
                    if (t+1)%1000 == 0:
                        print('Ran {} timesteps'.format(t+1))
                else:
                    if (t+1)%100 == 0:
                        print('Ran {} timesteps'.format(t+1))

                if done:
                    print("Episode finished after {} timesteps".format(t+1))
                    break

            if not done:
                print('Terminated episode after running for 10000 steps')

            R = 0
            policy_loss = []
            rewards = []
            print('Calculating rewards and loss function values...')
            for r in policy.rewards[::-1]:
                # R = r + args.gamma * R
                R = r + R
                rewards.insert(0, R)
            writer.add_scalar('0618-0015/reward', R, i_episode+1)
            writer.export_scalars_to_json("./0618-0015/{}-reward.json".format(i_episode+1))
            rewards = torch.Tensor(rewards)
            # rewards = (rewards - rewards.mean()) / (rewards.std() + eps)

            print('Doing backprops (one at a time) into policy...')
            for log_prob, reward in zip(policy.saved_log_probs, rewards):
                single_loss = -log_prob * reward
                single_loss.backward()

            # policy_loss = torch.cat(policy_loss).sum()
            # print('Doing backprop into policy...')
            # policy_loss.backward()
            optimizer.step()

            print('[Episode {}: {} minutes] Reward is {}'.format(i_episode+1, (time.time()-a)/60, R))
            
            if i_episode+1 % 200 == 1:
                torch.save(policy.state_dict(), './models/0618-0015-episode{}.pth'.format(i_episode+1))
                print('Saved model!')


    def make_action(self, observation, test=True): # used during test time
        """
        Return predicted action of your agent

        Input:
            observation: np.array
                current RGB screen of game, shape: (210, 160, 3)

        Return:
            action: int
                the predicted action from trained model
        """
        ##################
        # YOUR CODE HERE #
        ##################
        return self.env.get_random_action()


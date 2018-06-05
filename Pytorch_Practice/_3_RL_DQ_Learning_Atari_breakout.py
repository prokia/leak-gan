""" Script for creating an RL agent for the atari breakout game
    Use of OpenAI GYM for the Environment
    Use of PyTorch for the Neural Networks (Obviously :p :D)
"""

import argparse
import numpy as np
import torch as th
import time
import random
import gym
import os
import pickle

from collections import namedtuple

# Set the global Constants for the script
device = th.device("cuda" if th.cuda.is_available() else "cpu")
environment_name = "Breakout-v0"

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))


class ReplayMemory(object):
    """ A Buffer for implementing the Replay memory feature
    """

    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args):
        """Saves a transition."""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


class DQN(th.nn.Module):
    """ Convolutional neural network defining the Q function for the Agent
        Will be trained using Deep Q Learning
    """

    def __init__(self, channels, num_actions):
        """
        Constructor
        :param channels: input channel width of the image
        :param num_actions: number of actions to predict over
        """
        # super call
        super(DQN, self).__init__()

        # create state of the Network
        self.channels = channels
        self.num_actions = num_actions

        # define the modules needed for the network:
        from torch.nn import Conv2d, BatchNorm2d

        self.conv1 = Conv2d(self.channels, 8, kernel_size=(2, 2), stride=2, padding=(0, 1))
        self.bn1 = BatchNorm2d(8)
        self.conv2 = Conv2d(8, 16, kernel_size=(4, 4), stride=2, padding=(0, 0))
        self.bn2 = BatchNorm2d(16)
        self.conv3 = Conv2d(16, 32, kernel_size=(8, 8), stride=4, padding=(0, 0))
        self.bn3 = BatchNorm2d(32)

        # last classifying convolutional layer:
        self.final = Conv2d(32, self.num_actions, (10, 8), stride=1, padding=(0, 0))

    def forward(self, inp_x):
        """
        forward pass of the neural network
        :param inp_x: input image of the network [here => (180 x 146) grayscale image]
        :return: preds => predictions for taking an action
        """
        from torch.nn.functional import relu

        # reassign the tensor name (for simplicity)
        y = inp_x

        # define the computational pipeline
        y = relu(self.bn1(self.conv1(y)))
        y = relu(self.bn2(self.conv2(y)))
        y = relu(self.bn3(self.conv3(y)))

        # final classification layer
        raw_preds = self.final(y).view(-1, self.num_actions)  # squeeze the predictions

        # return the predictions
        return raw_preds


class Agent:
    """ The agent for the RL environment """

    def __init__(self, q_net, t_net, memory, batch_size=128, gamma=0.999,
                 eps_start=0.9, eps_end=0.05, eps_decay=210000, target_update=12,
                 learning_rate=0.01, clip_value=3):
        """
        constructor for the class
        """

        # set the state for the object
        self.q_net = q_net
        self.t_net = t_net
        self.memory = memory
        self.batch_size = batch_size
        self.gamma = gamma
        self.eps_start = eps_start
        self.eps_end = eps_end
        self.eps_decay = eps_decay
        self.target_update = target_update
        self.lr = learning_rate
        self.clip_value = clip_value

        # set a counter for number of steps done
        self.steps_done = 0

        # The target Net must have the same parameters as the q_net initially:
        self.t_net.load_state_dict(q_net.state_dict())
        self.t_net.eval()  # switch the target_net in evaluation mode

        # create an optimizer for the Agent
        self.optimizer = th.optim.RMSprop(self.q_net.parameters())

    def select_action(self, state):
        sample = random.random()
        eps_threshold = (self.eps_end + (self.eps_start - self.eps_end) *
                         np.exp(-1. * self.steps_done / self.eps_decay))
        self.steps_done += 1
        if sample > eps_threshold:
            with th.no_grad():
                return self.q_net(state).max(1)[1].view(1, 1)
        else:
            return th.tensor([[random.randrange(self.q_net.num_actions)]],
                             device=device, dtype=th.long)

    def optimize(self):
        """ Optimize the Q_Function Network """

        from torch.nn.functional import smooth_l1_loss

        if len(self.memory) < self.batch_size:
            # still need to fill the Memory with transitions
            return

        transitions = self.memory.sample(self.batch_size)
        # Transpose the batch (see http://stackoverflow.com/a/19343/3343043 for
        # detailed explanation).
        batch = Transition(*zip(*transitions))

        # Compute a mask of non-final states and concatenate the batch elements
        non_final_mask = th.tensor(tuple(map(lambda s: s is not None,
                                             batch.next_state)), device=device, dtype=th.uint8)
        non_final_next_states = th.cat([s for s in batch.next_state
                                        if s is not None])

        state_batch = th.cat(batch.state)
        action_batch = th.cat(batch.action)
        reward_batch = th.cat(batch.reward)

        # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
        # columns of actions taken
        state_action_values = self.q_net(state_batch).gather(1, action_batch)

        # Compute V(s_{t+1}) for all next states.
        next_state_values = th.zeros(self.batch_size, device=device)
        next_state_values[non_final_mask] = self.t_net(non_final_next_states).max(1)[0].detach()

        # Compute the expected Q values
        expected_state_action_values = (next_state_values * self.gamma) + reward_batch

        # Compute Huber loss
        loss = smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1))

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()

        # gradient clipping in the range => [-1, 1]
        for param in self.q_net.parameters():
            param.grad.data.clamp_(-self.clip_value, self.clip_value)
        self.optimizer.step()

        return loss.item()


def subtract_scaled_frames(fr1, fr2, alpha=0.7):
    """
    subtract the second frame from the first one after alpha scaling
    Generally used after preprocessing frames
    :param fr1: previous observation
    :param fr2: current observation
    :param alpha: scaling constant
    :return: sub => combined frame
    """
    return fr2 - (alpha * fr1)


def expand_and_convert(frame):
    return th.from_numpy(
        np.expand_dims(
            np.expand_dims(
                frame, axis=0
            ), axis=0
        )
    ).float().to(device)


def preprocess_frame(observation):
    """
    crop, remove colour and normalize the observation frames
    :param observation: observation Frame
    :return: nor => Preprocessed frame
    """
    # crop the observation
    crop = observation[30:, 7: -7, :]

    # make it black and white
    bnw = crop.mean(axis=-1)

    # normalize the images
    nor = bnw / 255

    return nor


def play_one_episode(agent, env, render=False, inverse_speed=0.05, save_dir="./Out"):
    # apply the monitor if render is false
    if not render:
        work_env = gym.wrappers.Monitor(env, directory=save_dir, resume=True)
    else:
        work_env = env

    # play and render one episode:
    current_screen = expand_and_convert(preprocess_frame(work_env.reset()))
    last_screen = th.zeros_like(current_screen, device=device, dtype=th.float)

    state = subtract_scaled_frames(last_screen, current_screen)

    done = False

    while not done:
        action = agent.select_action(state)
        obs, _, done, _ = work_env.step(action.item())

        last_screen = current_screen
        current_screen = expand_and_convert(preprocess_frame(obs))

        state = subtract_scaled_frames(last_screen, current_screen)

        if render:
            work_env.render()
        time.sleep(inverse_speed)


def train_agent(agent, env, num_episodes=100, feed_back_factor=100,
                save_after=100, save_dir="./Saved_Models", save=True):
    from itertools import count

    # start the training loop:
    for i_episode in range(num_episodes):
        # Initialize the environment and state
        current_screen = expand_and_convert(preprocess_frame(env.reset()))
        last_screen = th.zeros_like(current_screen, device=device, dtype=th.float)

        state = subtract_scaled_frames(last_screen, current_screen)

        losses = []
        rewards = []
        for t in count():
            # Select and perform an action
            action = agent.select_action(state)
            obs, reward, done, _ = env.step(action.item())
            reward = th.tensor([reward], device=device)

            # Observe new state
            last_screen = current_screen
            current_screen = expand_and_convert(preprocess_frame(obs))
            if not done:
                next_state = subtract_scaled_frames(last_screen, current_screen)
            else:
                next_state = None

            # Store the transition in memory
            agent.memory.push(state, action, next_state, reward)

            # Move to the next state
            state = next_state

            # Perform one step of the optimization (on the target network)
            current_loss = agent.optimize()
            rewards.append(reward)
            losses.append(current_loss if current_loss is not None else 0)
            if done:
                break

        print("Episode: %d Total_episode_Length: %d  Total_Loss: %.3f  Total_Reward: %d" %
              (i_episode, t, np.sum(losses), np.sum(rewards)))

        if i_episode % feed_back_factor == 0 or i_episode == 0:
            # play an episode for viewing
            play_one_episode(agent, env, render=not save)

        # save the model
        if i_episode % save_after == 0 or i_episode == 0:
            print("Saving Model ... ")
            with open(os.path.join(save_dir, "Checkpoint"), 'w') as chkp:
                chkp.write("total_episodes: " + str(i_episode + 1) + "\n")
                chkp.write("total_steps: " + str(agent.steps_done) + "\n")
            th.save(agent.q_net, os.path.join(save_dir, "Model"+str(i_episode+1)+".pth"),
                    pickle)

        # Update the target network
        if i_episode % agent.target_update == 0:
            agent.t_net.load_state_dict(agent.q_net.state_dict())

    print('Complete')


def parse_arguments():
    """
    command line argument parser
    :return: args => parsed arguments namespace
    """
    parser = argparse.ArgumentParser()

    parser.add_argument("--frames_alpha", action="store", type=float, default=0.7,
                        help="Value of \\alpha used for scaling input observations'")

    args = parser.parse_args()

    return args


def main(args):
    """
    Main function of the script
    :param args: parsed command line arguments
    :return: None
    """
    # get the Atari game environment:
    env = gym.make(environment_name)
    num_actions = env.action_space.n

    q_network = DQN(1, num_actions).to(device)
    target_network = DQN(1, num_actions).to(device)

    # create an agent:
    agent = Agent(q_network, target_network, ReplayMemory(10000), batch_size=256)

    # train the agent
    train_agent(agent, env, num_episodes=21000)

    env.close()


if __name__ == '__main__':
    main(parse_arguments())

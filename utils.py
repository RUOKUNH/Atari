import numpy as np
# import matplotlib.pyplot as plt
from net import *
import gym
from actor_critic import ActorCritic
import rl_utils

def random_policy(env_name):
    env = gym.make(env_name)
    state = env.reset()
    for s in range(1000):
        env.render()
        action = env.action_space.sample()
        state, reward, done, info = env.step(action)
        if done:
            print('dead in %d steps' % s)
            break
    env.close()


def train_DQN(env_name, lr, gamma, hidden_dim=None, net_type='DQN'):
    if hidden_dim is None:
        hidden_dim = [128]
    env = gym.make(env_name)
    num_episodes = 500
    epsilon = 0.01
    target_update = 10
    buffer_size = 10000
    minimal_size = 500
    batch_size = 64
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    random.seed(0)
    np.random.seed(0)
    env.seed(0)
    torch.manual_seed(0)
    replay_buffer = ReplayBuffer(buffer_size)
    state_dim = len(env.observation_space.high)
    action_dim = len(env.action_space.high)
    agent = DQN(state_dim, hidden_dim, action_dim, device, lr, gamma, epsilon, target_update, net_type=net_type)
    best_return = -np.inf
    return_list = []
    for i in range(10):
        with tqdm(total=int(num_episodes / 10), desc='Iteration %d' % i) as pbar:
            for i_episode in range(int(num_episodes / 10)):
                episode_return = 0
                state = env.reset()
                done = False
                while not done:
                    action = agent.take_action(state)
                    next_state, reward, done, _ = env.step(action)
                    replay_buffer.add(state, action, reward, next_state, done)
                    state = next_state
                    episode_return += reward
                    if replay_buffer.size() > minimal_size:  # 当buffer数据数量超过一定值后，才进行Q网络训练
                        b_s, b_a, b_r, b_ns, b_d = replay_buffer.sample(batch_size)
                        transition_dict = {'states': b_s, 'actions': b_a, 'next_states': b_ns, 'rewards': b_r,
                                           'dones': b_d}
                        agent.update(transition_dict)
                return_list.append(episode_return)
                if episode_return > best_return:
                    best_return = episode_return
                    agent.save_model('./models', env_name)
                if (i_episode + 1) % 10 == 0:
                    pbar.set_postfix({'episode': '%d' % (num_episodes / 10 * i + i_episode + 1),
                                      'return': '%.3f' % np.mean(return_list[-10:])})
                pbar.update(1)

    return agent


def train_DQN_Conv(env_name, lr, gamma):
    env = gym.make(env_name)
    state_channels = 10
    num_episodes = 500
    epsilon = 0.01
    target_update = 10
    buffer_size = 10000
    minimal_size = 500
    maximal_size = 1000
    batch_size = 16
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print(device)

    random.seed(0)
    np.random.seed(0)
    env.seed(0)
    torch.manual_seed(0)
    replay_buffer = ReplayBuffer(buffer_size)
    state_dim = len(env.observation_space.high)
    action_dim = len(env.action_space.high)
    agent = ConvDQN(state_dim, action_dim, device, lr, gamma, epsilon, target_update)
    best_return = -np.inf
    return_list = []
    state_list = []
    for i in range(10):
        with tqdm(total=int(num_episodes / 10), desc='Iteration %d' % i) as pbar:
            for i_episode in range(int(num_episodes / 10)):
                episode_return = 0
                state = env.reset()
                done = False
                while not done:
                    if len(state_list) < state_channels:
                        s0 = np.repeat(np.mean(state, axis=-1)[np.newaxis, ...], state_channels, 0)
                    action = agent.take_action(s0)
                    next_state, reward, done, _ = env.step(action)
                    if len(state_list) < state_channels:
                        state_list.append(np.mean(state, axis=-1))
                    else:
                        s0 = np.array(state_list)
                        state_list.pop(0)
                        state_list.append(np.mean(next_state, axis=-1))
                        s1 = np.array(state_list)
                        replay_buffer.add(s0, action, reward, s1, done)
                        if replay_buffer.size() > maximal_size:
                            replay_buffer.pop()
                        s0 = s1
                    state = next_state
                    episode_return += reward
                    if replay_buffer.size() > minimal_size:  # 当buffer数据数量超过一定值后，才进行Q网络训练
                        b_s, b_a, b_r, b_ns, b_d = replay_buffer.sample(batch_size)
                        transition_dict = {'states': b_s, 'actions': b_a, 'next_states': b_ns, 'rewards': b_r,
                                           'dones': b_d}
                        agent.update(transition_dict)
                return_list.append(episode_return)
                if episode_return > best_return:
                    best_return = episode_return
                    agent.save_model('./models', env_name)
                if (i_episode + 1) % 10 == 0:
                    pbar.set_postfix({'episode': '%d' % (num_episodes / 10 * i + i_episode + 1),
                                      'return': '%.3f' % np.mean(return_list[-10:])})
                pbar.update(1)

    return agent


def train_ac(env_name):
    actor_lr = 1e-3
    critic_lr = 1e-2
    num_episodes = 3000
    gamma = 0.98

    env = gym.make(env_name)
    env.seed(0)
    state_dim = len(env.observation_space.high)
    action_dim = len(env.action_space.high)
    agent = ActorCritic(state_dim, action_dim, actor_lr, critic_lr, gamma)
    best_return = -np.inf

    return_list = []
    for i in range(10):
        with tqdm(total=int(num_episodes / 10), desc='Iteration %d' % i) as pbar:
            for i_episode in range(int(num_episodes / 10)):
                episode_return = 0
                transition_dict = {'states': [], 'actions': [], 'next_states': [], 'rewards': [], 'dones': []}
                state = env.reset()
                done = False
                while not done:
                    action = agent.take_action(state)
                    next_state, reward, done, _ = env.step(action)
                    transition_dict['states'].append(state)
                    transition_dict['actions'].append(action)
                    transition_dict['next_states'].append(next_state)
                    transition_dict['rewards'].append(reward)
                    transition_dict['dones'].append(done)
                    state = next_state
                    episode_return += reward
                return_list.append(episode_return)
                if episode_return > best_return:
                    best_return = episode_return
                    agent.save_model('./models', env_name)
                agent.update(transition_dict)
                if (i_episode + 1) % 10 == 0:
                    pbar.set_postfix({'episode': '%d' % (num_episodes / 10 * i + i_episode + 1),
                                      'return': '%.3f' % np.mean(return_list[-10:])})
                pbar.update(1)
    return return_list


def test_ac(env_name):
    env = gym.make(env_name)
    model = torch.load(os.path.join('./models', env_name + '.pth'))
    state_dim, action_dim = model['dim']
    agent = ActorCritic(state_dim, action_dim, 1, 1, 1)
    agent.load_model(model, env_name)
    state = env.reset()
    for s in range(1000):
        env.render()
        action = agent.take_action(state)
        state, reward, done, info = env.step(action)

        if done:
            print('dead in %d steps' % s)
            break
    env.close()
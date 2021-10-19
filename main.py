import gym
from utils import *


def train(env_name):
    train_DQN(env_name, [256, 512, 1024])


def test(env_name):
    device = torch.device('cpu')
    env = gym.make(env_name)
    model = torch.load(os.path.join('./models', env_name+'.pth'))
    state_dim, hidden_dim, action_dim = model['dim']
    agent = DQN(state_dim, hidden_dim, action_dim, device)
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


def train_conv(env_name):
    train_DQN_Conv(env_name)


def test_conv(env_name):
    device = torch.device('cpu')
    env = gym.make(env_name)
    model = torch.load(os.path.join('./models', env_name + '_conv.pth'))
    state_dim, action_dim = model['dim']
    agent = ConvDQN(state_dim, action_dim, device)
    agent.load_model(model, env_name)
    state_list = []
    state_channels = 3
    state = env.reset()
    for s in range(1000):
        env.render()
        if len(state_list) < state_channels:
            state_list.append(np.mean(state, axis=-1))
            s0 = np.repeat(np.mean(state, axis=-1)[np.newaxis, ...], state_channels, 0)
        else:
            state_list.append(np.mean(state, axis=-1))
            state_list.pop(0)
            s0 = np.array(state_list)
        action = agent.take_action(s0)
        state, reward, done, info = env.step(action)

        if done:
            print('dead in %d steps' % s)
            break
    env.close()


if __name__ == '__main__':
    env_name = 'Breakout-v0'
    os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
    test_conv(env_name)

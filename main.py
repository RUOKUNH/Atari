import gym
from utils import *
import argparse


def train(env_name, input_type, lr, gamma, net_type='DQN'):
    if input_type == 'ram':
        train_DQN(env_name, lr, gamma, [128, 256, 512, 1024], net_type=net_type)
    else:
        train_DQN_Conv(env_name, lr, gamma)


def test_ram(env_name, net_type):
    device = torch.device('cpu')
    env = gym.make(env_name)
    model = torch.load(os.path.join('./models', env_name+'.pth'))
    state_dim, hidden_dim, action_dim = model['dim']
    agent = DQN(state_dim, hidden_dim, action_dim, device, net_type=net_type)
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


def test_image(env_name, net_type):
    device = torch.device('cpu')
    env = gym.make(env_name)
    model = torch.load(os.path.join('./models', env_name + '_conv.pth'))
    state_dim, action_dim = model['dim']
    agent = ConvDQN(state_dim, action_dim, device, net_type=net_type)
    agent.load_model(model, env_name)
    state_list = []
    state_channels = 10
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


def test(env_name, net_type, input_type):
    if input_type == 'ram':
        test_ram(env_name, net_type)
    else:
        test_image(env_name, net_type)


def main(args):
    env_name = 'Breakout-v0'
    os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
    if args.mode == 'train':
        train(env_name, net_type=args.net_type, input_type=args.input_type, lr=args.lr, gamma=args.gamma)
    else:
        test(env_name, net_type=args.net_type, input_type=args.input_type)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode',
                        help='train or test',
                        type=str)
    parser.add_argument('--net_type',
                        default='DQN',
                        help='net_type',
                        type=str)
    parser.add_argument('--input_type',
                        default='ram',
                        help='input type',
                        type=str)
    parser.add_argument('--lr',
                        default=1e-3,
                        type=float)
    parser.add_argument('--gamma',
                        default=0.98,
                        type=float)
    args = parser.parse_args()
    main(args)

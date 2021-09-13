import sys
sys.path.append('../')
sys.path.append('../../')

import time
import numpy as np
import pickle
import dmc2gym
import hydra
import torch
import utils
from logger import Logger
from replay_buffer import ReplayBuffer
from video import VideoRecorder
from tqdm import tqdm

torch.backends.cudnn.benchmark = True


def make_env(cfg):
    """Helper function to create dm_control environment"""
    if cfg.env == 'ball_in_cup_catch':
        domain_name = 'ball_in_cup'
        task_name = 'catch'
    elif cfg.env == 'point_mass_easy':
        domain_name = 'point_mass'
        task_name = 'easy'
    elif cfg.env == 'cartpole_two_poles':
        domain_name = 'cartpole'
        task_name = 'two_poles'
    elif cfg.env == 'cartpole_three_poles':
        domain_name = 'cartpole'
        task_name = 'three_poles'
    else:
        domain_name = cfg.env.split('_')[0]
        task_name = '_'.join(cfg.env.split('_')[1:])

    # per dreamer: https://github.com/danijar/dreamer/blob/02f0210f5991c7710826ca7881f19c64a012290c/wrappers.py#L26
    camera_id = 2 if domain_name == 'quadruped' else 0

    env = dmc2gym.make(domain_name=domain_name,
                       task_name=task_name,
                       seed=cfg.seed,
                       visualize_reward=False,
                       from_pixels=True,
                       height=cfg.image_size,
                       width=cfg.image_size,
                       frame_skip=cfg.action_repeat,
                       camera_id=camera_id)

    env = utils.FrameStack(env, k=cfg.frame_stack)

    env.seed(cfg.seed)
    assert env.action_space.low.min() >= -1
    assert env.action_space.high.max() <= 1

    return env


class Workspace(object):
    def __init__(self, cfg):
        self.work_dir = '<ENTER/LOCATION/HERE>'
        print(f'workspace: {self.work_dir}')

        self.cfg = cfg

        self.logger = Logger(self.work_dir,
                             save_tb=cfg.log_save_tb,
                             log_frequency=cfg.log_frequency_step,
                             agent=cfg.agent.name,
                             action_repeat=cfg.action_repeat)

        utils.set_seed_everywhere(cfg.seed)
        self.device = torch.device(cfg.device)
        self.env = make_env(cfg)

        cfg.agent.params.obs_shape = self.env.observation_space.shape
        cfg.agent.params.action_shape = self.env.action_space.shape
        cfg.agent.params.action_range = [
            float(self.env.action_space.low.min()),
            float(self.env.action_space.high.max())
        ]

        self.agent = hydra.utils.instantiate(cfg.agent)

        self.replay_buffer = ReplayBuffer(self.env.observation_space.shape,
                                          self.env.action_space.shape,
                                          cfg.replay_buffer_capacity,
                                          self.cfg.image_pad,
                                          self.device,
                                          self.cfg.env)

        self.video_recorder = VideoRecorder(
            self.work_dir if cfg.save_video else None)
        self.step = 0

    def evaluate(self):
        average_episode_reward = 0
        eps_reward = []

        for episode in range(self.cfg.num_eval_episodes):
            obs = self.env.reset()
            n_actions = 0

            self.video_recorder.init(enabled=(episode == 0))
            done = False
            episode_reward = 0
            episode_step = 0
            while not done:
                with utils.eval_mode(self.agent):
                    action = self.agent.act(obs, sample=False)
                    n_actions += 1

                obs, reward, done, info = self.env.step(action)
                self.video_recorder.record(self.env)
                episode_reward += reward
                episode_step += 1

            average_episode_reward += episode_reward
            self.video_recorder.save(f'{self.step}.mp4')

        average_episode_reward /= self.cfg.num_eval_episodes

        sd_episode_reward = np.std(eps_reward)

        self.logger.log('eval/episode_reward', average_episode_reward,
                        self.step)
        self.logger.dump(self.step)

        return average_episode_reward, sd_episode_reward

    def evaluate_gather_grads(self):
        """Need to gather the gradients of the RL task and KSL task"""
        if len(self.replay_buffer) < 500:
            return

        rl_m, rl_v, ksl_m, ksl_v = self.agent.get_gradients(self.replay_buffer)

        with open(f'<ENTER/LOCATION/HERE>/sep_optim-rl_m-{self.cfg.env}-{self.cfg.seed}-{self.step}.data', 'wb') as f:
            pickle.dump(rl_m, f)

        with open(f'<ENTER/LOCATION/HERE>/sep_optim-rl_v-{self.cfg.env}-{self.cfg.seed}-{self.step}.data', 'wb') as f:
            pickle.dump(rl_v, f)

        with open(f'<ENTER/LOCATION/HERE>/sep_optim-ksl_m-{self.cfg.env}-{self.cfg.seed}-{self.step}.data', 'wb') as f:
            pickle.dump(ksl_m, f)

        with open(f'<ENTER/LOCATION/HERE>/sep_optim-ksl_v-{self.cfg.env}-{self.cfg.seed}-{self.step}.data', 'wb') as f:
            pickle.dump(ksl_v, f)

    def run(self):
        print(f'Eval freq: {self.cfg.eval_frequency}')
        print(f'k: {self.agent.k}')
        print(f'lr: {self.cfg.lr}')

        episode, episode_reward, episode_step, done = 0, 0, 1, True
        start_time = time.time()

        eval_mean = []

        while self.step < (self.cfg.num_train_steps // self.cfg.action_repeat):
            if done:
                if self.step > 0:
                    self.logger.log('train/duration',
                                    time.time() - start_time, self.step)
                    start_time = time.time()
                    self.logger.dump(
                        self.step, save=(self.step > self.cfg.num_seed_steps))

                # evaluate agent periodically
                if self.step % self.cfg.eval_frequency == 0:
                    self.logger.log('eval/episode', episode, self.step)

                    means, sds = self.evaluate()
                    eval_mean.append(means)

                    print(f'OSL: {np.mean(self.agent.osl_loss_hist[-20000:])}')
                    print(f'R: {np.mean(self.agent.r_loss_hist[-20000:])}')

                    # print(f'Saving {(self.step * self.cfg.action_repeat) / 1000}k agent')
                    # torch.save(
                    #     self.agent.critic.encoder.state_dict(),
                    #     f'<ENTER/LOCATION/HERE>/ksl-optim-encoder-{(self.step * self.cfg.action_repeat) / 1000}k.pt'
                    # )

                    self.evaluate_gather_grads()

                self.logger.log('train/episode_reward', episode_reward,
                                self.step)

                obs = self.env.reset()
                done = False
                episode_reward = 0
                # TODO: at the very top, episode_step is init to 1 but here it is 0...
                episode_step = 0
                episode += 1

                self.logger.log('train/episode', episode, self.step)

            # sample action for data collection
            if self.step < self.cfg.num_seed_steps:
                action = self.env.action_space.sample()
            else:
                with utils.eval_mode(self.agent):
                    action = self.agent.act(obs, sample=True)

            self.agent.osl.train(True)

            # run training update
            if self.step >= self.cfg.num_seed_steps:
                for _ in range(self.cfg.num_train_iters):
                    self.agent.update(self.replay_buffer, self.logger,
                                      self.step)

            next_obs, reward, done, info = self.env.step(action)

            # allow infinite bootstrap
            # TODO: shouldn't DONE always be 0? replay buffer is NOT DONE when adding...
            done = float(done)
            done_no_max = 0 if episode_step + 1 == self.env._max_episode_steps else done
            episode_reward += reward

            if done:
                eeo = 1
            else:
                eeo = 0

            self.replay_buffer.add(obs, action, reward, next_obs, done,
                                   done_no_max, eeo)

            obs = next_obs
            episode_step += 1
            self.step += 1

        with open(f'<ENTER/LOCATION/HERE>/ksl-optim-r-{self.cfg.env}-s{self.cfg.seed}-b{self.cfg.batch_size}-k{self.cfg.agent.params.k}-p{self.cfg.p}-mean.data', 'wb') as f:
            pickle.dump(eval_mean, f)

        self.agent.save(
            dir='<ENTER/LOCATION/HERE>',
            extras=f'ksl-{self.cfg.env}-s{self.cfg.seed}-b{self.cfg.batch_size}-k{self.cfg.agent.params.k}-p{self.cfg.p}-500k'
        )


@hydra.main(config_path='config.yaml', strict=True)
def main(cfg):
    from train import Workspace as W

    workspace = W(cfg)
    workspace.run()


if __name__ == '__main__':
    main()

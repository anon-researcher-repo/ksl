import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import utils
import hydra

DROPOUT = 0.0
DROPOUT_FC = 0.0


def loss_fn(x, y):
    x = F.normalize(x, dim=-1, p=2)
    y = F.normalize(y, dim=-1, p=2)
    return 2 - 2 * (x * y).sum(dim=-1)


class Encoder(nn.Module):
    """Convolutional encoder for image-based observations."""
    def __init__(self, obs_shape, feature_dim):
        super().__init__()

        assert len(obs_shape) == 3
        self.num_layers = 4
        self.num_filters = 32
        self.output_dim = 35
        self.output_logits = True
        self.feature_dim = feature_dim

        self.convs = nn.ModuleList([
            nn.Conv2d(obs_shape[0], self.num_filters, 3, stride=2),
            nn.Dropout(DROPOUT),
            nn.Conv2d(self.num_filters, self.num_filters, 3, stride=1),
            nn.Dropout(DROPOUT),
            nn.Conv2d(self.num_filters, self.num_filters, 3, stride=1),
            nn.Dropout(DROPOUT),
            nn.Conv2d(self.num_filters, self.num_filters, 3, stride=1),
            nn.Dropout(DROPOUT)
        ])

        self.head = nn.Sequential(
            nn.Linear(self.num_filters * 35 * 35, self.feature_dim),
            nn.LayerNorm(self.feature_dim))

        self.outputs = dict()

    def forward_conv(self, obs):
        conv = obs / 255.

        for layer in self.convs:
            if 'stride' not in layer.__constants__:
                conv = layer(conv)
            else:
                conv = torch.relu(layer(conv))

        h = conv.view(conv.size(0), -1)
        return h

    def forward(self, obs, detach=False):
        h = self.forward_conv(obs)

        if detach:
            h = h.detach()

        out = self.head(h)

        if not self.output_logits:
            out = torch.tanh(out)

        self.outputs['out'] = out

        return out

    def copy_conv_weights_from(self, source):
        """Tie convolutional layers"""
        for i in range(len(self.convs)):
            if 'stride' not in self.convs[i].__constants__:
                pass
            else:
                utils.tie_weights(src=source.convs[i], trg=self.convs[i])

    def log(self, logger, step):
        for k, v in self.outputs.items():
            logger.log_histogram(f'train_encoder/{k}_hist', v, step)
            if len(v.shape) > 2:
                logger.log_image(f'train_encoder/{k}_img', v[0], step)

        for i in range(self.num_layers):
            logger.log_param(f'train_encoder/conv{i + 1}', self.convs[i], step)


class Actor(nn.Module):
    """torch.distributions implementation of an diagonal Gaussian policy."""
    def __init__(self, encoder_cfg, action_shape, hidden_dim, hidden_depth,
                 log_std_bounds):
        super().__init__()

        self.encoder = hydra.utils.instantiate(encoder_cfg)

        self.log_std_bounds = log_std_bounds
        self.trunk = utils.mlp(self.encoder.feature_dim, hidden_dim,
                               2 * action_shape[0], hidden_depth)

        self.outputs = dict()

    def forward(self, obs, detach_encoder=False):
        obs = self.encoder(obs, detach=detach_encoder)

        mu, log_std = self.trunk(obs).chunk(2, dim=-1)

        # constrain log_std inside [log_std_min, log_std_max]
        log_std = torch.tanh(log_std)
        log_std_min, log_std_max = self.log_std_bounds
        log_std = log_std_min + 0.5 * (log_std_max - log_std_min) * (log_std +
                                                                     1)
        std = log_std.exp()

        self.outputs['mu'] = mu
        self.outputs['std'] = std

        dist = utils.SquashedNormal(mu, std)
        return dist

    def noise(self, obs, n, detach_encoder=False):
        obs = self.encoder(obs, detach=detach_encoder)

        obs[0][np.random.choice(range(len(obs[0])), n, replace=False)] = 0

        mu, log_std = self.trunk(obs).chunk(2, dim=-1)

        # constrain log_std inside [log_std_min, log_std_max]
        log_std = torch.tanh(log_std)
        log_std_min, log_std_max = self.log_std_bounds
        log_std = log_std_min + 0.5 * (log_std_max - log_std_min) * (log_std + 1)
        std = log_std.exp()

        self.outputs['mu'] = mu
        self.outputs['std'] = std

        dist = utils.SquashedNormal(mu, std)
        return dist

    def log(self, logger, step):
        for k, v in self.outputs.items():
            logger.log_histogram(f'train_actor/{k}_hist', v, step)

        for i, m in enumerate(self.trunk):
            if type(m) == nn.Linear:
                logger.log_param(f'train_actor/fc{i}', m, step)


class Critic(nn.Module):
    """Critic network, employes double Q-learning."""
    def __init__(self, encoder_cfg, action_shape, hidden_dim, hidden_depth):
        super().__init__()

        self.encoder = hydra.utils.instantiate(encoder_cfg)

        self.Q1 = utils.mlp(self.encoder.feature_dim + action_shape[0],
                            hidden_dim, 1, hidden_depth)
        self.Q2 = utils.mlp(self.encoder.feature_dim + action_shape[0],
                            hidden_dim, 1, hidden_depth)

        self.outputs = dict()

    def forward(self, obs, action, detach_encoder=False):
        assert obs.size(0) == action.size(0)
        obs = self.encoder(obs, detach=detach_encoder)

        obs_action = torch.cat([obs, action], dim=-1)
        q1 = self.Q1(obs_action)
        q2 = self.Q2(obs_action)

        self.outputs['q1'] = q1
        self.outputs['q2'] = q2

        return q1, q2

    def log(self, logger, step):
        self.encoder.log(logger, step)

        for k, v in self.outputs.items():
            logger.log_histogram(f'train_critic/{k}_hist', v, step)

        assert len(self.Q1) == len(self.Q2)
        for i, (m1, m2) in enumerate(zip(self.Q1, self.Q2)):
            assert type(m1) == type(m2)
            if type(m1) is nn.Linear:
                logger.log_param(f'train_critic/q1_fc{i}', m1, step)
                logger.log_param(f'train_critic/q2_fc{i}', m2, step)


class DenseTrans(nn.Module):
    def __init__(self, critic):
        super().__init__()

        # Uncomment for knowledge-sharing ablation
        # Change [0:1] to [0:2} for l=2
        # self.q1 = critic.Q1[0:1]
        # self.q2 = critic.Q2[0:1]

        self.q1 = nn.Linear(critic.Q1[0].in_features, critic.Q1[0].out_features)
        self.q2 = nn.Linear(critic.Q2[0].in_features, critic.Q2[0].out_features)

        self.dense_head = nn.ModuleList([
            nn.Linear(1024, 512),
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Dropout(DROPOUT_FC),
            nn.Linear(512, 50)
        ])

    def forward(self, h, a):
        h = torch.cat([h, a], dim=-1)

        h1 = self.q1(h)
        h2 = self.q2(h)

        h = (h1 + h2) / 2

        for layer in self.dense_head:
            h = layer(h)

        return h


class ProjectionHead(nn.Module):
    def __init__(self):
        super().__init__()

        self.proj_head = nn.ModuleList([
            nn.Linear(50, 512),
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Dropout(DROPOUT_FC),
            nn.Linear(512, 50),
        ])

    def forward(self, x):
        for layer in self.proj_head:
            x = layer(x)
        return x


class KSL(nn.Module):
    def __init__(self, critic_online, critic_momentum, action_shape):
        super().__init__()
        self.encoder_online = critic_online.encoder
        self.encoder_momentum = critic_momentum.encoder
        self.transition_model = DenseTrans(critic_online)
        self.proj_online = ProjectionHead()
        self.proj_momentum = ProjectionHead()
        self.proj_momentum.load_state_dict(self.proj_online.state_dict())
        self.Wz = nn.Linear(50, 50, bias=False)
        self.Wsingle = nn.Linear(50 + action_shape, 50)

    def encode(self, s, s_):
        h = self.encoder_online(s)
        h_ = self.encoder_momentum(s_).detach()

        return h, h_

    def transition(self, h, a):
        h = self.transition_model(h, a)
        return h

    def projection(self, h, h_):
        projection = self.proj_online(h)
        projection_ = self.proj_momentum(h_).detach()

        return projection, projection_

    def predict(self, projection):
        z_hat = self.Wz(projection)

        return z_hat


class KSLAgent:
    """k-Step Latent Agent"""
    def __init__(self, obs_shape, action_shape, action_range, device,
                 encoder_cfg, critic_cfg, actor_cfg, discount,
                 init_temperature, lr, actor_update_frequency, critic_tau,
                 critic_target_update_frequency, batch_size, ksl_update_frequency, k):
        self.action_range = action_range
        self.device = device
        self.discount = discount
        self.critic_tau = critic_tau
        self.actor_update_frequency = actor_update_frequency
        self.critic_target_update_frequency = critic_target_update_frequency
        self.batch_size = batch_size
        self.ksl_update_frequency = ksl_update_frequency
        self.k = k

        self.actor = hydra.utils.instantiate(actor_cfg).to(self.device)

        self.critic = hydra.utils.instantiate(critic_cfg).to(self.device)
        self.critic_target = hydra.utils.instantiate(critic_cfg).to(
            self.device)
        self.critic_target.load_state_dict(self.critic.state_dict())

        # tie conv layers between actor and critic
        self.actor.encoder.copy_conv_weights_from(self.critic.encoder)

        self.log_alpha = torch.tensor(np.log(init_temperature)).to(device)
        self.log_alpha.requires_grad = True

        # set target entropy to -|A|
        self.target_entropy = -action_shape[0]

        self.ksl = KSL(self.critic, self.critic_target, action_shape[0]).to(self.device)

        # optimizers
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=lr)

        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=lr)

        self.log_alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=lr)

        self.ksl_optimizer = torch.optim.Adam(self.ksl.parameters(), lr=lr)

        self.encoder_optimizer = torch.optim.Adam(
            self.critic.encoder.parameters(), lr=lr
        )

        self.train()

        self.critic_target.train()

        self.loss_fn = loss_fn

        self.ksl_loss_hist = []

    def train(self, training=True):
        self.training = training
        self.actor.train(training)
        self.critic.train(training)
        self.critic_target.train(training)
        self.ksl.train(training)

    @property
    def alpha(self):
        return self.log_alpha.exp()

    def act(self, obs, sample=False):
        obs = torch.FloatTensor(obs).to(self.device)
        obs = obs.unsqueeze(0)
        dist = self.actor(obs)
        action = dist.sample() if sample else dist.mean
        action = action.clamp(*self.action_range)
        assert action.ndim == 2 and action.shape[0] == 1
        return utils.to_np(action[0])

    def act_noise(self, obs, n, sample=False):
        obs = torch.FloatTensor(obs).to(self.device)
        obs = obs.unsqueeze(0)
        dist = self.actor.noise(obs, n=n)
        action = dist.sample() if sample else dist.mean
        action = action.clamp(*self.action_range)
        assert action.ndim == 2 and action.shape[0] == 1
        return utils.to_np(action[0])

    def update_critic(self, obs, obs_aug, action, reward, next_obs,
                      next_obs_aug, not_done, logger, step):
        with torch.no_grad():
            dist = self.actor(next_obs)
            next_action = dist.rsample()
            log_prob = dist.log_prob(next_action).sum(-1, keepdim=True)
            target_Q1, target_Q2 = self.critic_target(next_obs, next_action)
            target_V = torch.min(target_Q1,
                                 target_Q2) - self.alpha.detach() * log_prob
            target_Q = reward + (not_done * self.discount * target_V)

            dist_aug = self.actor(next_obs_aug)
            next_action_aug = dist_aug.rsample()
            log_prob_aug = dist_aug.log_prob(next_action_aug).sum(-1,
                                                                  keepdim=True)
            target_Q1, target_Q2 = self.critic_target(next_obs_aug,
                                                      next_action_aug)

            target_V = torch.min(target_Q1, target_Q2) - self.alpha.detach() * log_prob_aug

            target_Q_aug = reward + (not_done * self.discount * target_V)

            target_Q = (target_Q + target_Q_aug) / 2

        current_Q1, current_Q2 = self.critic(obs, action)
        critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(
            current_Q2, target_Q)

        Q1_aug, Q2_aug = self.critic(obs_aug, action)

        critic_loss += F.mse_loss(Q1_aug, target_Q) + F.mse_loss(
            Q2_aug, target_Q)

        logger.log('train_critic/loss', critic_loss, step)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        self.critic.log(logger, step)

    def update_actor_and_alpha(self, obs, logger, step):
        # detach conv filters, so we don't update them with the actor loss
        dist = self.actor(obs, detach_encoder=True)
        action = dist.rsample()
        log_prob = dist.log_prob(action).sum(-1, keepdim=True)
        # detach conv filters, so we don't update them with the actor loss
        actor_Q1, actor_Q2 = self.critic(obs, action, detach_encoder=True)

        actor_Q = torch.min(actor_Q1, actor_Q2)

        actor_loss = (self.alpha.detach() * log_prob - actor_Q).mean()

        logger.log('train_actor/loss', actor_loss, step)
        logger.log('train_actor/target_entropy', self.target_entropy, step)
        logger.log('train_actor/entropy', -log_prob.mean(), step)

        # optimize the actor
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        self.actor.log(logger, step)

        self.log_alpha_optimizer.zero_grad()
        alpha_loss = (self.alpha *
                      (-log_prob - self.target_entropy).detach()).mean()
        logger.log('train_alpha/loss', alpha_loss, step)
        logger.log('train_alpha/value', self.alpha, step)
        alpha_loss.backward()
        self.log_alpha_optimizer.step()

    def update_ksl_traj(self, replay_buffer):
        self.ksl.train(True)

        obses, actions, obses_next, rewards = replay_buffer.sample_traj(self.batch_size, self.k)

        loss = 0

        z_o = self.ksl.encoder_online(obses[:, 0, :, :, :])

        for i in range(self.k):
            z_m = self.ksl.encoder_momentum(obses_next[:, i, :, :, :]).detach()

            z_o = self.ksl.transition(z_o, actions[:, i])

            z_bar_o = self.ksl.proj_online(z_o)
            z_bar_m = self.ksl.proj_momentum(z_m).detach()

            z_hat_o = self.ksl.predict(z_bar_o)

            loss += self.loss_fn(z_hat_o, z_bar_m).mean()

        self.ksl_loss_hist.append(loss.item())

        self.ksl_optimizer.zero_grad()
        self.encoder_optimizer.zero_grad()

        loss.backward()

        self.encoder_optimizer.step()
        self.ksl_optimizer.step()

    def update(self, replay_buffer, logger, step):
        obs, action, reward, next_obs, not_done, obs_aug, next_obs_aug = replay_buffer.sample(
            self.batch_size)

        logger.log('train/batch_reward', reward.mean(), step)

        if step % self.ksl_update_frequency == 0:
            self.update_ksl_traj(replay_buffer)
            
        self.update_critic(obs, obs_aug, action, reward, next_obs,
                           next_obs_aug, not_done, logger, step)

        if step % self.actor_update_frequency == 0:
            self.update_actor_and_alpha(obs, logger, step)

        if step % self.critic_target_update_frequency == 0:
            utils.soft_update_params(self.critic.Q1, self.critic_target.Q1,
                                     0.01)
            utils.soft_update_params(self.critic.Q2, self.critic_target.Q2,
                                     0.01)
            utils.soft_update_params(self.ksl.proj_online, self.ksl.proj_momentum,
                                     0.05)
            utils.soft_update_params(self.ksl.encoder_online, self.ksl.encoder_momentum,
                                     0.05)

    def save(self, dir, extras):
        torch.save(
            self.actor.state_dict(), dir + extras + '_actor.pt'
        )

        torch.save(
            self.critic.state_dict(), dir + extras + '_critic.pt'
        )

        torch.save(
            self.ksl.state_dict(), dir + extras + '_ksl.pt'
        )

    def load(self, dir, extras):
        self.actor.load_state_dict(
            torch.load(dir + extras + '_actor.pt')
        )

        self.critic.load_state_dict(
            torch.load(dir + extras + '_critic.pt')
        )

        self.ksl.load_state_dict(
            torch.load(dir + extras + '_ksl.pt')
        )

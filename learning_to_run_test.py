import os
import sys

import click
import numpy as np
import json

from mpi4py import MPI

import tensorflow as tf

import baselines.her.experiment.config as config
from baselines import logger
from baselines.common import set_global_seeds
from baselines.her.rollout import RolloutWorker
from baselines.her.util import mpi_fork
from baselines.her.experiment.train import mpi_average, load_stats

from subprocess import CalledProcessError
from importlib import import_module


goal_step_size = 1  # use this as a global counter to be able to update the value in _sample_her_transitions


def get_random_goals(goals, n_goals):
    res = np.random.permutation(goals)
    if len(res) > n_goals:
        return res[:n_goals]
    return (res * n_goals)[:n_goals]


def log(logger, epoch, evaluator, policy, rollout_worker, rank):
    logger.record_tabular('epoch', epoch)
    for key, val in evaluator.logs('test'):
        logger.record_tabular(key, mpi_average(val))
    for key, val in rollout_worker.logs('train'):
        logger.record_tabular(key, mpi_average(val))
    for key, val in policy.logs():
        logger.record_tabular(key, mpi_average(val))
    if rank == 0:
        logger.dump_tabular()


def save_policy(logger, saver, epoch, best_success_rate, success_rate, policy, evaluator, policy_path,
                model_name, epoch_log_path, rank):
    if rank == 0 and success_rate >= best_success_rate:
        best_success_rate = success_rate
        logger.info(
            'New best success rate: {}. Saving policy to {} ...'.format(best_success_rate, policy_path + model_name))
        pth = saver.save(policy.sess, policy_path + model_name)
        evaluator.save_policy(policy_path + 'policy.pkl')
        try:
            with open(epoch_log_path, 'a') as file:
                file.write(str(epoch) + ' ' + str(best_success_rate) + '\n')
        except Exception as e:
            logger.info(e)
        print("saved policy to {}".format(pth))


def make_running_env(visualize=False):
    _osim = import_module('osim.env')
    # envname needs to be set outside
    env = getattr(_osim, make_running_env.envname)
    env = env(visualize=visualize)
    # needs attr _max_episode_steps for configuration (of what? T never used in rolloutworker/ ddpg -> for HER?
    env.time_limit = 20  # default is 1000
    env._max_episode_steps = env.time_limit
    return env


def make_her_function(replay_strategy, replay_k, reward_fun):
    '''
    copied from baselines.her.her
    but we need extra control over what goal is chosen
    '''
    future_p = 1-(1. / (1+replay_k))
    def _sample_her_transitions(episode_batch, batch_size_in_transitions):
        """episode_batch is {key: array(buffer_size x T x dim_key)}
        """
        global goal_step_size
        T = episode_batch['u'].shape[1]
        rollout_batch_size = episode_batch['u'].shape[0]
        batch_size = batch_size_in_transitions

        # Select which episodes and time steps to use.
        episode_idxs = np.random.randint(0, rollout_batch_size, batch_size)  # select episode
        t_samples = np.random.randint(T-goal_step_size, size=batch_size)  # select time step
        transitions = {key: episode_batch[key][episode_idxs, t_samples].copy()
                       for key in episode_batch.keys()}

        # Select future time indexes proportional with probability future_p. These
        # will be used for HER replay by substituting in future goals.a
        her_indexes = np.where(np.random.uniform(size=batch_size) < future_p)  # select what indices to give her trtmnt
        future_offset = np.array([goal_step_size]*batch_size, dtype=int)  # np.random.uniform(size=batch_size) * (T - t_samples)
        # future_offset = future_offset.astype(int)
        future_t = (t_samples + 1 + future_offset)[her_indexes]

        # Replace goal with achieved goal but only for the previously-selected
        # HER transitions (as defined by her_indexes). For the other transitions,
        # keep the original goal.
        future_ag = episode_batch['ag'][episode_idxs[her_indexes], future_t]
        goal_length = episode_batch['ag'][0][0].shape[0]
        transitions['g'][her_indexes] = future_ag
        # TODO until here this is the vanilla her function from baselines, just using goal_step_size instead of random
        # TODO how are the transitions actually structured ?
        # TODO move the goals into the observations ?
        # TODO her paper: new transition (s_t||g', a_t, r'_t, s_t+1||g')
        # transitions['o'][:, goal_length:] = transitions['g']  # put
        # transitions['o_2'][:, goal_length:] = transitions['g']

        # Reconstruct info dictionary for reward  computation.
        info = {}
        for key, value in transitions.items():
            if key.startswith('info_'):
                info[key.replace('info_', '')] = value

        # Re-compute reward since we may have substituted the goal.
        reward_params = {k: transitions[k] for k in ['ag_2', 'g']}
        reward_params['info'] = info
        transitions['r'] = reward_fun(**reward_params)

        transitions = {k: transitions[k].reshape(batch_size, *transitions[k].shape[1:])
                       for k in transitions.keys()}

        assert (transitions['u'].shape[0] == batch_size_in_transitions)

        return transitions
    return _sample_her_transitions


def train(env, policy, rollout_worker,
          n_epochs, n_test_rollouts, n_cycles, n_batches, policy_save_interval, evaluator, policy_path,
          save_policies=True, model_name='model.ckpt', **kwargs):

    global goal_step_size
    saver = tf.train.Saver()
    if os.path.isfile(policy_path+model_name):
        saver.restore(policy.sess, policy_path+model_name)
        logger.info("Successfully restored policy from {}".format(policy_path))
    else:
        logger.info("no policy {} found in {}".format(model_name, policy_path))
        if not os.path.exists(policy_path):
            logger.info('creating directory {}'.format(policy_path))
            os.mkdir(policy_path)

    epoch_log_path = policy_path+'epoch.txt'

    print("generating first goals")
    episode = rollout_worker.generate_rollouts()  # sample some transitions with dummy goal [0]*n
    goals = [ag[goal_step_size] for ag in episode['ag']]  # just goal_step_size'th entry in episode as first "goal"
    new_rollouts_per_epoch = 1
    trained_epochs, best_success_rate = load_stats(epoch_log_path)  # use load_stats function for continuing training

    rank = MPI.COMM_WORLD.Get_rank()
    for epoch in range(trained_epochs+1, n_epochs):
        logger.info('starting epoch: {}'.format(epoch))
        for g in np.random.permutation(goals)[:new_rollouts_per_epoch]: # not for each g in goals, that would grow quite fast
            for e in rollout_worker.envs:
                e.goal = g
            episode = rollout_worker.generate_rollouts()
            # and store the achieved goal in step t = goal_step size in goals
            for ag in episode['ag']:
                goals.append(ag[goal_step_size])
            policy.store_episode(episode,  # get trajectories with real goals [g] in observation
                                 update_stats=True)  # TODO set to default update_stats=True again

        for _ in range(n_cycles):
            policy.train()
        policy.update_target_net()

        # evaluate
        evaluator.clear_history()
        goals_evaluation = get_random_goals(goals, n_test_rollouts)
        for g in goals_evaluation:
            for e in evaluator.envs:
                e.goal = g
            evaluator.generate_rollouts()

        # save the policy if it's better than the previous ones
        success_rate = mpi_average(evaluator.current_success_rate())

        # record logs
        log(logger, epoch, evaluator, policy, rollout_worker, rank)
        if success_rate > 0.9:
            goal_step_size += 1
            logger.info('Increased goal_step_size to {}'.format(goal_step_size))

        if save_policies:
            save_policy(logger, saver, epoch, best_success_rate, success_rate, policy, evaluator, policy_path,
                model_name, epoch_log_path, rank)  # not beatiful but declutters this part


        # make sure that different threads have different seeds
        local_uniform = np.random.uniform(size=(1,))
        root_uniform = local_uniform.copy()
        MPI.COMM_WORLD.Bcast(root_uniform, root=0)
        if rank != 0:
            assert local_uniform[0] != root_uniform[0]

    return kwargs


def launch(
        env, logdir, n_epochs, num_cpu, seed, replay_strategy, policy_save_interval, clip_return, policy_path,
        with_forces, plot_forces,
        scope=None, override_params={}, save_policies=True, one_hot_encoding=None
):

    # Fork for multi-CPU MPI implementation.
    if num_cpu > 1:
        try:
            whoami = mpi_fork(num_cpu, ['--bind-to', 'core'])
        except CalledProcessError:
            # fancy version of mpi call failed, try simple version
            whoami = mpi_fork(num_cpu)

        if whoami == 'parent':
            sys.exit(0)
        import baselines.common.tf_util as U
        U.single_threaded_session().__enter__()
    rank = MPI.COMM_WORLD.Get_rank()

    # Configure logging
    if logdir is None:
        logdir = './logs/'
    if rank == 0:
        if logdir or logger.get_dir() is None:
            logger.configure(dir=logdir)
    else:
        logger.configure()
    logdir = logger.get_dir()
    assert logdir is not None
    os.makedirs(logdir, exist_ok=True)

    # Seed everything.
    rank_seed = seed + 1000000 * rank
    set_global_seeds(rank_seed)

    # Prepare params.
    params = config.DEFAULT_PARAMS
    params['with_forces'] = with_forces
    params['plot_forces'] = plot_forces
    params['env_name'] = env
    params['replay_strategy'] = replay_strategy
    if env in config.DEFAULT_ENV_PARAMS:
        params.update(config.DEFAULT_ENV_PARAMS[env])  # merge env-specific parameters in
    params.update(**override_params)  # makes it possible to override any parameter
    with open(os.path.join(logger.get_dir(), 'params.json'), 'w') as f:
        json.dump(params, f)
    make_running_env.envname = env
    params['make_env'] = make_running_env
    params = config.prepare_params(params)
    config.log_params(params, logger=logger)

    dims = config.configure_dims(params)

    policy = config.configure_ddpg(dims=dims, params=params,
                                   clip_return=clip_return, scope=scope,
                                   make_her_function=make_her_function
                                   )

    rollout_params = {
        'exploit': False,
        'use_target_net': False,
        'use_demo_states': True,
        'compute_Q': False,
        'with_forces': with_forces,
        'plot_forces': plot_forces,
        'T': params['T'],
    }

    eval_params = {
        'exploit': True,
        'use_target_net': params['test_with_polyak'],  # ? is set to False in standard config anyways
        'use_demo_states': False,
        'compute_Q': True,
        'with_forces': with_forces,
        'plot_forces': plot_forces,
        'T': params['T'],
    }

    for name in ['T', 'rollout_batch_size', 'gamma', 'noise_eps', 'random_eps']:
        rollout_params[name] = params[name]
        eval_params[name] = params[name]

    rollout_worker = RolloutWorker(params['make_env'], policy, dims, logger, **rollout_params)
    rollout_worker.seed(rank_seed)
    evaluator = RolloutWorker(params['make_env'], policy, dims, logger, **eval_params)
    evaluator.seed(rank_seed)

    train(
        env=env, logdir=logdir, policy=policy, rollout_worker=rollout_worker,
        n_epochs=n_epochs, n_test_rollouts=params['n_test_rollouts'], evaluator=evaluator,
        n_cycles=params['n_cycles'], n_batches=params['n_batches'],
        policy_save_interval=policy_save_interval, save_policies=save_policies,
        policy_path=policy_path)



@click.command()
@click.option('--env', type=str, default='L2RunEnvHER', help='the name of the OpenAI Gym environment that you want to train on')
@click.option('--logdir', type=str, default=None, help='the path to where logs and policy pickles should go. If not specified, creates a folder in /tmp/')
@click.option('--n_epochs', type=int, default=50, help='the number of training epochs to run')
@click.option('--num_cpu', type=int, default=1, help='the number of CPU cores to use (using MPI)')
@click.option('--seed', type=int, default=0, help='the random seed used to seed both the environment and the training code')
@click.option('--policy_save_interval', type=int, default=5, help='the interval with which policy pickles are saved. If set to 0, only the best and latest policy will be pickled.')
@click.option('--replay_strategy', type=click.Choice(['future', 'none']), default='future', help='the HER replay strategy to be used. "future" uses HER, "none" disables HER.')
@click.option('--clip_return', type=int, default=1, help='whether or not returns should be clipped')
@click.option('--with_forces', type=bool, default=False)
@click.option('--plot_forces', type=bool, default=False)
@click.option('--policy_path', type=str, default='./checkpoint/', help='path to policy to be loaded and trained')
@click.option('--scope', type=str, default=None, help='name of scope for tf')
def main(**kwargs):
    launch(**kwargs)


if __name__ == '__main__':
    main()

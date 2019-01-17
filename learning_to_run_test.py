import os
import sys

import click
import numpy as np
import json
import pickle

from mpi4py import MPI

import tensorflow as tf

import baselines.her.experiment.config as config
from baselines import logger
from baselines.common import set_global_seeds
from baselines.her.rollout import RolloutWorker
from baselines.her.util import mpi_fork
from baselines.her.experiment.train import mpi_average, load_stats

from baselines.her.her import make_sample_her_transitions

from subprocess import CalledProcessError
from importlib import import_module

from environments import RunEnv2HER, RunEnv2

goal_step_size = 1  # use this as a global counter to be able to update the value in _sample_her_transitions


def sample_goal(path, env=None, num_samples=1, offset=12):# strategy="fourth", goal_type="goal_mass"):
    assert(env is not None), "env was None"
    assert(os.path.isfile(path)), "path didn't point to a file"

    with open(path, 'rb') as file:
        res = []
        iteration_number = 0
        unpickled = pickle.Unpickler(file)
        data = []
        #
        # while True:
        #     if iteration_number == num_samples:
        #         break
        #     iteration_number += 1
        #     try:
        #         for i in range(offset-1):
        #             _ = unpickled.load()
        #         data = unpickled.load()
        #         goal = env.get_achieved_goal(data)
        #         res.append(goal)
        #     except EOFError:
        #         break
        for _ in range(10):
            try:
                data = unpickled.load()
                goal = env.get_achieved_goal(data)
                res.append(goal)
            except EOFError:
                break

    print('successfully loaded {} goals'.format(len(res)))
    return res


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

def get_args():
    # taken from L2R.run_experiment (0123Andrew/L2R.git), to parametrize RunEnv2HER
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--modeldim', dest='modeldim', action='store', default='3D', choices=('3D', '2D'), type=str)
    parser.add_argument('--prosthetic', dest='prosthetic', action='store', default=1, type=int)
    parser.add_argument('--proj', type=str, default='3DPro37', help='dict to projection (version): 2D35, 3DPro35, 3DPro37, 3D39, 3DPro37_2, 3D39_2')
    parser.add_argument('--accuracy', dest='accuracy', action='store', default=5e-5, type=float)
    parser.add_argument('--difficulty', dest='difficulty', action='store', default=0, type=int)
    parser.add_argument('--episodes', type=int, default=37, help="Number of test episodes.")
    parser.add_argument('--critic_layers', nargs='+', type=int, default=[128, 128], help="critic hidden layer sizes as tuple")  # 512
    parser.add_argument('--actor_layers', nargs='+', type=int, default=[128, 128], help="actor hidden layer sizes as tuple")  # 512
    parser.add_argument('--layer_norm', action='store_true', help="Use layer normalization.")
    parser.add_argument('--skip_frame', type=int, default=3, help='skip_frame')
    parser.add_argument('--weights', type=str, default=None, help='weights to load')
    parser.add_argument('--plot', type=int, default=0, help='online plot of an observation')
    parser.add_argument('--saveplots', type=int, default=0, help='png of an episode')
    parser.add_argument('--reward_func', type=str, default='3D_pen01_VelMass', help='3D_pen01_VelMass  or  3D_penAdd_VelMass  or  3D_pen01_VelMassFoot or 3D_penAdd_VelMassFoot2')
    parser.add_argument('--logdir', type=str, default=None, help='the path to where logs and policy pickles should go. If not specified, creates a folder in /tmp/')
    parser.add_argument('--n_epochs', type=int, default=50, help='the number of training epochs to run')
    parser.add_argument('--num_cpu', type=int, default=1, help='the number of CPU cores to use (using MPI)')
    parser.add_argument('--seed', type=int, default=0, help='the random seed used to seed both the environment and the training code')
    parser.add_argument('--policy_save_interval', type=int, default=5, help='the interval with which policy pickles are saved. If set to 0, only the best and latest policy will be pickled.')
    parser.add_argument('--clip_return', type=int, default=1, help='whether or not returns should be clipped')
    parser.add_argument('--policy_path', type=str, default='./checkpoint/', help='path to policy to be loaded and trained')
    parser.add_argument('--scope', type=str, default=None, help='name of scope for tf')
    parser.add_argument('--goaltype', type=str, default='pos_mass')
    args = parser.parse_args()
    args.modeldim = args.modeldim.upper()

    args.weights = 'weights/weights_2D_X_53.pkl'
    args.critic_layers = [64, 64]
    args.actor_layers = [64, 64]
    args.saveplots = 0
    args.modeldim = '2D'
    args.prosthetic = 0
    args.episodes=20
    args.saveobs = 1
    return args


def make_running_env(visualize=False):
    assert(hasattr(make_running_env, 'envname'))
    if make_running_env.envname == 'RunEnv2HER':
        assert(hasattr(make_running_env, 'goaltype'))
        env = RunEnv2HER
        args = get_args()
        args.proj= '2Dpos'  # '3DPro37'
        env = env(goaltype=make_running_env.goaltype, visualize=True, model=args.modeldim, prosthetic=args.prosthetic, difficulty=args.difficulty, skip_frame=args.skip_frame, args=args)#goaltype=make_running_env.goaltype)
    elif make_running_env.envname == 'L2RunEnvHER':
        _osim = import_module('osim.env')
        # envname needs to be set outside
        env = getattr(_osim, make_running_env.envname)
        env = env(visualize=True)
    else:
        raise ValueError('env not recognized, '
                       'must be one of[RunEnv2HER, L2RunEnvHER], but was {}'.format(make_running_env.envname))
    # needs attr _max_episode_steps for configuration (of what? T never used in rolloutworker/ ddpg -> for HER?
    env.time_limit   = 100  # default is 1000
    env._max_episode_steps = env.time_limit
    return env



def train(env, policy, rollout_worker,
          n_epochs, n_test_rollouts, n_cycles, n_batches, policy_save_interval, evaluator, policy_path,
          save_policies=True, model_name='model.ckpt', **kwargs):

    testing = True
    if testing:
        n_test_rollouts = 1

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
    # load goals from observations in path
    goals = sample_goal(path=os.getcwd()+'/data/observations_2019-01-09 22:13:01.435718.dat',
                        env=rollout_worker.envs[0], offset=12, num_samples=10)
    new_rollouts_per_epoch = 5 if not testing else 1
    trained_epochs, best_success_rate = load_stats(epoch_log_path)  # use load_stats function for continuing training

    rank = MPI.COMM_WORLD.Get_rank()
    for epoch in range(trained_epochs+1, n_epochs):
        logger.info('starting epoch: {}'.format(epoch))
        if rank == 0:
            print("\tgenerate episode")
        for g in np.random.permutation(goals)[:new_rollouts_per_epoch]: # not for each g in goals, that would grow quite fast
            for e in rollout_worker.envs:
                e.goal = g
            episode = rollout_worker.generate_rollouts()
            policy.store_episode(episode) # get trajectories with real goals [g] in observation
            if rank == 0:
                print("\ttrain")
            for _ in range(n_cycles):
                policy.train()
            policy.update_target_net()

        # evaluate
        evaluator.clear_history()
        # goals_evaluation = get_random_goals(goals, n_test_rollouts)
        goals_evaluation = np.random.permutation(goals)[:n_test_rollouts]  # automatically handles out of bounds
        # goals_evaluation = get_random_goals(goals, 2)
        if rank == 0:
            print("\tevaluate")
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
        with_forces, plot_forces, goaltype='',
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
    assert logdir is not None, "logdir was None"
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
    # make_running_env.envname = 'L2RunEnvHER'  # select what env, see make_running_env method! L2RunEnvHER or  RunEnv2HER
    make_running_env.envname = 'RunEnv2HER'  # select what env, see make_running_env method! L2RunEnvHER or  RunEnv2HER
    make_running_env.goaltype = goaltype
    params['make_env'] = make_running_env
    params = config.prepare_params(params)
    config.log_params(params, logger=logger)

    dims = config.configure_dims(params)

    policy = config.configure_ddpg(dims=dims, params=params,
                                   clip_return=clip_return, scope=scope,
                                   make_her_function=make_sample_her_transitions
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
@click.option('--goaltype', type=str, default='pos_mass')
def main(**kwargs):
    launch(**kwargs)


if __name__ == '__main__':
    main()

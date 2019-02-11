import os
import sys
import pickle

import click
import numpy as np
import json
from argparse import Namespace
from baselines import logger
import pandas as pd
from environments import RunEnv2HER

def get_dummy_action(action_size=(22,1), low=-1, high=1):
    assert low <= high
    return (low - high)*np.random.random_sample(action_size) + high

def play(policy, env, goals, logdir):
    if not os.path.exists(logdir):
        os.mkdir(logdir)
    if logdir is not None:
        logger.configure(dir=logdir)

    n_simulations = len(goals)
    max_sim_steps = 100

    for n, g in enumerate(goals):
        logger.info("STARTING SIMULATION RUN {}/{}".format(n+1, n_simulations))
        d = {'Q': [], 'o': [], 'u': [], 'g': [], 'ag': []}
        env.reset()
        env.goals = g

        o = env.get_observation()
        o, g, ag = o['observation'], o['desired_goal'], o['achieved_goal']
        terminal = False
        sim_step = 0
        while not terminal:
            u, Q = policy.get_actions(
                o, ag, g,
                compute_Q=True,
                noise_eps=0, random_eps=0, use_target_net=False
            )

            d['Q'].append(Q)
            d['o'].append(o)
            d['u'].append(u)
            d['g'].append(g)
            d['ag'].append(ag)

            o, r, _, info = env.step(u)
            # logger.info("Q: {} success: {} \n g: {} \n ag: {}".format(Q, info['is_success'], g, ag))
            env.render()
            o, g, ag = o['observation'], o['desired_goal'], o['achieved_goal']
            logger.info("")
            sim_step += 1
            logger.info('SIM STEP {}'.format(sim_step))
            if info['is_success'] or sim_step >= max_sim_steps:
                logger.log('TERMINAL STATE REACHED')
                terminal = True
        pth = logdir+'/sim_log_{}.csv'.format(n)
        with open(pth, 'w') as fp:
            logger.info('save recorded data to {}'.format(pth))
            df = pd.DataFrame(d)
            df.to_csv(fp, index=False)  # formatting looks awful
    pass

def launch(base_path, policy, env_conf, logdir):

    logger.info("Working dir {}".format(os.getcwd()))
    policy_pth = os.path.join(base_path, policy)
    env_conf_pth = os.path.join(base_path, env_conf)

    # init and load stuff
    try:
        env_conf_json = open(env_conf_pth, 'r')
        env_conf = json.load(env_conf_json)
        goals = env_conf['goals'].copy()
        goals = np.array([np.array(g) for g in goals])
        env_name = env_conf['name']
        del env_conf['goals']
        del env_conf['name']
        args = Namespace()
        args.__dict__.update(env_conf)
    except:
        raise FileNotFoundError('Config file could not be loaded from {}'.format(env_conf_pth))

    env = None

    if env_name == 'RunEnv2HER':
        env = RunEnv2HER\
                (goaltype=None, visualize=True, model=args.modeldim,
                  prosthetic=args.prosthetic, difficulty=args.difficulty,
                  skip_frame=args.skip_frame, args=args)
    elif False:  # other env
        pass
    else:
        raise ValueError('environment from config not supported')

    try:
        pickle_file = open(policy_pth, 'rb')
        policy = pickle.Unpickler(pickle_file).load()
    except Exception as e:
        raise FileNotFoundError("Policy could not be loaded from {}".format(policy_pth))


    play(policy=policy, env=env, goals=goals, logdir=logdir)

@click.command()
@click.option('--env_conf', type=str, default='env_conf.json', help='Path to .json that contains a dict with environment configs')
@click.option('--base_path', type=str, default='./checkpoint')
@click.option('--policy', type=str, default='policy.pkl', help='Path to policy.pkl')
@click.option('--logdir', type=str, default='./logs')
# @click.option()
def main(**kwargs):
    launch(**kwargs)

if __name__ == '__main__':
    main()
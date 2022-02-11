import gym
import numpy as np
import torch as T
# each thread gets its own individual actor critic 
from a3c import ActorCritic
from icm import ICM
from memory import Memory
from utils import plot_learning_curve

def worker(name, input_shape, n_actions, global_agent, global_icm,
           optimizer, icm_optimizer, env_id, n_threads, icm=False):
    T_MAX = 20

    local_agent = ActorCritic(input_shape, n_actions)

    if icm:
        local_icm = ICM(input_shape, n_actions)
        # just a string for printing debug information to the terminal && saving our plot 
        algo = 'ICM'
    else:
        intrinsic_reward = T.zeros(1)
        algo = 'A3C'
    # each agent gets each own memory 
    memory = Memory()
    # each own environment
    env = gym.make(env_id)
    # how many time steps we have, the episode , the score, the average score
    t_steps, max_eps, episode, scores, avg_score = 0, 1000, 0, [], 0

    while episode < max_eps:
        obs = env.reset()
        # make your hidden state for the actor critic a3c
        hx = T.zeros(1, 256)
        # we need a score, a terminal flag and the number of steps taken withing the episode
        # every 20 steps in an episode we want to excecute the learning function
        score, done, ep_steps = 0, False, 0
        while not done:
            state = T.tensor([obs], dtype=T.float)
            # feed forward our state and our hidden state to the local agent to get the action we want to take, value for that state, log_prob for that action
            action, value, log_prob, hx = local_agent(state, hx)
            # take your action
            obs_, reward, done, info = env.step(action)
            # increment total steps, episode steps, increase your score
            t_steps += 1
            ep_steps += 1
            score += reward
            reward = 0  # turn off extrinsic rewards
            memory.remember(obs, action, reward, obs_, value, log_prob)
            obs = obs_
            # LEARNING
            # every 20 steps or when the game is done
            if ep_steps % T_MAX == 0 or done:
                states, actions, rewards, new_states, values, log_probs = \
                        memory.sample_memory()
                # If we are doing icm them we want to calculate our loss according to icm
                if icm:
                    intrinsic_reward, L_I, L_F = \
                            local_icm.calc_loss(states, new_states, actions)
                # loss according to our a3c agent
                loss = local_agent.calc_loss(obs, hx, done, rewards, values,
                                             log_probs, intrinsic_reward)

                optimizer.zero_grad()
                # detach hx bc it gives us a reward trying to backpropagate through the graph twice 
                hx = hx.detach_()
                if icm:
                    icm_optimizer.zero_grad()
                   # backpropagate your loss 
                    (L_I + L_F).backward()
                # backpropagate your loss from your a3c
                loss.backward()
                T.nn.utils.clip_grad_norm_(local_agent.parameters(), 40)

                for local_param, global_param in zip(
                                        local_agent.parameters(),
                                        global_agent.parameters()):
                    global_param._grad = local_param.grad
                optimizer.step()
                # take our gradients from our local agent and upload them to the global agent
                local_agent.load_state_dict(global_agent.state_dict())

                if icm:
                    for local_param, global_param in zip(
                                            local_icm.parameters(),
                                            global_icm.parameters()):
                        global_param._grad = local_param.grad
                    icm_optimizer.step()
                    local_icm.load_state_dict(global_icm.state_dict())
                    # clear memory at the end of the episode
                memory.clear_memory()
        # at the end of each episode
        if name == '1':
            # append our score to our scores array
            scores.append(score)
            avg_score = np.mean(scores[-100:])
            print('{} episode {} thread {} of {} steps {:.2f}M score {:.2f} '
                  'intrinsic_reward {:.2f} avg score (100) {:.1f}'.format(
                      algo, episode, name, n_threads,
                      t_steps/1e6, score,
                      T.sum(intrinsic_reward),
                      avg_score))
        episode += 1
    # at the end of all the episodes    
    if name == '1':     
        x = [z for z in range(episode)]
        # for the first agent we want to plot our graph
        fname = algo + '_CartPole_no_rewards.png'
        plot_learning_curve(x, scores, fname)

import torch.multiprocessing as mp
from a3c import ActorCritic
from icm import ICM
from shared_adam import SharedAdam
from worker import worker
# Worker function is what handles all the main functionalities and is called by the multiprocessing function here
# Functionality to parallelise our environments and agents
class ParallelEnv:
    def __init__(self, env_id, input_shape, n_actions, icm, n_threads=8):
        # icm is a boolean : whether or not we want to use the icm algorithm
        # name all of our threads
        names = [str(i) for i in range(1, n_threads+1)]
        # define global a3cs and global icms
        global_actor_critic = ActorCritic(input_shape, n_actions)
        global_actor_critic.share_memory()
        # Share the memory of all of our global agents amongst all the local agents
        ''' This is the basic idea of a3c: You gonna have a whole bunch of independent agents in their own thread interacting with their own environments,
        they are gonna do their own learning and upload their learning to the global actor critic'''
        global_optim = SharedAdam(global_actor_critic.parameters())

        if not icm:
            global_icm = None
            global_icm_optim = None
        else:
            global_icm = ICM(input_shape, n_actions)
            global_icm.share_memory()
            global_icm_optim = SharedAdam(global_icm.parameters())
        # Define our processors, the target is the function we wanna call for each individual thread 
        self.ps = [mp.Process(target=worker,
                              args=(name, input_shape, n_actions, 
                                    global_actor_critic, global_icm,
                                    global_optim, global_icm_optim, env_id,
                                    n_threads, icm))
                   for name in names]

        [p.start() for p in self.ps]
        [p.join() for p in self.ps]

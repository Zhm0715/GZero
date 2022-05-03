import os
import threading
import time
import timeit
import pprint
from collections import deque
import numpy as np

import torch

torch.set_printoptions(profile="full")
from torch import multiprocessing as mp
from torch import nn

from .file_writer import FileWriter
from .models import Model
from .utils import get_batch, log, create_env, create_buffers, create_optimizers, act

mean_episode_return_buf = {p: deque(maxlen=100) for p in ['A', 'B', 'C', 'D']}

def compute_loss(logits, targets):
    loss = ((logits.squeeze(-1) - targets) ** 2).mean()
    return loss

def learn(position,
          actor_models,
          model,
          batch,
          optimizer,
          flags,
          lock,
          show_model=None,
          show_optimizer=None):
    """
    执行学习(优化)步骤.
    Performs a learning (optimization) step.
    """
    # learn ones every 360 game
    if flags.training_device != "cpu":
        device = torch.device('cuda:' + str(flags.training_device))
    else:
        device = torch.device('cpu')
    show_obs_x_no_action = batch['show_obs_x_no_action'].to(device)
    show_obs_action = batch['show_obs_action'].to(device)
    show_obs_x = torch.cat((show_obs_x_no_action, show_obs_action), dim=2).float()
    show_obs_x = torch.flatten(show_obs_x, 0, 1)
    show_obs_z = None
    obs_x_no_action = batch['obs_x_no_action'].to(device)
    obs_action = batch['obs_action'].to(device)
    torch.autograd.set_detect_anomaly(True)
    obs_x = torch.cat((obs_x_no_action, obs_action), dim=2).float()
    obs_x = torch.flatten(obs_x, 0, 1)
    obs_z = torch.flatten(batch['obs_z'].to(device), 0, 1).float()
    target = torch.flatten(batch['target'].to(device), 0, 1)
    # show_target = torch.flatten(batch['show_target'].to(device), 0, 1)
    episode_returns = batch['episode_return'][batch['done']]
    mean_episode_return_buf[position].append(torch.mean(episode_returns).to(device))
    with lock:
        learner_outputs = model(obs_z, obs_x, return_value=True)
        show_learner_output = show_model(show_obs_z, show_obs_x, return_value=True)
        show_loss = compute_loss(show_learner_output['values'], target)
        loss = compute_loss(learner_outputs['values'], target)
        stats = {
            'mean_episode_return_' + position: torch.mean(
                torch.stack([_r for _r in mean_episode_return_buf[position]])).item(),
            'loss_' + position: loss.item(),
            'loss_show' + position: show_loss.item(),
        }
        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), flags.max_grad_norm)
        optimizer.step()
        show_optimizer.zero_grad()
        show_loss.backward()
        nn.utils.clip_grad_norm_(show_model.parameters(), flags.max_grad_norm)
        show_optimizer.step()
        for actor_model in actor_models.values():
            actor_model.get_model(position).load_state_dict(model.state_dict())
        actor_model.get_model("show"+position).load_state_dict(show_model.state_dict())
        return stats


def train(flags):
    """
    This is the main funtion for training. It will first
    initilize everything, such as buffers, optimizers, etc.
    Then it will start subprocesses as actors. Then, it will call
    learning function with  multiple threads.
    """
    if not flags.actor_device_cpu or flags.training_device != 'cpu':
        if not torch.cuda.is_available():
            raise AssertionError(
                "CUDA not available. If you have GPUs, please specify the ID after `--gpu_devices`. Otherwise, please train with CPU with `python3 train.py --actor_device_cpu --training_device cpu`")
    """
    flags.xpid: train id
       
    """
    plogger = FileWriter(
        xpid=flags.xpid,
        xp_args=flags.__dict__,
        rootdir=flags.savedir,
    )
    checkpointpath = os.path.expandvars(
        os.path.expanduser('%s/%s/%s' % (flags.savedir, flags.xpid, 'model.tar')))
    T = flags.unroll_length
    B = flags.batch_size

    if flags.actor_device_cpu:
        device_iterator = ['cpu']
    else:
        device_iterator = range(flags.num_actor_devices)
        assert flags.num_actor_devices <= len(
            flags.gpu_devices.split(',')), 'The number of actor devices can not exceed the number of available devices'

    # Initialize actor models
    models = {}
    for device in device_iterator:
        model = Model(device=device)
        model.share_memory()
        model.eval()
        models[device] = model
    # print("models: " + str(models))     # just one device
    # Initialize buffers 创建buffer
    buffers = create_buffers(flags, device_iterator)
    # Initialize queues
    actor_processes = []
    # 多线程产生器
    ctx = mp.get_context('spawn')
    free_queue = {}
    full_queue = {}

    # 为每一个device创建训练上下文
    for device in device_iterator:
        _free_queue = {'A': ctx.SimpleQueue(), 'B': ctx.SimpleQueue(), 'C': ctx.SimpleQueue(), 'D': ctx.SimpleQueue()}
        _full_queue = {'A': ctx.SimpleQueue(), 'B': ctx.SimpleQueue(), 'C': ctx.SimpleQueue(), 'D': ctx.SimpleQueue()}
        free_queue[device] = _free_queue
        full_queue[device] = _full_queue

    # Learner model for training
    learner_model = Model(device=flags.training_device)

    # Create optimizers
    optimizers = create_optimizers(flags, learner_model)

    # Stat Keys
    stat_keys = [
        'mean_episode_return_A',
        'loss_A',
        'mean_episode_return_B',
        'loss_B',
        'mean_episode_return_C',
        'loss_C',
        'mean_episode_return_D',
        'loss_D',
        'loss_showA',
        'loss_showB',
        'loss_showC',
        'loss_showD',
    ]
    frames, stats = 0, {k: 0 for k in stat_keys}
    position_frames = {'A': 0, 'B': 0, 'C': 0, 'D': 0}

    # Load models if any
    if flags.load_model and os.path.exists(checkpointpath):
        checkpoint_states = torch.load(
            checkpointpath,
            map_location=("cuda:" + str(flags.training_device) if flags.training_device != "cpu" else "cpu")
        )
        for k in ['A', 'B', 'C', 'D']:
            learner_model.get_model(k).load_state_dict(checkpoint_states["model_state_dict"][k])
            optimizers[k].load_state_dict(checkpoint_states["optimizer_state_dict"][k])
            for device in device_iterator:
                models[device].get_model(k).load_state_dict(learner_model.get_model(k).state_dict())
        stats = checkpoint_states["stats"]
        frames = checkpoint_states["frames"]
        position_frames = checkpoint_states["position_frames"]
        log.info(f"Resuming preempted job, current stats:\n{stats}")

    # Starting actor processes
    for device in device_iterator:
        num_actors = flags.num_actors
        for i in range(flags.num_actors):
            actor = ctx.Process(
                target=act,
                args=(i, device, free_queue[device], full_queue[device], models[device], buffers[device], flags))
            actor.start()
            actor_processes.append(actor)

    def batch_and_learn(i, device, position, local_lock, position_lock, lock=threading.Lock()):
        """
        线程目标为学习过程。
        Thread target for the learning process.
        """
        # nonlocal声明的变量不是局部变量,也不是全局变量,而是外部嵌套函数内的变量。
        nonlocal frames, position_frames, stats
        while frames < flags.total_frames:
            batch = get_batch(free_queue[device][position], full_queue[device][position], buffers[device][position],
                              flags, local_lock)
            _stats = learn(position, models, learner_model.get_model(position), batch,
                           optimizers[position], flags, position_lock,
                           show_model=learner_model.get_model("show"+position), show_optimizer=optimizers.get("show"+position))
            with lock:
                for k in _stats:
                    stats[k] = _stats[k]
                to_log = dict(frames=frames)
                to_log.update({k: stats[k] for k in stat_keys})
                plogger.log(to_log)
                frames += T * B
                position_frames[position] += T * B

    for device in device_iterator:
        for m in range(flags.num_buffers):
            free_queue[device]['A'].put(m)
            free_queue[device]['B'].put(m)
            free_queue[device]['C'].put(m)
            free_queue[device]['D'].put(m)

    threads = []
    locks = {}
    for device in device_iterator:
        locks[device] = {'A': threading.Lock(), 'B': threading.Lock(), 'C': threading.Lock(), 'D': threading.Lock()}
    position_locks = {'A': threading.Lock(), 'B': threading.Lock(), 'C': threading.Lock(), 'D': threading.Lock()}

    for device in device_iterator:
        for i in range(flags.num_threads):
            for position in ['A', 'B', 'C', 'D']:
                thread = threading.Thread(
                    target=batch_and_learn, name='batch-and-learn-%d' % i,
                    args=(i, device, position, locks[device][position], position_locks[position]))
                thread.start()
                threads.append(thread)

    def checkpoint(frames):
        if flags.disable_checkpoint:
            return
        log.info('Saving checkpoint to %s', checkpointpath)
        _models = learner_model.get_models()
        torch.save({
            'model_state_dict': {k: _models[k].state_dict() for k in _models},
            'optimizer_state_dict': {k: optimizers[k].state_dict() for k in optimizers},
            "stats": stats,
            'flags': vars(flags),
            'frames': frames,
            'position_frames': position_frames
        }, checkpointpath)

        # Save the weights for evaluation purpose
        for position in ['A', 'B', 'C', 'D', 'showA', 'showB', 'showC', 'showD']:
            model_weights_dir = os.path.expandvars(os.path.expanduser(
                '%s/%s/%s' % (flags.savedir, flags.xpid, position + '_weights_' + str(frames) + '.ckpt')))
            torch.save(learner_model.get_model(position).state_dict(), model_weights_dir)

    fps_log = []
    timer = timeit.default_timer
    try:
        last_checkpoint_time = timer() - flags.save_interval * 60
        while frames < flags.total_frames:
            start_frames = frames
            position_start_frames = {k: position_frames[k] for k in position_frames}
            start_time = timer()
            time.sleep(5)

            if timer() - last_checkpoint_time > flags.save_interval * 60:
                checkpoint(frames)
                last_checkpoint_time = timer()
            end_time = timer()

            fps = (frames - start_frames) / (end_time - start_time)
            fps_log.append(fps)
            if len(fps_log) > 24:
                fps_log = fps_log[1:]
            fps_avg = np.mean(fps_log)

            position_fps = {k: (position_frames[k] - position_start_frames[k]) / (end_time - start_time) for k in
                            position_frames}
            log.info(
                'After %i (A:%i B:%i C:%i, D:%i) frames: @ %.1f fps (avg@ %.1f fps) (A:%.1f B:%.1f C:%.1f, D:%.if) Stats:\n%s',
                frames,
                position_frames['A'],
                position_frames['B'],
                position_frames['C'],
                position_frames['D'],
                fps,
                fps_avg,
                position_fps['A'],
                position_fps['B'],
                position_fps['C'],
                position_fps['D'],
                pprint.pformat(stats))

    except KeyboardInterrupt:
        return
    else:
        for thread in threads:
            thread.join()
        log.info('Learning finished after %d frames.', frames)

    checkpoint(frames)
    plogger.close()

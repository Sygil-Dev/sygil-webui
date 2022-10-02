""" Scheduler to handle moving models on and off the device"""
import logging
import queue
import sys
import threading
import uuid
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Dict, Generator, Optional

import torch
from readerwriterlock import rwlock

from .model import Model

logging.basicConfig(stream=sys.stdout)
logger = logging.getLogger("ModelRepo.Schedulers")


@dataclass
class TaskInfo:
    model: Model
    lock: rwlock.RWLockWrite = field(default_factory=rwlock.RWLockWrite)
    wlock: rwlock.Lockable = None
    waiters_cnt: int = 0

    def __repr__(self):
        return f"TaskInfo: {self.model.name} (On Device: {self.on_device}). Waiters: {self.waiters_cnt}. rlock: {self.lock.v_read_count}  wlock: {self.lock.v_write_count}"

    def __post_init__(self):
        self.wlock = self.lock.gen_wlock()
        self.wlock.acquire()  # Keep held as long as off-device

    @property
    def on_device(self):
        return not self.wlock.locked()


@dataclass
class LoadModelCmd:
    task_info: TaskInfo


class Scheduler:
    def __init__(self):
        self._tasks: Dict[str, TaskInfo] = {}
        self._device_tasks: Dict[str, str] = {}
        self._thread = threading.Thread(name="ModelRepo Model Mover", target=self._model_thread)
        self._device_queue: queue.Queue[LoadModelCmd] = queue.Queue()
        self._tls: threading.local = threading.local()
        self._thread.start()

    def register(self, model: Model) -> str:
        """Registers a model with the scheduler,
        returns a token to identify the model to the scheduler

        Args:
            model (Model): model to be handled by the scheduler

        Returns:
            str: token to identify the model on the scheduler
        """
        key = uuid.uuid4().hex
        new_task = TaskInfo(model=model)
        self._tasks[key] = new_task
        self._device_tasks.setdefault(model.device.type, set()).add(key)
        return key

    def prepare_for_task(self, task: TaskInfo):
        """ Perform any actions necessary to prepare for the task

        Args:
            task_info (TaskInfo): the task to be loaded onto the device
        """

        raise NotImplementedError("Policy must be implemented in a derived class")

    @contextmanager
    def on_device_context(self, token: str) -> Generator[None, None, None]:
        """ Context manager that ensures calls to the task occur on device

        Args:
           token [str]: token returned when registering the task
        """
        prev_frame_info: TaskInfo = getattr(self._tls, "frame_task_info", None)
        prev_frame_lock: rwlock.RWLockable = getattr(self._tls, "frame_lock", None)

        next_frame_info: TaskInfo = self._tasks[token]

        # Release this thread's lock on the current frame
        if prev_frame_info:
            prev_frame_lock.release()

        # Now make sure the next frame's model is on device
        next_frame_lock = self._get_task_device_lock(next_frame_info)

        # Push the next frame
        self._tls.frame_lock = next_frame_lock
        self._tls.frame_task_info = next_frame_info
        yield  # actual function call
        # Restore the previous frame
        self._tls.frame_lock = prev_frame_lock
        self._tls.frame_task_info = prev_frame_info

        # Release the just-exited frame
        next_frame_lock.release()

        if prev_frame_info:
            self._get_task_device_lock(prev_frame_info, rlock=prev_frame_lock)

    def _model_thread(self):
        """ Background thread to move models to and fro """
        logger.info("Model Loader Thread Started")
        while True:
            try:
                try:
                    cmd = self._device_queue.get(timeout=1)
                    task = cmd.task_info
                except queue.Empty:
                    # Periodically check for 'waiters' that got missed
                    waiting_tasks = [task for task in self._tasks.values() if task.waiters_cnt > 0]
                    for task in waiting_tasks:
                        logger.info(f"Re-queuing waiting task {task}")
                        self._device_queue.put(LoadModelCmd(task_info=task))
                    continue

                logger.debug(f"ModelLoader woke up for {cmd}")
                if task.on_device:
                    continue

                # Run scheduling policy
                self.prepare_for_task(task)

                self._move_to_device(task)
            except Exception as e:
                logger.error(e)

    def _print_task_list(self):
        """Prints a list of all active tasks"""
        logger.info("Current models:")
        for key, task in self._tasks.items():
            if task.on_device:
                logger.info(f"{key}: {task}")

    def _move_off_device(self, task: TaskInfo) -> None:
        """ Moves a task off the device, blocking until complete """
        if not task.on_device:
            return
        logger.debug(f"Acquiring wlock to move {task} off device")
        task.wlock.acquire()  # no new readers, wait for active to finish
        task.model.move_model(to_device=False)

    def _move_to_device(self, task: TaskInfo) -> None:
        """ Moves a task on to the device """
        if task.on_device:
            return
        task.model.move_model(to_device=True)
        task.wlock.release()
        logger.debug(f"Released wlock after moving {task} to device")

    def _get_task_device_lock(self, task_info: TaskInfo, rlock: Optional[rwlock.Lockable] = None) -> rwlock.Lockable:
        """Moves the model to the device, if necessary, and returns an already-locked lock which will
        ensure the model is available on the device until the lock is released.

        Args:
            task_info (TaskInfo): task to use on device
            rlock (Optional[rwlock.Lockable]): lock to use instead of generating a new one from the task

        Returns:
            rwlock.Lockable: an already-acquired lock which guards the model from being unloaded
        """
        if rlock is None:
            rlock = task_info.lock.gen_rlock()

        # Immediately acquire the lock if already on device, or else block until loading is done
        if not rlock.acquire(blocking=False):
            task_info.waiters_cnt += 1
            self._device_queue.put(LoadModelCmd(task_info=task_info))
            rlock.acquire(blocking=True)
            task_info.waiters_cnt -= 1
        return rlock


class OneAtATimeScheduler(Scheduler):
    def __init__(self, keep_family: bool = True):
        """Initialize the One-at-a-Time scheduler, which only allows one
        model to be loaded at a time

        Args:
            keep_family (bool, optional): Allow children/parent models to be loaded together. Defaults to True.
        """
        super().__init__()
        self._keep_family = keep_family

    def prepare_for_task(self, task: TaskInfo):
        # Only consider models that share the same device
        device_task_keys = self._device_tasks[task.model.device.type]
        device_tasks = [self._tasks[x] for x in device_task_keys]

        for iter_task in device_tasks:
            if iter_task is not task:
                # Optimization: Don't unload same 'family'
                iter_is_parent = task.model.name.startswith(iter_task.model.name)
                iter_is_child = iter_task.model.name.startswith(task.model.name)
                if not (self._keep_family and (iter_is_parent or iter_is_child)):
                    self._move_off_device(iter_task)
        torch_gc()


def torch_gc():
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()

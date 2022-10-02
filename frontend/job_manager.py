# This file is part of stable-diffusion-webui (https://github.com/sd-webui/stable-diffusion-webui/).

# Copyright 2022 sd-webui team.
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.

# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>. 
''' Provides simple job management for gradio, allowing viewing and stopping in-progress multi-batch generations '''
from __future__ import annotations
import gradio as gr
from gradio.components import Component, Gallery, Slider
from threading import Event, Timer
from typing import Callable, List, Dict, Tuple, Optional, Any
from dataclasses import dataclass, field
from functools import partial
from PIL.Image import Image
import uuid
import traceback
import time


@dataclass(eq=True, frozen=True)
class FuncKey:
    job_id: str
    func: Callable


@dataclass(eq=True, frozen=True)
class JobKey:
    func_key: FuncKey
    session_key: str


@dataclass
class JobInfo:
    inputs: List[Component]
    func: Callable
    session_key: str
    job_token: Optional[int] = None
    images: List[Image] = field(default_factory=list)
    active_image: Image = None
    rec_steps_enabled: bool = False
    rec_steps_imgs: List[Image] = field(default_factory=list)
    rec_steps_intrvl: int = None
    rec_steps_to_gallery: bool = False
    rec_steps_to_file: bool = False
    should_stop: Event = field(default_factory=Event)
    refresh_active_image_requested: Event = field(default_factory=Event)
    refresh_active_image_done: Event = field(default_factory=Event)
    stop_cur_iter: Event = field(default_factory=Event)
    active_iteration_cnt: int = field(default_factory=int)
    job_status: str = field(default_factory=str)
    finished: bool = False
    started: bool = False
    timestamp: float = None
    removed_output_idxs: List[int] = field(default_factory=list)


@dataclass
class SessionInfo:
    jobs: Dict[FuncKey, JobInfo] = field(default_factory=dict)
    finished_jobs: Dict[FuncKey, JobInfo] = field(default_factory=dict)


@dataclass
class QueueItem:
    wait_event: Event


def triggerChangeEvent():
    return uuid.uuid4().hex


@dataclass
class JobManagerUi:
    def wrap_func(
            self,
            func: Callable,
            inputs: List[Component],
            outputs: List[Component]) -> Tuple[Callable, List[Component], List[Component]]:
        ''' Takes a gradio event listener function and its input/outputs and returns wrapped replacements which will
            be managed by JobManager
        Parameters:
        func (Callable) the original event listener to be wrapped.
                        This listener should be modified to take a 'job_info' parameter which, if not None, should can
                        be used by the function to check for stop events and to store intermediate image results
        inputs (List[Component]) the original inputs
        outputs (List[Component]) the original outputs. The first gallery, if any, will be used for refreshing images
        refresh_btn: (gr.Button, optional) a button to use for updating the gallery with intermediate results
        stop_btn: (gr.Button, optional) a button to use for stopping the function
        status_text: (gr.Textbox) a textbox to display job status updates

        Returns:
        Tuple(newFunc (Callable), newInputs (List[Component]), newOutputs (List[Component]), which should be used as
        replacements for the passed in function, inputs and outputs
        '''
        return self._job_manager._wrap_func(
            func=func, inputs=inputs, outputs=outputs,
            job_ui=self
        )

    _refresh_btn: gr.Button
    _stop_btn: gr.Button
    _status_text: gr.Textbox
    _stop_all_session_btn: gr.Button
    _free_done_sessions_btn: gr.Button
    _active_image: gr.Image
    _active_image_stop_btn: gr.Button
    _active_image_refresh_btn: gr.Button
    _rec_steps_intrvl_sldr: gr.Slider
    _rec_steps_checkbox: gr.Checkbox
    _save_rec_steps_to_gallery_chkbx: gr.Checkbox
    _save_rec_steps_to_file_chkbx: gr.Checkbox
    _job_manager: JobManager


class JobManager:
    JOB_MAX_START_TIME = 5.0  # How long can a job be stuck 'starting' before assuming it isn't running

    def __init__(self, max_jobs: int):
        self._max_jobs: int = max_jobs
        self._avail_job_tokens: List[Any] = list(range(max_jobs))
        self._job_queue: List[QueueItem] = []
        self._sessions: Dict[str, SessionInfo] = {}
        self._session_key: gr.JSON = None

    def draw_gradio_ui(self) -> JobManagerUi:
        ''' draws the job manager ui in gradio
            Returns:
            ui (JobManagerUi): object which can connect functions to the ui
        '''
        assert gr.context.Context.block is not None, "draw_gradio_ui must be called within a 'gr.Blocks' 'with' context"
        with gr.Tabs():
            with gr.TabItem("Job Controls"):
                with gr.Row():
                    stop_btn = gr.Button("Stop All Batches", elem_id="stop", variant="secondary")
                    refresh_btn = gr.Button("Refresh Finished Batches", elem_id="refresh", variant="secondary")
                status_text = gr.Textbox(placeholder="Job Status", interactive=False, show_label=False)
                with gr.Row():
                    active_image_stop_btn = gr.Button("Skip Active Batch", variant="secondary")
                    active_image_refresh_btn = gr.Button("View Batch Progress", variant="secondary")
                active_image = gr.Image(type="pil", interactive=False, visible=False, elem_id="active_iteration_image")
            with gr.TabItem("Batch Progress Settings"):
                with gr.Row():
                    record_steps_checkbox = gr.Checkbox(value=False, label="Enable Batch Progress Grid")
                    record_steps_interval_slider = gr.Slider(
                        value=3, label="Record Interval (steps)", minimum=1, maximum=25, step=1)
                with gr.Row() as record_steps_box:
                    steps_to_gallery_checkbox = gr.Checkbox(value=False, label="Save Progress Grid to Gallery")
                    steps_to_file_checkbox = gr.Checkbox(value=False, label="Save Progress Grid to File")
            with gr.TabItem("Maintenance"):
                with gr.Row():
                    gr.Markdown(
                        "Stop all concurrent sessions, or free memory associated with jobs which were finished after the browser was closed")
                with gr.Row():
                    stop_all_sessions_btn = gr.Button(
                        "Stop All Sessions", elem_id="stop_all", variant="secondary"
                    )
                    free_done_sessions_btn = gr.Button(
                        "Clear Finished Jobs", elem_id="clear_finished", variant="secondary"
                    )

        return JobManagerUi(_refresh_btn=refresh_btn, _stop_btn=stop_btn, _status_text=status_text,
                            _stop_all_session_btn=stop_all_sessions_btn, _free_done_sessions_btn=free_done_sessions_btn,
                            _active_image=active_image, _active_image_stop_btn=active_image_stop_btn,
                            _active_image_refresh_btn=active_image_refresh_btn,
                            _rec_steps_checkbox=record_steps_checkbox,
                            _save_rec_steps_to_gallery_chkbx=steps_to_gallery_checkbox,
                            _save_rec_steps_to_file_chkbx=steps_to_file_checkbox,
                            _rec_steps_intrvl_sldr=record_steps_interval_slider, _job_manager=self)

    def clear_all_finished_jobs(self):
        ''' Removes all currently finished jobs, across all sessions.
            Useful to free memory if a job is started and the browser is closed
            before it finishes '''
        for session in self._sessions.values():
            session.finished_jobs.clear()

    def stop_all_jobs(self):
        ''' Stops all active jobs, across all sessions'''
        for session in self._sessions.values():
            for job in session.jobs.values():
                job.should_stop.set()
                job.stop_cur_iter.set()

    def _get_job_token(self, block: bool = False) -> Optional[int]:
        ''' Attempts to acquire a job token, optionally blocking until available '''
        token = None
        while token is None:
            try:
                token = self._avail_job_tokens.pop()
                break
            except IndexError:
                pass

            if not block:
                break

            # No token and requested to block, so queue up
            wait_event = Event()
            self._job_queue.append(QueueItem(wait_event))
            wait_event.wait()

        return token

    def _release_job_token(self, token: int) -> None:
        ''' Returns a job token to allow another job to start '''
        self._avail_job_tokens.append(token)
        self._run_queued_jobs()

    def _refresh_func(self, func_key: FuncKey, session_key: str) -> List[Component]:
        ''' Updates information from the active job '''
        session_info, job_info = self._get_call_info(func_key, session_key)
        if job_info is None:
            return [None, f"Session {session_key} was not running function {func_key}"]
        return [triggerChangeEvent(), job_info.job_status]

    def _stop_wrapped_func(self, func_key: FuncKey, session_key: str) -> List[Component]:
        ''' Marks that the job should be stopped'''
        session_info, job_info = self._get_call_info(func_key, session_key)
        if job_info is None:
            return f"Session {session_key} was not running function {func_key}"
        job_info.should_stop.set()
        return "Stopping after current batch finishes"

    def _refresh_cur_iter_func(self, func_key: FuncKey, session_key: str) -> List[Component]:
        ''' Updates information from the active iteration '''
        session_info, job_info = self._get_call_info(func_key, session_key)
        if job_info is None:
            return [None, f"Session {session_key} was not running function {func_key}"]

        job_info.refresh_active_image_requested.set()
        if job_info.refresh_active_image_done.wait(timeout=20.0):
            job_info.refresh_active_image_done.clear()
            return [gr.Image.update(value=job_info.active_image, visible=True), f"Sample iteration {job_info.active_iteration_cnt}"]
        return [gr.Image.update(visible=False), "Timed out getting image"]

    def _stop_cur_iter_func(self, func_key: FuncKey, session_key: str) -> List[Component]:
        ''' Marks that the active iteration should be stopped'''
        session_info, job_info = self._get_call_info(func_key, session_key)
        if job_info is None:
            return [None, f"Session {session_key} was not running function {func_key}"]
        job_info.stop_cur_iter.set()
        return [gr.Image.update(visible=False), "Stopping current iteration"]

    def _get_call_info(self, func_key: FuncKey, session_key: str) -> Tuple[SessionInfo, JobInfo]:
        ''' Helper to get the SessionInfo and JobInfo. '''
        session_info = self._sessions.get(session_key, None)
        if not session_info:
            print(f"Couldn't find session {session_key} for call to {func_key}")
            return None, None

        job_info = session_info.jobs.get(func_key, None)
        if not job_info:
            job_info = session_info.finished_jobs.get(func_key, None)
        if not job_info:
            print(f"Couldn't find job {func_key} in session {session_key}")
            return session_info, None

        return session_info, job_info

    def _run_queued_jobs(self) -> None:
        ''' Runs queued jobs for any available slots '''
        if self._avail_job_tokens:
            try:
                # Notify next queued job it may begin
                queue_item = self._job_queue.pop(0)
                queue_item.wait_event.set()

                # Check again in a few seconds, just in case the queued
                # waiter closed the browser while still queued
                Timer(3.0, self._run_queued_jobs).start()
            except IndexError:
                pass  # No queued jobs

    def _pre_call_func(
            self, func_key: FuncKey, output_dummy_obj: Component, refresh_btn: gr.Button, stop_btn: gr.Button,
            status_text: gr.Textbox, active_image: gr.Image, active_refresh_btn: gr.Button, active_stop_btn: gr.Button,
            session_key: str) -> List[Component]:
        ''' Called when a job is about to start '''
        session_info, job_info = self._get_call_info(func_key, session_key)

        # If we didn't already get a token then queue up for one
        if job_info.job_token is None:
            job_info.job_token = self._get_job_token(block=True)

        # Buttons don't seem to update unless value is set on them as well...
        return {output_dummy_obj: triggerChangeEvent(),
                refresh_btn: gr.Button.update(variant="primary", value=refresh_btn.value),
                stop_btn: gr.Button.update(variant="primary", value=stop_btn.value),
                status_text: gr.Textbox.update(value="Generation has started. Click 'Refresh' to see finished images, 'View Batch Progress' for active images"),
                active_refresh_btn: gr.Button.update(variant="primary", value=active_refresh_btn.value),
                active_stop_btn: gr.Button.update(variant="primary", value=active_stop_btn.value),
                }

    def _call_func(self, func_key: FuncKey, session_key: str) -> List[Component]:
        ''' Runs the real function with job management. '''
        session_info, job_info = self._get_call_info(func_key, session_key)
        if session_info is None or job_info is None:
            return []

        job_info.started = True
        try:
            if job_info.should_stop.is_set():
                raise Exception(f"Job {job_info} requested a stop before execution began")
            outputs = job_info.func(*job_info.inputs, job_info=job_info)
        except Exception as e:
            job_info.job_status = f"Error: {e}"
            print(f"Exception processing job {job_info}: {e}\n{traceback.format_exc()}")
            raise
        finally:
            job_info.finished = True
            session_info.finished_jobs[func_key] = session_info.jobs.pop(func_key)
            self._release_job_token(job_info.job_token)

        # Filter the function output for any removed outputs
        filtered_output = []
        for idx, output in enumerate(outputs):
            if idx not in job_info.removed_output_idxs:
                filtered_output.append(output)

        # The wrapper added a dummy JSON output. Append a random text string
        # to fire the dummy objects 'change' event to notify that the job is done
        filtered_output.append(triggerChangeEvent())

        return tuple(filtered_output)

    def _post_call_func(
            self, func_key: FuncKey, output_dummy_obj: Component, refresh_btn: gr.Button, stop_btn: gr.Button,
            status_text: gr.Textbox, active_image: gr.Image, active_refresh_btn: gr.Button, active_stop_btn: gr.Button,
            session_key: str) -> List[Component]:
        ''' Called when a job completes '''
        return {output_dummy_obj: triggerChangeEvent(),
                refresh_btn: gr.Button.update(variant="secondary", value=refresh_btn.value),
                stop_btn: gr.Button.update(variant="secondary", value=stop_btn.value),
                status_text: gr.Textbox.update(value="Generation has finished!"),
                active_refresh_btn: gr.Button.update(variant="secondary", value=active_refresh_btn.value),
                active_stop_btn: gr.Button.update(variant="secondary", value=active_stop_btn.value),
                active_image: gr.Image.update(visible=False)
                }

    def _update_gallery_event(self, func_key: FuncKey, session_key: str) -> List[Component]:
        ''' Updates the gallery with results from the given job.
            Frees the images after return if the job is finished.
            Triggered by changing the update_gallery_obj dummy object '''
        session_info, job_info = self._get_call_info(func_key, session_key)
        if session_info is None or job_info is None:
            return []

        return job_info.images

    def _wrap_func(self, func: Callable, inputs: List[Component],
                   outputs: List[Component],
                   job_ui: JobManagerUi) -> Tuple[Callable, List[Component]]:
        ''' handles JobManageUI's wrap_func'''

        assert gr.context.Context.block is not None, "wrap_func must be called within a 'gr.Blocks' 'with' context"

        # Create a unique key for this job
        func_key = FuncKey(job_id=uuid.uuid4().hex, func=func)

        # Create a unique session key (next gradio release can use gr.State, see https://gradio.app/state_in_blocks/)
        if self._session_key is None:
            # When this gradio object is received as an event handler input it will resolve to a unique per-session id
            self._session_key = gr.JSON(value=lambda: uuid.uuid4().hex, visible=False,
                                        elem_id="JobManagerDummyObject_sessionKey")

        # Pull the gallery out of the original outputs and assign it to the gallery update dummy object
        gallery_comp = None
        removed_idxs = []
        for idx, comp in enumerate(outputs):
            if isinstance(comp, Gallery):
                removed_idxs.append(idx)
                gallery_comp = comp
                del outputs[idx]
                break

        # Create dummy objects
        update_gallery_obj = gr.JSON(visible=False, elem_id="JobManagerDummyObject")
        update_gallery_obj.change(
            partial(self._update_gallery_event, func_key),
            [self._session_key],
            [gallery_comp],
            queue=False
        )

        if job_ui._refresh_btn:
            job_ui._refresh_btn.variant = 'secondary'
            job_ui._refresh_btn.click(
                partial(self._refresh_func, func_key),
                [self._session_key],
                [update_gallery_obj, job_ui._status_text],
                queue=False
            )

        if job_ui._stop_btn:
            job_ui._stop_btn.variant = 'secondary'
            job_ui._stop_btn.click(
                partial(self._stop_wrapped_func, func_key),
                [self._session_key],
                [job_ui._status_text],
                queue=False
            )

        if job_ui._active_image and job_ui._active_image_refresh_btn:
            job_ui._active_image_refresh_btn.click(
                partial(self._refresh_cur_iter_func, func_key),
                [self._session_key],
                [job_ui._active_image, job_ui._status_text],
                queue=False
            )

        if job_ui._active_image_stop_btn:
            job_ui._active_image_stop_btn.click(
                partial(self._stop_cur_iter_func, func_key),
                [self._session_key],
                [job_ui._active_image, job_ui._status_text],
                queue=False
            )

        if job_ui._stop_all_session_btn:
            job_ui._stop_all_session_btn.click(
                self.stop_all_jobs, [], [],
                queue=False
            )

        if job_ui._free_done_sessions_btn:
            job_ui._free_done_sessions_btn.click(
                self.clear_all_finished_jobs, [], [],
                queue=False
            )

        # (ab)use gr.JSON to forward events.
        # The gr.JSON object will fire its 'change' event when it is modified by being the output
        # of another component. This allows a method to forward events and allow multiple components
        # to update the gallery (without locking it).

        # For example, the update_gallery_obj will update the gallery as in output of its 'change' event.
        # When its content changes it will update the gallery with the most recent images available from
        # the JobInfo. Now, eg, testComponent can have update_gallery_obj as an output and write random text
        # to it. This will trigger an update to the gallery, but testComponent didn't need to have
        # update_gallery_obj listed as an output, which would have locked it.

        # Since some parameters are optional it makes sense to use the 'dict' return value type, which requires
        # the Component as a key... so group together the UI components that the event listeners are going to update
        # to make it easy to append to function calls and outputs
        job_ui_params = [job_ui._refresh_btn, job_ui._stop_btn, job_ui._status_text,
                         job_ui._active_image, job_ui._active_image_refresh_btn, job_ui._active_image_stop_btn]
        job_ui_outputs = [comp for comp in job_ui_params if comp is not None]

        # Here a chain is constructed that will make a 'pre' call, a 'run' call, and a 'post' call,
        # to be able to update the UI before and after, as well as run the actual call
        post_call_dummyobj = gr.JSON(visible=False, elem_id="JobManagerDummyObject_postCall")
        post_call_dummyobj.change(
            partial(self._post_call_func, func_key, update_gallery_obj, *job_ui_params),
            [self._session_key],
            [update_gallery_obj] + job_ui_outputs,
            queue=False
        )

        call_dummyobj = gr.JSON(visible=False, elem_id="JobManagerDummyObject_runCall")
        call_dummyobj.change(
            partial(self._call_func, func_key),
            [self._session_key],
            outputs + [post_call_dummyobj],
            queue=False
        )

        pre_call_dummyobj = gr.JSON(visible=False, elem_id="JobManagerDummyObject_preCall")
        pre_call_dummyobj.change(
            partial(self._pre_call_func, func_key, call_dummyobj, *job_ui_params),
            [self._session_key],
            [call_dummyobj] + job_ui_outputs,
            queue=False
        )

        # Add any components that we want the runtime values for
        added_inputs = [self._session_key, job_ui._rec_steps_checkbox, job_ui._save_rec_steps_to_gallery_chkbx,
                        job_ui._save_rec_steps_to_file_chkbx, job_ui._rec_steps_intrvl_sldr]

        # Now replace the original function with one that creates a JobInfo and triggers the dummy obj
        def wrapped_func(*wrapped_inputs):
            # Remove the added_inputs (pop opposite order of list)

            wrapped_inputs = list(wrapped_inputs)
            rec_steps_interval: int = wrapped_inputs.pop()
            save_rec_steps_file: bool = wrapped_inputs.pop()
            save_rec_steps_grid: bool = wrapped_inputs.pop()
            record_steps_enabled: bool = wrapped_inputs.pop()
            session_key: str = wrapped_inputs.pop()
            job_inputs = tuple(wrapped_inputs)

            # Get or create a session for this key
            session_info = self._sessions.setdefault(session_key, SessionInfo())

            # Is this session already running this job?
            if func_key in session_info.jobs:
                job_info = session_info.jobs[func_key]
                # If the job seems stuck in 'starting' then go ahead and toss it
                if not job_info.started and time.time() > job_info.timestamp + JobManager.JOB_MAX_START_TIME:
                    job_info.should_stop.set()
                    job_info.stop_cur_iter.set()
                    session_info.jobs.pop(func_key)
                    return {job_ui._status_text: "Canceled possibly hung job. Try again"}
                return {job_ui._status_text: "This session is already running that function!"}

            # Is this a new run of a previously finished job? Clear old info
            if func_key in session_info.finished_jobs:
                session_info.finished_jobs.pop(func_key)

            job_token = self._get_job_token(block=False)
            job = JobInfo(
                inputs=job_inputs, func=func, removed_output_idxs=removed_idxs, session_key=session_key,
                job_token=job_token, rec_steps_enabled=record_steps_enabled, rec_steps_intrvl=rec_steps_interval,
                rec_steps_to_gallery=save_rec_steps_grid, rec_steps_to_file=save_rec_steps_file, timestamp=time.time())
            session_info.jobs[func_key] = job

            ret = {pre_call_dummyobj: triggerChangeEvent()}
            if job_token is None:
                ret[job_ui._status_text] = "Job is queued"
            return ret

        return wrapped_func, inputs + added_inputs, [pre_call_dummyobj, job_ui._status_text]

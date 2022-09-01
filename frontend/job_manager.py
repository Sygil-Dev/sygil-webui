''' Provides simple job management for gradio, allowing viewing and canceling in-progress multi-batch generations '''
import gradio as gr
from gradio.components import Component, Gallery
from threading import Semaphore, Event
from typing import Callable, List, Dict, Tuple, Optional
from dataclasses import dataclass, field
from functools import partial
from PIL.Image import Image
import uuid

# TODO: Session management
# TODO: Maximum jobs
# TODO: UI needs to show busy
# TODO: Auto-Refresh


@dataclass
class JobInfo:
    inputs: List[Component]
    func: Callable
    session_key: str
    images: List[Image] = field(default_factory=list)
    should_stop: Event = field(default_factory=Event)
    job_status: str = field(default_factory=str)
    finished: bool = False
    removed_output_idxs: List[int] = field(default_factory=list)


@dataclass(eq=True, frozen=True)
class JobKey:
    job_id: str
    func: Callable


class JobManager:
    def __init__(self, max_jobs: int):
        self._max_jobs: int = max_jobs
        self._jobs_avail: Semaphore = Semaphore(max_jobs)
        self._jobs: Dict[JobKey, JobInfo] = {}
        self._session_key: gr.JSON = None

    def _run_wrapped_func(self, job_key: JobKey) -> List[Component]:
        ''' Runs the real function with job management. '''
        job_info = self._jobs.get(job_key)
        if not job_info:
            return []

        outputs = job_info.func(*job_info.inputs, job_info=job_info)

        # Filter the function output for any removed outputs
        filtered_output = []
        for idx, output in enumerate(outputs):
            if idx not in job_info.removed_output_idxs:
                filtered_output.append(output)

        job_info.finished = True

        # The wrapper added a dummy JSON output. Append a random text string
        # to fire the dummy objects 'change' event to notify that the job is done
        filtered_output.append(uuid.uuid4().hex)

        return tuple(filtered_output)

    def _refresh_func(self, job_key: JobKey) -> List[Component]:
        ''' Updates information from the active job '''
        job_info = self._jobs.get(job_key)
        if not job_info:
            return [None, f"Job key not found: {job_key}"]

        return [uuid.uuid4().hex, job_info.job_status]

    def _cancel_wrapped_func(self, job_key: JobKey) -> List[Component]:
        ''' Marks that the job should be stopped'''
        job_info = self._jobs.get(job_key)
        if job_info:
            job_info.should_stop.set()
        return []

    def _update_gallery_event(self, job_key: JobKey) -> List[Component]:
        ''' Updates the gallery with results from the given job_key.
            Removes the job if it was finished.
            Triggered by changing the update_gallery_obj dummy object '''
        job_info = self._jobs.get(job_key)
        if not job_info:
            return []
        if job_info.finished:
            self._jobs.pop(job_key)
        return job_info.images

    def wrap_func(
            self, func: Callable, inputs: List[Component],
            outputs: List[Component],
            refresh_btn: gr.Button = None, cancel_btn: gr.Button = None, status_text: Optional[gr.Textbox] = None) -> Tuple[
            Callable, List[Component]]:
        ''' Takes a gradio event listener function and its input/outputs and returns wrapped replacements which will
            be managed by JobManager
        Parameters:
        func (Callable) the original event listener to be wrapped.
                        This listener should be modified to take a 'job_info' parameter which, if not None, should can
                        be used by the function to check for cancel events and to store intermediate image results
        inputs (List[Component]) the original inputs
        outputs (List[Component]) the original outputs. The first gallery, if any, will be used for refreshing images
        refresh_btn: (gr.Button, optional) a button to use for updating the gallery with intermediate results
        cancel_btn: (gr.Button, optional) a button to use for cancelling the function
        status_text: (gr.Textbox) a textbox to display job status updates

        Returns:
        Tuple(newFunc (Callable), newInputs (List[Component]), newOutputs (List[Component]), which should be used as
        replacements for the passed in function, inputs and outputs
        '''
        assert gr.context.Context.block is not None, "wrap_func must be called within a 'gr.Blocks' 'with' context"

        # Create a unique key for this job
        job_key = JobKey(job_id=uuid.uuid4().hex, func=func)

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

        # Add the session key to the inputs
        inputs += [self._session_key]

        # Create dummy objects
        update_gallery_obj = gr.JSON(visible=False, elem_id="JobManagerDummyObject")
        update_gallery_obj.change(
            partial(self._update_gallery_event, job_key),
            [],
            [gallery_comp]
        )

        if refresh_btn:
            refresh_btn.click(
                partial(self._refresh_func, job_key),
                [],
                [update_gallery_obj, status_text]
            )

        # TODO: reject existing jobs
        if cancel_btn:
            cancel_btn.click(
                partial(self._cancel_wrapped_func, job_key),
                [],
                []
            )

        # (ab)use gr.JSON to forward events.
        # The gr.JSON object will fire its 'change' event when it is modified by being the output
        # of another component. No other component appears to operate this way. This allows a method
        # to forward events and allow multiple components to update the gallery (without locking it).

        # For example, the update_gallery_obj will update the gallery as in output of its 'change' event.
        # When its content changes it will update the gallery with the most recent images available from
        # the JobInfo. Now, eg, testComponent can have update_gallery_obj as an output and write random text
        # to it. This will trigger an update to the gallery, but testComponent didn't need to have
        # update_gallery_obj listed as an output, which would have locked it.

        # Create the dummy object to replace the original output list, to be triggered when the original function
        # would normally have been called.
        dummy_obj = gr.JSON(visible=False, elem_id="JobManagerDummyObject")
        dummy_obj.change(
            partial(self._run_wrapped_func, job_key),
            [],
            outputs + [update_gallery_obj]
        )

        # Now replace the original function with one that creates a JobInfo and triggers the dummy obj

        def wrapped_func(*inputs):
            session_key = inputs[-1]
            inputs = inputs[:-1]
            self._jobs[job_key] = JobInfo(inputs=inputs, func=func,
                                          removed_output_idxs=removed_idxs, session_key=session_key)
            return uuid.uuid4().hex  # ensures the 'change' event fires
        return wrapped_func, inputs, [dummy_obj]

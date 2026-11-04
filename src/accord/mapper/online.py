"""The online correction in a process of its own, beside the mapper, so that mapping never waits for it. The mapper hands
it each window's images (``revisit.prepare``) and the graph's nodes; the process solves whatever has arrived since its
last solve as one batch and publishes the corrections, and answers once more when the sequence ends. The engine starts
it (a daemonic process has no children); the two talk through the engine's queues."""
import multiprocessing as mp
from queue import Empty
import time
import traceback

from . import revisit


def correction_loop(inbox, outbox, cfg):
    """Until the mapper sends None: take every message waiting, measure and solve, publish
    {values, through (the last window taken), final, stats}."""
    online = revisit.OnlineCorrection(cfg)
    stats = dict(batches=0, windows=0, busy_s=0., largest_batch=0)
    through, final = None, False
    try:
        while not final:
            messages = [inbox.get()]
            while True:
                try:
                    messages.append(inbox.get_nowait())
                except Empty:
                    break
            final = any(m is None for m in messages)
            messages = [m for m in messages if m is not None]
            if messages:
                started = time.monotonic()
                values = online.update_batch([m['window'] for m in messages], messages[-1]['nodes'])
                through = messages[-1]['window'][0]
                stats['batches'] += 1
                stats['windows'] += len(messages)
                stats['largest_batch'] = max(stats['largest_batch'], len(messages))
                stats['busy_s'] += time.monotonic() - started
            outbox.put(dict(values=online.values, through=through, final=final, stats=dict(stats)))
    except Exception:
        outbox.put(dict(error=traceback.format_exc(), final=True))


class Client:
    """The mapper's side of the queues."""

    def __init__(self, inbox, outbox):
        self.inbox, self.outbox = inbox, outbox

    def submit(self, key, images, nodes):
        self.inbox.put(dict(window=(key, images), nodes=nodes))

    def latest(self):
        """The newest answer since the last call, or None."""
        last = None
        while True:
            try:
                last = self.outbox.get_nowait()
            except Empty:
                return last

    def finish(self, timeout_s):
        """Tell the process the sequence has ended; its final answer, or None if it does not come in ``timeout_s``."""
        self.inbox.put(None)
        until = time.monotonic() + timeout_s
        while time.monotonic() < until:
            try:
                answer = self.outbox.get(timeout=.1)
            except Empty:
                continue
            if answer.get('final'):
                return answer
        return None


class Process:
    """The correction process, started by the engine; ``queues`` go to the vision process."""

    def __init__(self, cfg):
        ctx = mp.get_context('spawn')
        self.queues = (ctx.Queue(), ctx.Queue())
        self.process = ctx.Process(target=correction_loop, args=(*self.queues, cfg), daemon=True)
        self.process.start()

    def close(self, grace_s=5.):
        self.process.join(timeout=grace_s)
        if self.process.is_alive():
            self.process.terminate()
            self.process.join(timeout=1.)
        for queue in self.queues:
            queue.cancel_join_thread()
            queue.close()

"""Wrappers for forwarding stdout/stderr over zmq"""

# Copyright (c) IPython Development Team.
# Distributed under the terms of the Modified BSD License.

from __future__ import annotations

import atexit
import contextvars
import io
import os
import sys
import threading
import traceback
import warnings
from binascii import b2a_hex
from collections import defaultdict, deque
from io import StringIO, TextIOBase
from threading import local
from typing import Any, Callable

import zmq
import zmq_anyio
from anyio import sleep
from jupyter_client.session import extract_header

from .thread import BaseThread

# -----------------------------------------------------------------------------
# Globals
# -----------------------------------------------------------------------------

_PARENT = 0
_CHILD = 1

PIPE_BUFFER_SIZE = 1000

# -----------------------------------------------------------------------------
# IO classes
# -----------------------------------------------------------------------------


class IOPubThread:
    """An object for sending IOPub messages in a background thread

    Prevents a blocking main thread from delaying output from threads.

    IOPubThread(pub_socket).background_socket is a Socket-API-providing object
    whose IO is always run in a thread.
    """

    def __init__(self, socket: zmq_anyio.Socket, pipe=False):
        """Create IOPub thread

        Parameters
        ----------
        socket : zmq.PUB Socket
            the socket on which messages will be sent.
        pipe : bool
            Whether this process should listen for IOPub messages
            piped from subprocesses.
        """
        # ensure all of our sockets as sync zmq.Sockets
        # don't create async wrappers until we are within the appropriate coroutines
        self.socket: zmq_anyio.Socket = socket
        self._context = socket.context

        self.background_socket = BackgroundSocket(self)
        self._main_pid = os.getpid()
        self._pipe_flag = pipe
        if pipe:
            self._setup_pipe_in()
        self._local = threading.local()
        self._events: deque[Callable[..., Any]] = deque()
        self._event_pipes: dict[threading.Thread, Any] = {}
        self._event_pipe_gc_lock: threading.Lock = threading.Lock()
        self._event_pipe_gc_seconds: float = 10
        self._setup_event_pipe()
        tasks = [self._handle_event, self._run_event_pipe_gc, self.socket.start]
        if pipe:
            tasks.append(self._handle_pipe_msgs)
        self.thread = BaseThread(name="IOPub", daemon=True)
        for task in tasks:
            self.thread.start_soon(task)

    def _setup_event_pipe(self):
        """Create the PULL socket listening for events that should fire in this thread."""
        self._pipe_in0 = self._context.socket(zmq.PULL)
        self._pipe_in0.linger = 0

        _uuid = b2a_hex(os.urandom(16)).decode("ascii")
        iface = self._event_interface = "inproc://%s" % _uuid
        self._pipe_in0.bind(iface)

    async def _run_event_pipe_gc(self):
        """Task to run event pipe gc continuously"""
        while True:
            await sleep(self._event_pipe_gc_seconds)
            try:
                await self._event_pipe_gc()
            except Exception as e:
                print(f"Exception in IOPubThread._event_pipe_gc: {e}", file=sys.__stderr__)

    async def _event_pipe_gc(self):
        """run a single garbage collection on event pipes"""
        if not self._event_pipes:
            # don't acquire the lock if there's nothing to do
            return
        with self._event_pipe_gc_lock:
            for thread, socket in list(self._event_pipes.items()):
                if not thread.is_alive():
                    socket.close()
                    del self._event_pipes[thread]

    @property
    def _event_pipe(self):
        """thread-local event pipe for signaling events that should be processed in the thread"""
        try:
            event_pipe = self._local.event_pipe
        except AttributeError:
            # new thread, new event pipe
            # create sync base socket
            event_pipe = self._context.socket(zmq.PUSH)
            event_pipe.linger = 0
            event_pipe.connect(self._event_interface)
            self._local.event_pipe = event_pipe
            # associate event pipes to their threads
            # so they can be closed explicitly
            # implicit close on __del__ throws a ResourceWarning
            with self._event_pipe_gc_lock:
                self._event_pipes[threading.current_thread()] = event_pipe
        return event_pipe

    async def _handle_event(self):
        """Handle an event on the event pipe

        Content of the message is ignored.

        Whenever *an* event arrives on the event stream,
        *all* waiting events are processed in order.
        """
        pipe_in = zmq_anyio.Socket(self._pipe_in0)
        async with pipe_in:
            try:
                while True:
                    await pipe_in.arecv().wait()
                    # freeze event count so new writes don't extend the queue
                    # while we are processing
                    n_events = len(self._events)
                    for _ in range(n_events):
                        event_f = self._events.popleft()
                        event_f()
            except Exception:
                if self.thread.stopped.is_set():
                    return
                raise

    def _setup_pipe_in(self):
        """setup listening pipe for IOPub from forked subprocesses"""
        # use UUID to authenticate pipe messages
        self._pipe_uuid = os.urandom(16)

        self._pipe_in1 = zmq_anyio.Socket(self._context.socket(zmq.PULL))
        self._pipe_in1.linger = 0

        try:
            self._pipe_port = self._pipe_in1.bind_to_random_port("tcp://127.0.0.1")
        except zmq.ZMQError as e:
            warnings.warn(
                "Couldn't bind IOPub Pipe to 127.0.0.1: %s" % e
                + "\nsubprocess output will be unavailable.",
                stacklevel=2,
            )
            self._pipe_flag = False
            self._pipe_in1.close()
            return

    async def _handle_pipe_msgs(self):
        """handle pipe messages from a subprocess"""
        async with self._pipe_in1:
            try:
                while True:
                    await self._handle_pipe_msg()
            except Exception:
                if self.thread.stopped.is_set():
                    return
                raise

    async def _handle_pipe_msg(self, msg=None):
        """handle a pipe message from a subprocess"""
        msg = msg or await self._pipe_in1.arecv_multipart().wait()
        if not self._pipe_flag or not self._is_main_process():
            return
        if msg[0] != self._pipe_uuid:
            print("Bad pipe message: %s", msg, file=sys.__stderr__)
            return
        self.send_multipart(msg[1:])

    def _setup_pipe_out(self):
        # must be new context after fork
        ctx = zmq.Context()
        pipe_out = ctx.socket(zmq.PUSH)
        pipe_out.linger = 3000  # 3s timeout for pipe_out sends before discarding the message
        pipe_out.connect("tcp://127.0.0.1:%i" % self._pipe_port)
        return ctx, pipe_out

    def _is_main_process(self):
        return os.getpid() == self._main_pid

    def _check_mp_mode(self):
        """check for forks, and switch to zmq pipeline if necessary"""
        if not self._pipe_flag or self._is_main_process():
            return _PARENT
        return _CHILD

    def start(self):
        """Start the IOPub thread"""
        self.thread.start()
        # make sure we don't prevent process exit
        # I'm not sure why setting daemon=True above isn't enough, but it doesn't appear to be.
        atexit.register(self.stop)

    def stop(self):
        """Stop the IOPub thread"""
        if not self.thread.is_alive():
            return
        self.thread.stop()

        self.thread.join(timeout=30)
        if self.thread.is_alive():
            # avoid infinite hang if stop fails
            msg = "IOPub thread did not terminate in 30 seconds"
            raise TimeoutError(msg)
        # close *all* event pipes, created in any thread
        # event pipes can only be used from other threads while self.thread.is_alive()
        # so after thread.join, this should be safe
        for _thread, event_pipe in self._event_pipes.items():
            event_pipe.close()

    def close(self):
        """Close the IOPub thread."""
        if self.closed:
            return
        self._pipe_in0.close()
        if self._pipe_flag:
            self._pipe_in1.close()
        if self.socket is not None:
            self.socket.close()
        self.socket = None

    @property
    def closed(self):
        return self.socket is None

    def schedule(self, f):
        """Schedule a function to be called in our IO thread.

        If the thread is not running, call immediately.
        """
        if self.thread.is_alive():
            self._events.append(f)
            # wake event thread (message content is ignored)
            try:
                self._event_pipe.send(b"")
            except RuntimeError:
                pass

        else:
            f()

    def send_multipart(self, *args, **kwargs):
        """send_multipart schedules actual zmq send in my thread.

        If my thread isn't running (e.g. forked process), send immediately.
        """
        self.schedule(lambda: self._really_send(*args, **kwargs))

    def _really_send(self, msg, *args, **kwargs):
        """The callback that actually sends messages"""
        if self.closed:
            return

        mp_mode = self._check_mp_mode()
        if mp_mode != _CHILD:
            # we are the main parent process, do a regular send
            assert self.socket is not None
            self.socket.send_multipart(msg, *args, **kwargs)
        else:
            # we are a child, pipe to parent process
            # new context/socket for every pipe-out
            # since forks don't teardown politely, use ctx.term to ensure send has completed
            ctx, pipe_out = self._setup_pipe_out()
            pipe_out.send_multipart([self._pipe_uuid, *msg], *args, **kwargs)
            pipe_out.close()
            ctx.term()


class BackgroundSocket:
    """Wrapper around IOPub thread that provides zmq send[_multipart]"""

    io_thread = None

    def __init__(self, io_thread):
        """Initialize the socket."""
        self.io_thread = io_thread

    def __getattr__(self, attr):
        """Wrap socket attr access for backward-compatibility"""
        if attr.startswith("__") and attr.endswith("__"):
            # don't wrap magic methods
            super().__getattr__(attr)  # type:ignore[misc]
        assert self.io_thread is not None
        if hasattr(self.io_thread.socket, attr):
            warnings.warn(
                f"Accessing zmq Socket attribute {attr} on BackgroundSocket"
                f" is deprecated since ipykernel 4.3.0"
                f" use .io_thread.socket.{attr}",
                DeprecationWarning,
                stacklevel=2,
            )
            return getattr(self.io_thread.socket, attr)
        return super().__getattr__(attr)  # type:ignore[misc]

    def __setattr__(self, attr, value):
        """Set an attribute on the socket."""
        if attr == "io_thread" or (attr.startswith("__") and attr.endswith("__")):
            super().__setattr__(attr, value)
        else:
            warnings.warn(
                f"Setting zmq Socket attribute {attr} on BackgroundSocket"
                f" is deprecated since ipykernel 4.3.0"
                f" use .io_thread.socket.{attr}",
                DeprecationWarning,
                stacklevel=2,
            )
            assert self.io_thread is not None
            setattr(self.io_thread.socket, attr, value)

    def send(self, msg, *args, **kwargs):
        """Send a message to the socket."""
        return self.send_multipart([msg], *args, **kwargs)

    def send_multipart(self, *args, **kwargs):
        """Schedule send in IO thread"""
        assert self.io_thread is not None
        return self.io_thread.send_multipart(*args, **kwargs)


class OutStream(TextIOBase):
    """A file like object that publishes the stream to a 0MQ PUB socket.

    Output is handed off to an IO Thread
    """

    # timeout for flush to avoid infinite hang
    # in case of misbehavior
    flush_timeout = 10
    # The time interval between automatic flushes, in seconds.
    flush_interval = 0.2
    topic = None
    encoding = "UTF-8"
    _exc: Any = None

    def fileno(self):
        """
        Things like subprocess will peak and write to the fileno() of stderr/stdout.
        """
        if getattr(self, "_original_stdstream_fd", None) is not None:
            return self._original_stdstream_fd
        msg = "fileno"
        raise io.UnsupportedOperation(msg)

    def _watch_pipe_fd(self):
        """
        We've redirected standards streams 0 and 1 into a pipe.

        We need to watch in a thread and redirect them to the right places.

        1) the ZMQ channels to show in notebook interfaces,
        2) the original stdout/err, to capture errors in terminals.

        We cannot schedule this on the ioloop thread, as this might be blocking.

        """

        try:
            bts = os.read(self._fid, PIPE_BUFFER_SIZE)
            while bts and self._should_watch:
                self.write(bts.decode(errors="replace"))
                os.write(self._original_stdstream_copy, bts)
                bts = os.read(self._fid, PIPE_BUFFER_SIZE)
        except Exception:
            self._exc = sys.exc_info()

    def __init__(
        self,
        session,
        pub_thread,
        name,
        pipe=None,
        echo=None,
        *,
        watchfd=True,
        isatty=False,
    ):
        """
        Parameters
        ----------
        session : object
            the session object
        pub_thread : threading.Thread
            the publication thread
        name : str {'stderr', 'stdout'}
            the name of the standard stream to replace
        pipe : object
            the pipe object
        echo : bool
            whether to echo output
        watchfd : bool (default, True)
            Watch the file descriptor corresponding to the replaced stream.
            This is useful if you know some underlying code will write directly
            the file descriptor by its number. It will spawn a watching thread,
            that will swap the give file descriptor for a pipe, read from the
            pipe, and insert this into the current Stream.
        isatty : bool (default, False)
            Indication of whether this stream has terminal capabilities (e.g. can handle colors)

        """
        if pipe is not None:
            warnings.warn(
                "pipe argument to OutStream is deprecated and ignored since ipykernel 4.2.3.",
                DeprecationWarning,
                stacklevel=2,
            )
        # This is necessary for compatibility with Python built-in streams
        self.session = session
        self._has_thread = False
        self.watch_fd_thread = None
        if not isinstance(pub_thread, IOPubThread):
            # Backward-compat: given socket, not thread. Wrap in a thread.
            warnings.warn(
                "Since IPykernel 4.3, OutStream should be created with "
                "IOPubThread, not %r" % pub_thread,
                DeprecationWarning,
                stacklevel=2,
            )
            pub_thread = IOPubThread(pub_thread)
            pub_thread.start()
            self._has_thread = True
        self.pub_thread = pub_thread
        self.name = name
        self.topic = b"stream." + name.encode()
        self._parent_header: contextvars.ContextVar[dict[str, Any]] = contextvars.ContextVar(
            "parent_header"
        )
        self._parent_header.set({})
        self._thread_to_parent = {}
        self._thread_to_parent_header = {}
        self._parent_header_global = {}
        self._main_pid = os.getpid()
        self._flush_pending = False
        self._subprocess_flush_pending = False
        self._buffer_lock = threading.RLock()
        self._buffers = defaultdict(StringIO)
        self.echo = None
        self._isatty = bool(isatty)
        self._should_watch = False
        self._local = local()

        if (
            (
                watchfd
                and (
                    (sys.platform.startswith("linux") or sys.platform.startswith("darwin"))
                    # Pytest set its own capture. Don't redirect from within pytest.
                    and ("PYTEST_CURRENT_TEST" not in os.environ)
                )
            )
            # allow forcing watchfd (mainly for tests)
            or watchfd == "force"
        ):
            self._should_watch = True
            self._setup_stream_redirects(name)

        if echo:
            if hasattr(echo, "read") and hasattr(echo, "write"):
                # make sure we aren't trying to echo on the FD we're watching!
                # that would cause an infinite loop, always echoing on itself
                if self._should_watch:
                    try:
                        echo_fd = echo.fileno()
                    except Exception:
                        echo_fd = None

                    if echo_fd is not None and echo_fd == self._original_stdstream_fd:
                        # echo on the _copy_ we made during
                        # this is the actual terminal FD now
                        echo = io.TextIOWrapper(
                            io.FileIO(self._original_stdstream_copy, "w", closefd=False)
                        )
                self.echo = echo
            else:
                msg = "echo argument must be a file-like object"
                raise ValueError(msg)

    @property
    def parent_header(self):
        try:
            # asyncio-specific
            return self._parent_header.get()
        except LookupError:
            try:
                # thread-specific
                identity = threading.current_thread().ident
                # retrieve the outermost (oldest ancestor,
                # discounting the kernel thread) thread identity
                while identity in self._thread_to_parent:
                    identity = self._thread_to_parent[identity]
                # use the header of the oldest ancestor
                return self._thread_to_parent_header[identity]
            except KeyError:
                # global (fallback)
                return self._parent_header_global

    @parent_header.setter
    def parent_header(self, value):
        self._parent_header_global = value
        return self._parent_header.set(value)

    def isatty(self):
        """Return a bool indicating whether this is an 'interactive' stream.

        Returns:
            Boolean
        """
        return self._isatty

    def _setup_stream_redirects(self, name):
        pr, pw = os.pipe()
        fno = self._original_stdstream_fd = getattr(sys, name).fileno()
        self._original_stdstream_copy = os.dup(fno)
        os.dup2(pw, fno)

        self._fid = pr

        self._exc = None
        self.watch_fd_thread = threading.Thread(target=self._watch_pipe_fd)
        self.watch_fd_thread.daemon = True
        self.watch_fd_thread.start()

    def _is_main_process(self):
        return os.getpid() == self._main_pid

    def set_parent(self, parent):
        """Set the parent header."""
        self.parent_header = extract_header(parent)

    def close(self):
        """Close the stream."""
        if self._should_watch:
            self._should_watch = False
            # thread won't wake unless there's something to read
            # writing something after _should_watch will not be echoed
            if self.watch_fd_thread is not None and self.watch_fd_thread.is_alive():
                os.write(self._original_stdstream_fd, b"\0")
                self.watch_fd_thread.join()
            self.echo = None
            # restore original FDs
            os.dup2(self._original_stdstream_copy, self._original_stdstream_fd)
            os.close(self._original_stdstream_copy)
        if self._exc:
            etype, value, tb = self._exc
            traceback.print_exception(etype, value, tb)
        if self._has_thread:
            self.pub_thread.stop()
        self.pub_thread = None

    @property
    def closed(self):
        return self.pub_thread is None

    def _schedule_flush(self):
        """schedule a flush in the IO thread

        call this on write, to indicate that flush should be called soon.
        """
        if self._flush_pending:
            return
        self._flush_pending = True

        # add_timeout has to be handed to the io thread via event pipe
        self.pub_thread.schedule(self._flush)

    def flush(self):
        """trigger actual zmq send

        send will happen in the background thread
        """
        if (
            self.pub_thread
            and self.pub_thread.thread is not None
            and self.pub_thread.thread.is_alive()
            and self.pub_thread.thread.ident != threading.current_thread().ident
        ):
            # request flush on the background thread
            self.pub_thread.schedule(self._flush)
            # wait for flush to actually get through, if we can.
            evt = threading.Event()
            self.pub_thread.schedule(evt.set)
            # and give a timeout to avoid
            if not evt.wait(self.flush_timeout):
                # write directly to __stderr__ instead of warning because
                # if this is happening sys.stderr may be the problem.
                print("IOStream.flush timed out", file=sys.__stderr__)
        else:
            self._flush()

    def _flush(self):
        """This is where the actual send happens.

        _flush should generally be called in the IO thread,
        unless the thread has been destroyed (e.g. forked subprocess).
        """
        self._flush_pending = False
        self._subprocess_flush_pending = False

        if self.echo is not None:
            try:
                self.echo.flush()
            except OSError as e:
                if self.echo is not sys.__stderr__:
                    print(f"Flush failed: {e}", file=sys.__stderr__)

        for parent, data in self._flush_buffers():
            if data:
                # FIXME: this disables Session's fork-safe check,
                # since pub_thread is itself fork-safe.
                # There should be a better way to do this.
                self.session.pid = os.getpid()
                content = {"name": self.name, "text": data}
                msg = self.session.msg("stream", content, parent=parent)

                # Each transform either returns a new
                # message or None. If None is returned,
                # the message has been 'used' and we return.
                for hook in self._hooks:
                    msg = hook(msg)
                    if msg is None:
                        return

                self.session.send(
                    self.pub_thread,
                    msg,
                    ident=self.topic,
                )

    def write(self, string: str) -> int:
        """Write to current stream after encoding if necessary

        Returns
        -------
        len : int
            number of items from input parameter written to stream.

        """
        parent = self.parent_header

        if not isinstance(string, str):
            msg = f"write() argument must be str, not {type(string)}"  # type:ignore[unreachable]
            raise TypeError(msg)

        if self.echo is not None:
            try:
                self.echo.write(string)
            except OSError as e:
                if self.echo is not sys.__stderr__:
                    print(f"Write failed: {e}", file=sys.__stderr__)

        if self.pub_thread is None:
            msg = "I/O operation on closed file"
            raise ValueError(msg)

        is_child = not self._is_main_process()
        # only touch the buffer in the IO thread to avoid races
        with self._buffer_lock:
            self._buffers[frozenset(parent.items())].write(string)
        if is_child:
            # mp.Pool cannot be trusted to flush promptly (or ever),
            # and this helps.
            if self._subprocess_flush_pending:
                return 0
            self._subprocess_flush_pending = True
            # We can not rely on self._io_loop.call_later from a subprocess
            self.pub_thread.schedule(self._flush)
        else:
            self._schedule_flush()

        return len(string)

    def writelines(self, sequence):
        """Write lines to the stream."""
        if self.pub_thread is None:
            msg = "I/O operation on closed file"
            raise ValueError(msg)
        for string in sequence:
            self.write(string)

    def writable(self):
        """Test whether the stream is writable."""
        return True

    def _flush_buffers(self):
        """clear the current buffer and return the current buffer data."""
        buffers = self._rotate_buffers()
        for frozen_parent, buffer in buffers.items():
            data = buffer.getvalue()
            buffer.close()
            yield dict(frozen_parent), data

    def _rotate_buffers(self):
        """Returns the current buffer and replaces it with an empty buffer."""
        with self._buffer_lock:
            old_buffers = self._buffers
            self._buffers = defaultdict(StringIO)
        return old_buffers

    @property
    def _hooks(self):
        if not hasattr(self._local, "hooks"):
            # create new list for a new thread
            self._local.hooks = []
        return self._local.hooks

    def register_hook(self, hook):
        """
        Registers a hook with the thread-local storage.

        Parameters
        ----------
        hook : Any callable object

        Returns
        -------
        Either a publishable message, or `None`.
        The hook callable must return a message from
        the __call__ method if they still require the
        `session.send` method to be called after transformation.
        Returning `None` will halt that execution path, and
        session.send will not be called.
        """
        self._hooks.append(hook)

    def unregister_hook(self, hook):
        """
        Un-registers a hook with the thread-local storage.

        Parameters
        ----------
        hook : Any callable object which has previously been
            registered as a hook.

        Returns
        -------
        bool - `True` if the hook was removed, `False` if it wasn't
            found.
        """
        try:
            self._hooks.remove(hook)
            return True
        except ValueError:
            return False

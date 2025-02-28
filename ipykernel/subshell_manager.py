"""Manager of subshells in a kernel."""

# Copyright (c) IPython Development Team.
# Distributed under the terms of the Modified BSD License.
from __future__ import annotations

import typing as t
import uuid
from dataclasses import dataclass
from functools import partial
from threading import Lock, current_thread, main_thread

import zmq
import zmq_anyio
from anyio import create_memory_object_stream, create_task_group
from anyio.abc import TaskGroup

from .subshell import SubshellThread
from .thread import SHELL_CHANNEL_THREAD_NAME, BaseThread


@dataclass
class Subshell:
    thread: SubshellThread
    shell_channel_socket: zmq_anyio.Socket


class SubshellManager:
    """A manager of subshells.

    Controls the lifetimes of subshell threads and their associated ZMQ sockets.
    Runs mostly in the shell channel thread.

    Care needed with threadsafe access here.  All write access to the cache occurs in
    the shell channel thread so there is only ever one write access at any one time.
    Reading of cache information can be performed by other threads, so all reads are
    protected by a lock so that they are atomic.

    Sending reply messages via the shell_socket is wrapped by another lock to protect
    against multiple subshells attempting to send at the same time.
    """

    def __init__(
        self,
        context: zmq.Context,  # type: ignore[type-arg]
        shell_socket: zmq_anyio.Socket,
    ):
        assert current_thread() == main_thread()

        self._context: zmq.Context = context  # type: ignore[type-arg]
        self._shell_socket = shell_socket
        self._cache: dict[str, Subshell] = {}
        self._lock_cache = Lock()
        self._lock_shell_socket = Lock()

        # Inproc pair sockets for control channel and main shell (parent subshell).
        # Each inproc pair has a "shell_channel" socket used in the shell channel
        # thread, and an "other" socket used in the other thread.
        self.__control_shell_channel_socket: zmq_anyio.Socket | None = None
        self.__control_other_socket: zmq_anyio.Socket | None = None
        self.__parent_shell_channel_socket: zmq_anyio.Socket | None = None
        self.__parent_other_socket: zmq_anyio.Socket | None = None

        # anyio memory object stream for async queue-like communication between tasks.
        # Used by _create_subshell to tell listen_from_subshells to spawn a new task.
        self._send_stream, self._receive_stream = create_memory_object_stream[str]()

    @property
    def _control_shell_channel_socket(self) -> zmq_anyio.Socket:
        if self.__control_shell_channel_socket is None:
            self.__control_shell_channel_socket = self._create_inproc_pair_socket("control", True)
        return self.__control_shell_channel_socket

    @property
    def _control_other_socket(self) -> zmq_anyio.Socket:
        if self.__control_other_socket is None:
            self.__control_other_socket = self._create_inproc_pair_socket("control", False)
        return self.__control_other_socket

    @property
    def _parent_shell_channel_socket(self) -> zmq_anyio.Socket:
        if self.__parent_shell_channel_socket is None:
            self.__parent_shell_channel_socket = self._create_inproc_pair_socket(None, True)
        return self.__parent_shell_channel_socket

    @property
    def _parent_other_socket(self) -> zmq_anyio.Socket:
        if self.__parent_other_socket is None:
            self.__parent_other_socket = self._create_inproc_pair_socket(None, False)
        return self.__parent_other_socket

    def close(self) -> None:
        """Stop all subshells and close all resources."""
        assert current_thread().name == SHELL_CHANNEL_THREAD_NAME

        self._send_stream.close()
        self._receive_stream.close()

        for socket in (
            self.__control_shell_channel_socket,
            self.__control_other_socket,
            self.__parent_shell_channel_socket,
            self.__parent_other_socket,
        ):
            if socket is not None:
                socket.close()

        with self._lock_cache:
            while True:
                try:
                    _, subshell = self._cache.popitem()
                except KeyError:
                    break
                self._stop_subshell(subshell)

    async def get_control_other_socket(self, thread: BaseThread) -> zmq_anyio.Socket:
        if not self._control_other_socket.started.is_set():
            thread.task_group.start_soon(self._control_other_socket.start)
            await self._control_other_socket.started.wait()
        return self._control_other_socket

    def get_other_socket(self, subshell_id: str | None) -> zmq_anyio.Socket:
        """Return the other inproc pair socket for a subshell.

        This socket is accessed from the subshell thread.
        """
        if subshell_id is None:
            return self._parent_other_socket
        with self._lock_cache:
            socket = self._cache[subshell_id].thread._pair_socket
            assert socket is not None
            return socket

    def get_shell_channel_socket(self, subshell_id: str | None) -> zmq_anyio.Socket:
        """Return the shell channel inproc pair socket for a subshell.

        This socket is accessed from the shell channel thread.
        """
        if subshell_id is None:
            return self._parent_shell_channel_socket
        with self._lock_cache:
            return self._cache[subshell_id].shell_channel_socket

    def list_subshell(self) -> list[str]:
        """Return list of current subshell ids.

        Can be called by any subshell using %subshell magic.
        """
        with self._lock_cache:
            return list(self._cache)

    async def listen_from_control(self, subshell_task: t.Any) -> None:
        """Listen for messages on the control inproc socket, handle those messages and
        return replies on the same socket.  Runs in the shell channel thread.
        """
        assert current_thread().name == SHELL_CHANNEL_THREAD_NAME

        socket = self._control_shell_channel_socket
        async with socket:
            while True:
                request = await socket.arecv_json().wait()
                reply = await self._process_control_request(request, subshell_task)
                await socket.asend_json(reply).wait()

    async def listen_from_subshells(self) -> None:
        """Listen for reply messages on inproc sockets of all subshells and resend
        those messages to the client via the shell_socket.

        Runs in the shell channel thread.
        """
        assert current_thread().name == SHELL_CHANNEL_THREAD_NAME

        async with create_task_group() as tg:
            tg.start_soon(self._listen_for_subshell_reply, None, tg)
            async for subshell_id in self._receive_stream:
                tg.start_soon(self._listen_for_subshell_reply, subshell_id, tg)

    def subshell_id_from_thread_id(self, thread_id: int) -> str | None:
        """Return subshell_id of the specified thread_id.

        Raises RuntimeError if thread_id is not the main shell or a subshell.

        Only used by %subshell magic so does not have to be fast/cached.
        """
        with self._lock_cache:
            if thread_id == main_thread().ident:
                return None
            for id, subshell in self._cache.items():
                if subshell.thread.ident == thread_id:
                    return id
            msg = f"Thread id {thread_id!r} does not correspond to a subshell of this kernel"
            raise RuntimeError(msg)

    def _create_inproc_pair_socket(
        self, name: str | None, shell_channel_end: bool
    ) -> zmq_anyio.Socket:
        """Create and return a single ZMQ inproc pair socket."""
        address = self._get_inproc_socket_address(name)
        socket = zmq_anyio.Socket(self._context, zmq.PAIR)
        if shell_channel_end:
            socket.bind(address)
        else:
            socket.connect(address)
        return socket

    async def _create_subshell(self, subshell_task: t.Any) -> str:
        """Create and start a new subshell thread."""
        assert current_thread().name == SHELL_CHANNEL_THREAD_NAME

        subshell_id = str(uuid.uuid4())
        thread = SubshellThread(subshell_id)

        with self._lock_cache:
            assert subshell_id not in self._cache
            shell_channel_socket = self._create_inproc_pair_socket(subshell_id, True)
            self._cache[subshell_id] = Subshell(thread, shell_channel_socket)

        # Tell task running listen_from_subshells to create a new task to listen for
        # reply messages from the new subshell to resend to the client.
        await self._send_stream.send(subshell_id)

        address = self._get_inproc_socket_address(subshell_id)
        thread.start_soon(partial(thread.create_pair_socket, self._context, address))
        thread.start_soon(partial(subshell_task, subshell_id))
        thread.start()

        return subshell_id

    def _delete_subshell(self, subshell_id: str) -> None:
        """Delete subshell identified by subshell_id.

        Raises key error if subshell_id not in cache.
        """
        assert current_thread().name == SHELL_CHANNEL_THREAD_NAME

        with self._lock_cache:
            subshell = self._cache.pop(subshell_id)

        self._stop_subshell(subshell)

    def _get_inproc_socket_address(self, name: str | None) -> str:
        full_name = f"subshell-{name}" if name else "subshell"
        return f"inproc://{full_name}"

    def _get_shell_channel_socket(self, subshell_id: str | None) -> zmq_anyio.Socket:
        if subshell_id is None:
            return self._parent_shell_channel_socket
        with self._lock_cache:
            return self._cache[subshell_id].shell_channel_socket

    def _is_subshell(self, subshell_id: str | None) -> bool:
        if subshell_id is None:
            return True
        with self._lock_cache:
            return subshell_id in self._cache

    async def _listen_for_subshell_reply(
        self, subshell_id: str | None, task_group: TaskGroup
    ) -> None:
        """Listen for reply messages on specified subshell inproc socket and
        resend to the client via the shell_socket.

        Runs in the shell channel thread.
        """
        assert current_thread().name == SHELL_CHANNEL_THREAD_NAME

        shell_channel_socket = self._get_shell_channel_socket(subshell_id)

        task_group.start_soon(shell_channel_socket.start)
        await shell_channel_socket.started.wait()
        try:
            while True:
                msg = await shell_channel_socket.arecv_multipart(copy=False).wait()
                with self._lock_shell_socket:
                    await self._shell_socket.asend_multipart(msg).wait()
        except BaseException:
            if not self._is_subshell(subshell_id):
                # Subshell no longer exists so exit gracefully
                return
            raise

    async def _process_control_request(
        self, request: dict[str, t.Any], subshell_task: t.Any
    ) -> dict[str, t.Any]:
        """Process a control request message received on the control inproc
        socket and return the reply.  Runs in the shell channel thread.
        """
        assert current_thread().name == SHELL_CHANNEL_THREAD_NAME

        try:
            type = request["type"]
            reply: dict[str, t.Any] = {"status": "ok"}

            if type == "create":
                reply["subshell_id"] = await self._create_subshell(subshell_task)
            elif type == "delete":
                subshell_id = request["subshell_id"]
                self._delete_subshell(subshell_id)
            elif type == "list":
                reply["subshell_id"] = self.list_subshell()
            else:
                msg = f"Unrecognised message type {type!r}"
                raise RuntimeError(msg)
        except BaseException as err:
            reply = {
                "status": "error",
                "evalue": str(err),
            }
        return reply

    def _stop_subshell(self, subshell: Subshell) -> None:
        """Stop a subshell thread and close all of its resources."""
        assert current_thread().name == SHELL_CHANNEL_THREAD_NAME

        thread = subshell.thread
        if thread.is_alive():
            thread.stop()
            thread.join()

        # Closing the shell_channel_socket terminates the task that is listening on it.
        subshell.shell_channel_socket.close()

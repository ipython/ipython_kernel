"""Defines a dummy socket implementing (part of) the zmq.Socket interface."""

# Copyright (c) IPython Development Team.
# Distributed under the terms of the Modified BSD License.

from math import inf

import zmq
import zmq.asyncio
from anyio import create_memory_object_stream
from traitlets import HasTraits, Instance

# -----------------------------------------------------------------------------
# Dummy socket class
# -----------------------------------------------------------------------------


class DummySocket(HasTraits):
    """A dummy socket implementing (part of) the zmq.asyncio.Socket interface."""

    context = Instance(zmq.asyncio.Context)

    def _context_default(self):
        return zmq.asyncio.Context()

    # -------------------------------------------------------------------------
    # Socket interface
    # -------------------------------------------------------------------------

    def __init__(self, is_shell, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.is_shell = is_shell
        self.on_recv = None
        if is_shell:
            self.in_send_stream, self.in_receive_stream = create_memory_object_stream[dict](
                max_buffer_size=inf
            )
            self.out_send_stream, self.out_receive_stream = create_memory_object_stream[dict](
                max_buffer_size=inf
            )

    def put(self, msg):
        self.in_send_stream.send_nowait(msg)

    async def get(self):
        return await self.out_receive_stream.receive()

    async def recv_multipart(self, flags=0, copy=True, track=False):
        """Recv a multipart message."""
        return await self.in_receive_stream.receive()

    def send_multipart(self, msg_parts, flags=0, copy=True, track=False):
        """Send a multipart message."""
        if self.is_shell:
            self.out_send_stream.send_nowait(msg_parts)
        if self.on_recv is not None:
            self.on_recv(msg_parts)

    def flush(self, timeout=1.0):
        """no-op to comply with stream API"""

    async def poll(self, timeout=0):
        assert timeout == 0
        statistics = self.in_receive_stream.statistics()
        return statistics.current_buffer_used != 0

    def close(self):
        if self.is_shell:
            self.in_send_stream.close()
            self.in_receive_stream.close()
            self.out_send_stream.close()
            self.out_receive_stream.close()

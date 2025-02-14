"""A thread for a subshell."""

from threading import current_thread

import zmq
import zmq_anyio

from .thread import BaseThread


class SubshellThread(BaseThread):
    """A thread for a subshell."""

    def __init__(self, subshell_id: str, **kwargs):
        """Initialize the thread."""
        super().__init__(name=f"subshell-{subshell_id}", **kwargs)

        # Inproc PAIR socket, for communication with shell channel thread.
        self._pair_socket: zmq_anyio.Socket | None = None

    async def create_pair_socket(
        self,
        context: zmq.Context,  # type: ignore[type-arg]
        address: str,
    ) -> None:
        """Create inproc PAIR socket, for communication with shell channel thread.

        Should be called from this thread, so usually via start_soon before the
        thread is started.
        """
        assert current_thread() == self
        self._pair_socket = zmq_anyio.Socket(context, zmq.PAIR)
        self._pair_socket.connect(address)
        self.start_soon(self._pair_socket.start)

    def run(self) -> None:
        try:
            super().run()
        finally:
            if self._pair_socket is not None:
                self._pair_socket.close()
                self._pair_socket = None

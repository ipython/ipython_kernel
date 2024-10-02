"""A thread for a shell channel."""
from .subshell_manager import SubshellManager
from .thread import SHELL_CHANNEL_THREAD_NAME, BaseThread


class ShellChannelThread(BaseThread):
    """A thread for a shell channel.

    Communicates with shell/subshell threads via pairs of ZMQ inproc sockets.
    """

    def __init__(self, context, shell_socket, **kwargs):
        """Initialize the thread."""
        super().__init__(name=SHELL_CHANNEL_THREAD_NAME, **kwargs)
        self._manager: SubshellManager | None = None
        self._context = context
        self._shell_socket = shell_socket

    @property
    def manager(self):
        # Lazy initialisation.
        if self._manager is None:
            self._manager = SubshellManager(self._context, self._shell_socket)
        return self._manager

    def run(self):
        """Run the thread."""
        try:
            super().run()
        finally:
            if self._manager:
                self._manager.close()

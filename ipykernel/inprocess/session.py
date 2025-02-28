from typing import Any

from jupyter_client.session import Session as _Session


class Session(_Session):
    # superclass is not async.
    async def recv(  # type: ignore[override]
        self, socket, mode: int = 0, content: bool = True, copy=True
    ) -> Any:
        """
        mode, content, copy have no effect, but are present for superclass compatibility

        """
        return await socket.arecv_multipart().wait()

    def send(
        self,
        socket,
        msg_or_type,
        content=None,
        parent=None,
        ident=None,
        buffers=None,
        track=False,
        header=None,
        metadata=None,
    ):
        if isinstance(msg_or_type, str):
            msg = self.msg(
                msg_or_type,
                content=content,
                parent=parent,
                header=header,
                metadata=metadata,
            )
        else:
            # We got a Message or message dict, not a msg_type so don't
            # build a new Message.
            msg = msg_or_type
            buffers = buffers or msg.get("buffers", [])

        socket.send_multipart(msg)
        return msg

    def feed_identities(self, msg, copy=True):
        return "", msg

    def deserialize(self, msg, content=True, copy=True):
        return msg

import json
import os
import threading
import time

import pytest
from jupyter_core.paths import secure_write
from traitlets.config.loader import Config

from ipykernel.kernelapp import IPKernelApp

from .conftest import MockKernel
from .utils import TemporaryWorkingDirectory


@pytest.mark.skipif(os.name == "nt", reason="requires ipc")
def test_init_ipc_socket():
    app = IPKernelApp(transport="ipc")
    app.init_sockets()
    app.cleanup_connection_file()
    app.close()


def test_blackhole():
    app = IPKernelApp()
    app.no_stderr = True
    app.no_stdout = True
    app.init_blackhole()
    app.close()


def test_start_app():
    app = IPKernelApp()
    app.kernel = MockKernel()

    def trigger_stop():
        time.sleep(1)
        app.stop()

    thread = threading.Thread(target=trigger_stop)
    t0 = time.time()
    thread.start()
    app.init_sockets()
    app.start()
    t1 = time.time()
    assert t1 - t0 >= 1
    app.cleanup_connection_file()
    app.kernel.destroy()
    app.close()


@pytest.mark.skipif(os.name == "nt", reason="permission errors on windows")
def test_merge_connection_file():
    cfg = Config()
    with TemporaryWorkingDirectory() as d:
        cfg.ProfileDir.location = d
        cf = os.path.join(d, "kernel.json")
        initial_connection_info = {
            "ip": "*",
            "transport": "tcp",
            "shell_port": 0,
            "hb_port": 0,
            "iopub_port": 0,
            "stdin_port": 0,
            "control_port": 53555,
            "key": "abc123",
            "signature_scheme": "hmac-sha256",
            "kernel_name": "My Kernel",
        }
        # We cannot use connect.write_connection_file since
        # it replaces port number 0 with a random port
        # and we want IPKernelApp to do that replacement.
        with secure_write(cf) as f:
            json.dump(initial_connection_info, f)
        assert os.path.exists(cf)

        app = IPKernelApp(config=cfg, connection_file=cf)

        # Calling app.initialize() does not work in the test, so we call the relevant functions that initialize() calls
        # We must pass in an empty argv, otherwise the default is to try to parse the test runner's argv
        super(IPKernelApp, app).initialize(argv=[""])
        app.init_connection_file()
        app.init_sockets()
        app.init_heartbeat()
        app.write_connection_file()

        # Initialize should have merged the actual connection info
        # with the connection info in the file
        assert cf == app.abs_connection_file
        assert os.path.exists(cf)

        with open(cf) as f:
            new_connection_info = json.load(f)

        # ports originally set as 0 have been replaced
        for port in ("shell", "hb", "iopub", "stdin"):
            key = f"{port}_port"
            # We initially had the port as 0
            assert initial_connection_info[key] == 0
            # the port is not 0 now
            assert new_connection_info[key] > 0
            # the port matches the port the kernel actually used
            assert new_connection_info[key] == getattr(app, key), f"{key}"
            del new_connection_info[key]
            del initial_connection_info[key]

        # The wildcard ip address was also replaced
        assert new_connection_info["ip"] != "*"
        del new_connection_info["ip"]
        del initial_connection_info["ip"]

        # everything else in the connection file is the same
        assert initial_connection_info == new_connection_info

        app.close()
        os.remove(cf)


@pytest.mark.skip("Something wrong with CI")
@pytest.mark.parametrize("anyio_backend", ["trio"])
async def test_trio_loop(anyio_backend):
    import trio

    app = IPKernelApp(trio_loop=True)

    async def trigger_stop():
        await trio.sleep(1)
        app.stop()

    app.kernel = MockKernel()
    app.init_sockets()
    async with trio.open_nursery() as nursery:
        nursery.start_soon(lambda: app._start("trio"))
        nursery.start_soon(trigger_stop)
    app.cleanup_connection_file()
    app.kernel.destroy()
    app.close()

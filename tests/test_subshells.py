"""Test kernel subshells."""

# Copyright (c) IPython Development Team.
# Distributed under the terms of the Modified BSD License.
from __future__ import annotations

import platform
import time
from datetime import datetime, timedelta

import pytest
from jupyter_client.blocking.client import BlockingKernelClient

from .utils import TIMEOUT, get_replies, get_reply, new_kernel

# Helpers


def create_subshell_helper(kc: BlockingKernelClient):
    msg = kc.session.msg("create_subshell_request")
    kc.control_channel.send(msg)
    msg_id = msg["header"]["msg_id"]
    reply = get_reply(kc, msg_id, TIMEOUT, channel="control")
    return reply["content"]


def delete_subshell_helper(kc: BlockingKernelClient, subshell_id: str):
    msg = kc.session.msg("delete_subshell_request", {"subshell_id": subshell_id})
    kc.control_channel.send(msg)
    msg_id = msg["header"]["msg_id"]
    reply = get_reply(kc, msg_id, TIMEOUT, channel="control")
    return reply["content"]


def list_subshell_helper(kc: BlockingKernelClient):
    msg = kc.session.msg("list_subshell_request")
    kc.control_channel.send(msg)
    msg_id = msg["header"]["msg_id"]
    reply = get_reply(kc, msg_id, TIMEOUT, channel="control")
    return reply["content"]


def execute_request_subshell_id(
    kc: BlockingKernelClient, code: str, subshell_id: str | None, terminator: str = "\n"
):
    msg = kc.session.msg("execute_request", {"code": code})
    msg["header"]["subshell_id"] = subshell_id
    msg_id = msg["msg_id"]
    kc.shell_channel.send(msg)
    stdout = ""
    while True:
        msg = kc.get_iopub_msg()
        # Get the stream messages corresponding to msg_id
        if (
            msg["msg_type"] == "stream"
            and msg["parent_header"]["msg_id"] == msg_id
            and msg["content"]["name"] == "stdout"
        ):
            stdout += msg["content"]["text"]
            if stdout.endswith(terminator):
                break
    return stdout.strip()


def execute_thread_count(kc: BlockingKernelClient) -> int:
    code = "import threading as t; print(t.active_count())"
    return int(execute_request_subshell_id(kc, code, None))


def execute_thread_ids(kc: BlockingKernelClient, subshell_id: str | None = None) -> tuple[str, str]:
    code = "import threading as t; print(t.get_ident(), t.main_thread().ident)"
    return execute_request_subshell_id(kc, code, subshell_id).split()


# Tests


def test_supported():
    with new_kernel() as kc:
        msg_id = kc.kernel_info()
        reply = get_reply(kc, msg_id, TIMEOUT)
        assert "supported_features" in reply["content"]
        assert "kernel subshells" in reply["content"]["supported_features"]


def test_subshell_id_lifetime():
    with new_kernel() as kc:
        assert list_subshell_helper(kc)["subshell_id"] == []
        subshell_id = create_subshell_helper(kc)["subshell_id"]
        assert list_subshell_helper(kc)["subshell_id"] == [subshell_id]
        delete_subshell_helper(kc, subshell_id)
        assert list_subshell_helper(kc)["subshell_id"] == []


def test_delete_non_existent():
    with new_kernel() as kc:
        reply = delete_subshell_helper(kc, "unknown_subshell_id")
        assert reply["status"] == "error"
        assert "evalue" in reply


def test_thread_counts():
    with new_kernel() as kc:
        nthreads = execute_thread_count(kc)

        subshell_id = create_subshell_helper(kc)["subshell_id"]
        nthreads2 = execute_thread_count(kc)
        assert nthreads2 > nthreads

        delete_subshell_helper(kc, subshell_id)
        nthreads3 = execute_thread_count(kc)
        assert nthreads3 == nthreads


def test_thread_ids():
    with new_kernel() as kc:
        subshell_id = create_subshell_helper(kc)["subshell_id"]

        thread_id, main_thread_id = execute_thread_ids(kc)
        assert thread_id == main_thread_id

        thread_id, main_thread_id = execute_thread_ids(kc, subshell_id)
        assert thread_id != main_thread_id

        delete_subshell_helper(kc, subshell_id)


@pytest.mark.xfail(
    strict=False, reason="this randomly fail and make downstream testing less useful"
)
@pytest.mark.parametrize("are_subshells", [(False, True), (True, False), (True, True)])
@pytest.mark.parametrize("overlap", [True, False])
def test_run_concurrently_sequence(are_subshells, overlap):
    with new_kernel() as kc:
        subshell_ids = [
            create_subshell_helper(kc)["subshell_id"] if is_subshell else None
            for is_subshell in are_subshells
        ]

        # Import time module before running time-sensitive subshell code
        # and use threading.Barrier to synchronise start of subshell code.
        execute_request_subshell_id(
            kc, "import threading as t, time; b=t.Barrier(2); print('ok')", None
        )

        sleep = 0.2
        if overlap:
            codes = [
                f"b.wait(); start0=True; end0=False; time.sleep({sleep}); end0=True",
                f"b.wait(); time.sleep({sleep / 2}); assert start0; assert not end0; time.sleep({sleep}); assert end0",
            ]
        else:
            codes = [
                f"b.wait(); start0=True; end0=False; time.sleep({sleep}); assert end1",
                f"b.wait(); time.sleep({sleep / 2}); assert start0; assert not end0; end1=True",
            ]

        msgs = []
        for subshell_id, code in zip(subshell_ids, codes, strict=False):
            msg = kc.session.msg("execute_request", {"code": code})
            msg["header"]["subshell_id"] = subshell_id
            kc.shell_channel.send(msg)
            msgs.append(msg)

        replies = get_replies(kc, [msg["msg_id"] for msg in msgs])

        for subshell_id in subshell_ids:
            if subshell_id:
                delete_subshell_helper(kc, subshell_id)

        for reply in replies:
            assert reply["content"]["status"] == "ok", reply


@pytest.mark.parametrize("include_main_shell", [True, False])
def test_run_concurrently_timing(include_main_shell):
    with new_kernel() as kc:
        subshell_ids = [
            None if include_main_shell else create_subshell_helper(kc)["subshell_id"],
            create_subshell_helper(kc)["subshell_id"],
        ]

        # Import time module before running time-sensitive subshell code
        # and use threading.Barrier to synchronise start of subshell code.
        execute_request_subshell_id(
            kc, "import threading as t, time; b=t.Barrier(2); print('ok')", None
        )

        times = (0.2, 0.2)
        # Prepare messages, times are sleep times in seconds.
        # Identical times for both subshells is a harder test as preparing and sending
        # the execute_reply messages may overlap.
        msgs = []
        for id, sleep in zip(subshell_ids, times, strict=False):
            code = f"b.wait(); time.sleep({sleep})"
            msg = kc.session.msg("execute_request", {"code": code})
            msg["header"]["subshell_id"] = id
            msgs.append(msg)

        # Send messages
        start = datetime.now()
        for msg in msgs:
            kc.shell_channel.send(msg)

        _ = get_replies(kc, [msg["msg_id"] for msg in msgs])
        end = datetime.now()

        for subshell_id in subshell_ids:
            if subshell_id:
                delete_subshell_helper(kc, subshell_id)

        duration = end - start
        assert duration >= timedelta(seconds=max(times))
        # Care is needed with this test as runtime conditions such as gathering
        # coverage can slow it down causing the following assert to fail.
        # The sleep time of 0.2 is empirically determined to run OK in CI, but
        # consider increasing it if the following fails.
        assert duration < timedelta(seconds=sum(times))


@pytest.mark.xfail(strict=False, reason="subshell still sometime give different results")
def test_execution_count():
    with new_kernel() as kc:
        subshell_id = create_subshell_helper(kc)["subshell_id"]

        # Import time module before running time-sensitive subshell code
        # and use threading.Barrier to synchronise start of subshell code.
        execute_request_subshell_id(
            kc, "import threading as t, time; b=t.Barrier(2); print('ok')", None
        )

        # Prepare messages
        times = (0.2, 0.1, 0.4, 0.15)  # Sleep seconds
        msgs = []
        for i, (id, sleep) in enumerate(
            zip((None, subshell_id, None, subshell_id), times, strict=False)
        ):
            code = f"b.wait(); time.sleep({sleep})" if i < 2 else f"time.sleep({sleep})"
            msg = kc.session.msg("execute_request", {"code": code})
            msg["header"]["subshell_id"] = id
            msgs.append(msg)

        for msg in msgs:
            kc.shell_channel.send(msg)

        # Wait for replies, may be in any order.
        replies = get_replies(kc, [msg["msg_id"] for msg in msgs])

        delete_subshell_helper(kc, subshell_id)

        execution_counts = [r["content"]["execution_count"] for r in replies]
        ec = execution_counts[0]
        assert execution_counts == [ec, ec - 1, ec + 2, ec + 1]


def test_create_while_execute():
    with new_kernel() as kc:
        # Send request to execute code on main subshell.
        msg = kc.session.msg("execute_request", {"code": "import time; time.sleep(0.05)"})
        kc.shell_channel.send(msg)

        # Create subshell via control channel.
        control_msg = kc.session.msg("create_subshell_request")
        kc.control_channel.send(control_msg)
        control_reply = get_reply(kc, control_msg["header"]["msg_id"], TIMEOUT, channel="control")
        subshell_id = control_reply["content"]["subshell_id"]
        control_date = control_reply["header"]["date"]

        # Get result message from main subshell.
        shell_date = get_reply(kc, msg["msg_id"])["header"]["date"]

        delete_subshell_helper(kc, subshell_id)

        assert control_date < shell_date


@pytest.mark.skipif(
    platform.python_implementation() == "PyPy",
    reason="does not work on PyPy",
)
def test_shutdown_with_subshell():
    # Based on test_kernel.py::test_shutdown
    with new_kernel() as kc:
        km = kc.parent
        subshell_id = create_subshell_helper(kc)["subshell_id"]
        assert list_subshell_helper(kc)["subshell_id"] == [subshell_id]
        kc.shutdown()
        for _ in range(100):  # 10 s timeout
            if km.is_alive():
                time.sleep(0.1)
            else:
                break
        assert not km.is_alive()

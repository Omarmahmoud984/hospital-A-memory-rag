#!/usr/bin/env python3

"""
agent/agent.py
--------------
MediCore Hospital Network -- MCP Client
"""

import argparse
import asyncio
import json
import os
import sys


if sys.platform == "win32":
    asyncio.set_event_loop_policy(
        asyncio.WindowsProactorEventLoopPolicy()
    )


sys.path.insert(0, os.path.dirname(__file__))

from mcp_protocol import JsonRpcEndpoint


DEFAULT_SERVER_ARGS = [
    sys.executable,
    "-u",
    os.path.join(
        os.path.dirname(__file__),
        "..",
        "mcp_server",
        "MCP.py"
    ),
]

SERVER_ARGS = (
    os.environ.get("MCP_SERVER_CMD", "").split()
    or DEFAULT_SERVER_ARGS
)
class MediCoreAgent:

    def __init__(self, auto_confirm=False):

        self.proc = None
        self.endpoint = None
        self._reader_task = None
        self.server_capabilities = {}
        self.tools = []

        self.auto_confirm = auto_confirm
        self.scripted_answers = []


    async def start(self):

        print("PYTHON USED:", sys.executable)

        print("Starting MCP Server...")

        self.proc = await asyncio.create_subprocess_exec(
            *SERVER_ARGS,

            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )


        if self.proc.stdin is None or self.proc.stdout is None:
            raise RuntimeError(
                "MCP server pipes were not created"
            )


        self.endpoint = JsonRpcEndpoint(
            self.proc.stdout,
            self.proc.stdin,

            request_handler=self._handle_server_request,
            notification_handler=self._handle_server_notification,

            name="medicore-client",
        )

        self._stderr_task = asyncio.create_task(
                    self._read_server_errors()
                )

        self._reader_task = asyncio.create_task(
            self.endpoint.run()
        )


        print("Initializing MCP protocol...")


        result = await self.endpoint.send_request(
            "initialize",
            {
                "protocolVersion": "2025-06-18",

                "capabilities": {
                    "elicitation": {},
                    "sampling": {}
                },

                "clientInfo": {
                    "name": "medicore-agent",
                    "version": "0.1.0"
                }
            }
        )


        self.server_capabilities = (
            result.get("capabilities", {})
        )


        await self.endpoint.send_notification(
            "initialized",
            {}
        )
        await self._refresh_tools()


        print("MCP Client Ready")



    async def _read_server_errors(self):

        if self.proc and self.proc.stderr:

            while True:

                line = await self.proc.stderr.readline()

                if not line:
                    break

                print(
                    "[SERVER]",
                    line.decode(errors="ignore").strip()
                )

    async def stop(self):

        if self._reader_task:
            self._reader_task.cancel()
            try:
                await self._reader_task
            except asyncio.CancelledError:
                pass

        if self._stderr_task:
            self._stderr_task.cancel()
            try:
                await self._stderr_task
            except asyncio.CancelledError:
                pass

        if self.proc:

            if self.proc.stdin:
                self.proc.stdin.close()

            self.proc.terminate()

            try:
                await asyncio.wait_for(self.proc.wait(), timeout=3)
            except asyncio.TimeoutError:
                self.proc.kill()
                await self.proc.wait()
            await asyncio.sleep(0.1)
    async def _refresh_tools(self):

            result = await self.endpoint.send_request(
                "tools/list",
                {}
            )

            self.tools = result.get(
                "tools",
                []
            )



    async def _handle_server_request(
            self,
            method,
            params
    ):

        if method == "elicitation/create":

            return {
                "action": "accept",
                "content": {
                    "confirm": True
                }
            }


        if method == "sampling/createMessage":

            return {
                "role": "assistant",
                "content": {
                    "type": "text",
                    "text": "offline response"
                }
            }


        raise Exception(
            f"Unsupported request {method}"
        )



    async def _handle_server_notification(
            self,
            method,
            params
    ):

        print(
            "[NOTIFICATION]",
            method,
            params
        )



    async def call_tool(
            self,
            name,
            arguments
    ):

        return await self.endpoint.send_request(
            "tools/call",
            {
                "name": name,
                "arguments": arguments
            }
        )



def decide_next_tool_call(message):

    text = message.lower()


    if "icu beds" in text:

        return {
            "name": "get_available_icu_beds",
            "arguments": {}
        }


    if "patient details" in text:

        return {
            "name": "get_patient_details",
            "arguments": {
                "patient_id": 1
            }
        }


    if "admission" in text:

        return {
            "name": "create_admission",
            "arguments": {
                "admission": {
                    "patient_id": 1,
                    "doctor_id": 1,
                    "room_id": None,
                    "status": "Active"
                }
            }
        }


    if "capacity" in text:

        return {
            "name": "get_hospital_capacity",
            "arguments": {}
        }


    return None

DEMO_SCRIPT = [

    "Which ICU beds are available?",

    "Get patient details",

    "Create admission",

    "Get hospital capacity"

]



async def run_demo():

    agent = MediCoreAgent(
        auto_confirm=True
    )


    await agent.start()


    print(
        "\nCapabilities:"
    )

    print(
        json.dumps(
            agent.server_capabilities,
            indent=2
        )
    )


    print(
        "\nTools:"
    )

    print(
        [
            t["name"]
            for t in agent.tools
        ]
    )


    for msg in DEMO_SCRIPT:

        print(
            "\nUSER:",
            msg
        )


        call = decide_next_tool_call(
            msg
        )


        if call:

            print(
                "Calling:",
                call["name"]
            )


            result = await agent.call_tool(
                call["name"],
                call["arguments"]
            )


            print(
                "RESULT:",
                result
            )



    await agent.stop()




async def run_interactive():

    agent = MediCoreAgent()

    await agent.start()


    while True:

        msg = input(
            "\nyou> "
        )


        if msg == "quit":
            break


        call = decide_next_tool_call(
            msg
        )


        if call:

            result = await agent.call_tool(
                call["name"],
                call["arguments"]
            )

            print(result)



    await agent.stop()



if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--demo",
        action="store_true"
    )

    args = parser.parse_args()


    asyncio.run(
        run_demo()
        if args.demo
        else run_interactive()
    )

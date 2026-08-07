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


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from mcp_protocol import JsonRpcEndpoint
from memory.short_term import ShortTermMemory, Message, MessageRole
from memory.scratchpad import Scratchpad, PlanStep
from memory.episodic import EpisodicMemory, EventCategory
from memory.router import PromoteOrDropRouter
from memory.semantic import SemanticMemory
from memory.consolidation import SemanticConsolidationEngine


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

        # --- Initialize 4-Layer Memory Subsystem ---
        self.episodic_memory = EpisodicMemory()
        self.semantic_memory = SemanticMemory()
        self.router = PromoteOrDropRouter(episodic_memory=self.episodic_memory)
        self.short_term_memory = ShortTermMemory(
            capacity=10,
            overflow_callback=self.router.evaluate_and_route,
        )
        self.scratchpad = Scratchpad()
        self.consolidation_engine = SemanticConsolidationEngine(
            episodic_memory=self.episodic_memory,
            semantic_memory=self.semantic_memory,
        )


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

    async def execute_agent_turn(self, user_text: str):
        """
        Executes a full turn demonstrating Memory & Tool interaction:
        1. Logs user input to Short-Term Memory.
        2. Updates Scratchpad plan & goals.
        3. Queries Semantic/Episodic memory for active knowledge.
        4. Calls MCP tool if appropriate.
        5. Logs tool output to Short-Term Memory & Scratchpad.
        6. Runs periodic consolidation pass.
        """
        # Step 1: Push user message to Short-Term Memory
        user_msg = Message(role=MessageRole.USER, content=user_text)
        self.short_term_memory.add_message(user_msg)

        # Step 2: Update Scratchpad working state
        self.scratchpad.set_goal(f"Address user query: '{user_text}'")
        self.scratchpad.add_reasoning_step(f"Evaluating intent for input: {user_text}")

        # Step 3: Query Semantic Memory & Episodic Memory
        recalled_facts = self.semantic_memory.search_knowledge(user_text)
        recent_episodes = self.episodic_memory.get_recent_events(limit=3)

        print(f"\n[MEMORY CONTEXT]")
        print(f"  - Active Short-Term Buffer: {self.short_term_memory.count} messages ({self.short_term_memory.total_tokens} tokens)")
        print(f"  - Recalled Semantic Facts: {[f.object_value for f in recalled_facts]}")
        print(f"  - Recent Episodic Events: {[e.summary for e in recent_episodes]}")

        # Step 4: Determine tool call
        call = decide_next_tool_call(user_text)
        result = None
        if call:
            print(f"  - Agent Scratchpad Plan: Step 1 -> Execute tool '{call['name']}'")
            self.scratchpad.set_execution_plan([
                PlanStep(step_number=1, description=f"Call tool {call['name']}", status="in_progress")
            ])

            tool_msg = Message(role=MessageRole.TOOL_CALL, content=json.dumps(call))
            self.short_term_memory.add_message(tool_msg)

            # Call MCP tool
            result = await self.call_tool(call["name"], call["arguments"])

            obs_msg = Message(role=MessageRole.TOOL_OBSERVATION, content=json.dumps(result))
            self.short_term_memory.add_message(obs_msg)

            self.scratchpad.store_partial_tool_result(call["name"], result)
            self.scratchpad.update_plan_step(step_number=1, status="completed", result=str(result))

        # Step 5: Consolidation Pass over Episodic Memory
        cons_report = self.consolidation_engine.run_consolidation(min_importance=0.5)
        print(f"  - Periodic Consolidation: {cons_report.facts_created} created, {cons_report.facts_updated} updated, {cons_report.facts_superseded} superseded")

        return result



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
        print("\n=======================================================")
        print("USER:", msg)
        result = await agent.execute_agent_turn(msg)
        print("AGENT RESULT:", result)



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

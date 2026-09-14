# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Copyright (c) 2026 DarbotLabs. (LMTuner modifications)
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""lmcli mcp — serve MCP 2.0 (2026-07-28) stateless Streamable HTTP + stdio."""

from argparse import ArgumentParser

from olive.cli.base import BaseOliveCLICommand, add_telemetry_options


class McpCommand(BaseOliveCLICommand):
    @staticmethod
    def register_subcommand(parser: ArgumentParser):
        sub_parser = parser.add_parser(
            "mcp",
            help="Serve MCP 2.0 (2026-07-28) on stdio or stateless Streamable HTTP at /mcp.",
        )
        sub_parser.add_argument(
            "--transport",
            choices=["stdio", "http"],
            default="stdio",
            help="stdio JSON-RPC (default) or Streamable HTTP POST /mcp.",
        )
        sub_parser.add_argument("--host", default="127.0.0.1", help="Bind address for HTTP (default 127.0.0.1).")
        sub_parser.add_argument("--port", type=int, default=8765, help="Bind port for HTTP (default 8765).")
        add_telemetry_options(sub_parser)
        sub_parser.set_defaults(func=McpCommand)

    def run(self):
        from olive.protocols import MCP_HTTP_URL, MCP_PROTOCOL_VERSION, MCP_SPEC_URL, MCP_TASKS_URL
        from olive.protocols.mcp import serve_http, serve_stdio

        print(f"LMTuner MCP {MCP_PROTOCOL_VERSION} server")
        print(f"  spec: {MCP_SPEC_URL}")
        print(f"  streamable HTTP: {MCP_HTTP_URL}")
        print(f"  tasks: {MCP_TASKS_URL}")
        if self.args.transport == "http":
            print(f"  endpoint: http://{self.args.host}:{self.args.port}/mcp")
            print("  every POST is self-contained; no initialize handshake; no Mcp-Session-Id.")
            serve_http(self.args.host, self.args.port)
        else:
            print("  transport: stdio JSON-RPC")
            serve_stdio()

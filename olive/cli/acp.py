# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Copyright (c) 2026 DarbotLabs. (LMTuner modifications)
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""lmcli acp — serve the Zed Agent Client Protocol."""

from argparse import ArgumentParser

from olive.cli.base import BaseOliveCLICommand, add_telemetry_options


class AcpCommand(BaseOliveCLICommand):
    @staticmethod
    def register_subcommand(parser: ArgumentParser):
        sub_parser = parser.add_parser(
            "acp",
            help="Serve the Zed Agent Client Protocol (stdio or Streamable HTTP/WS on /acp).",
        )
        sub_parser.add_argument(
            "--transport",
            choices=["stdio", "http"],
            default="stdio",
            help="stdio JSON-RPC (default) or Streamable HTTP + WebSocket on /acp.",
        )
        sub_parser.add_argument("--host", default="127.0.0.1", help="Bind address for HTTP (default 127.0.0.1).")
        sub_parser.add_argument("--port", type=int, default=8000, help="Bind port for HTTP (default 8000).")
        add_telemetry_options(sub_parser)
        sub_parser.set_defaults(func=AcpCommand)

    def run(self):
        from olive.protocols import ACP_HTTP_RFD_URL, ACP_PYTHON_SDK_URL, ACP_SPEC_URL
        from olive.protocols.acp import serve_http, serve_stdio

        print("LMTuner ACP server")
        print(f"  spec: {ACP_SPEC_URL}")
        print(f"  streamable HTTP RFD: {ACP_HTTP_RFD_URL}")
        print(f"  python SDK: {ACP_PYTHON_SDK_URL}")
        if self.args.transport == "http":
            print(f"  endpoint: http://{self.args.host}:{self.args.port}/acp")
            print("  initialize POST -> 200 + Acp-Connection-Id; other POST -> 202; GET SSE; DELETE ends.")
            print("  HTTP/2 required (Hypercorn). Do not use uvicorn for Streamable HTTP.")
            serve_http(self.args.host, self.args.port)
        else:
            print("  transport: stdio JSON-RPC")
            serve_stdio()

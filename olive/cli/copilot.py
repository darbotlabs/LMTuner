# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Copyright (c) 2024 DarbotLabs. (LMTuner modifications)
# Licensed under the MIT License.
# --------------------------------------------------------------------------
import json
import logging

from olive.cli.base import BaseOliveCLICommand

logger = logging.getLogger(__name__)


class CopilotCommand(BaseOliveCLICommand):
    """Command for GitHub Copilot integration and assistance."""

    @staticmethod
    def register_subcommand(parser):
        sub_parser = parser.add_parser(
            "copilot",
            help="GitHub Copilot integration and AI assistance for LMTuner optimization",
        )
        sub_parser.add_argument(
            "--info",
            action="store_true",
            help="Display information about LMTuner framework and GitHub Copilot integration",
        )
        sub_parser.add_argument(
            "--suggest",
            type=str,
            metavar="MODEL_TYPE",
            help="Get optimization suggestions for a specific model type (e.g., 'llama', 'phi', 'qwen')",
        )
        sub_parser.add_argument(
            "--best-practices",
            action="store_true",
            help="Display best practices for model optimization with LMTuner",
        )
        sub_parser.add_argument(
            "--example",
            type=str,
            metavar="COMMAND",
            choices=["optimize", "finetune", "quantize", "capture-onnx"],
            help="Show example usage for a specific command",
        )
        sub_parser.set_defaults(func=CopilotCommand)

    def run(self):
        """Execute the copilot command."""
        if self.args.info:
            self._show_info()
        elif self.args.suggest:
            self._suggest_optimization(self.args.suggest)
        elif self.args.best_practices:
            self._show_best_practices()
        elif self.args.example:
            self._show_example(self.args.example)
        else:
            print("LMTuner Copilot Integration")
            print("=" * 60)
            print("Use --help to see available options")
            print("\nQuick start:")
            print("  lmcli copilot --info              # Framework information")
            print("  lmcli copilot --suggest llama     # Get optimization suggestions")
            print("  lmcli copilot --best-practices    # View best practices")
            print("  lmcli copilot --example optimize  # Show command examples")

    def _show_info(self):
        """Display information about LMTuner and GitHub Copilot integration."""
        info = {
            "framework": "LMTuner",
            "description": "AI Model Optimization Toolkit for ONNX Runtime (Fork of Microsoft Olive)",
            "version": self._get_version(),
            "features": [
                "Automatic model optimization with 40+ built-in techniques",
                "Fine-tuning with PEFT (Parameter-Efficient Fine-Tuning)",
                "Model quantization (INT4, INT8, FP16)",
                "ONNX graph capture and optimization",
                "Multi-hardware support (CPU, GPU, NPU)",
                "GitHub Copilot integration for AI-assisted optimization",
            ],
            "supported_architectures": [
                "Llama",
                "Phi",
                "Qwen",
                "Gemma",
                "Mistral",
                "GPT",
                "BERT",
                "T5",
                "and more via HuggingFace",
            ],
            "github_copilot_features": [
                "AI-powered optimization suggestions",
                "Best practices recommendations",
                "Interactive command examples",
                "Framework guidance and documentation",
            ],
        }

        print("LMTuner Framework Information")
        print("=" * 60)
        print(json.dumps(info, indent=2))
        print("\n" + "=" * 60)
        print("\nFor more information, visit:")
        print("  Documentation: https://microsoft.github.io/Olive/")
        print("  GitHub: https://github.com/darbotlabs/LMTuner")

    def _suggest_optimization(self, model_type: str):
        """Provide optimization suggestions for a specific model type."""
        print(f"\n🤖 LMTuner Optimization Suggestions for '{model_type}' models")
        print("=" * 60)

        suggestions = {
            "llama": {
                "recommended_precision": "int4",
                "optimization_passes": ["GPTQ quantization", "ONNX graph optimization", "Fusion passes"],
                "example_command": """lmcli optimize \\
    --model_name_or_path meta-llama/Llama-2-7b-hf \\
    --precision int4 \\
    --output_path models/llama-7b-int4""",
                "tips": [
                    "Use INT4 quantization for best performance/quality balance",
                    "Enable graph fusion for faster inference",
                    "Consider using ONNX Runtime GenAI for deployment",
                ],
            },
            "phi": {
                "recommended_precision": "int4",
                "optimization_passes": ["GPTQ quantization", "ONNX optimization", "Layer fusion"],
                "example_command": """lmcli optimize \\
    --model_name_or_path microsoft/phi-2 \\
    --precision int4 \\
    --output_path models/phi-2-int4""",
                "tips": [
                    "Phi models benefit from aggressive quantization",
                    "Use smaller batch sizes for memory efficiency",
                    "Test on your target hardware before deployment",
                ],
            },
            "qwen": {
                "recommended_precision": "int4",
                "optimization_passes": ["GPTQ quantization", "Graph optimization"],
                "example_command": """lmcli optimize \\
    --model_name_or_path Qwen/Qwen2.5-0.5B-Instruct \\
    --precision int4 \\
    --output_path models/qwen-int4""",
                "tips": [
                    "Qwen models support multilingual optimization",
                    "Consider fine-tuning before quantization for better quality",
                    "Use auto-opt for automatic pass selection",
                ],
            },
        }

        model_suggestions = suggestions.get(
            model_type.lower(),
            {
                "recommended_precision": "int4 or int8",
                "optimization_passes": ["Quantization", "ONNX optimization"],
                "example_command": f"""lmcli optimize \\
    --model_name_or_path <your-model> \\
    --precision int4 \\
    --output_path models/optimized""",
                "tips": [
                    "Start with auto-opt for automatic optimization",
                    "Test different precision levels (int4, int8, fp16)",
                    "Validate accuracy after optimization",
                ],
            },
        )

        print(f"\n📊 Recommended Precision: {model_suggestions['recommended_precision']}")
        print(f"\n🔧 Optimization Passes:")
        for pass_name in model_suggestions["optimization_passes"]:
            print(f"  • {pass_name}")

        print(f"\n💡 Example Command:")
        print(model_suggestions["example_command"])

        print(f"\n✨ Tips:")
        for tip in model_suggestions["tips"]:
            print(f"  • {tip}")

    def _show_best_practices(self):
        """Display best practices for model optimization."""
        print("\n📚 LMTuner Best Practices for Model Optimization")
        print("=" * 60)

        practices = [
            {
                "category": "Pre-Optimization",
                "items": [
                    "Understand your target hardware constraints (CPU, GPU, NPU)",
                    "Benchmark baseline model performance before optimization",
                    "Define accuracy thresholds and performance targets",
                    "Prepare representative evaluation datasets",
                ],
            },
            {
                "category": "Optimization Process",
                "items": [
                    "Start with auto-opt for automatic pass selection",
                    "Use dry-run mode to validate configurations",
                    "Test multiple precision levels (int4, int8, fp16)",
                    "Enable search to find optimal hyperparameters",
                    "Monitor accuracy degradation during quantization",
                ],
            },
            {
                "category": "Post-Optimization",
                "items": [
                    "Validate optimized model on representative data",
                    "Benchmark inference performance on target hardware",
                    "Compare accuracy metrics with baseline",
                    "Test edge cases and failure modes",
                    "Document optimization parameters for reproducibility",
                ],
            },
            {
                "category": "Deployment",
                "items": [
                    "Use ONNX Runtime for cross-platform deployment",
                    "Consider using ONNX Runtime GenAI for generative models",
                    "Implement proper error handling and fallbacks",
                    "Monitor model performance in production",
                    "Set up CI/CD for model updates",
                ],
            },
        ]

        for practice in practices:
            print(f"\n{practice['category']}:")
            for item in practice["items"]:
                print(f"  ✓ {item}")

    def _show_example(self, command: str):
        """Show example usage for a specific command."""
        print(f"\n📖 Example Usage: {command}")
        print("=" * 60)

        examples = {
            "optimize": """
# Basic optimization with auto-opt
lmcli optimize \\
    --model_name_or_path Qwen/Qwen2.5-0.5B-Instruct \\
    --precision int4 \\
    --output_path models/qwen-optimized

# Advanced optimization with search
lmcli optimize \\
    --model_name_or_path meta-llama/Llama-2-7b-hf \\
    --precision int4 \\
    --enable_search tpe \\
    --device gpu \\
    --output_path models/llama-optimized

# Dry run to validate configuration
lmcli optimize \\
    --model_name_or_path microsoft/phi-2 \\
    --precision int4 \\
    --dry_run
""",
            "finetune": """
# Fine-tune a model with PEFT
lmcli finetune \\
    --model_name_or_path meta-llama/Llama-2-7b-hf \\
    --data_name wikitext \\
    --text_template "### Text: {text}" \\
    --max_steps 100 \\
    --output_path models/llama-finetuned

# Fine-tune with custom dataset
lmcli finetune \\
    --model_name_or_path microsoft/phi-2 \\
    --data_name custom_dataset \\
    --data_files train.json \\
    --use_chat_template \\
    --output_path models/phi-finetuned
""",
            "quantize": """
# Quantize to INT4
lmcli quantize \\
    --model_name_or_path Qwen/Qwen2.5-0.5B-Instruct \\
    --algorithm gptq \\
    --precision int4 \\
    --output_path models/qwen-int4

# Quantize to INT8 with calibration
lmcli quantize \\
    --model_name_or_path meta-llama/Llama-2-7b-hf \\
    --algorithm quantization \\
    --precision int8 \\
    --data_name wikitext \\
    --output_path models/llama-int8
""",
            "capture-onnx": """
# Capture ONNX graph from HuggingFace model
lmcli capture-onnx-graph \\
    --model_name_or_path microsoft/phi-2 \\
    --task text-generation \\
    --output_path models/phi-2-onnx

# Capture with specific precision
lmcli capture-onnx-graph \\
    --model_name_or_path Qwen/Qwen2.5-0.5B-Instruct \\
    --precision fp16 \\
    --output_path models/qwen-onnx
""",
        }

        print(examples.get(command, "No example available for this command."))

    def _get_version(self):
        """Get the LMTuner version."""
        try:
            import olive

            return olive.__version__
        except Exception:
            return "unknown"

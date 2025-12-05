<div align="center">
██╗        ███╗    ███╗   ████████╗   ██╗    ██╗   ███╗    ██╗   ███████╗   ██████╗  
██║        ████╗  ████║   ╚══██╔══╝   ██║    ██║   ████╗   ██║   ██╔════╝   ██╔══██╗  
██║        ██╔████╔██║       ██║      ██║    ██║   ██╔██╗  ██║   █████╗     ██████╔╝  
██║        ██║╚██╔╝██║       ██║      ██║    ██║   ██║╚██╗ ██║   ██╔══╝     ██╔══██╗  
███████╗   ██║ ╚═╝ ██║       ██║       ╚██████╔╝   ██║ ╚█████║   ███████╗   ██║  ██║  
╚══════╝   ╚═╝     ╚═╝       ╚═╝        ╚═════╝    ╚═╝  ╚═══╝    ╚══════╝   ╚═╝  ╚═╝
  
[![GitHub](https://img.shields.io/badge/GitHub-LMTuner-blue)](https://github.com/darbotlabs/LMTuner)
[![Documentation](https://img.shields.io/website/https/microsoft.github.io/Olive?down_color=red&down_message=offline&up_message=online)](https://microsoft.github.io/Olive/)

## AI Model Optimization Toolkit for the ONNX Runtime
</div>

LMTuner (Language Model Tuner) is a fork of Microsoft Olive with significant framework improvements and enhancements. Given a model and targeted hardware, LMTuner composes the best suitable optimization techniques to output the most efficient ONNX model(s) for inferencing on the cloud or edge, while taking a set of constraints such as accuracy and latency into consideration.

## Getting Started

###  Quickstart
If you prefer using the command line directly instead of Jupyter notebooks, we've outlined the quickstart commands here.

#### 1. Install LMTuner CLI
We recommend installing LMTuner in a [virtual environment](https://docs.python.org/3/library/venv.html) or a [conda environment](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html).

```
pip install lmcli[auto-opt]
pip install transformers onnxruntime-genai
```
> [!NOTE]
> LMTuner has optional dependencies that can be installed to enable additional features. Please refer to [LMTuner package configuration](./olive/olive_config.json) for the list of extras and their dependencies.

#### 2. Automatic Optimizer

In this quickstart you'll be optimizing [Qwen/Qwen2.5-0.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct), which has many model files in the Hugging Face repo for different precisions that are not required by LMTuner.

Run the automatic optimization:

```bash
lmcli optimize \
    --model_name_or_path Qwen/Qwen2.5-0.5B-Instruct \
    --precision int4 \
    --output_path models/qwen
```

>[!TIP]
><details>
><summary>PowerShell Users</summary>
>Line continuation between Bash and PowerShell are not interchangable. If you are using PowerShell, then you can copy-and-paste the following command that uses compatible line continuation.
>
>```powershell
>lmcli optimize `
>    --model_name_or_path Qwen/Qwen2.5-0.5B-Instruct `
>    --output_path models/qwen `
>    --precision int4
>```
</details>
<br>

The automatic optimizer will:

1. Acquire the model from the the Hugging Face model repo.
1. Quantize the model to `int4` using GPTQ.
1. Capture the ONNX Graph and store the weights in an ONNX data file.
1. Optimize the ONNX Graph.

LMTuner can automatically optimize popular model *architectures* like Llama, Phi, Qwen, Gemma, etc out-of-the-box - [see detailed list here](https://huggingface.co/docs/optimum/en/exporters/onnx/overview). Also, you can optimize other model architectures by providing details on the input/outputs of the model (`io_config`).


#### 3. Inference on the ONNX Runtime

The ONNX Runtime (ORT) is a fast and light-weight cross-platform inference engine with bindings for popular programming language such as Python, C/C++, C#, Java, JavaScript, etc. ORT enables you to infuse AI models into your applications so that inference is handled on-device.

The sample chat app to run is found as [model-chat.py](https://github.com/microsoft/onnxruntime-genai/blob/main/examples/python/model-chat.py) in the [onnxruntime-genai](https://github.com/microsoft/onnxruntime-genai/) Github repository.

##  Learn more

- [LMTuner GitHub](https://github.com/darbotlabs/LMTuner)
- [Original Olive Documentation](https://microsoft.github.io/Olive)
- [GitHub Copilot Integration](https://github.com/darbotlabs/LMTuner#copilot-integration)

## GitHub Copilot Integration

LMTuner includes native GitHub Copilot integration for AI-assisted model optimization:

```bash
# Get framework information
lmcli copilot --info

# Get optimization suggestions for specific models
lmcli copilot --suggest llama
lmcli copilot --suggest phi
lmcli copilot --suggest qwen

# View best practices
lmcli copilot --best-practices

# Show command examples
lmcli copilot --example optimize
lmcli copilot --example finetune
```

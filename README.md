<div align="center">
██╗        ███╗    ███╗   ████████╗   ██╗    ██╗   ███╗    ██╗   ███████╗   ██████╗  
██║        ████╗  ████║   ╚══██╔══╝   ██║    ██║   ████╗   ██║   ██╔════╝   ██╔══██╗  
██║        ██╔████╔██║       ██║      ██║    ██║   ██╔██╗  ██║   █████╗     ██████╔╝  
██║        ██║╚██╔╝██║       ██║      ██║    ██║   ██║╚██╗ ██║   ██╔══╝     ██╔══██╗  
███████╗   ██║ ╚═╝ ██║       ██║       ╚██████╔╝   ██║ ╚█████║   ███████╗   ██║  ██║  
╚══════╝   ╚═╝     ╚═╝       ╚═╝        ╚═════╝    ╚═╝  ╚═══╝    ╚══════╝   ╚═╝  ╚═╝
  
[![PyPI release](https://img.shields.io/pypi/v/lmtuner-ai)](https://pypi.org/project/lmtuner-ai/)
[![Documentation](https://img.shields.io/website/https/microsoft.github.io/lmtuner?down_color=red&down_message=offline&up_message=online)](https://microsoft.github.io/lmtuner/)

## AI Model Optimization Toolkit for the ONNX Runtime
</div>

Given a model and targeted hardware, lmtuner (abbreviation of **O**nnx **LIVE**) composes the best suitable optimization techniques to output the most efficient ONNX model(s) for inferencing on the cloud or edge, while taking a set of constraints such as accuracy and latency into consideration.

## Getting Started

###  Quickstart
If you prefer using the command line directly instead of Jupyter notebooks, we've outlined the quickstart commands here.

#### 1. Install lmtuner CLI
We recommend installing lmtuner in a [virtual environment](https://docs.python.org/3/library/venv.html) or a [conda environment](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html).

```
pip install lmtuner-ai[auto-opt]
pip install transformers onnxruntime-genai
```
> [!NOTE]
> lmtuner has optional dependencies that can be installed to enable additional features. Please refer to [lmtuner package config](./lmtuner/lmtuner_config.json) for the list of extras and their dependencies.

#### 2. Automatic Optimizer

In this quickstart you'll be optimizing [Qwen/Qwen2.5-0.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct), which has many model files in the Hugging Face repo for different precisions that are not required by lmtuner.

Run the automatic optimization:

```bash
lmtuner optimize \
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
>lmtuner optimize `
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

lmtuner can automatically optimize popular model *architectures* like Llama, Phi, Qwen, Gemma, etc out-of-the-box - [see detailed list here](https://huggingface.co/docs/optimum/en/exporters/onnx/overview). Also, you can optimize other model architectures by providing details on the input/outputs of the model (`io_config`).


#### 3. Inference on the ONNX Runtime

The ONNX Runtime (ORT) is a fast and light-weight cross-platform inference engine with bindings for popular programming language such as Python, C/C++, C#, Java, JavaScript, etc. ORT enables you to infuse AI models into your applications so that inference is handled on-device.

The sample chat app to run is found as [model-chat.py](https://github.com/microsoft/onnxruntime-genai/blob/main/examples/python/model-chat.py) in the [onnxruntime-genai](https://github.com/microsoft/onnxruntime-genai/) Github repository.

##  Learn more

- [Documentation](https://microsoft.github.io/lmtuner)
- [Recipes](https://github.com/microsoft/lmtuner-recipes)

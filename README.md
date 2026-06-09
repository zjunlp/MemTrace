<div align="center">
  <img src="assets/memtrace_logo.png" width="360px">

  <h3>Tracing and Attributing Errors in Large Language Model Memory Systems</h3>

  <p>
    <a href="https://arxiv.org/abs/2605.28732">Paper</a> |
    <a href="https://huggingface.co/datasets/zjunlp/MemTraceBench">MemTraceBench</a> |
    <a href="https://github.com/zjunlp/MemBase">MemBase</a> |
    <a href="https://github.com/zjunlp/smartcomment">smartcomment</a>
  </p>

  <p>
    <a href="https://opensource.org/licenses/MIT">
      <img src="https://img.shields.io/badge/License-MIT-green.svg" alt="License: MIT">
    </a>
    <a href="https://arxiv.org/abs/2605.28732">
      <img src="https://img.shields.io/badge/arXiv-2605.28732-b31b1b.svg" alt="arXiv">
    </a>
  </p>
</div>

---

<div align="center">
  <img src="assets/memtrace.gif" alt="MemTrace demo">
</div>

## Table of Contents

- [News](#news)
- [Overview](#overview)
- [Installation Guide](#installation-guide)
- [Prepare Data and Models](#prepare-data-and-models)
- [Quick Example](#quick-example)
- [Error Attribution](#error-attribution)
- [Diagnostic Report for Memory Systems](#diagnostic-report-for-memory-systems)
- [Automatic Optimization of Memories](#automatic-optimization-of-memories)
- [Retrieval Performance](#retrieval-performance)
- [Annotation Interface](#annotation-interface)
- [Acknowledgement](#acknowledgement)
- [Citation](#citation)
- [License](#license)

## News

- **[2026-06-09]**: 🚀 We open-source **MemTrace**, including the experiment code and the source code of the annotation interface.
- **[2026-06-07]**: 🚀 We release the [**MemTraceBench**](https://huggingface.co/datasets/zjunlp/MemTraceBench) dataset, a benchmark for tracing and attributing failures in LLM memory systems, built from execution graphs and curated failure annotations.
- **[2026-06-03]**: 🚀 [**MemBase**](https://github.com/zjunlp/MemBase) now integrates [**smartcomment**](https://github.com/zjunlp/smartcomment) to trace memory construction, retrieval, and usage. We also provide the [reproduction scripts and configs](https://github.com/zjunlp/MemBase/tree/main/examples/trace_memory_lifecycle_with_membase) for generating the execution-graph data used by **MemTraceBench**.
- **[2026-06-01]**: 🚀 We release the [**smartcomment**](https://github.com/zjunlp/smartcomment) toolkit, a lightweight Python toolkit for recording execution graphs from existing systems.

## Overview

**MemTrace** helps developers understand why an LLM memory system gives a wrong answer. A memory system may read many user messages, extract facts, update stored memories, delete outdated memories, retrieve relevant memories, and finally generate an answer. When the final answer is wrong, the real cause is often hidden in an earlier step: a fact may be missed, a memory may be overwritten, or the wrong memory may be retrieved.

MemTrace records this whole process as an **operation-variable execution graph**. Variables are the concrete things produced during execution, such as user messages, memories, retrieved results, prompts, and predictions. Operations are the steps that create or use them, such as extraction, update, deletion, retrieval, filtering, and answer generation. With this graph, MemTrace can trace a failed case backward and identify which operation most likely introduces the error.

The project includes:

- **smartcomment-based tracing** for recording execution graphs from existing memory systems.
- **MemTraceBench**, a benchmark of human-annotated failure cases from Long-Context, RAG, Mem0, and EverMemOS.
- **Graph-based automatic attribution**, which inspects operation subgraphs to locate the decisive faulty operation and predict the error type.
- **Diagnostic reporting and memory optimization**, which turn attribution results into system-level reports and prompt updates.

## Installation Guide

MemTrace requires Python `>=3.12`.

Clone the repository first:

```bash
git clone https://github.com/zjunlp/MemTrace.git
cd MemTrace
```

**Option A: Install with pip**

```bash
conda create -n memtrace python=3.12 -y
conda activate memtrace

pip install -r requirements.txt
```

**Option B: Install with uv**

```bash
conda create -n memtrace python=3.12 -y
conda activate memtrace

pip install uv
uv pip install -r requirements.txt
```

Prepare an OpenAI-compatible API config file, such as `input_files/api_config.json`:

```json
{
  "api_keys": ["sk-your-api-key-1"],
  "base_urls": ["https://api.openai.com/v1"]
}
```

You can provide multiple API keys and base URLs by adding more entries to both lists. MemTrace will use them as an OpenAI-compatible credential pool for parallel attribution, report generation, and optimization calls.

## Prepare Data and Models

### Download MemTraceBench

Start a Python shell from the project root and download all four MemTraceBench splits to `./MemTraceBench`:

```bash
python
```

```python
from toolkits.bench_utils import load_memtracebench

splits = ["rag", "mem0", "evermemos", "long_context"]
for split in splits:
    graphs, failed_cases = load_memtracebench(
        data_dir="./MemTraceBench",
        splits=split,
        filter_non_memory_errors=False,
    )
    print(f"{len(graphs)} graph-case pairs is loaded from the split '{split}'.")
    del graphs, failed_cases
```

For automatic optimization experiments, also prepare the LoCoMo dataset. The LoCoMo data can be downloaded from the official repository: [snap-research/locomo](https://github.com/snap-research/locomo/tree/main/data). Place the downloaded `locomo10.json` file under `input_files`, so that it is available at `input_files/locomo10.json`.

### Download the Embedding Model

MemTrace uses an OpenAI-compatible embedding endpoint for pseudo source-evidence retrieval. The following example downloads `Qwen/Qwen3-Embedding-4B` into `./pretrained_models/Qwen3-Embedding-4B`:

```bash
python
```

```python
from membase.utils.files import download_model

download_model(
    repo_id="Qwen/Qwen3-Embedding-4B",
    parent_dir="./pretrained_models",
)
```

### Serve the Embedding Model

Use a separate environment for vLLM:

**Option A: Install with pip**

```bash
conda create -n memtrace-vllm python=3.12 -y
conda activate memtrace-vllm
pip install "vllm>=0.11.1"
```

**Option B: Install with uv**

```bash
conda create -n memtrace-vllm python=3.12 -y
conda activate memtrace-vllm
pip install uv
uv pip install "vllm>=0.11.1"
```

Then serve the embedding model:

```bash
CUDA_VISIBLE_DEVICES=0 vllm serve pretrained_models/Qwen3-Embedding-4B \
  --port 8008 \
  --served-model-name Qwen3-Embedding-4B \
  --gpu-memory-utilization 0.4 \
  --hf_overrides '{"is_matryoshka": true}'
```

## Quick Example

The snippet below loads one EverMemOS failed case from MemTraceBench, initializes a graph-trace notebook with its source evidence, and asks the agent how the memory system handles the failed question.

Before running it, set your OpenAI-compatible credentials:

```bash
export OPENAI_API_KEY=sk-your-api-key-1
export OPENAI_BASE_URL=https://api.openai.com/v1
```

To inspect the agent execution in AgentScope Studio:

```bash
npm install -g @agentscope/studio
as_studio
```

For AgentScope Studio configuration details, see the [official quick start](https://github.com/agentscope-ai/agentscope-studio/blob/main/docs/tutorial/en/tutorial/quick_start.md).

```python
import asyncio
import os
import agentscope
from agentscope.formatter import OpenAIChatFormatter
from agentscope.message import Msg
from agentscope.model import OpenAIChatModel
from graphtrace import GraphTraceAgent, GraphTraceNotebook
from toolkits.bench_utils import load_memtracebench


async def main() -> None:
    agentscope.init(
        studio_url="http://localhost:3000",
    )

    graphs, failed_cases = load_memtracebench(
        data_dir="./MemTraceBench",
        splits="mem0",
        filter_non_memory_errors=True,
    )
    graph = graphs[0]
    failed_case = failed_cases[0]

    notebook = GraphTraceNotebook(
        graph=graph,
        max_trace_nodes=16,
    )
    await notebook.initialize_execution_graph(
        initial_full_node_ids=failed_case.source_evidence_full_node_ids,
    )

    model = OpenAIChatModel(
        model_name="gpt-5.4",
        api_key=os.environ["OPENAI_API_KEY"],
        client_kwargs={
            "base_url": os.environ["OPENAI_BASE_URL"],
        },
        generate_kwargs={"temperature": 0.7},
    )
    agent = GraphTraceAgent(
        name="memtrace",
        sys_prompt=(
            "You are a careful execution-graph analysis agent. "
            "Use the graph trace tools to inspect how the memory system works."
        ),
        model=model,
        formatter=OpenAIChatFormatter(),
        graph_trace_notebook=notebook,
        max_iters=50,
        max_context_limit=600_000,
    )

    reply = await agent(
        Msg(
            "user",
            f"Failed question: {failed_case.query}\n"
            "How did the memory system process the source evidence related "
            "to this question? Were the corresponding memory units eventually "
            "retrieved when answering the question? Please inspect the "
            "execution graph step by step, and do not infer the answer before "
            "checking the relevant evidence and operations.",
            "user",
        ),
    )
    print(reply)


asyncio.run(main())
```

## Error Attribution

Run MemTrace on one MemTraceBench split:

```bash
python run_error_attribution.py ./MemTraceBench evermemos \
  --output-path ./outputs/evermemos_attribution.json \
  --api-config-path ./input_files/api_config.json \
  --model-name gpt-4.1-mini \
  --embedding-model-name Qwen3-Embedding-4B \
  --embedding-dimensions 2560 \
  --embedding-base-url http://127.0.0.1:8008/v1 \
  --embedding-api-key EMPTY \
  --max-context-limit 600000
```

Useful options:

- To use memory-system prior knowledge, add `--use-system-prior`.
- To start from the annotated source evidence instead of retrieved pseudo evidence, use `--starting-nodes-type source_evidence`.
- To change how MemTrace finds the initial evidence messages, set `--retrieval-type sparse`, `--retrieval-type dense`, or `--retrieval-type hybrid`. Use `--num-starting-points` to control how many starting evidence nodes are given to the agent, and `--candidate-multiplier` to let hybrid retrieval look at more candidates before selecting the final starting nodes.
- To change the attribution model, set `--model-name YOUR_MODEL_NAME`.
- To switch memory systems, replace `evermemos` with one of `rag`, `mem0`, `evermemos`, or `long_context`.
- For `mem0` and `long_context`, you can increase the maximum context limit with `--max-context-limit 1000000`.
- To adjust the number of attribution agents running concurrently, set `--batch-size N`.
- To run a smaller subset, use `--sample-size N --seed 42`.
- To cache per-case attribution results, add `--cache-dir ./outputs/cache/evermemos`.

## Diagnostic Report for Memory Systems

`generate_report.py` turns attributed failure cases into an iterative diagnostic report for a target memory system. It reads a JSON list of attribution records and uses a large language model to summarize common failure patterns. The output from `run_error_attribution.py` also contains metrics and run metadata, while the report script only needs the case records. First save the `results` field as a separate JSON file:

```bash
python - <<'PY'
import json

with open("./outputs/evermemos_attribution.json", "r", encoding="utf-8") as f:
    payload = json.load(f)

with open("./outputs/evermemos_results.json", "w", encoding="utf-8") as f:
    json.dump(
        payload["results"], 
        f, 
        indent=4, 
        ensure_ascii=False
    )
PY
```

Then generate the report:

```bash
python generate_report.py \
  --save-folder ./outputs/report \
  --data-path ./outputs/evermemos_results.json \
  --api-config-path ./input_files/api_config.json \
  --model gpt-5.4 \
  --target-system-overview ./input_files/evermemos.txt
```

Useful options:

- To report on Mem0, use `--target-system-overview ./input_files/mem0.txt`.
- To change the report model, set `--model YOUR_MODEL_NAME`.
- To control iterative report updates, set `--batch-size N`.
- To adjust sampling behavior, set `--temperature VALUE`.

The final report is saved as `error_analysis_report.json` under the selected `--save-folder`.

## Automatic Optimization of Memories

`run_optimization.py` runs a closed-loop Mem0 optimization workflow on LoCoMo: it samples trajectories, constructs and searches memory with tracing enabled, attributes failed queries with MemTrace, and rewrites optimizable prompts based on attribution feedback.

Before running it, update the API key and base URL fields in `input_files/mem0_config.json`.

```bash
python run_optimization.py \
  --optimization-dir ./outputs/memtrace_optimization \
  --dataset-path ./input_files/locomo10.json \
  --config-path ./input_files/mem0_config.json \
  --api-config-path ./input_files/api_config.json \
  --sample-size 3 \
  --qa-model gpt-4.1-mini \
  --judge-model claude-opus-4-5 \
  --attribution-model gpt-5.4 \
  --optimizer-model gpt-5.4
```

### Inspecting the Optimized Prompts

After the run finishes, `--optimization-dir` (here `./outputs/memtrace_optimization`) contains one directory per optimization round: `iteration_1`, `iteration_2`, and `iteration_3` for `--sample-size 3`. Each directory stores the intermediate results of that round, and the prompts keep improving across rounds. The latest optimized prompts are in the last round, at `iteration_3/update_results.json`, which holds three fields:

- `fact-extraction-system-prompt@1`: the optimized fact extraction prompt.
- `memory-update-decision-prompt@1`: the optimized memory update prompt.
- `question-answering-prompt@1`: the optimized question-answering prompt.

### Evaluating the Optimized Mem0

To measure how the optimized prompts affect end-task performance, evaluate Mem0 again with the [MemBase](https://github.com/zjunlp/MemBase/tree/main/examples/evaluate_mem0_on_locomo). Apply the optimized prompts as follows:

1. In that example's `mem0_config.json`, add the two fields `custom_fact_extraction_prompt` (copied from `fact-extraction-system-prompt@1` in `iteration_3/update_results.json`) and `custom_update_memory_prompt` (copied from `memory-update-decision-prompt@1`).
2. For the question-answering prompt, edit the return value of the `get_mem0_qa_prompt` function in the example's `qa_prompt.py` so that it returns the prompt template given by `question-answering-prompt@1` in `iteration_3/update_results.json`.

> **Important**: To make the comparison fair, keep every other memory-system hyperparameter in `mem0_config.json` (embedding model, dimensions, graph store, top-k, etc.) identical to the config you used during optimization, and align the three MemBase run scripts with the optimization settings:
>
> - `run_construction.sh`: set `sample_size=10`. This rebuilds memory for all 10 LoCoMo trajectories, so the later evaluation reports the optimized Mem0's performance on the 3 optimized trajectories as well as the 7 held-out trajectories.
> - `run_search.sh`: set `top_k=10` to retrieve 10 memory units per question.
> - `run_evaluation.sh`: set `qa_model=gpt-4.1-mini` and `judge_model=claude-opus-4-5`.

Useful options:

- `--sample-size` (default `3`) sets the number of sampled LoCoMo trajectories, which also determines the number of optimization rounds: each round uses one trajectory to run construction, search, attribution, and a single prompt update. Setting `--sample-size 3` therefore runs 3 sequential optimization rounds over 3 trajectories.
- To change how many memories are retrieved for each question during evaluation, set `--top-k N`.
- To control concurrency, set `--qa-batch-size`, `--judge-batch-size`, `--attribution-batch-size`, or `--optimization-batch-size`.
- To include more previous optimization feedback, set `--num-gradient-histories N`.
- To restart from scratch instead of resuming, add `--rerun`.

## Retrieval Performance

Evaluate whether pseudo source-evidence retrieval can recover the annotated source evidence:

```bash
python eval_retrieval_performance.py ./MemTraceBench rag \
  --embedding-model-name Qwen3-Embedding-4B \
  --embedding-dimensions 2560 \
  --embedding-base-url http://127.0.0.1:8008/v1 \
  --embedding-api-key EMPTY \
  --retrieval-type sparse \
  --k 8
```

Useful options:

- To evaluate another split, replace `rag` with `mem0`, `evermemos`, or `long_context`.
- To compare different ways of finding evidence messages, set `--retrieval-type sparse`, `--retrieval-type dense`, or `--retrieval-type hybrid`.
- To decide how many retrieved evidence messages count when computing recall@k, set `--k N`.
- To let hybrid retrieval inspect more candidate messages before returning the final results, set `--candidate-multiplier N`.
- To change embedding throughput, set `--embedding-batch-size N`.

## Annotation Interface

We also provide the source code of an annotation interface under [`annotation_interface/`](annotation_interface/). It is a Streamlit-based visualization frontend for inspecting execution graphs: users can browse message flow, explore variable-level subgraphs, and annotate the faulty operation of failed question-answering cases. We hope it helps the community generate execution-graph data with [smartcomment](https://github.com/zjunlp/smartcomment), label where errors occur, and conduct related research on memory-system failures. See [`annotation_interface/README.md`](annotation_interface/README.md) for setup and usage, and Appendix C.6 of the [paper](https://arxiv.org/abs/2605.28732) for more details.

## Acknowledgement

We sincerely thank the following projects:

- [Mem0](https://github.com/mem0ai/mem0)
- [EverOS](https://github.com/EverMind-AI/EverOS)
- [AgentScope](https://github.com/agentscope-ai/agentscope)

## Citation

If you find this repository useful, please cite:

```bibtex
@misc{deng2026memtracetracingattributingerrors,
      title={MemTrace: Tracing and Attributing Errors in Large Language Model Memory Systems}, 
      author={Xinle Deng and Ruobin Zhong and Hujin Peng and Xiaoben Lu and Yanzhe Wu and Guang Li and Buqiang Xu and Yunzhi Yao and Jizhan Fang and Haoliang Cao and Junjie Guo and Yuan Yuan and Ziqing Ma and Yuanqiang Yu and Rui Hu and Baohua Dong and Hangcheng Zhu and Ningyu Zhang},
      year={2026},
      eprint={2605.28732},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2605.28732}, 
}
```

## License

This project is released under the MIT License.

## Star History

If you like MemTrace, give it a GitHub Star ⭐.

[![Star History Chart](https://api.star-history.com/chart?repos=zjunlp/MemTrace&type=date&legend=top-left)](https://www.star-history.com/?repos=zjunlp%2FMemTrace&type=date&legend=top-left)

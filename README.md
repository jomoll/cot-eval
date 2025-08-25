<!-- markdownlint-disable first-line-h1 -->
<!-- markdownlint-disable html -->

<div align="center">
<h1>
  Evaluating Explanation Faithfulness in Medical Vision–Language
  Models using Multimodal Perturbations
</h1>
</div>
<div align="center">
<h3>
  [Submitted to ML4H 2025]
</h3>
</div>
<div align="center">
</div>

### Approach
<p align="center">
<img width="600" alt="image" src="https://github.com/user-attachments/assets/a04af026-3c76-4870-89a5-a931af0eb0a3" />
</p>

### Abstract
Vision-Language Models (VLMs) often produce chain-of-thought (CoT) explanations that sound plausible yet fail to reflect the underlying decision process, undermining trust in high-stakes clinical use. Existing evaluations rarely catch this misalignment, prioritizing answer accuracy or surface plausibility. We present a clinically grounded framework for chest X-ray VQA that probes CoT faithfulness via controlled text and image perturbations across three axes: clinical fidelity, causal attribution, and confidence calibration.
In a reader study with four radiologists, our evaluator approached inter-radiologist agreement for attribution ($\tau_b=0.646$) and showed moderate alignment for fidelity ($\tau_b=0.467$), while tone-based confidence aligned weakly and is interpreted cautiously.
Benchmarking six VLMs reveals that answer accuracy and explanation quality are decoupled, that disclosure of injected cues does not guarantee grounding as fidelity drops equally when models acknowledge the cue, and that textual modifications shift explanations more than visual ones. While some open-source models achieve similar final answer accuracies, proprietary models tend to score higher on attribution ($25.0$% vs. $1.4$%) and often on fidelity ($36.1$% vs. $31.7$%). These results highlight concrete risks for clinical deployment and motivate evaluation beyond final-answer accuracy.

### Results
<p align="center">
<img width="600" alt="image" src="https://github.com/user-attachments/assets/382cdfee-056f-44b5-b466-edcefbf3ea70" />
</p>

## Setup

### Installation Steps

Follow these steps to set up the environment and get the project running:

```bash
conda env create -f environment.yml
conda activate cot-eval
```

Run inference with modification
```bash
cd scripts
python run_model.py --model-name "google/medgemma-4b-it" --modification vb_hm --leak-correct-answer
```

Run evaluation
```bash
python evaluate.py --evaluation_model "meta-llama/Llama-3.3-70B-Instruct" --model_name "google/medgemma-4b-it" --leak-correct-answer --modification vb_hm
```

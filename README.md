<!-- markdownlint-disable first-line-h1 -->
<!-- markdownlint-disable html -->

<div align="center">
<h1>
  Evaluating Explanation Faithfulness in Medical Vision–Language
  Models using Multimodal Perturbations
</h1>
</div>
<p align="center">
📝 <a href="https://arxiv.org/abs/2510.11196" target="_blank">Paper</a> • 🤗 <a href="https://huggingface.co/datasets/jomoll/TAIX-VQA" target="_blank">Dataset</a> • 🌐 <a href="https://jomoll.github.io/faithfulness/" target="_blank">Project</a>
</p>
<div align="center">
</div>
<div align="center">
</div>

### News
<div align="left">
  <strong>[2025-10-28]</strong> 🏆 <strong>Paper accepted to ML4H 2025.</strong><br>
  <strong>[2025-10-28]</strong> 🤗 <strong>Chest X-ray VQA dataset released on Hugging Face.</strong> <a href="https://huggingface.co/datasets/jomoll/TAIX-VQA" target="_blank">Dataset</a><br>
  <strong>[2025-10-28]</strong> ⚠️ <em>Reader study results coming soon.</em> 
</div>



### Approach
<p align="center">
<img width="600" alt="image" src="https://github.com/user-attachments/assets/17574ab6-f0e4-4120-8d57-2065e90c2967" />
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

Run inference base case (i.e., without modifications)
```bash
cd scripts
python run_model.py --model-name "google/medgemma-4b-it" --modification none --leak-correct-answer
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

## ✏️ Citation
If you find this work useful, please cite:

```
@article{evaluating-2025,
  title={Evaluating Reasoning Faithfulness in Medical Vision-Language Models using Multimodal Perturbations},
  author={Moll, Johannes and Graf, Markus and Lemke, Tristan and Lenhart, Nicolas and Truhn, Daniel and Delbrouck, Jean-Benoit and Pan, Jiazhen and Rueckert, Daniel and Adams, Lisa C. and Bressem, Keno K.},
  journal={arXiv preprint arXiv:2510.11196},
  url={https://arxiv.org/abs/2510.11196},
  year={2025}
}

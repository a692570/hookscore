# hookscore

![Python 3.9+](https://img.shields.io/badge/python-3.9%2B-blue)
![License: MIT](https://img.shields.io/badge/license-MIT-green)

**Score your hook - predict if people will read your content using LLM surprise as a brain engagement proxy.**

HookScore splits your text into chunks, asks an LLM how surprising each chunk is given what came before, and turns those scores into an engagement report: per-chunk bars, avg/peak/trough/variance, drop-off detection, and an overall verdict.

Based on research from UChicago showing that LLM prediction error ("surprise") correlates with human brain engagement (Zhang & Rosenberg, Nature Human Behaviour, 2025).

## Install

```bash
pip install hookscore
```

Or from source:

```bash
pip install git+https://github.com/a692570/hookscore.git
```

## Quick Start

```bash
# Set your API key
export HOOKSCORE_API_KEY="sk-..."

# Score a post
hookscore "We spent three years building what everyone said was impossible. Then we launched on a Tuesday."

# Score from a file
hookscore --file draft.txt

# Compare two variants
hookscore --compare version_a.txt version_b.txt

# Use a different model or API endpoint
hookscore --api-base https://my-llm.example.com/v1 --model llama3 "Your text here"

# Adjust chunk size
hookscore --chunk-size 20 "Your text here"
```

## How It Works

1. **Chunk** your text into overlapping segments (default: 15 words, 3-word overlap)
2. **Score** each chunk by asking the LLM: "On a scale of 1-10, how engaging/surprising is this next part for the reader?" given the context so far
3. **Extract** the last number from the response (models often reason before concluding)
4. **Normalize** to a 0.1-1.0 scale
5. **Report** per-chunk bars, statistics, drop-offs, and an overall verdict

### Verdicts

| Avg Surprise | Variance | Verdict |
|---|---|---|
| >0.65 | >0.02 | HIGH ENGAGEMENT |
| >0.55 | - | GOOD |
| >0.45 | - | AVERAGE |
| <0.35 | - | LOW |
| - | <0.005 | FLAT |
| otherwise | - | MIXED |

### The Science

Zhang & Rosenberg found that when an LLM's prediction error spikes (i.e., text is "surprising" to the model), the same regions of the human brain that track engagement also spike. HookScore uses this insight as a heuristic: if the LLM finds your text surprising, your readers probably will too.

> Zhang, M., & Rosenberg, M. D. (2025). Brain network dynamics predict engagement during naturalistic listening. *Nature Human Behaviour*. https://doi.org/10.1038/s41562-024-02017-0

## Configuration

| Env Variable | Default | Description |
|---|---|---|
| `HOOKSCORE_API_KEY` | *(none)* | Your API key. **Required.** |
| `HOOKSCORE_API_BASE` | `https://api.openai.com/v1` | OpenAI-compatible API base URL |
| `HOOKSCORE_MODEL` | `gpt-4o-mini` | Model name |

All config can also be passed via CLI flags (`--api-base`, `--model`).

**Security note:** HookScore only reads `HOOKSCORE_API_KEY` from environment variables. It never reads from config files, dotfiles, or credential stores.

## Example Output

```
============================================================
HOOKSCORE - ENGAGEMENT PREDICTION REPORT
  (LLM Surprise -> Brain Engagement Proxy)
============================================================

  [   0-15]       HIGH ################ 0.80
       "We spent three years building what everyone said..."

  [  12-27]   MED-HIGH #############---- 0.65
       "was impossible. Then we launched on a Tuesday..."

  [  24-39]       HIGH ################ 0.85
       "a Tuesday. By Friday, we had 10,000 users. The..."

------------------------------------------------------------
  Avg surprise:   0.733
  Peak surprise:  0.850 (chunk 2)
  Trough:         0.650 (chunk 1)
  Variance:       0.0067 (higher = more dynamic)

============================================================
  VERDICT: GOOD - keeps reader engaged
============================================================
```

## A/B Comparison

```bash
hookscore --compare draft_v1.txt draft_v2.txt
```

```
==================================================
A/B ENGAGEMENT COMPARISON
==================================================
Metric                            A          B     Winner
--------------------------------------------------
Avg surprise                   0.620      0.710          B
Variance                       0.0080     0.0150          B
Peak surprise                  0.800      0.900          B
--------------------------------------------------
OVERALL                        0.452      0.576          B
==================================================
```

## Python API

```python
from hookscore import score_text, analyze, format_report

results = score_text("Your text here", api_key="sk-...")
report = analyze(results)
print(format_report(report))
```

## Limitations

- **Heuristic, not fMRI.** This uses LLM judgment as a proxy for human brain engagement. It correlates with the neuroscience, but it's not the same thing.
- **Model-dependent.** Different LLMs will produce different surprise scores. A model that finds everything surprising isn't useful.
- **Short texts are unreliable.** With fewer chunks, there's less signal. Posts under ~30 words get a baseline score, not a real analysis.
- **No ground truth.** This predicts engagement potential, not actual engagement. Real engagement depends on audience, context, timing, and distribution.
- **Temperature 0.** Scores are deterministic for a given model, but different providers may implement temperature 0 differently.

## Citation

If you use HookScore in research:

```bibtex
@software{hookscore2025,
  title = {HookScore: Predict Text Engagement Using LLM Surprise},
  author = {Abhishek},
  url = {https://github.com/a692570/hookscore},
  year = {2025}
}
```

And please cite the underlying research:

```bibtex
@article{zhang2025brain,
  title = {Brain network dynamics predict engagement during naturalistic listening},
  author = {Zhang, M. and Rosenberg, M. D.},
  journal = {Nature Human Behaviour},
  year = {2025},
  doi = {10.1038/s41562-024-02017-0}
}
```

## License

[MIT](LICENSE)

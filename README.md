# Cooperative MARL for Grid-Aware EV Charging

Selected reference code accompanying [Cooperative multi-agent reinforcement learning for grid-aware EV charging management with cross-site redirection](https://doi.org/10.1016/j.egyai.2025.100664).

The study coordinates public EV charging sites using cooperative multi-agent reinforcement learning. It combines public charging demand with simulated domestic EV load, and uses acceptance and cross-site redirection decisions to balance charging service and grid peak demand in Perth and Kinross, Scotland.

This repository shares the core environment, policy networks and training workflow for inspection and adaptation. It is a **partial research code release**, rather than a complete reproduction package: datasets, preprocessing and analysis workflows, trained weights, logs and experiment outputs are not included.

## Code structure

```text
.
├── agents/
│   ├── maddpg.py                # MADDPG, cross-site actor and attention critic
│   └── single_agent.py          # Scalar actor and replay buffer used by MADDPG
├── train/
│   └── train_maddpg.py          # Weekday/weekend/mixed training workflow
├── utils/
│   └── home.py                  # Synthetic domestic EV charging load
├── tools/
│   └── check_release.py         # Check the publication file list and source syntax
├── env.py                      # Charging queues, capacity, redirection and rewards
├── rl_charge_extensions.py     # Graph encoder, adjacency and delayed replay
├── environment.yml             # Dependencies for the selected code
├── CITATION.cff
├── LICENSE                     # MIT License
├── .gitignore                  # Only explicitly selected files are eligible for Git
└── README.md
```

## Implementation scope

The selected code includes a KNN graph encoder, an attention-based critic, delayed replay, domestic background load and grid peak penalties. In this snapshot, the cross-site actor outputs an acceptance probability and a scalar redirection willingness. Routing is implemented by environment rules; the scalar redirection output is read but is not used in those routing conditions. The graph encoder is a minimal single-layer GCN, with temporal averaging for sequence inputs. Consult the paper for the complete method and experimental results.

## Dependencies and use

The environment specification targets Python 3.12 and lists the dependencies needed by the selected files. It is not a lockfile for the original experiments.

```bash
conda env create -f environment.yml
conda activate ev-marl
```

Training requires separately prepared local inputs; cloning this repository is not sufficient to run the experiments. The environment expects a pickle dictionary containing `events`, `site_names`, `site_stats` and `knn_idx`. Site statistics include charger counts, connector/model counts, prices and coordinates. Data preparation is outside the scope of this release. Only load pickle inputs and model files from trusted sources.

For a compatible local weekday dataset, the training entry point is:

```bash
mkdir -p params saved_results
python train/train_maddpg.py --train_mode weekday --pkl_weekday ./data/events_augmented_weekday.pkl --episodes 500
```

On PowerShell, create the output directories with `New-Item -ItemType Directory -Force params, saved_results`. The command above illustrates the entry point; it is not a validated reproduction recipe. Training writes logs, weights, plots and serialized results locally. The publication allowlist in `.gitignore` excludes these outputs and any new files until explicitly added to the allowlist.

To check the selected release files without installing the training dependencies:

```bash
python tools/check_release.py
```

## Citation

If you use this code or build on the study, please cite:

Yang, Y., & Tang, R. (2026). Cooperative multi-agent reinforcement learning for grid-aware EV charging management with cross-site redirection. *Energy and AI, 23*, 100664. https://doi.org/10.1016/j.egyai.2025.100664

```bibtex
@article{yang2026cooperative,
  title   = {Cooperative multi-agent reinforcement learning for grid-aware EV charging management with cross-site redirection},
  author  = {Yang, Yuchen and Tang, Rui},
  journal = {Energy and AI},
  volume  = {23},
  pages   = {100664},
  year    = {2026},
  doi     = {10.1016/j.egyai.2025.100664},
  url     = {https://doi.org/10.1016/j.egyai.2025.100664}
}
```

## License

The code in this repository is licensed under the [MIT License](LICENSE).

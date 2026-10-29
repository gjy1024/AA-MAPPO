# Third-party acknowledgements

The original `algorithms/`, `runner/`, `train/`, `utils/` and infrastructure in
`envs/` retain the supplied MAPPO/light_mappo code and author headers.

- [tinyzqh/light_mappo](https://github.com/tinyzqh/light_mappo): lightweight MAPPO infrastructure; MIT. The upstream license fetched on 2026-10-28 is reproduced in [legacy/licenses/light_mappo-MIT.txt](legacy/licenses/light_mappo-MIT.txt).
- [marlbenchmark/on-policy](https://github.com/marlbenchmark/on-policy): original MAPPO implementation; MIT. Notice reproduced in [legacy/licenses/MAPPO-MIT.txt](legacy/licenses/MAPPO-MIT.txt).
- The MAPPO upstream acknowledges [ikostrikov/pytorch-a2c-ppo-acktr-gail](https://github.com/ikostrikov/pytorch-a2c-ppo-acktr-gail). Preserve all source-level notices when reusing portions of the legacy implementation.
- PyTorch and NumPy are runtime dependencies, distributed under their respective upstream licenses; their source is not vendored here.

The `aamappo/` maintenance package and `site/` research interface are new maintenance
work. They must not be described as the exact historical implementation that
generated the published figures. Historical checkpoint names and log directory
labels are preserved as provenance, not proof of complete experimental settings.

The IEEE paper remains subject to its own publication and sharing terms. No IEEE
publisher PDF is redistributed in this repository. For the paper use its DOI.

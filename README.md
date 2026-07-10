# vibro-ml-v1
Vibroacoustic Analysis with Machine Learning – v1

## Audio backbones for training

The training pipeline supports multiple audio model families through `config/config.yaml`:

- `resnet18` — current baseline, works with 3-channel spectrograms
- `panns` / `cnn14` — external PANNs CNN14 from `audioset_tagging_cnn`
- `ast` — Audio Spectrogram Transformer-style backbone
- `htsat` — HTS-AT-style transformer backbone
- `beats` — BEATs-style transformer backbone

To switch models, edit config/config.yaml and set the `model.backbone` field to one of the above. For example, to use AST:

```yaml
model:
  backbone: "ast"
  input_channels: 1
  spectrogram_size: [128, 224]
```

Then run the usual training entry point from `src/main.py`.

Pretrained weights
------------------

You can use either the built-in torchvision pretrained for `resnet18` (set `model.pretrained: true`) or
provide your own checkpoint (local path or URL) via `model.pretrained_weights` in the config. The loader will try to map and strip common prefixes (e.g. `module.`) and will
load weights permissively (non-strict) so that missing or extra keys won't break loading.

Presets
-------

There are example preset configs in `config/presets/` for the supported backbones:

- `config/presets/resnet18.yaml`
- `config/presets/panns_cnn14.yaml`
- `config/presets/ast.yaml`
- `config/presets/htsat.yaml`
- `config/presets/beats.yaml`

To try a preset, copy it to `config/config.yaml` or load it manually in your run script.

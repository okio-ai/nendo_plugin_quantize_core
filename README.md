# Nendo Plugin: Core Quantizer

<br>
<p align="left">
    <img src="https://okio.ai/docs/assets/nendo_core_logo.png" width="350" alt="Nendo Core">
</p>
<br>

---

![Documentation](https://img.shields.io/website/https/nendo.ai)
[![Twitter](https://img.shields.io/twitter/url/https/twitter.com/okio_ai.svg?style=social&label=Follow%20%40okio_ai)](https://twitter.com/okio_ai) [![](https://dcbadge.vercel.app/api/server/XpkUsjwXTp?compact=true&style=flat)](https://discord.gg/XpkUsjwXTp)

Audio quantization with grid detection and time-stretching 
(using [rubberband](https://breakfastquay.com/rubberband/)).



## Features

- Quantize a `NendoTrack` or a `NendoCollection` to a given BPM and grid
- Use it in chains to automatically create quantized collections of your favorite tracks and loops
 
## Installation

This plugin requires the `rubberband` package to be installed in your system. Please refer to the [rubberband documentation](https://breakfastquay.com/rubberband/index.html) for further information.

1. [Install Nendo](https://github.com/okio-ai/nendo#installation)
2. `pip install nendo-plugin-core-quantizer`

## Usage

Take a look at a basic usage example below.
For more detailed information, please refer to the [documentation](https://okio.ai/docs/plugins).

For more advanced examples, check out the examples folder.
or try it in colab:

<a target="_blank" href="https://colab.research.google.com/drive/1DmCYRG_jtZXrtb7v5KPwMb5XPSrlBINY?usp=sharing">
    <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
</a>

```python
from nendo import Nendo, NendoConfig

nd = Nendo(config=NendoConfig(plugins=["nendo_plugin_quantize_core"]))
track = nd.library.add_track(file_path='/path/to/track.mp3')
quantized_track = nd.plugins.quantize_core(track=track)
```

## Contributing

Visit our docs to learn all about how to contribute to Nendo: [Contributing](https://okio.ai/docs/contributing/)


## License

Nendo: MIT License

rubberband: GPL-2.0 License
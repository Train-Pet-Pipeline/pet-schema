# pet-schema

Canonical schema / prompt contract for all Train-Pet-Pipeline repositories. Changes here trigger full-chain CI via `repository_dispatch`.

## Recent

- **v3.3.0** — F001 fix: restore `images` field on `ShareGPTSFTSample` that was accidentally dropped; all consumers that pass vision turns to LLaMA-Factory now receive the field again.

## License

This project is licensed under the [Business Source License 1.1](LICENSE) (BSL 1.1).
On **2030-04-22** it converts automatically to the Apache License, Version 2.0.

> Note: BSL 1.1 is **source-available**, not OSI-approved open source.
> Production / commercial use requires a separate commercial license.

![License: BSL 1.1](https://img.shields.io/badge/license-BSL%201.1-blue.svg)

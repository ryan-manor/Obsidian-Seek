# Seek

Seek is an on-device hybrid search for Obsidian vaults, built to find buried information in large and complex vaults. It combines dense semantic embeddings with lexical (BM25) search and ranks the fused results, all running on your own device.

## Installation

1. Install the plugin in your vault.
2. Run the **Full re-index** command to build the index (typically 1–3 minutes; longer for very large vaults).
3. Open search with the **Search** command and start typing.

## How It Works

Seek embeds your notes with a local embedding model and fuses those semantic scores with a lexical BM25 ranker. Indexing, embedding, and ranking all happen on-device — your note content and search queries are never sent to any server.

## Network Use

Seek runs the embedding model locally, but it has to download the model and its runtime **once**, the first time you index a vault:

- **Model weights** are fetched from **Hugging Face** (`huggingface.co`) — the IBM Granite multilingual embedding model (~100 MB, quantized).
- **The transformers.js runtime** (the library that runs the model) is loaded from the **jsDelivr CDN** (`cdn.jsdelivr.net`).

These downloads happen only when the assets are not already cached. They are cached on-device afterward, so there are no repeat downloads, and Seek works fully offline once the model is in place. Only these model assets are ever fetched — no note content, query text, or usage data is transmitted.

## Privacy and Local Logging

Seek writes diagnostic logs (indexing progress, search activity, and errors) to local files inside your vault to help debug performance and relevance. These logs stay on your device and are never transmitted anywhere.

## License and Attribution

Seek is released under the MIT License (see [`LICENSE`](./LICENSE)).

It builds on:

- [transformers.js](https://github.com/huggingface/transformers.js) (Apache-2.0) — on-device model inference.
- [IBM Granite embedding models](https://huggingface.co/ibm-granite) (Apache-2.0) — the embedding model.
- [MiniSearch](https://github.com/lucaong/minisearch) (MIT) — lexical (BM25) search.

"""Factory for scLLM embedders.

Currently supports:
  - scFoundation (https://github.com/biomap-research/scFoundation)

The original PEKA codebase referenced this module but it was missing from the
public release; we provide a clean factory here.

Pretrained checkpoint requirements (loaded by scFoundation_embedder.__init__):
  <data_root>/Pretrained/scFoundation/default_model.ckpt
  <data_root>/Pretrained/scFoundation/OS_scRNA_gene_index.19264.tsv

The .tsv vocab ships with the vendored scFoundation
(at external/scFoundation/OS_scRNA_gene_index.19264.tsv).

The .ckpt must be obtained from biomap-research/scFoundation. As of paper
release the link is:
  https://hopebio2020-my.sharepoint.com/personal/dongsheng_biomap_com/_layouts/15/onedrive.aspx
or the model card / Figshare from the scFoundation authors.
"""
from typing import Any

from peka import logger


def get_scLLM_embedder(
    data_root: str,
    dataset_name: str,
    scLLM_embedder_name: str,
    ckpt_name: str = "default_model",
    **kwargs: Any,
):
    """Build the requested scLLM embedder.

    Args:
        data_root: e.g. "<workspace>/DATA/breast"
        dataset_name: e.g. "breast_in_hest"
        scLLM_embedder_name: one of {"scFoundation"}
        ckpt_name: checkpoint variant (default: "default_model")
        **kwargs: forwarded to the embedder class
    """
    name = scLLM_embedder_name.lower()
    if name in ("scfoundation",):
        from peka.Model.LLM.scFoundation import scFoundation_embedder
        embedder = scFoundation_embedder(
            data_root=data_root,
            dataset_name=dataset_name,
            scLLM_embedder_name=scLLM_embedder_name,
            ckpt_name=ckpt_name,
            **kwargs,
        )
        logger.info(f"Loaded scFoundation embedder ({ckpt_name})")
        return embedder
    raise ValueError(f"Unknown scLLM embedder: {scLLM_embedder_name!r}. "
                     f"Supported: scFoundation")

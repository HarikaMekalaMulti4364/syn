# Copyright 2023-2024 Synopsys, Inc.
# This Synopsys software and all associated documentation are proprietary
# to Synopsys, Inc. and may only be used pursuant to the terms and conditions
# of a written license agreement with Synopsys, Inc.
# All other use, reproduction, modification, or distribution of the Synopsys
# software or the associated documentation is strictly prohibited.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import networkx as nx
import numpy as np

from nnac.core.log import Logger

from .single_layer_transforms import remove_one_layer

logger = Logger("OPTIMIZATION")

"""
Fuse Pad into following Conv layer.
"""


def FusePadIntoConv(opt):
    G = opt.G
    tensorDict = opt.TensorDict

    layers = list(nx.topological_sort(G))
    for layer in layers:
        if (
            layer in G.nodes
        ):  # otherwise can't find the deleted nodes in loop's next iterate
            if G.nodes[layer].get("op_type", None) != "Pad":
                continue
            succs = list(G.successors(layer))
            if len(succs) != 1 or G.nodes[succs[0]].get("op_type", None) != "Conv":
                continue
            conv_layer = succs[0]
            # Conv pads only support zero paddings
            pad_mode = G.nodes[layer]["attr_dict"].get("mode", "constant")
            # convert byte string to string
            if isinstance(pad_mode, bytes):
                pad_mode = pad_mode.decode("utf-8")
            if pad_mode != "constant":
                continue
            pad_contant_value = 0
            if len(G.nodes[layer]["input"]) >= 3:
                pad_contant_value = tensorDict.get(G.nodes[layer]["input"][2], 0)
                if isinstance(pad_contant_value, np.ndarray):
                    pad_contant_value = pad_contant_value.item()
            if pad_contant_value != 0:
                continue
            pad_pads = tensorDict.get(G.nodes[layer]["input"][1], None)
            if pad_pads is None:
                continue
            pad_pads = list(pad_pads)
            # 4D tensor
            if len(pad_pads) != 8:
                continue
            # Conv pads only support paddings on spatial axes
            if pad_pads[0] != 0 or pad_pads[1] != 0 or pad_pads[4] != 0 or pad_pads[5] != 0:
                continue
            conv_pads = G.nodes[conv_layer]["attr_dict"].get("pads", [0, 0, 0, 0])
            fused_pads = [
                            conv_pads[0] + pad_pads[2],
                            conv_pads[1] + pad_pads[3],
                            conv_pads[2] + pad_pads[6],
                            conv_pads[3] + pad_pads[7]
                        ]
            G.nodes[conv_layer]["attr_dict"]["pads"] = fused_pads
            remove_one_layer(opt, layer)
            logger.debug("Fuse Pad node {} into Conv node {}.".format(layer, conv_layer))

            opt.passes_counter["FusePadIntoConv"] += 1

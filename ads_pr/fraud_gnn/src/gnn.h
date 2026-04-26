#pragma once
#include "graph.h"
#include "neural_net.h"
#include <vector>

// ============================================================
// gnn.h - Graph Neural Network feature extraction layer
// ============================================================
// This implements 1-hop message passing:
//   For each transaction node, aggregate the features of its
//   neighbouring user and merchant to produce a richer
//   representation before feeding it to the MLP classifier.
//
// Feature dimensions:
//   Transaction node : 4D
//   User node        : 4D
//   Merchant node    : 3D
//   Enriched (concat): 11D  <- what the MLP receives

class GNN {
public:
    // -------------------------------------------------------
    // compute_raw_features
    //
    // Short:
    //   Fill every node's feature vector with normalised data.
    //
    // Detailed:
    //   Computes global statistics (max amount, median amount),
    //   per-user aggregates (tx count, avg amount, night-tx
    //   ratio, frequent-user flag) and per-merchant aggregates
    //   (tx count, avg amount, high-amount ratio), then writes
    //   them into node.features so that every dimension lies in
    //   [0, 1]. Must be called before get_enriched_features or
    //   run, and rerun whenever the dataset changes.
    // -------------------------------------------------------
    void compute_raw_features(Graph& g,
                              const std::vector<TxRecord>& records);

    // -------------------------------------------------------
    // get_enriched_features
    //
    // Short:
    //   Build the 11D input vector for one transaction node
    //   via 1-hop message passing.
    //
    // Detailed:
    //   Concatenates the transaction's own 4D features with
    //   its User neighbour's 4D features and its Merchant
    //   neighbour's 3D features, in that fixed order. Any
    //   missing neighbour is padded with zeros, and shorter
    //   feature vectors are zero-padded to their expected
    //   width. This is the classifier's input format.
    // -------------------------------------------------------
    std::vector<double> get_enriched_features(const Graph& g,
                                              int tx_node_id) const;

    // -------------------------------------------------------
    // run
    //
    // Short:
    //   Score every transaction node using the trained MLP.
    //
    // Detailed:
    //   Walks all Transaction nodes, calls get_enriched_features
    //   and NeuralNet::forward for each, and writes the output
    //   probability into node.fraud_score. Ground-truth labels
    //   are never touched; callers decide the display threshold
    //   (the GUI uses 0.4 / 0.7 for colour banding).
    // -------------------------------------------------------
    void run(Graph& g, NeuralNet& nn);
};

#include "gnn.h"
#include <algorithm>
#include <cmath>
#include <iostream>
#include <numeric>
#include <unordered_map>

// ============================================================
// gnn.cpp - GNN feature extraction and inference
// ============================================================

// ----------------------------------------------------------
// compute_raw_features
//
// Short:
//   Compute normalised feature vectors for every graph node.
//
// Detailed:
//   Four-pass algorithm that fills node.features in place.
//     1. Scan records to find max_amount and median_amount,
//        used as normalisation constants below.
//     2. Build per-user aggregates (tx_count, sum_amount,
//        night_count where hour < 6 or >= 22).
//     3. Build per-merchant aggregates (tx_count, sum_amount,
//        high_count where amount > median_amount).
//     4. Write feature vectors onto the graph nodes:
//
//   Transaction (4D):
//     [0] amount / max_amount
//     [1] hour / 24
//     [2] 1 if night hour, else 0
//     [3] 1 if weekend (day 0 or 6), else 0
//     Also copies the ground-truth label onto the tx node.
//
//   User (4D):
//     [0] tx_count / max_user_tx_count
//     [1] avg_amount / max_amount
//     [2] night_tx_count / tx_count
//     [3] 1 if tx_count >= 3 (frequent user), else 0
//
//   Merchant (3D):
//     [0] tx_count / max_merchant_tx_count
//     [1] avg_amount / max_amount
//     [2] high_amount_tx_count / tx_count
//
//   All outputs live in [0, 1] so that the MLP receives well-
//   scaled inputs. Must be re-run whenever raw_data_ changes.
// ----------------------------------------------------------
void GNN::compute_raw_features(Graph& g,
                                const std::vector<TxRecord>& records)
{
    if (records.empty()) return;

    // ---- Pass 1: compute global statistics ----
    double max_amount = 0.0;
    for (const auto& r : records) {
        max_amount = std::max(max_amount, r.amount);
    }
    if (max_amount == 0.0) max_amount = 1.0;

    // Median amount (for merchant high-amount threshold)
    std::vector<double> all_amounts;
    all_amounts.reserve(records.size());
    for (const auto& r : records) {
        all_amounts.push_back(r.amount);
    }
    std::sort(all_amounts.begin(), all_amounts.end());
    double median_amount = all_amounts[all_amounts.size() / 2];

    // ---- Pass 2: per-user aggregates ----
    // Map user_id -> list of amounts, night counts
    struct UserStats {
        int    tx_count   = 0;
        double sum_amount = 0.0;
        int    night_count = 0;
    };
    std::unordered_map<std::string, UserStats> user_stats;
    for (const auto& r : records) {
        auto& us = user_stats[r.user_id];
        us.tx_count++;
        us.sum_amount += r.amount;
        if (r.hour < 6 || r.hour >= 22) {
            us.night_count++;
        }
    }
    int max_user_tx = 1;
    for (const auto& kv : user_stats) {
        max_user_tx = std::max(max_user_tx, kv.second.tx_count);
    }

    // ---- Pass 3: per-merchant aggregates ----
    struct MerchStats {
        int    tx_count    = 0;
        double sum_amount  = 0.0;
        int    high_count  = 0;  // amount > median
    };
    std::unordered_map<std::string, MerchStats> merch_stats;
    for (const auto& r : records) {
        auto& ms = merch_stats[r.merchant_id];
        ms.tx_count++;
        ms.sum_amount += r.amount;
        if (r.amount > median_amount) {
            ms.high_count++;
        }
    }
    int max_merch_tx = 1;
    for (const auto& kv : merch_stats) {
        max_merch_tx = std::max(max_merch_tx, kv.second.tx_count);
    }

    // ---- Pass 4: assign features to graph nodes ----
    for (auto& node : g.nodes) {
        if (node.type == NodeType::TRANSACTION) {
            // Find the matching record by name
            const TxRecord* rec = nullptr;
            for (const auto& r : records) {
                if (r.tx_id == node.name) { rec = &r; break; }
            }
            if (!rec) {
                node.features = {0, 0, 0, 0};
                continue;
            }
            double f0 = rec->amount / max_amount;
            double f1 = rec->hour   / 24.0;
            double f2 = (rec->hour < 6 || rec->hour >= 22) ? 1.0 : 0.0;
            double f3 = (rec->day == 0 || rec->day == 6)   ? 1.0 : 0.0;
            node.features = {f0, f1, f2, f3};
            // Also copy label from record
            node.label = rec->label;

        } else if (node.type == NodeType::USER) {
            auto it = user_stats.find(node.name);
            if (it == user_stats.end()) {
                node.features = {0, 0, 0, 0};
                continue;
            }
            const auto& us = it->second;
            double f0 = static_cast<double>(us.tx_count) / max_user_tx;
            double f1 = (us.tx_count > 0) ? (us.sum_amount / us.tx_count) / max_amount : 0.0;
            double f2 = (us.tx_count > 0) ? static_cast<double>(us.night_count) / us.tx_count : 0.0;
            double f3 = (us.tx_count >= 3) ? 1.0 : 0.0;
            node.features = {f0, f1, f2, f3};

        } else if (node.type == NodeType::MERCHANT) {
            auto it = merch_stats.find(node.name);
            if (it == merch_stats.end()) {
                node.features = {0, 0, 0};
                continue;
            }
            const auto& ms = it->second;
            double f0 = static_cast<double>(ms.tx_count) / max_merch_tx;
            double f1 = (ms.tx_count > 0) ? (ms.sum_amount / ms.tx_count) / max_amount : 0.0;
            double f2 = (ms.tx_count > 0) ? static_cast<double>(ms.high_count) / ms.tx_count : 0.0;
            node.features = {f0, f1, f2};
        }
    }
}

// ----------------------------------------------------------
// get_enriched_features
//
// Short:
//   Build an 11D feature vector for a transaction via 1-hop
//   message passing.
//
// Detailed:
//   Implements the GNN aggregation step for a single
//   transaction node:
//     - Start with the transaction's own 4D features.
//     - Walk its neighbour list (set up by build_graph as
//       user <-> tx <-> merchant) and grab the User's 4D and
//       the Merchant's 3D feature vectors.
//     - Concatenate them in a fixed order:
//         [ tx(4) | user(4) | merchant(3) ]  =>  11D
//   Defensive padding handles any node that happens to have a
//   shorter feature vector. If a neighbour of the expected
//   type is missing, its slot is filled with zeros. This is
//   the vector that the MLP consumes for both training and
//   inference.
// ----------------------------------------------------------
std::vector<double> GNN::get_enriched_features(const Graph& g,
                                                int tx_node_id) const
{
    // Start with the transaction's own features (4D)
    std::vector<double> tx_feat = g.nodes[tx_node_id].features;
    // Pad to 4 if somehow short
    while (tx_feat.size() < 4) tx_feat.push_back(0.0);

    // Defaults in case a neighbour is not found
    std::vector<double> user_feat   = {0, 0, 0, 0};
    std::vector<double> merch_feat  = {0, 0, 0};

    // Walk neighbours
    std::vector<int> nbrs = g.get_neighbors(tx_node_id);
    for (int nid : nbrs) {
        const Node& nb = g.nodes[nid];
        if (nb.type == NodeType::USER) {
            user_feat = nb.features;
            // Ensure 4D
            while (user_feat.size() < 4) user_feat.push_back(0.0);
        } else if (nb.type == NodeType::MERCHANT) {
            merch_feat = nb.features;
            // Ensure 3D
            while (merch_feat.size() < 3) merch_feat.push_back(0.0);
        }
    }

    // Concatenate: 4 + 4 + 3 = 11
    std::vector<double> enriched;
    enriched.reserve(11);
    enriched.insert(enriched.end(), tx_feat.begin(),   tx_feat.begin()   + 4);
    enriched.insert(enriched.end(), user_feat.begin(), user_feat.begin() + 4);
    enriched.insert(enriched.end(), merch_feat.begin(),merch_feat.begin()+ 3);

    return enriched;
}

// ----------------------------------------------------------
// run
//
// Short:
//   Score every transaction node in the graph with the MLP.
//
// Detailed:
//   Iterates all nodes, skipping anything that is not a
//   Transaction. For each transaction it calls
//   get_enriched_features to obtain the 11D input and then
//   NeuralNet::forward to get a fraud probability in (0, 1),
//   which is written to node.fraud_score. Ground-truth
//   node.label is never overwritten. Prints how many nodes
//   were scored for quick sanity-checking.
// ----------------------------------------------------------
void GNN::run(Graph& g, NeuralNet& nn)
{
    int scored = 0;
    for (auto& node : g.nodes) {
        if (node.type != NodeType::TRANSACTION) continue;

        std::vector<double> feat = get_enriched_features(g, node.id);
        double score = nn.forward(feat);
        node.fraud_score = score;

        // Threshold at 0.5 for binary classification
        // Note: we don't overwrite node.label (ground truth stays)
        ++scored;
    }
    std::cout << "[GNN] Scored " << scored << " transaction nodes.\n";
}

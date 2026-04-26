#include "detector.h"
#include <algorithm>
#include <fstream>
#include <iostream>
#include <sstream>

// ============================================================
// detector.cpp - FraudDetector implementation
// ============================================================

// ----------------------------------------------------------
// FraudDetector (constructor)
//
// Short:
//   Build an untrained FraudDetector wired to an 11->8 MLP.
//
// Detailed:
//   Creates the owned NeuralNet with input=11 (the enriched
//   GNN feature vector) and hidden=8 neurons, marks the model
//   as untrained, and seeds rand() with a fixed value (42) so
//   that every run produces the same Xavier weight init and
//   therefore reproducible training results.
// ----------------------------------------------------------
FraudDetector::FraudDetector()
    : nn_(11, 8), is_trained_(false)
{
    // Seed random for weight initialisation
    std::srand(42);
}

// ----------------------------------------------------------
// load_csv
//
// Short:
//   Parse a transactions CSV file into the raw_data_ vector.
//
// Detailed:
//   Opens `filename`, skips the header row, and reads each
//   subsequent comma-separated row into a TxRecord
//   (tx_id, user_id, merchant_id, amount, hour, day, label).
//   Malformed rows are logged and skipped rather than aborting
//   the load. Any previous contents of raw_data_ are cleared
//   first. Returns true if at least one record was loaded.
// ----------------------------------------------------------
bool FraudDetector::load_csv(const std::string& filename)
{
    std::ifstream f(filename);
    if (!f.is_open()) {
        std::cerr << "[Detector] Cannot open CSV: " << filename << "\n";
        return false;
    }

    raw_data_.clear();

    std::string line;
    // Skip header
    if (!std::getline(f, line)) {
        std::cerr << "[Detector] CSV file is empty.\n";
        return false;
    }

    int row = 0;
    while (std::getline(f, line)) {
        if (line.empty()) continue;
        std::istringstream ss(line);
        std::string token;
        TxRecord rec;
        try {
            std::getline(ss, rec.tx_id,       ',');
            std::getline(ss, rec.user_id,     ',');
            std::getline(ss, rec.merchant_id, ',');
            std::getline(ss, token, ','); rec.amount = std::stod(token);
            std::getline(ss, token, ','); rec.hour   = std::stoi(token);
            std::getline(ss, token, ','); rec.day    = std::stoi(token);
            std::getline(ss, token, ','); rec.label  = std::stoi(token);
        } catch (...) {
            std::cerr << "[Detector] Failed to parse CSV row " << row << ": " << line << "\n";
            continue;
        }
        raw_data_.push_back(rec);
        ++row;
    }

    std::cout << "[Detector] Loaded " << raw_data_.size()
              << " records from " << filename << "\n";
    return !raw_data_.empty();
}

// ----------------------------------------------------------
// build_graph
//
// Short:
//   Construct the heterogeneous graph from raw_data_.
//
// Detailed:
//   Iterates every TxRecord and, via get_or_create, adds (or
//   reuses) a User, a Merchant, and a Transaction node with
//   placeholder feature vectors that GNN::compute_raw_features
//   will overwrite later. Two directed edges are added per
//   record: user -> transaction and transaction -> merchant,
//   so that 1-hop neighbour look-ups from a transaction reach
//   both its user and its merchant. Finally triggers
//   compute_layout so the nodes have canvas coordinates ready
//   for the GUI.
// ----------------------------------------------------------
void FraudDetector::build_graph()
{
    graph_.clear();

    for (const auto& r : raw_data_) {
        // User node (4D placeholder; features filled by GNN)
        int uid = graph_.get_or_create(r.user_id, NodeType::USER, {0,0,0,0});

        // Merchant node (3D placeholder)
        int mid = graph_.get_or_create(r.merchant_id, NodeType::MERCHANT, {0,0,0});

        // Transaction node (4D placeholder)
        int tid = graph_.get_or_create(r.tx_id, NodeType::TRANSACTION, {0,0,0,0});
        graph_.nodes[tid].label = r.label;

        // Edges: user -> transaction, transaction -> merchant
        graph_.add_edge(uid, tid);
        graph_.add_edge(tid, mid);
    }

    graph_.compute_layout(CANVAS_W, CANVAS_H);

    std::cout << "[Detector] Graph built: "
              << graph_.nodes.size() << " nodes, "
              << graph_.edges.size() << " edges.\n";
}

// ----------------------------------------------------------
// compute_features
//
// Short:
//   Delegate to the GNN to compute and normalise node features.
//
// Detailed:
//   Thin wrapper around GNN::compute_raw_features that fills
//   the feature vectors on every Transaction (4D), User (4D)
//   and Merchant (3D) node. Must be called after build_graph
//   and before train/detect, since the classifier consumes
//   those features via get_enriched_features.
// ----------------------------------------------------------
void FraudDetector::compute_features()
{
    gnn_.compute_raw_features(graph_, raw_data_);
    std::cout << "[Detector] Features computed for all nodes.\n";
}

// ----------------------------------------------------------
// train
//
// Short:
//   Train the MLP on every labelled transaction in the graph.
//
// Detailed:
//   Walks all Transaction nodes, skips those whose label is
//   unknown (-1), pulls each one's 11D enriched feature vector
//   via GNN::get_enriched_features and assembles matched
//   (X, y) training arrays. Aborts with a warning if no
//   labelled samples exist; otherwise invokes NeuralNet::train
//   for 200 epochs of SGD at learning rate 0.05 and flips
//   is_trained_ to true on success.
// ----------------------------------------------------------
void FraudDetector::train()
{
    std::vector<std::vector<double>> X;
    std::vector<double>              y;

    for (const auto& node : graph_.nodes) {
        if (node.type != NodeType::TRANSACTION) continue;
        if (node.label < 0) continue;   // skip unlabelled

        std::vector<double> feat = gnn_.get_enriched_features(graph_, node.id);
        X.push_back(feat);
        y.push_back(static_cast<double>(node.label));
    }

    if (X.empty()) {
        std::cerr << "[Detector] No labelled training samples available.\n";
        return;
    }

    std::cout << "[Detector] Training with " << X.size() << " samples...\n";
    nn_.train(X, y, 200, 0.05);
    is_trained_ = true;
}

// ----------------------------------------------------------
// detect
//
// Short:
//   Score every transaction node with the trained MLP.
//
// Detailed:
//   Refuses to run (logs an error and returns) if the model
//   has not been trained yet. Otherwise forwards to
//   GNN::run, which iterates every Transaction node, feeds
//   its enriched 11D vector through NeuralNet::forward, and
//   writes the resulting probability into node.fraud_score for
//   subsequent display and result reporting.
// ----------------------------------------------------------
void FraudDetector::detect()
{
    if (!is_trained_) {
        std::cerr << "[Detector] Model not trained yet.\n";
        return;
    }
    gnn_.run(graph_, nn_);
}

// ----------------------------------------------------------
// add_transaction
//
// Short:
//   Append a new transaction, rebuild the graph, and rescore.
//
// Detailed:
//   Used by the GUI's "Add Transaction" form. The caller
//   supplies all fields; label defaults to -1 (unknown).
//   Because user/merchant aggregates and normalisation
//   constants may change when the new row is added, the
//   simplest correct behaviour is to rebuild the entire graph
//   and recompute all features. If the MLP has already been
//   trained, detect() is then invoked so the new transaction
//   and any affected neighbours get fresh fraud scores.
// ----------------------------------------------------------
void FraudDetector::add_transaction(const std::string& tx_id,
                                     const std::string& user_id,
                                     const std::string& merchant_id,
                                     double amount, int hour, int day,
                                     int label)
{
    TxRecord r;
    r.tx_id       = tx_id;
    r.user_id     = user_id;
    r.merchant_id = merchant_id;
    r.amount      = amount;
    r.hour        = hour;
    r.day         = day;
    r.label       = label;
    raw_data_.push_back(r);

    // Rebuild everything from scratch for simplicity
    build_graph();
    compute_features();

    if (is_trained_) {
        detect();
    }

    std::cout << "[Detector] Added transaction " << tx_id << ".\n";
}

// ----------------------------------------------------------
// get_results
//
// Short:
//   Return (tx_name, fraud_score) pairs sorted by risk.
//
// Detailed:
//   Iterates the graph and collects one entry per Transaction
//   node containing its human-readable name (e.g. "T07") and
//   its current fraud_score. The result vector is sorted in
//   descending order so the highest-risk transactions appear
//   first; unscored entries (score == -1.0) therefore fall to
//   the bottom. The graph is not modified.
// ----------------------------------------------------------
std::vector<std::pair<std::string, double>>
FraudDetector::get_results() const
{
    std::vector<std::pair<std::string, double>> results;

    for (const auto& node : graph_.nodes) {
        if (node.type != NodeType::TRANSACTION) continue;
        results.emplace_back(node.name, node.fraud_score);
    }

    // Sort by fraud score descending (-1 = unscored goes to end)
    std::sort(results.begin(), results.end(),
              [](const std::pair<std::string,double>& a,
                 const std::pair<std::string,double>& b) {
                  return a.second > b.second;
              });

    return results;
}

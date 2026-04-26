#pragma once
#include "graph.h"
#include "neural_net.h"
#include "gnn.h"
#include <string>
#include <vector>

// ============================================================
// detector.h - High-level coordinator for fraud detection
// ============================================================
// FraudDetector owns the graph, the neural network and the GNN
// layer.  The GUI calls its methods in the correct order:
//   1. load_csv        -> populates raw_data
//   2. build_graph     -> creates graph nodes/edges
//   3. compute_features-> normalises and stores features
//   4. train           -> SGD training of the MLP
//   5. detect          -> scores every transaction node

class FraudDetector {
public:
    // -------------------------------------------------------
    // FraudDetector (constructor)
    //
    // Short:
    //   Build an untrained detector with a fixed-seed MLP.
    //
    // Detailed:
    //   Allocates an internal NeuralNet with 11 inputs (the
    //   enriched feature vector produced by 1-hop GNN message
    //   passing) and 8 hidden neurons. Seeds rand() with 42 so
    //   that Xavier weight initialisation is reproducible.
    // -------------------------------------------------------
    FraudDetector();

    // -------------------------------------------------------
    // load_csv
    //
    // Short:
    //   Parse a transactions CSV file into raw_data_.
    //
    // Detailed:
    //   Skips the header row and reads each subsequent row into
    //   a TxRecord (tx_id, user_id, merchant_id, amount, hour,
    //   day, label). Any previous rows are cleared. Malformed
    //   rows are logged and skipped. Returns true iff at least
    //   one record was loaded.
    // -------------------------------------------------------
    bool load_csv(const std::string& filename);

    // -------------------------------------------------------
    // build_graph
    //
    // Short:
    //   Build the heterogeneous graph from raw_data_.
    //
    // Detailed:
    //   Clears any existing graph and, for every TxRecord,
    //   gets-or-creates a User node, a Merchant node and a
    //   Transaction node, then adds the directed edges
    //   user -> transaction and transaction -> merchant. Feature
    //   vectors are created as zero placeholders here — the
    //   real values are written later by compute_features.
    //   Finally calls Graph::compute_layout so the GUI can
    //   render the nodes in three tidy columns.
    // -------------------------------------------------------
    void build_graph();

    // -------------------------------------------------------
    // compute_features
    //
    // Short:
    //   Fill every node's feature vector via the GNN layer.
    //
    // Detailed:
    //   Thin wrapper that calls GNN::compute_raw_features on
    //   the graph and raw_data_. Must be invoked after
    //   build_graph and before train/detect, since both consume
    //   the feature vectors through get_enriched_features.
    // -------------------------------------------------------
    void compute_features();

    // -------------------------------------------------------
    // train
    //
    // Short:
    //   Train the MLP on every labelled transaction.
    //
    // Detailed:
    //   Builds the (X, y) training set from Transaction nodes
    //   with a known label (0=legit, 1=fraud) by calling
    //   get_enriched_features on each. Invokes NeuralNet::train
    //   (200 epochs SGD, lr=0.05) and sets is_trained_ to true.
    //   If no labelled samples are present, logs an error and
    //   does nothing.
    // -------------------------------------------------------
    void train();

    // -------------------------------------------------------
    // detect
    //
    // Short:
    //   Score every transaction node with the trained MLP.
    //
    // Detailed:
    //   Refuses to run if the model is still untrained.
    //   Otherwise delegates to GNN::run, which forwards each
    //   transaction's 11D enriched vector through the network
    //   and writes the output probability to node.fraud_score.
    // -------------------------------------------------------
    void detect();

    // -------------------------------------------------------
    // add_transaction
    //
    // Short:
    //   Append a new transaction and rescore if already trained.
    //
    // Detailed:
    //   Used by the GUI "Add Transaction" form. Appends a
    //   TxRecord (label defaults to -1 = unknown) and rebuilds
    //   the graph plus all features because the new row can
    //   shift user/merchant aggregates and normalisation
    //   constants. If the model is already trained, detect() is
    //   called so the new transaction — and any neighbours
    //   affected by the aggregate shift — get fresh scores.
    // -------------------------------------------------------
    void add_transaction(const std::string& tx_id,
                         const std::string& user_id,
                         const std::string& merchant_id,
                         double amount,
                         int    hour,
                         int    day,
                         int    label = -1);

    // -------------------------------------------------------
    // get_results
    //
    // Short:
    //   Return (tx_name, fraud_score) pairs sorted by risk.
    //
    // Detailed:
    //   Collects one entry per Transaction node and sorts by
    //   fraud_score descending so the most suspicious ones come
    //   first; unscored nodes (score == -1.0) therefore sink to
    //   the bottom. The graph is not modified.
    // -------------------------------------------------------
    std::vector<std::pair<std::string, double>> get_results() const;

    // -------------------------------------------------------
    // get_graph (const / non-const)
    //
    // Short:
    //   Expose the owned Graph so the GUI can draw it.
    //
    // Detailed:
    //   Returns a reference to the internal graph_ member. Two
    //   overloads are provided: the const version for read-only
    //   callers (e.g. the Cairo draw callback), and a mutable
    //   one for advanced use. Callers must not keep the
    //   reference past the next graph rebuild.
    // -------------------------------------------------------
    const Graph& get_graph()      const { return graph_;      }
    Graph&       get_graph()            { return graph_;      }

    // -------------------------------------------------------
    // is_trained
    //
    // Short:
    //   Report whether train() has completed at least once.
    //
    // Detailed:
    //   Used by the GUI to gate the "Run Detection" button and
    //   to decide whether add_transaction should automatically
    //   rescore on insert.
    // -------------------------------------------------------
    bool         is_trained()     const { return is_trained_; }

    // -------------------------------------------------------
    // get_data_size
    //
    // Short:
    //   Return how many TxRecords are currently loaded.
    //
    // Detailed:
    //   Count of rows in raw_data_, cast to int for easy use
    //   in GUI status messages. Does not reflect the number of
    //   graph nodes (which is users + merchants + transactions).
    // -------------------------------------------------------
    int          get_data_size()  const { return static_cast<int>(raw_data_.size()); }

private:
    Graph                graph_;
    NeuralNet            nn_;        // input=11, hidden=8
    GNN                  gnn_;
    std::vector<TxRecord> raw_data_;
    bool                 is_trained_;

    // Canvas dimensions used for layout computation
    static constexpr int CANVAS_W = 620;
    static constexpr int CANVAS_H = 580;
};

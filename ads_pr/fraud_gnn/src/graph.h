#pragma once
#include <string>
#include <vector>
#include <unordered_map>

// ============================================================
// graph.h - Graph data structures for the fraud detection GNN
// ============================================================
// We model the financial transaction network as a heterogeneous
// graph with three types of nodes: users, merchants, and
// transactions.  Edges connect users to their transactions and
// transactions to the merchants they involve.

// ----------------------------------------------------------
// Node type enumeration
// ----------------------------------------------------------
enum class NodeType {
    USER        = 0,
    MERCHANT    = 1,
    TRANSACTION = 2
};

// ----------------------------------------------------------
// Node: one vertex in the transaction graph
// ----------------------------------------------------------
struct Node {
    int                 id;          // Unique integer index
    std::string         name;        // Human-readable name (e.g. "U1", "M3", "T07")
    NodeType            type;        // USER / MERCHANT / TRANSACTION
    std::vector<double> features;    // Raw feature vector (length depends on type)
    double              fraud_score; // Output of neural net (0-1); -1 if not yet scored
    int                 label;       // Ground-truth: 0=legit, 1=fraud, -1=unknown
    double              gx, gy;      // Canvas coordinates for visualisation
};

// ----------------------------------------------------------
// Edge: a directed edge src -> dst in the graph
// ----------------------------------------------------------
struct Edge {
    int src;
    int dst;
};

// ----------------------------------------------------------
// TxRecord: one row from the CSV file
// ----------------------------------------------------------
struct TxRecord {
    std::string tx_id;
    std::string user_id;
    std::string merchant_id;
    double      amount;
    int         hour;
    int         day;
    int         label;   // 0=legit, 1=fraud, -1=unknown
};

// ----------------------------------------------------------
// Graph: the main container for nodes and edges
// ----------------------------------------------------------
class Graph {
public:
    std::vector<Node>                    nodes;
    std::vector<Edge>                    edges;
    std::unordered_map<std::string, int> name_to_id;

    // -------------------------------------------------------
    // add_node
    //
    // Short:
    //   Create a new node and return its integer id.
    //
    // Detailed:
    //   Appends a Node to the internal vector, sets
    //   fraud_score = -1 (unscored) and label = -1 (unknown),
    //   and registers the name in name_to_id so later calls
    //   can look it up by string.
    // -------------------------------------------------------
    int  add_node(const std::string& name, NodeType type,
                  const std::vector<double>& features);

    // -------------------------------------------------------
    // get_or_create
    //
    // Short:
    //   Return the existing node id for `name`, or add a new one.
    //
    // Detailed:
    //   Idempotent helper used by build_graph so that users or
    //   merchants appearing in many rows are only added to the
    //   graph once. If `name` is already known, its id is
    //   returned and type/features are ignored.
    // -------------------------------------------------------
    int  get_or_create(const std::string& name, NodeType type,
                       const std::vector<double>& features);

    // -------------------------------------------------------
    // add_edge
    //
    // Short:
    //   Append a directed edge src -> dst (no dedup).
    //
    // Detailed:
    //   Callers are responsible for avoiding duplicate edges.
    //   Although edges carry a direction, get_neighbors
    //   traverses both ends, so queries behave as if the graph
    //   were undirected.
    // -------------------------------------------------------
    void add_edge(int src, int dst);

    // -------------------------------------------------------
    // get_neighbors
    //
    // Short:
    //   Return the ids of all nodes adjacent to `id`.
    //
    // Detailed:
    //   O(E) linear scan over the edge list that returns dst
    //   whenever src == id and src whenever dst == id, giving
    //   an undirected neighbour view suitable for 1-hop GNN
    //   aggregation.
    // -------------------------------------------------------
    std::vector<int> get_neighbors(int id) const;

    // -------------------------------------------------------
    // compute_layout
    //
    // Short:
    //   Assign canvas (gx, gy) coordinates to every node.
    //
    // Detailed:
    //   Deterministic three-column layout: users on the left,
    //   transactions in the middle, merchants on the right.
    //   Within each column nodes are spread evenly on the
    //   vertical axis between fixed top/bottom margins. No
    //   force-directed physics — it's fast and predictable.
    // -------------------------------------------------------
    void compute_layout(int canvas_width, int canvas_height);

    // -------------------------------------------------------
    // clear
    //
    // Short:
    //   Reset the graph to an empty state, keeping the object
    //   reusable.
    //
    // Detailed:
    //   Empties nodes, edges and name_to_id. Invoked at the
    //   start of every (re)build so stale state cannot leak
    //   between runs.
    // -------------------------------------------------------
    void clear();
};

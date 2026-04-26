#include "graph.h"
#include <cmath>

// ============================================================
// graph.cpp - Implementation of the Graph data structure
// ============================================================

// ----------------------------------------------------------
// add_node
//
// Short:
//   Create a new graph node and return its integer id.
//
// Detailed:
//   Appends a new Node to the nodes vector. The node gets its
//   id from the current vector size (so ids match positions),
//   and is initialised with the supplied name/type/features,
//   fraud_score = -1.0 (unscored), label = -1 (unknown) and
//   zeroed canvas coordinates. The name is also registered in
//   name_to_id so future get_or_create calls can find it.
// ----------------------------------------------------------
int Graph::add_node(const std::string& name, NodeType type,
                    const std::vector<double>& features)
{
    int idx = static_cast<int>(nodes.size());
    Node n;
    n.id          = idx;
    n.name        = name;
    n.type        = type;
    n.features    = features;
    n.fraud_score = -1.0;   // not yet scored
    n.label       = -1;     // unknown by default
    n.gx          = 0.0;
    n.gy          = 0.0;
    nodes.push_back(n);
    name_to_id[name] = idx;
    return idx;
}

// ----------------------------------------------------------
// get_or_create
//
// Short:
//   Look up a node by name or add it if not present.
//
// Detailed:
//   First consults the name_to_id hash map. If the name is
//   already registered, its existing id is returned and the
//   type/features arguments are ignored — so users and
//   merchants that appear in multiple transactions are only
//   added once. Otherwise add_node is called to create a
//   fresh node and its new id is returned.
// ----------------------------------------------------------
int Graph::get_or_create(const std::string& name, NodeType type,
                         const std::vector<double>& features)
{
    auto it = name_to_id.find(name);
    if (it != name_to_id.end()) {
        return it->second;
    }
    return add_node(name, type, features);
}

// ----------------------------------------------------------
// add_edge
//
// Short:
//   Append a directed edge src -> dst to the graph.
//
// Detailed:
//   No duplicate check is performed; callers are responsible
//   for avoiding redundant edges. Although edges are stored
//   with a direction, get_neighbors traverses them in both
//   directions so, for look-up purposes, the graph behaves as
//   undirected.
// ----------------------------------------------------------
void Graph::add_edge(int src, int dst)
{
    edges.push_back({src, dst});
}

// ----------------------------------------------------------
// get_neighbors
//
// Short:
//   Return the ids of all nodes adjacent to the given node.
//
// Detailed:
//   Linear scan of the edge list (O(E)). For each edge, if
//   the given id appears as src then dst is returned, and vice
//   versa — so the call treats the directed edge store as an
//   undirected adjacency. Simple but sufficient for the small
//   demonstration dataset; replace with an adjacency list if
//   the graph grows large.
// ----------------------------------------------------------
std::vector<int> Graph::get_neighbors(int id) const
{
    std::vector<int> result;
    for (const auto& e : edges) {
        if (e.src == id) {
            result.push_back(e.dst);
        }
        // Also traverse the reverse direction so the graph is
        // effectively undirected for neighbour look-ups.
        if (e.dst == id) {
            result.push_back(e.src);
        }
    }
    return result;
}

// ----------------------------------------------------------
// compute_layout
//
// Short:
//   Assign canvas (gx, gy) coordinates for every node.
//
// Detailed:
//   Three-column layout driven by canvas_width and
//   canvas_height:
//     Users      -> left column   (x ≈ 18 % of width)
//     Transactions -> middle column (x ≈ 50 %)
//     Merchants  -> right column  (x ≈ 82 %)
//   Within each column the nodes are spread evenly along the
//   vertical axis inside [top=60, bot=height-60], using the
//   midpoint when only a single node is present. No physics
//   simulation — the layout is deterministic and inexpensive,
//   and matches the edge structure user -> tx -> merchant.
// ----------------------------------------------------------
void Graph::compute_layout(int canvas_width, int canvas_height)
{
    // Separate nodes by type
    std::vector<int> users, merchants, transactions;
    for (const auto& n : nodes) {
        switch (n.type) {
            case NodeType::USER:        users.push_back(n.id);        break;
            case NodeType::MERCHANT:    merchants.push_back(n.id);    break;
            case NodeType::TRANSACTION: transactions.push_back(n.id); break;
        }
    }

    // ------------------------------------------------------
    // spread (lambda)
    //
    // Short:
    //   Place a list of node ids evenly along a vertical line.
    //
    // Detailed:
    //   Writes (x_pos, y) into each listed node, where y runs
    //   from `top` down to `bot`. A single node is centred at
    //   the midpoint of the range, otherwise the nodes are
    //   distributed at equal spacing that exactly hits the two
    //   endpoints.
    // ------------------------------------------------------
    auto spread = [&](const std::vector<int>& ids, double x_pos,
                      double top, double bot) {
        int n = static_cast<int>(ids.size());
        if (n == 0) return;
        for (int i = 0; i < n; ++i) {
            double y = (n == 1)
                       ? (top + bot) / 2.0
                       : top + i * (bot - top) / (n - 1);
            nodes[ids[i]].gx = x_pos;
            nodes[ids[i]].gy = y;
        }
    };

    double top = 60.0;
    double bot = canvas_height - 60.0;

    // Use fractions of canvas_width for the three columns
    double x_user   = canvas_width * 0.18;   // ~110 for 620px
    double x_tx     = canvas_width * 0.50;   // ~310
    double x_merch  = canvas_width * 0.82;   // ~508

    spread(users,        x_user,  top, bot);
    spread(transactions, x_tx,    top, bot);
    spread(merchants,    x_merch, top, bot);
}

// ----------------------------------------------------------
// clear
//
// Short:
//   Reset the graph back to an empty state.
//
// Detailed:
//   Empties the nodes vector, the edges vector and the
//   name_to_id map. Called by FraudDetector::build_graph at
//   the start of every (re)build so that stale state cannot
//   leak between loads — especially important after
//   add_transaction, which rebuilds everything.
// ----------------------------------------------------------
void Graph::clear()
{
    nodes.clear();
    edges.clear();
    name_to_id.clear();
}

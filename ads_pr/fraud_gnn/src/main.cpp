#include "detector.h"
#include "gui.h"

// ============================================================
// main.cpp - Entry point for the Fraud Detection GNN app
// ============================================================
// Build steps:
//   cd fraud_gnn
//   mkdir build && cd build
//   cmake .. && make
//   ./fraud_gnn

// ----------------------------------------------------------
// main
//
// Short:
//   Application entry point: boot GTK, load data, launch GUI.
//
// Detailed:
//   Initialises GTK before any widget work happens (a strict
//   requirement of the toolkit). Constructs the FraudDetector,
//   which in turn creates the graph, GNN and neural net.
//   Attempts to load "../data/transactions.csv" (works when
//   running from the build/ directory) and falls back to
//   "data/transactions.csv" (works when running from the
//   project root). On a successful load, builds the graph and
//   computes the normalised features so the canvas can render
//   immediately. Finally hands control to run_gui, which
//   blocks inside gtk_main until the window closes. Always
//   returns 0 — error reporting happens via stderr inside the
//   lower-level functions.
// ----------------------------------------------------------
int main(int argc, char* argv[])
{
    // Initialise GTK (must come before any GTK/GDK calls)
    gtk_init(&argc, &argv);

    // Create the main detector (owns graph + neural net + GNN)
    FraudDetector detector;

    // Load transaction data.
    // Try both relative paths depending on where the binary is run from.
    bool loaded = detector.load_csv("../data/transactions.csv");
    if (!loaded) {
        loaded = detector.load_csv("data/transactions.csv");
    }

    if (loaded) {
        // Build the graph from raw data
        detector.build_graph();
        // Compute and normalise features for every node
        detector.compute_features();
    }

    // Launch the GTK GUI; this blocks until the window is closed.
    run_gui(&detector);

    return 0;
}

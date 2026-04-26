# Financial Fraud Detection using Graph Neural Networks (GNN)

A university-level C++17 project demonstrating how graph-structured financial data can be
used to detect fraudulent transactions with a Graph Neural Network.

---

## Table of Contents

1. Project Overview
2. Learning Goals
3. What is a Graph Neural Network?
4. Why GNNs Work for Fraud Detection
5. Architecture Walkthrough
6. How to Build and Run
7. How to Use the GUI
8. Code Walkthrough
9. The Mathematics of Backpropagation
10. Interpreting Results
11. Ideas for Extension

---

## 1. Project Overview

This project models a financial transaction network as a **heterogeneous graph**:

- **User nodes** – people making payments
- **Merchant nodes** – shops / services receiving payments
- **Transaction nodes** – individual payment events

Each transaction connects a user to a merchant.  A small multi-layer perceptron (MLP) is
trained to classify each transaction as *legitimate* (0) or *fraudulent* (1) using features
that are enriched by 1-hop message passing across the graph — the key idea behind GNNs.

---

## 2. Learning Goals

After studying this project you should understand:

- How to represent a real-world problem as a graph
- Feature engineering for heterogeneous graphs
- The concept of message passing in GNNs
- Forward and backward passes in an MLP
- Stochastic gradient descent (SGD) with binary cross-entropy loss
- How to wire a data-processing backend to a GTK3 GUI
- Basic Cairo 2D drawing

---

## 3. What is a Graph Neural Network?

A **Graph Neural Network** is a type of neural network that operates directly on
graph-structured data.  Instead of treating each sample independently (as a standard MLP
would), a GNN allows each node to *communicate* with its neighbours before being classified.

The fundamental operation is **message passing**:

```
for each node v:
    aggregate information from neighbours of v
    update v's representation using the aggregated message
```

In this project we use the simplest possible version:
- Each **transaction** node collects the feature vectors of its user and merchant neighbours.
- The concatenated vector is fed to the MLP.

This single round of aggregation is called a **1-hop** GNN (or a 1-layer GraphSAGE-style
aggregation with concatenation).

---

## 4. Why GNNs Work for Fraud Detection

Fraud rarely happens in isolation:

- A fraudulent user tends to make multiple high-value, late-night transactions.
- Certain merchants appear repeatedly in fraud patterns.
- A transaction that looks borderline on its own becomes obviously suspicious when you
  see that the same user made 4 similar transactions in the last hour.

The graph structure naturally captures these relationships.  By aggregating neighbour
features, the model sees *context* that flat feature vectors would miss.

---

## 5. Architecture Walkthrough

```
CSV File
   |
   v
FraudDetector::load_csv()
   |  Parses rows into TxRecord structs
   v
FraudDetector::build_graph()
   |  Creates User / Merchant / Transaction nodes
   |  Adds edges:  User -> Transaction -> Merchant
   v
GNN::compute_raw_features()
   |  Normalises transaction, user and merchant features
   v
                    ┌─────────────────────────────────┐
  For each          │  1-hop Message Passing           │
  transaction:      │                                  │
                    │  tx_feat   (4D)                  │
                    │       +                          │
                    │  user_feat (4D)   <- neighbour   │
                    │       +                          │
                    │  merch_feat(3D)   <- neighbour   │
                    │       =                          │
                    │  enriched  (11D)                 │
                    └─────────────────────────────────┘
                                   |
                                   v
                    ┌─────────────────────────────────┐
                    │  MLP  (NeuralNet)                │
                    │                                  │
                    │  Input  (11) -> Dense+ReLU (8)   │
                    │             -> Dense+Sigmoid(1)  │
                    │                                  │
                    │  Output: fraud probability 0..1  │
                    └─────────────────────────────────┘
                                   |
                                   v
                    fraud_score stored in Node
                    Visualised as coloured diamond:
                      green  < 40%  (safe)
                      yellow 40-70% (suspicious)
                      red   >= 70%  (fraud)
```

---

## 6. How to Build and Run

### Prerequisites

Install the GTK3 development libraries:

```bash
# Debian / Ubuntu
sudo apt-get install libgtk-3-dev cmake g++ pkg-config

# Fedora / RHEL
sudo dnf install gtk3-devel cmake gcc-c++ pkg-config

# Arch Linux
sudo pacman -S gtk3 cmake base-devel pkg-config
```

### Build

```bash
cd fraud_gnn
mkdir build && cd build
cmake ..
make
```

### Run

```bash
# From inside fraud_gnn/build/
./fraud_gnn
```

The application looks for `../data/transactions.csv` relative to the working directory,
then falls back to `data/transactions.csv`.

---

## 7. How to Use the GUI

### Main window layout

```
+----------------------------------------------------+
| Financial Fraud Detection - GNN Visualiser         |
+--------------------------------------+-------------+
|                                      |             |
|  Transaction Graph Canvas            | Right Panel |
|                                      |             |
|  Blue circles   = Users              | [Add Tx]    |
|  Orange rects   = Merchants          |             |
|  Diamonds       = Transactions       | [Train]     |
|    Green  < 40% = Safe               | [Detect]    |
|    Yellow 40-70%=Suspicious          | [Reset]     |
|    Red   >= 70% = Fraud              |             |
|    Gray         = Unscored           | Results     |
|                                      | text view   |
+--------------------------------------+-------------+
| Status bar                                         |
+----------------------------------------------------+
```

### Step-by-step workflow

1. **Start** – the graph is drawn automatically from the CSV.
2. Click **Train Model** – runs 200 epochs of SGD on the labelled transactions.
   Watch the terminal for loss values.
3. Click **Run Detection** – scores all transactions and colours the diamonds.
4. **Add Transaction** – fill in user ID, merchant ID, amount, hour (0-23),
   day (0=Monday, 6=Sunday) and click "Add Transaction".  If the model is
   already trained the new transaction is scored immediately.
5. Click **Reset** – reloads the CSV and clears all scores.

---

## 8. Code Walkthrough

### graph.h / graph.cpp

Defines the graph data structure.  Key points:

- `NodeType` enum distinguishes users, merchants and transactions.
- `Node.fraud_score` starts at `-1.0` (unscored).
- `get_neighbors(id)` iterates all edges in O(E) — simple but sufficient for our dataset.
- `compute_layout()` uses fixed x-columns (users left, transactions centre, merchants right)
  and distributes nodes evenly within each column's vertical range.

### neural_net.h / neural_net.cpp

A 2-layer MLP with:

- **Xavier uniform** weight initialisation: keeps activations from vanishing or exploding at
  the start of training.
- **ReLU** hidden activation: avoids the vanishing gradient problem.
- **Sigmoid** output: squashes the scalar output to (0,1) — a probability.
- **Binary cross-entropy** loss:  `L = -y*log(p) - (1-y)*log(1-p)`
- **SGD** optimiser: one weight update per training sample.

### gnn.h / gnn.cpp

Implements feature computation and 1-hop message passing:

- `compute_raw_features` normalises all values to [0,1] so that the MLP receives
  well-scaled inputs.
- `get_enriched_features` walks the neighbour list of a transaction node and
  concatenates user + merchant features.
- `run` calls `forward()` for every transaction and stores the probability in `node.fraud_score`.

### detector.h / detector.cpp

The coordinator that ties everything together.  The GUI only calls `FraudDetector` methods
— it never touches the graph or neural net directly.

### gui.h / gui.cpp

GTK3 + Cairo rendering:

- `on_draw` is the Cairo drawing callback.  It draws edges as grey lines, user nodes as
  blue circles, merchant nodes as orange rectangles, and transaction nodes as diamonds
  coloured by their fraud score.
- All GTK signal callbacks receive an `AppState*` cast from `gpointer`.
- `gtk_widget_queue_draw(drawing_area)` schedules a redraw after any state change.

---

## 9. The Mathematics of Backpropagation

### Forward pass

```
z1[j] = b1[j] + Σ_i  W1[j,i] * x[i]        (hidden pre-activation)
a1[j] = max(0, z1[j])                         (ReLU)
z2    = b2    + Σ_j  W2[j]   * a1[j]         (output pre-activation)
p     = 1 / (1 + exp(-z2))                    (sigmoid -> probability)
```

### Loss

Binary cross-entropy for a single sample with true label y ∈ {0,1}:

```
L = -y * log(p) - (1-y) * log(1-p)
```

### Backward pass (chain rule)

The elegant cancellation when sigmoid meets cross-entropy:

```
dL/dz2 = p - y                                 (output delta)

dL/dW2[j]   = dL/dz2 * a1[j]
dL/db2       = dL/dz2

dL/da1[j]   = dL/dz2 * W2[j]
dL/dz1[j]   = dL/da1[j] * relu'(z1[j])        (* 1 if z1>0, else 0)

dL/dW1[j,i] = dL/dz1[j] * x[i]
dL/db1[j]   = dL/dz1[j]
```

### Weight update (SGD)

```
W  ← W - lr * dL/dW
```

where `lr` is the learning rate (default 0.05).

---

## 10. Interpreting Results

| Fraud score | Colour  | Interpretation                                      |
|-------------|---------|-----------------------------------------------------|
| < 40%       | Green   | Likely legitimate                                   |
| 40 – 70%    | Yellow  | Suspicious — worth manual review                    |
| ≥ 70%       | Red     | Highly likely fraud                                 |
| Not shown   | Gray    | Transaction not yet scored (model not trained)      |

The dataset contains clear fraud signals:
- High amounts (≥ 750)
- Late-night hours (hour < 6)
- Weekend days (day 0 or 6)
- Users with multiple suspicious transactions (U4, U5)
- Merchant M3 appears in many fraud cases

After training and detection, transactions T05–T08, T14, T17, T19, T23 should score red
(fraud), while T01–T04, T09–T13, T15, T16, T18, T20–T22, T24–T25 should score green (safe).

---

## 11. Ideas for Extension

### Algorithmic improvements

1. **Multi-layer GNN** – stack two rounds of message passing so each transaction sees
   2-hop neighbours (e.g. other transactions by the same user).
2. **Attention aggregation** – weight neighbours by learned importance scores (Graph
   Attention Network, GAT).
3. **Mini-batch training** – shuffle and batch samples instead of pure online SGD.
4. **Adam optimiser** – replace SGD with Adam for faster convergence.
5. **Class weighting** – fraud is rare; weight the loss by inverse class frequency.

### Data / features

6. **Time-series features** – add velocity (how many transactions in last N hours).
7. **IP / device fingerprint** – add device node type to the graph.
8. **Real datasets** – try the PaySim or IEEE-CIS Fraud Detection datasets (Kaggle).

### Software / UI

9. **Export results** – save fraud scores to a CSV with a "Save Results" button.
10. **Zoom & pan** – add mouse-drag panning and scroll-wheel zoom to the canvas.
11. **Persistent weights** – add "Save Weights" / "Load Weights" buttons to avoid
    retraining every run.
12. **Live simulation** – feed a stream of transactions every few seconds and watch the
    graph update in real time.

# Fraud Detection GNN — Function-by-Function Project Summary

## 1. What the project is

`ads_pr/fraud_gnn` is a self-contained **C++17 desktop application** that demonstrates how a
**Graph Neural Network (GNN)** can detect fraudulent financial transactions. It is built as
an educational university-style project, with a GTK3 + Cairo GUI for interactive
visualisation.

It does **not** depend on PyTorch, DGL, TensorFlow or any ML library — every piece (graph
container, MLP, backprop, GNN aggregation, GUI rendering) is hand-written in plain C++.

### High-level pipeline

```
CSV file  ──►  raw_data_ (vector<TxRecord>)
                    │
                    ▼
            build_graph()           heterogeneous graph:
                    │               User ─► Tx ─► Merchant
                    ▼
            compute_features()      normalised per-node features
                    │
                    ▼
            train()                 200 epochs SGD on labelled tx
                    │
                    ▼
            detect()                1-hop message passing → MLP
                    │                fraud_score ∈ (0,1) per tx
                    ▼
            GUI redraw              diamonds coloured by score
                                     green / yellow / red
```

### Build & run

- Build system: **CMake**, depends on **GTK3** (`pkg-config gtk+-3.0`) and `libm`.
- Compiled binary: [ads_pr/fraud_gnn/build/fraud_gnn](ads_pr/fraud_gnn/build/fraud_gnn).
- Reads [ads_pr/fraud_gnn/data/transactions.csv](ads_pr/fraud_gnn/data/transactions.csv)
  (25 sample transactions, 14 users, 15 merchants, with fraud labels in the last column).

---

## 2. Module map

| File | Role |
|------|------|
| [main.cpp](ads_pr/fraud_gnn/src/main.cpp)             | Entry point — boots GTK, creates `FraudDetector`, loads CSV, hands off to GUI |
| [graph.h](ads_pr/fraud_gnn/src/graph.h) / [graph.cpp](ads_pr/fraud_gnn/src/graph.cpp) | Heterogeneous graph data structure |
| [neural_net.h](ads_pr/fraud_gnn/src/neural_net.h) / [neural_net.cpp](ads_pr/fraud_gnn/src/neural_net.cpp) | 2-layer MLP (11 → 8 → 1) with manual SGD backprop |
| [gnn.h](ads_pr/fraud_gnn/src/gnn.h) / [gnn.cpp](ads_pr/fraud_gnn/src/gnn.cpp)         | Feature engineering + 1-hop message passing |
| [detector.h](ads_pr/fraud_gnn/src/detector.h) / [detector.cpp](ads_pr/fraud_gnn/src/detector.cpp) | Coordinator that owns the Graph, NeuralNet and GNN |
| [gui.h](ads_pr/fraud_gnn/src/gui.h) / [gui.cpp](ads_pr/fraud_gnn/src/gui.cpp)         | GTK3 + Cairo interactive visualiser |
| [CMakeLists.txt](ads_pr/fraud_gnn/CMakeLists.txt)     | Build configuration |
| [GUIDE.md](ads_pr/fraud_gnn/GUIDE.md)                  | Long-form learning guide for the project |

---

## 3. Data model

### `TxRecord` ([graph.h:47-55](ads_pr/fraud_gnn/src/graph.h#L47-L55))
One row of the CSV: `tx_id`, `user_id`, `merchant_id`, `amount`, `hour`, `day`, `label`
(0 = legit, 1 = fraud, -1 = unknown).

### `NodeType` ([graph.h:17-21](ads_pr/fraud_gnn/src/graph.h#L17-L21))
`USER`, `MERCHANT`, `TRANSACTION` — the three node kinds in the heterogeneous graph.

### `Node` ([graph.h:26-34](ads_pr/fraud_gnn/src/graph.h#L26-L34))
A vertex carrying its `id`, human name (e.g. `"T07"`), type, normalised feature vector,
ground-truth `label`, model output `fraud_score` (-1.0 if unscored), and canvas coordinates
`(gx, gy)` for the GUI.

### `Edge` ([graph.h:39-42](ads_pr/fraud_gnn/src/graph.h#L39-L42))
A directed edge `src → dst`. The graph stores them directionally but treats them as
undirected for neighbour look-ups.

---

## 4. `Graph` — [graph.cpp](ads_pr/fraud_gnn/src/graph.cpp)

Container for nodes/edges with a `name → id` hash map.

- **`add_node(name, type, features)`** ([graph.cpp:22-38](ads_pr/fraud_gnn/src/graph.cpp#L22-L38))
  Appends a new node, gives it the next free integer id, sets `fraud_score=-1` and
  `label=-1`, and registers the name in `name_to_id`.

- **`get_or_create(name, type, features)`** ([graph.cpp:54-62](ads_pr/fraud_gnn/src/graph.cpp#L54-L62))
  Idempotent variant. Returns the existing id if `name` is already in the map; otherwise
  delegates to `add_node`. This is what makes a single user/merchant appear once even when
  referenced by many CSV rows.

- **`add_edge(src, dst)`** ([graph.cpp:77-80](ads_pr/fraud_gnn/src/graph.cpp#L77-L80))
  Pushes an edge — no de-duplication.

- **`get_neighbors(id)`** ([graph.cpp:96-110](ads_pr/fraud_gnn/src/graph.cpp#L96-L110))
  O(E) linear scan that returns both `dst` (when `src==id`) and `src` (when `dst==id`),
  giving an undirected adjacency. Adequate for the small dataset; would need an adjacency
  list for scale.

- **`compute_layout(canvas_w, canvas_h)`** ([graph.cpp:130-179](ads_pr/fraud_gnn/src/graph.cpp#L130-L179))
  Deterministic three-column layout: users at 18 % width on the left, transactions at
  50 % in the middle, merchants at 82 % on the right. Internally uses a `spread` lambda
  that distributes ids evenly between the top and bottom margins (or centres a single
  node).

- **`clear()`** ([graph.cpp:194-199](ads_pr/fraud_gnn/src/graph.cpp#L194-L199))
  Empties nodes, edges and the name map. Called at the start of every (re)build to avoid
  stale state.

---

## 5. `NeuralNet` — [neural_net.cpp](ads_pr/fraud_gnn/src/neural_net.cpp)

A hand-coded 2-layer MLP: **Input(11) → Dense+ReLU(8) → Dense+Sigmoid(1)**.
Trained with online SGD on binary cross-entropy.

- **Constructor** ([neural_net.cpp:48-73](ads_pr/fraud_gnn/src/neural_net.cpp#L48-L73))
  Allocates `W1[hidden][input]`, `W2[hidden]`, biases `b1`, `b2`. All weights are filled
  with **Xavier-uniform** samples from `Uniform[-√(6/(fan_in+fan_out)), +√(...)]` so the
  initial activations stay well-scaled. Activation caches `z1_, a1_, z2_, a2_` are
  pre-allocated so forward/backward passes do no further heap work.

- **`xavier(fan_in, fan_out)`** ([neural_net.cpp:27-33](ads_pr/fraud_gnn/src/neural_net.cpp#L27-L33))
  File-static helper that draws one Xavier-uniform initial weight using `rand()`
  (seeded once with 42 by `FraudDetector`).

- **`sigmoid(x)`** ([neural_net.cpp:88-94](ads_pr/fraud_gnn/src/neural_net.cpp#L88-L94))
  Numerically stable logistic that clamps `|x|` at 20 before `exp()` to prevent overflow.

- **`relu(x)` / `relu_deriv(x)`** ([neural_net.h:175,201](ads_pr/fraud_gnn/src/neural_net.h#L175-L201))
  Inline hidden-layer activation and its derivative.

- **`forward(x)`** ([neural_net.cpp:113-139](ads_pr/fraud_gnn/src/neural_net.cpp#L113-L139))
  Two-layer feed-forward:
  - `z1[j] = b1[j] + Σᵢ W1[j][i]·x[i]`, `a1[j] = ReLU(z1[j])`
  - `z2 = b2 + Σⱼ W2[j]·a1[j]`, `a2 = σ(z2)` ← returned as the fraud probability.
  Caches every intermediate so `backward_step` can use analytic gradients without a
  second forward pass. Throws on input-size mismatch.

- **`backward_step(x, y_true, lr)`** ([neural_net.cpp:162-192](ads_pr/fraud_gnn/src/neural_net.cpp#L162-L192))
  One online SGD update derived from sigmoid + binary cross-entropy:
  - `dL/dz2 = a2 - y_true`
  - `dL/dW2[j] = dz2 · a1[j]`, `dL/db2 = dz2`
  - `dL/da1[j] = dz2 · W2[j]`, `dL/dz1[j] = dL/da1[j] · ReLU'(z1[j])`
  - `dL/dW1[j,i] = dL/dz1[j] · x[i]`, `dL/db1[j] = dL/dz1[j]`
  - All updated in place: `W ← W - lr · dL/dW`.

- **`train(X, y, epochs=200, lr=0.05)`** ([neural_net.cpp:211-245](ads_pr/fraud_gnn/src/neural_net.cpp#L211-L245))
  Online SGD: per epoch, walks every sample, accumulates the BCE loss (with `p` clamped
  to `[1e-7, 1-1e-7]` to avoid `log(0)`), then calls `backward_step`. Logs the average
  loss at epoch 1 and every 20 epochs. The order is *not* shuffled between epochs.

- **`save_weights(filename)` / `load_weights(filename)`** ([neural_net.cpp:264-338](ads_pr/fraud_gnn/src/neural_net.cpp#L264-L338))
  Plain-text serialisation: line 1 holds `input_size hidden_size`, then `W1`, `b1`, `W2`,
  `b2` row by row. Load refuses to proceed if the saved sizes don't match the current
  model.

- **`get_input_size()` / `get_hidden_size()`** ([neural_net.h:129,141](ads_pr/fraud_gnn/src/neural_net.h#L129-L141))
  Trivial accessors used by sanity checks and diagnostics.

---

## 6. `GNN` — [gnn.cpp](ads_pr/fraud_gnn/src/gnn.cpp)

Owns the feature engineering and the 1-hop message passing.

- **`compute_raw_features(g, records)`** ([gnn.cpp:49-157](ads_pr/fraud_gnn/src/gnn.cpp#L49-L157))
  Four-pass algorithm that fills every `node.features` with values in `[0, 1]`:
  1. **Globals**: maximum amount and median amount across all records.
  2. **Per-user aggregates**: tx count, total amount, count of "night" transactions
     (`hour < 6` or `hour ≥ 22`).
  3. **Per-merchant aggregates**: tx count, total amount, count of "high-amount"
     transactions (`amount > median`).
  4. **Write features back to nodes**:
     - **Transaction (4D)**:
       `[amount/max_amount, hour/24, is_night, is_weekend]`. Also copies `label`.
     - **User (4D)**:
       `[tx_count/max_user_tx, avg_amount/max_amount, night_ratio, frequent_user_flag]`.
     - **Merchant (3D)**:
       `[tx_count/max_merch_tx, avg_amount/max_amount, high_amount_ratio]`.

- **`get_enriched_features(g, tx_node_id)`** ([gnn.cpp:181-216](ads_pr/fraud_gnn/src/gnn.cpp#L181-L216))
  The actual GNN aggregation step for a single transaction:
  - Start with the transaction's own 4D features.
  - Walk its neighbours; the first `USER` neighbour contributes its 4D vector and the
    first `MERCHANT` neighbour its 3D vector. Missing slots are zero-filled.
  - Concatenate as `[ tx(4) | user(4) | merchant(3) ]` → **11D** input to the MLP.

- **`run(g, nn)`** ([gnn.cpp:233-248](ads_pr/fraud_gnn/src/gnn.cpp#L233-L248))
  Inference loop: for every Transaction node, build the 11D enriched vector, push it
  through `nn.forward`, and store the resulting probability in `node.fraud_score`.
  Ground-truth `node.label` is left untouched.

---

## 7. `FraudDetector` — [detector.cpp](ads_pr/fraud_gnn/src/detector.cpp)

The single coordinator the GUI talks to. Owns a `Graph`, a `NeuralNet(11, 8)`, a `GNN`
instance and the `raw_data_` vector.

- **Constructor** ([detector.cpp:24-29](ads_pr/fraud_gnn/src/detector.cpp#L24-L29))
  Builds the 11→8→1 MLP and seeds `rand()` with 42 so Xavier weights — and therefore
  training results — are reproducible across runs.

- **`load_csv(filename)`** ([detector.cpp:45-87](ads_pr/fraud_gnn/src/detector.cpp#L45-L87))
  Clears `raw_data_`, skips the CSV header, and parses each subsequent comma-separated
  row into a `TxRecord`. Bad rows are logged and skipped, not fatal. Returns `true` if at
  least one record loaded.

- **`build_graph()`** ([detector.cpp:106-131](ads_pr/fraud_gnn/src/detector.cpp#L106-L131))
  Wipes the graph and walks `raw_data_`. For each row it `get_or_create`s the User,
  Merchant and Transaction nodes (with zero placeholder features), copies the row's
  ground-truth label onto the transaction node, and adds two directed edges:
  `user → transaction` and `transaction → merchant`. Finally calls `compute_layout` so
  the canvas can render the result immediately.

- **`compute_features()`** ([detector.cpp:146-150](ads_pr/fraud_gnn/src/detector.cpp#L146-L150))
  Thin wrapper that calls `gnn_.compute_raw_features(graph_, raw_data_)`.

- **`train()`** ([detector.cpp:167-189](ads_pr/fraud_gnn/src/detector.cpp#L167-L189))
  Iterates Transaction nodes that have a known label, builds the `(X, y)` arrays via
  `gnn_.get_enriched_features`, then calls `nn_.train(X, y, 200, 0.05)`. Sets
  `is_trained_ = true` on success; warns and returns if there are no labelled samples.

- **`detect()`** ([detector.cpp:205-212](ads_pr/fraud_gnn/src/detector.cpp#L205-L212))
  Refuses to run if the model is untrained; otherwise delegates to `gnn_.run` to score
  every transaction node.

- **`add_transaction(tx_id, user_id, merchant_id, amount, hour, day, label=-1)`**
  ([detector.cpp:230-255](ads_pr/fraud_gnn/src/detector.cpp#L230-L255))
  Appends a new `TxRecord`, then **rebuilds the entire graph** and **recomputes all
  features** because aggregates and normalisation constants may have shifted. If the MLP
  is already trained, runs `detect()` again so the new (and any affected neighbours) get
  fresh scores.

- **`get_results()`** ([detector.cpp:271-289](ads_pr/fraud_gnn/src/detector.cpp#L271-L289))
  Returns `(tx_name, fraud_score)` pairs sorted by fraud score descending. Unscored
  entries (`-1.0`) sink to the bottom.

- **`get_graph()` / `is_trained()` / `get_data_size()`** ([detector.h:162-189](ads_pr/fraud_gnn/src/detector.h#L162-L189))
  Trivial accessors used by the GUI.

---

## 8. `main` — [main.cpp](ads_pr/fraud_gnn/src/main.cpp)

- **`main(argc, argv)`** ([main.cpp:33-59](ads_pr/fraud_gnn/src/main.cpp#L33-L59))
  Initialises GTK, constructs the `FraudDetector`, tries `../data/transactions.csv`
  first (works when run from `build/`) then falls back to `data/transactions.csv` (works
  when run from project root). On a successful load it calls `build_graph` and
  `compute_features` so the canvas can render immediately, then enters the GTK main loop
  via `run_gui`.

---

## 9. `gui.cpp` — GTK3 + Cairo visualiser

Layout: 1000 × 750 window with a title bar at top, a 620 × 580 graph canvas on the left,
a 360 px right control panel, and a status bar at the bottom.

### State

- **`AppState`** ([gui.h:32-50](ads_pr/fraud_gnn/src/gui.h#L32-L50))
  Holds the `FraudDetector*`, every widget pointer the callbacks need, the five "Add
  Transaction" entry fields, and an auto-incrementing `next_tx_id` counter.

### Drawing helpers

- **`draw_text_centred(cr, cx, cy, text, font_size)`** ([gui.cpp:26-40](ads_pr/fraud_gnn/src/gui.cpp#L26-L40))
  Measures the glyph bounding box and moves the Cairo pen so the text is centred on
  `(cx, cy)`. Used for node labels.

- **`draw_text_at(cr, x, y, text, font_size)`** ([gui.cpp:55-64](ads_pr/fraud_gnn/src/gui.cpp#L55-L64))
  Left-aligned text starting at `(x, y)` (the baseline). Used for the legend.

### Cairo render callback

- **`on_draw(widget, cr, data)`** ([gui.cpp:88-238](ads_pr/fraud_gnn/src/gui.cpp#L88-L238))
  Repaints the whole canvas:
  1. Paint dark grey background (`#2b2b2b`).
  2. If the graph is empty, show a "No graph loaded" placeholder.
  3. Stroke every edge as a thin grey line between the endpoints' `(gx, gy)`.
  4. Draw each node by type:
     - **User** → filled blue circle (radius 20) with a darker border, white label
       inside, light-blue label above.
     - **Merchant** → filled orange rectangle (44 × 26) with a brown border, black label
       inside, light-orange label above.
     - **Transaction** → diamond (half-diagonal 18) coloured by `fraud_score`:
       gray (`<0`, unscored), green (`<0.4`, safe), yellow (`<0.7`, suspicious),
       red (`≥0.7`, fraud). Name above, score percentage below (or `?%`).
  5. Overlay a six-row colour legend in the top-left corner.

### Button callbacks

- **`on_train_clicked(button, data)`** ([gui.cpp:255-268](ads_pr/fraud_gnn/src/gui.cpp#L255-L268))
  Updates the status bar to "Training...", flushes pending GTK events so the label
  actually paints, calls `detector->train()` synchronously, then updates the status bar
  again and queues a canvas redraw. Diamonds will only change colour after the user
  clicks Run Detection next.

- **`on_detect_clicked(button, data)`** ([gui.cpp:286-329](ads_pr/fraud_gnn/src/gui.cpp#L286-L329))
  Refuses if not trained. Otherwise calls `detector->detect()`, pulls the sorted result
  list, and renders it into the right-side `GtkTextView` as
  `<tx_id>  <score>%  <verdict>` rows (`SAFE`, `SUSPICIOUS`, `FRAUD`, `Unscored`). Then
  queues the canvas for a redraw so the diamonds update.

- **`on_add_clicked(button, data)`** ([gui.cpp:349-408](ads_pr/fraud_gnn/src/gui.cpp#L349-L408))
  Reads the five entry fields, validates emptiness, numeric parse, hour ∈ [0,23] and
  day ∈ [0,6]. On success generates a fresh `TX<n>` id, calls
  `detector->add_transaction(...)` with `label=-1`, clears the amount field, and queues a
  redraw. Any validation error is reported via the status label.

- **`on_reset_clicked(button, data)`** ([gui.cpp:425-449](ads_pr/fraud_gnn/src/gui.cpp#L425-L449))
  Reloads `transactions.csv` (trying both `../data/` and `data/` paths), rebuilds the
  graph, recomputes features so all `fraud_score`s reset to `-1`, clears the result text
  view and queues a redraw.

### Top-level wiring

- **`create_gui(state)`** ([gui.cpp:472-610](ads_pr/fraud_gnn/src/gui.cpp#L472-L610))
  Builds the entire widget tree:
  - Top-level `GtkWindow` (1000 × 750), titled `Financial Fraud Detection - GNN
    Visualiser`.
  - Title `GtkLabel` with Pango markup at the top.
  - Horizontal pane: drawing area (left, wired to `on_draw`) and a 360 px right
    `GtkBox`.
  - Right panel:
    - "Add Transaction" frame containing a 6-row `GtkGrid` built via the `add_row`
      lambda ([gui.cpp:538-546](ads_pr/fraud_gnn/src/gui.cpp#L538-L546)) with placeholder
      hints; an "Add Transaction" button at the bottom.
    - `Train Model` (styled `suggested-action`), `Run Detection` and `Reset (Reload
      CSV)` buttons.
    - "Detection Results" frame containing a scrollable, monospace, read-only
      `GtkTextView`.
  - Status bar `GtkLabel` at the bottom.
  - Stashes every widget pointer the callbacks need into `AppState`.

- **`run_gui(detector)`** ([gui.cpp:628-640](ads_pr/fraud_gnn/src/gui.cpp#L628-L640))
  Heap-allocates an `AppState` (must outlive the call frame because GTK callbacks fire
  inside `gtk_main`), points it at the detector, sets `next_tx_id = 100` so user-added
  transactions don't collide with the CSV's `T01..T25`, builds the widget tree, shows
  it and enters `gtk_main()`. Frees `AppState` after the loop returns.

---

## 10. End-to-end execution trace

1. `main` initialises GTK, instantiates `FraudDetector` (Xavier seed 42).
2. `load_csv` parses 25 records from `transactions.csv`.
3. `build_graph` materialises 14 users + 15 merchants + 25 transactions = 54 nodes and
   50 edges, then calls `compute_layout` so the canvas can render.
4. `compute_features` writes per-node 3D/4D normalised vectors.
5. `run_gui` opens the window. The user clicks **Train Model** → `train()` builds the
   labelled `(X, y)` set (all 25 rows are labelled in the sample CSV), runs 200 epochs
   of online SGD at lr=0.05, prints the loss curve to the terminal, sets `is_trained_`.
6. The user clicks **Run Detection** → `detect()` → `gnn_.run()` forwards each
   transaction's 11D vector through the MLP, stores `fraud_score`, and the GUI redraws
   the diamonds in green/yellow/red.
7. Optional: the user types fields and clicks **Add Transaction** → a new `TX<n>` row
   is appended, the graph is rebuilt, features recomputed, and (if trained) detection
   reruns automatically so the new node is scored on the spot.
8. **Reset** reloads the CSV from disk and clears all scores.

Per the [GUIDE.md](ads_pr/fraud_gnn/GUIDE.md) expected output, transactions T05–T08, T14,
T17, T19, T23 should end up red (fraud), and the rest green (safe), reflecting the late-
night, high-amount, weekend pattern around users U4/U5 and merchant M3.

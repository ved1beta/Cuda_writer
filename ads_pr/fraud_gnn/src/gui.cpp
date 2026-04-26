#include "gui.h"
#include <cairo/cairo.h>
#include <cmath>
#include <cstdio>
#include <sstream>
#include <string>
#include <vector>

// ============================================================
// gui.cpp - GTK3 + Cairo GUI implementation
// ============================================================

// ----------------------------------------------------------
// draw_text_centred (static helper)
//
// Short:
//   Render a text string centred on the canvas point (cx, cy).
//
// Detailed:
//   Selects the "Sans" Cairo font family at the requested
//   size, measures the string with cairo_text_extents, and
//   moves the Cairo pen so that the geometric centre of the
//   glyph bounding box lands exactly on (cx, cy). Typical use
//   is labelling nodes at their central coordinate.
// ----------------------------------------------------------
static void draw_text_centred(cairo_t* cr, double cx, double cy,
                               const std::string& text, double font_size = 9.0)
{
    cairo_select_font_face(cr, "Sans",
                           CAIRO_FONT_SLANT_NORMAL,
                           CAIRO_FONT_WEIGHT_NORMAL);
    cairo_set_font_size(cr, font_size);

    cairo_text_extents_t ext;
    cairo_text_extents(cr, text.c_str(), &ext);
    cairo_move_to(cr,
                  cx - ext.width / 2.0 - ext.x_bearing,
                  cy - ext.height / 2.0 - ext.y_bearing);
    cairo_show_text(cr, text.c_str());
}

// ----------------------------------------------------------
// draw_text_at (static helper)
//
// Short:
//   Render a left-aligned text string starting at (x, y).
//
// Detailed:
//   Selects the "Sans" Cairo font family at the requested
//   size and moves the pen to (x, y) without any bearing
//   correction, so (x, y) becomes the text baseline origin.
//   Used for the legend entries which need consistent
//   left-alignment rather than centred placement.
// ----------------------------------------------------------
static void draw_text_at(cairo_t* cr, double x, double y,
                          const std::string& text, double font_size = 9.0)
{
    cairo_select_font_face(cr, "Sans",
                           CAIRO_FONT_SLANT_NORMAL,
                           CAIRO_FONT_WEIGHT_NORMAL);
    cairo_set_font_size(cr, font_size);
    cairo_move_to(cr, x, y);
    cairo_show_text(cr, text.c_str());
}

// ----------------------------------------------------------
// on_draw
//
// Short:
//   Cairo callback that renders the entire graph canvas.
//
// Detailed:
//   Invoked by GTK whenever the drawing area needs a repaint.
//   Sequence:
//     1. Paint the dark background.
//     2. If the graph is empty, show a placeholder message and
//        return.
//     3. Draw every edge as a thin grey line between the
//        pre-computed (gx, gy) of its endpoints.
//     4. Draw each node by type — User = blue circle,
//        Merchant = orange rectangle, Transaction = diamond
//        coloured by fraud_score (gray/green/yellow/red).
//     5. Show the transaction name above the diamond and its
//        fraud probability below; unscored diamonds show "?%".
//     6. Overlay a small legend in the top-left corner.
//   Reads state from the AppState pointer passed as user data.
// ----------------------------------------------------------
gboolean on_draw(GtkWidget* /*widget*/, cairo_t* cr, gpointer data)
{
    AppState*   state    = static_cast<AppState*>(data);
    const Graph& g       = state->detector->get_graph();

    // ---- Background ----
    cairo_set_source_rgb(cr, 0.17, 0.17, 0.17);  // dark gray #2b2b2b
    cairo_paint(cr);

    if (g.nodes.empty()) {
        // Nothing to draw yet
        cairo_set_source_rgb(cr, 0.7, 0.7, 0.7);
        draw_text_centred(cr, 310, 290, "No graph loaded", 14.0);
        return FALSE;
    }

    // ---- Draw edges ----
    cairo_set_source_rgb(cr, 0.45, 0.45, 0.45);
    cairo_set_line_width(cr, 1.0);
    for (const auto& e : g.edges) {
        const Node& src = g.nodes[e.src];
        const Node& dst = g.nodes[e.dst];
        cairo_move_to(cr, src.gx, src.gy);
        cairo_line_to(cr, dst.gx, dst.gy);
        cairo_stroke(cr);
    }

    // ---- Draw nodes ----
    for (const auto& node : g.nodes) {
        if (node.type == NodeType::USER) {
            // Blue circle, radius 20
            double r = 20.0;
            cairo_set_source_rgb(cr, 0.2, 0.4, 0.8);
            cairo_arc(cr, node.gx, node.gy, r, 0, 2 * M_PI);
            cairo_fill(cr);

            // Dark border
            cairo_set_source_rgb(cr, 0.1, 0.2, 0.5);
            cairo_arc(cr, node.gx, node.gy, r, 0, 2 * M_PI);
            cairo_set_line_width(cr, 1.5);
            cairo_stroke(cr);

            // Label inside circle
            cairo_set_source_rgb(cr, 1, 1, 1);
            draw_text_centred(cr, node.gx, node.gy, node.name, 8.0);

            // Name above
            cairo_set_source_rgb(cr, 0.8, 0.8, 1.0);
            draw_text_centred(cr, node.gx, node.gy - r - 5, node.name, 8.0);

        } else if (node.type == NodeType::MERCHANT) {
            // Orange rectangle 40 x 24
            double hw = 22.0, hh = 13.0;
            cairo_set_source_rgb(cr, 0.9, 0.5, 0.1);
            cairo_rectangle(cr, node.gx - hw, node.gy - hh, hw * 2, hh * 2);
            cairo_fill(cr);

            // Border
            cairo_set_source_rgb(cr, 0.6, 0.3, 0.0);
            cairo_rectangle(cr, node.gx - hw, node.gy - hh, hw * 2, hh * 2);
            cairo_set_line_width(cr, 1.5);
            cairo_stroke(cr);

            // Label inside
            cairo_set_source_rgb(cr, 0.0, 0.0, 0.0);
            draw_text_centred(cr, node.gx, node.gy, node.name, 8.0);

            // Name above
            cairo_set_source_rgb(cr, 1.0, 0.8, 0.5);
            draw_text_centred(cr, node.gx, node.gy - hh - 5, node.name, 8.0);

        } else {
            // Transaction: diamond shape
            double score = node.fraud_score;

            // Choose colour based on fraud score
            if (score < 0.0) {
                // Not yet scored -> gray
                cairo_set_source_rgb(cr, 0.5, 0.5, 0.5);
            } else if (score < 0.4) {
                // Safe -> green
                cairo_set_source_rgb(cr, 0.2, 0.7, 0.2);
            } else if (score < 0.7) {
                // Suspicious -> yellow
                cairo_set_source_rgb(cr, 0.9, 0.8, 0.1);
            } else {
                // Fraud -> red
                cairo_set_source_rgb(cr, 0.8, 0.1, 0.1);
            }

            // Draw diamond: 4 vertices top, right, bottom, left
            double d = 18.0;  // half-diagonal
            cairo_move_to(cr, node.gx,       node.gy - d);  // top
            cairo_line_to(cr, node.gx + d,   node.gy);      // right
            cairo_line_to(cr, node.gx,       node.gy + d);  // bottom
            cairo_line_to(cr, node.gx - d,   node.gy);      // left
            cairo_close_path(cr);
            cairo_fill(cr);

            // Dark border on diamond
            cairo_set_source_rgb(cr, 0.0, 0.0, 0.0);
            cairo_move_to(cr, node.gx,       node.gy - d);
            cairo_line_to(cr, node.gx + d,   node.gy);
            cairo_line_to(cr, node.gx,       node.gy + d);
            cairo_line_to(cr, node.gx - d,   node.gy);
            cairo_close_path(cr);
            cairo_set_line_width(cr, 1.2);
            cairo_stroke(cr);

            // Transaction name above the diamond
            cairo_set_source_rgb(cr, 0.9, 0.9, 0.9);
            draw_text_centred(cr, node.gx, node.gy - d - 5, node.name, 7.5);

            // Fraud score percentage below the diamond
            if (score >= 0.0) {
                char buf[16];
                std::snprintf(buf, sizeof(buf), "%.0f%%", score * 100.0);
                cairo_set_source_rgb(cr, 1.0, 1.0, 1.0);
                draw_text_centred(cr, node.gx, node.gy + d + 10, std::string(buf), 7.5);
            } else {
                cairo_set_source_rgb(cr, 0.6, 0.6, 0.6);
                draw_text_centred(cr, node.gx, node.gy + d + 10, "?%", 7.5);
            }
        }
    }

    // ---- Legend ----
    double lx = 10, ly = 10;
    cairo_set_font_size(cr, 8.0);

    // Legend entries
    struct LegendEntry { double r, g, b; const char* label; };
    LegendEntry entries[] = {
        {0.2, 0.4, 0.8, "User"},
        {0.9, 0.5, 0.1, "Merchant"},
        {0.2, 0.7, 0.2, "Tx: Safe (<40%)"},
        {0.9, 0.8, 0.1, "Tx: Suspect (40-70%)"},
        {0.8, 0.1, 0.1, "Tx: Fraud (>=70%)"},
        {0.5, 0.5, 0.5, "Tx: Unscored"},
    };
    for (const auto& e : entries) {
        cairo_set_source_rgb(cr, e.r, e.g, e.b);
        cairo_rectangle(cr, lx, ly, 10, 10);
        cairo_fill(cr);
        cairo_set_source_rgb(cr, 0.9, 0.9, 0.9);
        draw_text_at(cr, lx + 13, ly + 9, e.label, 8.0);
        ly += 14;
    }

    return FALSE;
}

// ----------------------------------------------------------
// on_train_clicked
//
// Short:
//   GTK callback for the "Train Model" button.
//
// Detailed:
//   Updates the status label to let the user know training is
//   running, then drains any pending GTK events so the label
//   actually appears before the main thread blocks inside
//   FraudDetector::train (this implementation is synchronous).
//   When training returns, the label is updated again and the
//   drawing area is queued for a redraw. Node colours will
//   only change after the user then clicks "Run Detection".
// ----------------------------------------------------------
void on_train_clicked(GtkWidget* /*button*/, gpointer data)
{
    AppState* state = static_cast<AppState*>(data);
    gtk_label_set_text(GTK_LABEL(state->status_label),
                       "Training model... please wait.");
    // Flush pending UI events so the label updates before we block
    while (gtk_events_pending()) gtk_main_iteration();

    state->detector->train();

    gtk_label_set_text(GTK_LABEL(state->status_label),
                       "Training complete! Now click 'Run Detection'.");
    gtk_widget_queue_draw(state->drawing_area);
}

// ----------------------------------------------------------
// on_detect_clicked
//
// Short:
//   GTK callback for the "Run Detection" button.
//
// Detailed:
//   Requires that the model has been trained; otherwise sets
//   a warning in the status bar and returns. When training is
//   in place it invokes FraudDetector::detect, then pulls the
//   sorted results list and renders it into the text view as
//   "<tx_id>  <score>%  <verdict>" rows. Verdicts follow the
//   same thresholds as the canvas colours: SAFE < 40 %,
//   SUSPICIOUS 40–70 %, FRAUD ≥ 70 %. Finally requests a
//   canvas redraw so the diamonds pick up their new colours.
// ----------------------------------------------------------
void on_detect_clicked(GtkWidget* /*button*/, gpointer data)
{
    AppState* state = static_cast<AppState*>(data);

    if (!state->detector->is_trained()) {
        gtk_label_set_text(GTK_LABEL(state->status_label),
                           "Please train the model first!");
        return;
    }

    state->detector->detect();

    // Update result text view
    auto results = state->detector->get_results();
    std::ostringstream oss;
    oss << "Transaction    Score    Verdict\n";
    oss << "-------------------------------\n";
    for (const auto& p : results) {
        double sc = p.second;
        const char* verdict;
        if (sc < 0)        verdict = "Unscored";
        else if (sc < 0.4) verdict = "SAFE";
        else if (sc < 0.7) verdict = "SUSPICIOUS";
        else               verdict = "FRAUD";

        char line[80];
        if (sc >= 0) {
            std::snprintf(line, sizeof(line), "%-12s   %5.1f%%   %s\n",
                          p.first.c_str(), sc * 100.0, verdict);
        } else {
            std::snprintf(line, sizeof(line), "%-12s    ---     %s\n",
                          p.first.c_str(), verdict);
        }
        oss << line;
    }

    GtkTextBuffer* buf =
        gtk_text_view_get_buffer(GTK_TEXT_VIEW(state->result_view));
    gtk_text_buffer_set_text(buf, oss.str().c_str(), -1);

    gtk_label_set_text(GTK_LABEL(state->status_label),
                       "Detection complete. Results shown on graph and in panel.");
    gtk_widget_queue_draw(state->drawing_area);
}

// ----------------------------------------------------------
// on_add_clicked
//
// Short:
//   GTK callback for the "Add Transaction" button.
//
// Detailed:
//   Reads the five entry widgets (user, merchant, amount,
//   hour, day), validates that all are filled and that the
//   numeric fields parse plus lie in their allowed ranges
//   (hour 0-23, day 0-6). On any failure it writes a helpful
//   message to the status label and returns without touching
//   the detector. Otherwise it allocates a fresh tx_id of the
//   form "TX<next_tx_id>", forwards to
//   FraudDetector::add_transaction (label = -1, unknown), and
//   redraws the canvas. The amount field is cleared to make
//   repeated entries easier.
// ----------------------------------------------------------
void on_add_clicked(GtkWidget* /*button*/, gpointer data)
{
    AppState* state = static_cast<AppState*>(data);

    const char* user_txt    = gtk_entry_get_text(GTK_ENTRY(state->user_entry));
    const char* merch_txt   = gtk_entry_get_text(GTK_ENTRY(state->merchant_entry));
    const char* amount_txt  = gtk_entry_get_text(GTK_ENTRY(state->amount_entry));
    const char* hour_txt    = gtk_entry_get_text(GTK_ENTRY(state->hour_entry));
    const char* day_txt     = gtk_entry_get_text(GTK_ENTRY(state->day_entry));

    if (!user_txt[0] || !merch_txt[0] || !amount_txt[0] ||
        !hour_txt[0] || !day_txt[0])
    {
        gtk_label_set_text(GTK_LABEL(state->status_label),
                           "Please fill all fields before adding.");
        return;
    }

    double amount = 0.0;
    int    hour   = 0, day = 0;
    try {
        amount = std::stod(amount_txt);
        hour   = std::stoi(hour_txt);
        day    = std::stoi(day_txt);
    } catch (...) {
        gtk_label_set_text(GTK_LABEL(state->status_label),
                           "Invalid number in amount/hour/day field.");
        return;
    }

    if (hour < 0 || hour > 23) {
        gtk_label_set_text(GTK_LABEL(state->status_label), "Hour must be 0-23.");
        return;
    }
    if (day < 0 || day > 6) {
        gtk_label_set_text(GTK_LABEL(state->status_label), "Day must be 0-6 (Mon-Sun).");
        return;
    }

    // Generate a unique transaction ID
    char tx_id[32];
    std::snprintf(tx_id, sizeof(tx_id), "TX%d", state->next_tx_id++);

    state->detector->add_transaction(tx_id,
                                     std::string(user_txt),
                                     std::string(merch_txt),
                                     amount, hour, day,
                                     -1 /* unknown label */);

    char msg[128];
    std::snprintf(msg, sizeof(msg),
                  "Added %s (user=%s, merchant=%s, amount=%.2f).",
                  tx_id, user_txt, merch_txt, amount);
    gtk_label_set_text(GTK_LABEL(state->status_label), msg);

    // Clear entry fields for convenience
    gtk_entry_set_text(GTK_ENTRY(state->amount_entry), "");

    gtk_widget_queue_draw(state->drawing_area);
}

// ----------------------------------------------------------
// on_reset_clicked
//
// Short:
//   GTK callback for the "Reset (Reload CSV)" button.
//
// Detailed:
//   Reloads the original dataset by trying
//   "../data/transactions.csv" first (build-directory layout)
//   and then "data/transactions.csv" as a fallback. On
//   success, rebuilds the graph and recomputes features so
//   all fraud scores drop back to -1 (unscored); on failure,
//   reports the problem in the status bar. Also clears the
//   result text view and redraws the canvas.
// ----------------------------------------------------------
void on_reset_clicked(GtkWidget* /*button*/, gpointer data)
{
    AppState* state = static_cast<AppState*>(data);

    // Attempt to reload from both possible paths
    bool ok = state->detector->load_csv("../data/transactions.csv");
    if (!ok) ok = state->detector->load_csv("data/transactions.csv");

    if (ok) {
        state->detector->build_graph();
        state->detector->compute_features();
        gtk_label_set_text(GTK_LABEL(state->status_label),
                           "Reset complete. Train and detect to score transactions.");
    } else {
        gtk_label_set_text(GTK_LABEL(state->status_label),
                           "Reset failed: could not reload CSV.");
    }

    // Clear results
    GtkTextBuffer* buf =
        gtk_text_view_get_buffer(GTK_TEXT_VIEW(state->result_view));
    gtk_text_buffer_set_text(buf, "", -1);

    gtk_widget_queue_draw(state->drawing_area);
}

// ----------------------------------------------------------
// create_gui
//
// Short:
//   Build every widget and return the top-level window.
//
// Detailed:
//   Constructs the window (1000 x 750), a root vertical box
//   with a title / main-hbox / status bar, the drawing area
//   on the left (620 x 580, wired to on_draw), and the right
//   panel containing:
//     - the "Add Transaction" form (five entry fields + Add
//       button, wired to on_add_clicked)
//     - Train / Detect / Reset buttons wired to their
//       respective callbacks
//     - a scrolled, monospace GtkTextView for the detection
//       results
//   All widgets that callbacks need to reach are stashed into
//   the shared AppState pointer. Returns the window without
//   showing it — the caller calls gtk_widget_show_all.
// ----------------------------------------------------------
GtkWidget* create_gui(AppState* state)
{
    // ---- Main window ----
    GtkWidget* window = gtk_window_new(GTK_WINDOW_TOPLEVEL);
    state->window = window;
    gtk_window_set_title(GTK_WINDOW(window),
                         "Financial Fraud Detection - GNN Visualiser");
    gtk_window_set_default_size(GTK_WINDOW(window), 1000, 750);
    gtk_container_set_border_width(GTK_CONTAINER(window), 5);
    g_signal_connect(window, "destroy",
                     G_CALLBACK(gtk_main_quit), NULL);

    // ---- Root vertical box: title / content / status ----
    GtkWidget* root_vbox = gtk_box_new(GTK_ORIENTATION_VERTICAL, 4);
    gtk_container_add(GTK_CONTAINER(window), root_vbox);

    // Title label
    GtkWidget* title = gtk_label_new(NULL);
    gtk_label_set_markup(GTK_LABEL(title),
        "<span font='14' weight='bold' foreground='#4488ff'>"
        "Financial Fraud Detection using Graph Neural Networks"
        "</span>");
    gtk_box_pack_start(GTK_BOX(root_vbox), title, FALSE, FALSE, 4);

    // ---- Horizontal pane: canvas + right panel ----
    GtkWidget* hbox = gtk_box_new(GTK_ORIENTATION_HORIZONTAL, 6);
    gtk_box_pack_start(GTK_BOX(root_vbox), hbox, TRUE, TRUE, 0);

    // ---- Graph canvas (left) ----
    GtkWidget* canvas_frame = gtk_frame_new("Transaction Graph");
    gtk_box_pack_start(GTK_BOX(hbox), canvas_frame, TRUE, TRUE, 0);

    state->drawing_area = gtk_drawing_area_new();
    gtk_widget_set_size_request(state->drawing_area, 620, 580);
    gtk_container_add(GTK_CONTAINER(canvas_frame), state->drawing_area);
    g_signal_connect(state->drawing_area, "draw",
                     G_CALLBACK(on_draw), (gpointer)state);

    // ---- Right panel ----
    GtkWidget* right_vbox = gtk_box_new(GTK_ORIENTATION_VERTICAL, 6);
    gtk_widget_set_size_request(right_vbox, 360, -1);
    gtk_box_pack_start(GTK_BOX(hbox), right_vbox, FALSE, FALSE, 0);

    // "Add Transaction" frame
    GtkWidget* add_frame = gtk_frame_new("Add Transaction");
    gtk_box_pack_start(GTK_BOX(right_vbox), add_frame, FALSE, FALSE, 0);

    GtkWidget* add_grid = gtk_grid_new();
    gtk_grid_set_row_spacing   (GTK_GRID(add_grid), 4);
    gtk_grid_set_column_spacing(GTK_GRID(add_grid), 6);
    gtk_container_set_border_width(GTK_CONTAINER(add_grid), 8);
    gtk_container_add(GTK_CONTAINER(add_frame), add_grid);

    // ------------------------------------------------------
    // add_row (lambda)
    //
    // Short:
    //   Attach one "label : entry" pair to the add_grid.
    //
    // Detailed:
    //   Creates a left-aligned GtkLabel for `lbl_text` at
    //   column 0 and a 12-character-wide GtkEntry at column 1
    //   on the given grid row. Writes the entry pointer out
    //   through `entry_out` so the callback can read its text
    //   later.
    // ------------------------------------------------------
    auto add_row = [&](int row, const char* lbl_text, GtkWidget** entry_out) {
        GtkWidget* lbl = gtk_label_new(lbl_text);
        gtk_widget_set_halign(lbl, GTK_ALIGN_START);
        gtk_grid_attach(GTK_GRID(add_grid), lbl, 0, row, 1, 1);
        GtkWidget* ent = gtk_entry_new();
        gtk_entry_set_width_chars(GTK_ENTRY(ent), 12);
        gtk_grid_attach(GTK_GRID(add_grid), ent, 1, row, 1, 1);
        *entry_out = ent;
    };

    add_row(0, "User ID:",     &state->user_entry);
    add_row(1, "Merchant ID:", &state->merchant_entry);
    add_row(2, "Amount:",      &state->amount_entry);
    add_row(3, "Hour (0-23):", &state->hour_entry);
    add_row(4, "Day (0-6):",   &state->day_entry);

    // Set placeholder text hints
    gtk_entry_set_placeholder_text(GTK_ENTRY(state->user_entry),     "e.g. U15");
    gtk_entry_set_placeholder_text(GTK_ENTRY(state->merchant_entry), "e.g. M3");
    gtk_entry_set_placeholder_text(GTK_ENTRY(state->amount_entry),   "e.g. 950.00");
    gtk_entry_set_placeholder_text(GTK_ENTRY(state->hour_entry),     "e.g. 3");
    gtk_entry_set_placeholder_text(GTK_ENTRY(state->day_entry),      "e.g. 6");

    GtkWidget* add_btn = gtk_button_new_with_label("Add Transaction");
    gtk_grid_attach(GTK_GRID(add_grid), add_btn, 0, 5, 2, 1);
    g_signal_connect(add_btn, "clicked",
                     G_CALLBACK(on_add_clicked), (gpointer)state);

    // ---- Control buttons ----
    GtkWidget* train_btn = gtk_button_new_with_label("Train Model");
    {
        // Style: green background
        GtkStyleContext* ctx = gtk_widget_get_style_context(train_btn);
        gtk_style_context_add_class(ctx, "suggested-action");
    }
    gtk_box_pack_start(GTK_BOX(right_vbox), train_btn, FALSE, FALSE, 0);
    g_signal_connect(train_btn, "clicked",
                     G_CALLBACK(on_train_clicked), (gpointer)state);

    GtkWidget* detect_btn = gtk_button_new_with_label("Run Detection");
    gtk_box_pack_start(GTK_BOX(right_vbox), detect_btn, FALSE, FALSE, 0);
    g_signal_connect(detect_btn, "clicked",
                     G_CALLBACK(on_detect_clicked), (gpointer)state);

    GtkWidget* reset_btn = gtk_button_new_with_label("Reset (Reload CSV)");
    gtk_box_pack_start(GTK_BOX(right_vbox), reset_btn, FALSE, FALSE, 0);
    g_signal_connect(reset_btn, "clicked",
                     G_CALLBACK(on_reset_clicked), (gpointer)state);

    // ---- Results panel ----
    GtkWidget* res_frame = gtk_frame_new("Detection Results");
    gtk_box_pack_start(GTK_BOX(right_vbox), res_frame, TRUE, TRUE, 0);

    GtkWidget* scroll = gtk_scrolled_window_new(NULL, NULL);
    gtk_scrolled_window_set_policy(GTK_SCROLLED_WINDOW(scroll),
                                   GTK_POLICY_AUTOMATIC,
                                   GTK_POLICY_AUTOMATIC);
    gtk_container_add(GTK_CONTAINER(res_frame), scroll);

    state->result_view = gtk_text_view_new();
    gtk_text_view_set_editable(GTK_TEXT_VIEW(state->result_view), FALSE);
    gtk_text_view_set_monospace(GTK_TEXT_VIEW(state->result_view), TRUE);
    gtk_text_view_set_wrap_mode(GTK_TEXT_VIEW(state->result_view), GTK_WRAP_NONE);
    gtk_container_add(GTK_CONTAINER(scroll), state->result_view);

    // ---- Status bar ----
    state->status_label = gtk_label_new(
        "Ready. Load CSV, build graph, then Train and Detect.");
    gtk_widget_set_halign(state->status_label, GTK_ALIGN_START);
    gtk_box_pack_start(GTK_BOX(root_vbox), state->status_label, FALSE, FALSE, 2);

    return window;
}

// ----------------------------------------------------------
// run_gui
//
// Short:
//   Top-level GUI entry point invoked from main().
//
// Detailed:
//   Heap-allocates an AppState so the struct outlives this
//   function's stack frame (needed because GTK callbacks run
//   after run_gui returns up the call chain inside gtk_main).
//   Wires the detector pointer, initialises the next_tx_id
//   counter at 100 so user-added transactions don't collide
//   with CSV ones, builds the widget tree via create_gui,
//   shows everything and enters the GTK main loop. The
//   AppState is freed after the window is closed.
// ----------------------------------------------------------
void run_gui(FraudDetector* detector)
{
    // Allocate AppState on the heap so it outlives this function's stack
    AppState* state = new AppState();
    state->detector   = detector;
    state->next_tx_id = 100;   // New transactions start at TX100

    GtkWidget* window = create_gui(state);
    gtk_widget_show_all(window);
    gtk_main();

    delete state;
}

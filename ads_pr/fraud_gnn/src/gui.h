#pragma once
#include <gtk/gtk.h>
#include "detector.h"

// ============================================================
// gui.h - GTK3 graphical user interface declarations
// ============================================================
//
// Layout (1000 x 750 window):
// +--------------------------------------------------+
// |  Title label                                     |
// +--------------------------------------+-----------+
// |                                      |           |
// |  Graph Canvas  (GtkDrawingArea)      | Right     |
// |  620 x 580 px                        | Panel     |
// |                                      | 360 px    |
// |                                      |           |
// +--------------------------------------+-----------+
// |  Status bar label                                |
// +--------------------------------------------------+
//
// Right panel (top-to-bottom):
//   - "Add Transaction" frame  (5 entry fields + [Add] button)
//   - [Train Model] button
//   - [Run Detection] button
//   - [Reset] button
//   - "Detection Results" frame  (scrolled GtkTextView)

// ----------------------------------------------------------
// AppState: passed as user-data to every GTK signal callback
// ----------------------------------------------------------
struct AppState {
    FraudDetector* detector;

    // Widgets that callbacks need to access
    GtkWidget* window;
    GtkWidget* drawing_area;
    GtkWidget* result_view;    // GtkTextView
    GtkWidget* status_label;

    // "Add Transaction" input fields
    GtkWidget* user_entry;
    GtkWidget* merchant_entry;
    GtkWidget* amount_entry;
    GtkWidget* hour_entry;
    GtkWidget* day_entry;

    // Auto-incrementing counter for new tx IDs
    int next_tx_id;
};

// ----------------------------------------------------------
// Signal callbacks (defined in gui.cpp)
// ----------------------------------------------------------

// ----------------------------------------------------------
// on_draw
//
// Short:
//   Cairo callback that repaints the transaction graph.
//
// Detailed:
//   Paints the background, edges, legend and every node —
//   users as blue circles, merchants as orange rectangles,
//   transactions as diamonds coloured by fraud_score. Reads
//   all state through the AppState* passed as user data.
// ----------------------------------------------------------
gboolean on_draw         (GtkWidget* widget, cairo_t* cr, gpointer data);

// ----------------------------------------------------------
// on_train_clicked
//
// Short:
//   Handle clicks on the "Train Model" button.
//
// Detailed:
//   Updates the status label, flushes pending GTK events so
//   the label appears, runs FraudDetector::train
//   synchronously, then requests a canvas redraw and restores
//   the status label.
// ----------------------------------------------------------
void     on_train_clicked(GtkWidget* button, gpointer data);

// ----------------------------------------------------------
// on_detect_clicked
//
// Short:
//   Handle clicks on the "Run Detection" button.
//
// Detailed:
//   Refuses to run if the model is not yet trained. Otherwise
//   calls FraudDetector::detect, formats the sorted result
//   list into the text view ("tx  score  verdict") and queues
//   the canvas for a redraw so the diamond colours refresh.
// ----------------------------------------------------------
void     on_detect_clicked(GtkWidget* button, gpointer data);

// ----------------------------------------------------------
// on_add_clicked
//
// Short:
//   Handle clicks on the "Add Transaction" button.
//
// Detailed:
//   Validates the five input fields (user, merchant, amount,
//   hour 0-23, day 0-6) and, if all are valid, forwards a new
//   auto-numbered "TXnn" transaction to
//   FraudDetector::add_transaction (label unknown). Errors
//   are reported via the status label.
// ----------------------------------------------------------
void     on_add_clicked  (GtkWidget* button, gpointer data);

// ----------------------------------------------------------
// on_reset_clicked
//
// Short:
//   Handle clicks on the "Reset (Reload CSV)" button.
//
// Detailed:
//   Reloads transactions.csv from the build-relative or
//   project-relative path, rebuilds the graph and features,
//   clears the result text view and redraws the canvas so all
//   fraud scores return to "?".
// ----------------------------------------------------------
void     on_reset_clicked(GtkWidget* button, gpointer data);

// ----------------------------------------------------------
// Public API
// ----------------------------------------------------------

// ----------------------------------------------------------
// create_gui
//
// Short:
//   Construct and return the top-level GTK window.
//
// Detailed:
//   Builds the complete widget hierarchy (title, drawing
//   area, add-transaction form, control buttons, results
//   view, status bar), wires signal handlers and stores
//   widget pointers back into the given AppState so callbacks
//   can reach them. The window is returned unshown so the
//   caller can show it at the appropriate moment.
// ----------------------------------------------------------
GtkWidget* create_gui(AppState* state);

// ----------------------------------------------------------
// run_gui
//
// Short:
//   Program entry point for the GUI; blocks in gtk_main.
//
// Detailed:
//   Heap-allocates an AppState (needed because GTK callbacks
//   outlive the stack frame of run_gui), points it at the
//   supplied detector, builds the widget tree via create_gui,
//   shows it and enters the GTK event loop. On loop exit the
//   AppState is freed.
// ----------------------------------------------------------
void run_gui(FraudDetector* detector);

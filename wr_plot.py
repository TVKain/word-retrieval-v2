from datetime import datetime
from matplotlib import pyplot as plt
from pathlib import Path

def plot_layer_accuracy(layer_accuracy: dict[int, float], save_folder: str, note: str = "") -> Path:
    """
    Plot layer accuracy from a dictionary and save the plot, with a boxed note
    displayed above the plot without overlapping it.
    
    Args:
        layer_accuracy (dict[int, float]): Mapping of layer index to accuracy value.
        save_folder (str): Directory to save the generated plot.
        note (str): Optional multi-line note to display above the plot.
    
    Returns:
        Path: Path to the saved plot image.
    """
    if not layer_accuracy:
        raise ValueError("layer_accuracy is empty.")
    
    # Sort by layer index
    sorted_layers = sorted(layer_accuracy.items(), key=lambda x: x[0])
    layers = [layer for layer, _ in sorted_layers]
    accuracies = [acc for _, acc in sorted_layers]
    
    # Create the plot with extra height if there's a note
    if note:
        n_lines = note.count("\n") + 1
        # Add extra height for the note (roughly 0.3 inches per line)
        extra_height = n_lines * 0.3
        fig_height = 5 + extra_height
    else:
        fig_height = 5
    
    fig, ax = plt.subplots(figsize=(12, fig_height))
    
    # Plot the data
    ax.plot(layers, accuracies, marker="o", label="Layer Accuracy", linewidth=2, markersize=6)
    
    # Labels and style
    ax.set_xlabel("Layer", fontsize=12)
    ax.set_ylabel("Accuracy", fontsize=12)
    ax.set_title("Word Retrieval Accuracy per Layer", fontsize=14, pad=20)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=11)
    
    if note:
        # Adjust subplot to leave space at top for the note
        # The bottom of the note area should be at least 0.02 above the title
        top_space_needed = (n_lines * 0.04) + 0.15  # Space for note + padding
        subplot_top = 1.0 - top_space_needed
        fig.subplots_adjust(top=subplot_top)
        
        # Position note in the space above the plot
        note_y_position = subplot_top + 0.08  # Small gap above the plot area
        
        fig.text(
            0.02, note_y_position,  # left-aligned, positioned above plot
            note,
            fontsize=10,
            ha='left',
            va='bottom',  # Changed to bottom alignment for more predictable positioning
            wrap=True,
            bbox=dict(boxstyle="round,pad=0.5", facecolor="lightblue", 
                     edgecolor="navy", alpha=0.8)
        )
    
    # Save the plot
    Path(save_folder).mkdir(exist_ok=True, parents=True)
    timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    plot_path = Path(save_folder) / f"{timestamp_str}_layer_accuracy_plot.png"
    plt.savefig(plot_path, bbox_inches="tight", dpi=150)
    plt.close()
    
    print(f"Plot saved to {plot_path}")
    return plot_path
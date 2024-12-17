import os
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import argparse
import matplotlib.pyplot as plt

def extract_loss_from_tensorboard(event_file):
    """
    Reads a TensorBoard event file and extracts the loss values recorded during training.

    Args:
        event_file (str): Path to the TensorBoard event file.
    
    Returns:
        list of tuple: A list of (step, loss_value) tuples recorded during training.
    """
    # Load the event file
    event_acc = EventAccumulator(event_file)
    event_acc.Reload()  # Load the file contents

    # Get all scalar tags
    tags = event_acc.Tags().get("scalars", [])
    print(f"Found scalar tags: {tags}")

    # Extract loss values (assuming loss is recorded under the correct scalar tag)
    loss_tag = "Epoch loss"  # Updated to the correct tag name
    if loss_tag not in tags:
        raise ValueError(f"Loss tag '{loss_tag}' not found in the event file.")

    loss_events = event_acc.Scalars(loss_tag)

    # Convert loss events to (step, loss_value) tuples
    loss_values = [(event.step, event.value) for event in loss_events]
    return loss_values

def plot_loss(loss_values):
    """
    Plots the loss values over training steps.

    Args:
        loss_values (list of tuple): A list of (step, loss_value) tuples.
    """
    steps = [x for (x, _) in loss_values]
    losses = [y for (_, y) in loss_values]

    plt.figure(figsize=(10, 6))
    plt.plot(steps, losses, marker='o', linestyle='-', color='blue')
    plt.title("Training Loss Over Steps")
    plt.xlabel("Step")
    plt.ylabel("Loss")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def main():
    # Path to your TensorBoard event file
    parser = argparse.ArgumentParser(description="Extract and plot training loss from TensorBoard event file.")
    parser.add_argument(
        "--event_file",
        type=str,
        required=True,
        help="Path to the TensorBoard event file.",
    )

    args = parser.parse_args()

    if not os.path.exists(args.event_file):
        print(f"Event file not found: {args.event_file}")
        return

    try:
        loss_values = extract_loss_from_tensorboard(args.event_file)
        if loss_values:
            print("Loss values recorded during training:")
            for step, loss in loss_values:
                print(f"Step {step}: Loss = {loss}")
            
            # Plot the loss
            plot_loss(loss_values)
        else:
            print("No loss values found in the event file.")
    except Exception as e:
        print(f"An error occurred: {e}")


if __name__ == "__main__":
    main()

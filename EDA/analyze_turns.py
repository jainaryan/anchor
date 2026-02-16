import json
import matplotlib.pyplot as plt
import os

# Define file paths
input_file = 'data/new_raw_data/mindmate_train.jsonl'
output_graph = 'EDA/turns_bar_graph.png'
output_samples = 'EDA/one_turn_samples.json'

# Initialize counters
turn_counts = {
    '1': 0,
    '2': 0,
    '3': 0,
    '3+': 0
}

one_turn_samples = []
sample_target = 10

try:
    with open(input_file, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                data = json.loads(line)
                text = data.get('text', '')
                
                # Count turns based on occurrences of "<|user|>"
                turns = text.count('<|user|>')
                
                if turns == 1:
                    turn_counts['1'] += 1
                    if len(one_turn_samples) < sample_target:
                        one_turn_samples.append(data)
                elif turns == 2:
                    turn_counts['2'] += 1
                elif turns == 3:
                    turn_counts['3'] += 1
                elif turns > 3:
                    turn_counts['3+'] += 1
                    
            except json.JSONDecodeError:
                print(f"Error decoding JSON line: {line[:50]}...")
            except Exception as e:
                print(f"Error processing line: {e}")

    # Generate Bar Graph
    labels = list(turn_counts.keys())
    values = list(turn_counts.values())

    plt.figure(figsize=(10, 6))
    bars = plt.bar(labels, values, color='skyblue')
    
    # Add counts above bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                 f'{height}',
                 ha='center', va='bottom')

    plt.xlabel('Number of Turns')
    plt.ylabel('Count of Conversations')
    plt.title('Conversation Turn Distribution')
    plt.savefig(output_graph)
    print(f"Graph saved to {output_graph}")

    # Save Samples
    with open(output_samples, 'w', encoding='utf-8') as f:
        json.dump(one_turn_samples, f, indent=4)
    print(f"Saved {len(one_turn_samples)} samples to {output_samples}")

    print("Analysis Complete.")
    print("Counts:", turn_counts)

except FileNotFoundError:
    print(f"Error: Input file not found at {input_file}")
except Exception as e:
    print(f"An unexpected error occurred: {e}")

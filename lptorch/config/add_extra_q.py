import json
import os
import sys

# Get the new method name from the command line arguments
new_method = sys.argv[1]

# Open the JSON file
with open(os.path.join(os.path.dirname(__file__), "extra_methods.json")) as f:
    data = json.load(f)
    extra_methods = data.get('extra_q_methods', [])

# Check if the new method already exists in the list
if new_method in extra_methods:
    print("Method already exists in file. Ignoring.")
else:
    # Add the new method to the list
    extra_methods.append(new_method)
    data['extra_q_methods'] = extra_methods

    # Write the updated data back to the file
    with open(os.path.join(os.path.dirname(__file__), "extra_methods.json"), 'w') as f:
        json.dump(data, f, indent=4)
    print("Method added to file.")
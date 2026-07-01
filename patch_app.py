import re
import ast

with open('app.py', 'r') as f:
    content = f.read()

# Define descriptions for each experience key
exp_descriptions = {
    "default": "Provides a balanced configuration suitable for general-purpose baseline tuning.",
    "fast": "Optimizes parameters to reduce compute time and memory footprint, prioritizing speed over maximum accuracy.",
    "accurate": "Maximizes model capacity and training duration to achieve the highest possible accuracy, regardless of compute cost.",
    "repeatable": "Increases regularization and simplifies the model structure to reduce variance and ensure consistent results across multiple runs.",
    "volatile": "Reduces regularization and increases model complexity to capture unique, highly non-linear patterns, at the risk of overfitting.",
    "multi_npu": "Scales up batch sizes and model capacity to fully utilize parallel compute environments with multiple NPUs."
}

# Find MODELS_SCHEMA
start_idx = content.find('MODELS_SCHEMA = {')
end_idx = content.find('\n\n# ==============================================================================\n# HELPER FUNCTIONS', start_idx)
if end_idx == -1:
    end_idx = len(content)

schema_str = content[start_idx:end_idx].strip()
# It might be tricky to parse safely with ast if there are comments.
# Let's do simple string replacement for the experiences.

new_content = content
for key, desc in exp_descriptions.items():
    # Find all occurrences of "key": {"name": "...", "params": {...}}
    # and add "description": "desc"
    pattern = r'("' + key + r'":\s*\{"name":\s*"[^"]*",\s*"params":\s*\{[^}]*\})\}'
    replacement = r'\1, "description": "' + desc + '"}'
    new_content = re.sub(pattern, replacement, new_content)

with open('app.py', 'w') as f:
    f.write(new_content)

print("Patched app.py")

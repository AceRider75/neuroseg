import os
import json
import torch
from pathlib import Path

print("="*70)
print(" MODEL STATUS DIAGNOSTIC")
print("="*70)

# 1. Check for model file
print("\n1️⃣  MODEL FILE")
print("-" * 70)

if os.path.exists('best_model.pth'):
    size_mb = os.path.getsize('best_model.pth') / 1e6
    print(f"✓ best_model.pth exists ({size_mb:.1f} MB)")
    
    # Try to load it
    try:
        model = torch.load('best_model.pth', map_location='cpu')
        print("✓ Model can be loaded successfully")
    except Exception as e:
        print(f"✗ Error loading model: {e}")
else:
    print("✗ best_model.pth NOT FOUND")
    print("  → Run Step 6 to train a fresh model")

# 2. Check for training history
print("\n2️⃣  TRAINING HISTORY")
print("-" * 70)

current_dice = None
history_source = None

if os.path.exists('training_history.json'):
    try:
        with open('training_history.json', 'r') as f:
            history = json.load(f)
        current_dice = max(history.get('val_dice', [0]))
        epochs = len(history.get('val_dice', []))
        history_source = "training_history.json"
        print(f"✓ Found training_history.json")
        print(f"  Epochs trained: {epochs}")
        print(f"  Best Dice: {current_dice:.4f} ({current_dice*100:.2f}%)")
    except Exception as e:
        print(f"⚠ training_history.json exists but error reading: {e}")

elif os.path.exists('training_history_resumed.json'):
    try:
        with open('training_history_resumed.json', 'r') as f:
            history = json.load(f)
        current_dice = max(history.get('val_dice', [0]))
        epochs = len(history.get('val_dice', []))
        history_source = "training_history_resumed.json"
        print(f"✓ Found training_history_resumed.json")
        print(f"  Epochs trained: {epochs}")
        print(f"  Best Dice: {current_dice:.4f} ({current_dice*100:.2f}%)")
    except Exception as e:
        print(f"⚠ training_history_resumed.json exists but error reading: {e}")
else:
    print("✗ No training_history.json or training_history_resumed.json found")
    print("  This means training was interrupted before saving history")

# 3. Check for visualization files
print("\n3️⃣  VISUALIZATION FILES")
print("-" * 70)

viz_files = [
    'training_metrics.png',
    'training_metrics_resumed.png',
    'training_history.json',
    'training_history_resumed.json'
]

found_any = False
for f in viz_files:
    if os.path.exists(f):
        size = os.path.getsize(f) / 1e6
        print(f"✓ {f} ({size:.1f} MB)" if size > 1 else f"✓ {f} ({size*1000:.0f} KB)")
        found_any = True

if not found_any:
    print("✗ No visualization files found (yet)")

# 4. Recommendations
print("\n4️⃣  RECOMMENDATIONS")
print("-" * 70)

if not os.path.exists('best_model.pth'):
    print("\n🔴 NO MODEL FOUND - START FROM SCRATCH")
    print("\n  Action: Run Step 6 (train_max_dice.py)")
    print("  This will train a new model from scratch")
    
elif current_dice is None:
    print("\n🟡 MODEL EXISTS BUT NO HISTORY - AUTO-DETECT PERFORMANCE")
    print(f"\n  Current status: Model exists but performance unknown")
    print(f"  (Training was interrupted before saving history.json)")
    print(f"\n  Action: Run resume_training.py")
    print(f"  This will:")
    print(f"    1. Run quick validation to measure current Dice")
    print(f"    2. Automatically detect model performance")
    print(f"    3. Continue training from that point")
    print(f"    4. Report current Dice when done")
    
elif current_dice >= 0.90:
    print("\n🟢 TARGET ALREADY REACHED!")
    print(f"\n  Current Dice: {current_dice:.4f} ({current_dice*100:.2f}%)")
    print(f"  Target: 0.90 (90.00%)")
    print(f"\n  ✓ Your model meets the requirements!")
    print(f"\n  Action: Model is ready for deployment")
    
else:
    gap = 0.90 - current_dice
    print("\n🟡 MODEL TRAINED BUT NEEDS IMPROVEMENT")
    print(f"\n  Current Dice: {current_dice:.4f} ({current_dice*100:.2f}%)")
    print(f"  Target: 0.90 (90.00%)")
    print(f"  Gap: {gap:.4f} ({gap*100:.2f}%)")
    print(f"\n  Action: Run resume_training.py")
    print(f"  This will:")
    print(f"    1. Load your checkpoint at {current_dice:.4f}")
    print(f"    2. Continue training for remaining epochs")
    print(f"    3. Should reach {current_dice + gap:.4f} (0.90+)")
    print(f"    4. Saves improved model to best_model.pth")

# 5. Quick info
print("\n5️⃣  QUICK INFO")
print("-" * 70)
print("\nFiles created/available:")

files_info = {
    'best_model.pth': 'Your trained model checkpoint',
    'training_history.json': 'Metrics from training',
    'training_history_resumed.json': 'Metrics from resumed training',
    'training_metrics.png': '6-plot visualization',
    'resume_training.py': 'Resume from checkpoint script',
    'train_max_dice.py': 'Train fresh model script'
}

for fname, desc in files_info.items():
    exists = "✓" if os.path.exists(fname) else "✗"
    print(f"  {exists} {fname:30s} - {desc}")

print("\n" + "="*70)

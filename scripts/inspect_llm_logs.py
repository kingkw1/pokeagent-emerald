#!/usr/bin/env python3
"""
Detailed LLM Log Inspector

Shows what's actually in the LLM logs to help diagnose issues.
"""

import json
import glob
from collections import defaultdict

print("="*80)
print("LLM LOG INSPECTOR")
print("="*80)
print()

# Find all log files
log_files = sorted(glob.glob("llm_logs/llm_log_*.jsonl"))
print(f"Total log files: {len(log_files)}")
print()

# Analyze ALL logs (not just recent - need to find the ones with actual calls)
entry_types = defaultdict(int)
interaction_types = defaultdict(int)
total_entries = 0
sample_entries = []
files_with_calls = []

for filepath in log_files:
    try:
        with open(filepath, "r") as f:
            for line in f:
                try:
                    entry = json.loads(line)
                    total_entries += 1
                    
                    entry_type = entry.get("type", "unknown")
                    entry_types[entry_type] += 1
                    
                    if "interaction_type" in entry:
                        interaction_types[entry["interaction_type"]] += 1
                    
                    # Track files with actual calls
                    if entry_type not in ["session_start"] and filepath not in files_with_calls:
                        files_with_calls.append(filepath)
                    
                    # Collect samples
                    if len(sample_entries) < 10 and entry_type not in ["session_start"]:
                        sample_entries.append({
                            "file": filepath.split("/")[-1],
                            "type": entry_type,
                            "interaction": entry.get("interaction_type", "N/A"),
                            "has_response": "response" in entry,
                            "has_error": "error" in entry
                        })
                        
                except json.JSONDecodeError:
                    pass
    except Exception as e:
        print(f"Error reading {filepath}: {e}")

print("="*80)
print("ENTRY TYPE DISTRIBUTION")
print("="*80)
for entry_type, count in sorted(entry_types.items(), key=lambda x: -x[1]):
    pct = (count / total_entries * 100) if total_entries > 0 else 0
    print(f"   {entry_type:30s} : {count:6d} ({pct:5.1f}%)")
print()
print(f"Files with actual VLM calls: {len(files_with_calls)}")
if files_with_calls:
    print("Recent files with calls:")
    for f in files_with_calls[-10:]:
        print(f"   {f}")
print()

if interaction_types:
    print("="*80)
    print("INTERACTION TYPE DISTRIBUTION")
    print("="*80)
    for interaction_type, count in sorted(interaction_types.items(), key=lambda x: -x[1]):
        print(f"   {interaction_type}")
    print()

if sample_entries:
    print("="*80)
    print("SAMPLE ENTRIES (Non-Session-Start)")
    print("="*80)
    for i, sample in enumerate(sample_entries, 1):
        print(f"\n{i}. File: {sample['file']}")
        print(f"   Type: {sample['type']}")
        print(f"   Interaction: {sample['interaction']}")
        print(f"   Has Response: {sample['has_response']}")
        print(f"   Has Error: {sample['has_error']}")
    print()

# Diagnosis
print("="*80)
print("DIAGNOSIS")
print("="*80)
print()

success_entries = entry_types.get("llm_call", 0) + entry_types.get("response", 0)
error_entries = entry_types.get("error", 0)
session_entries = entry_types.get("session_start", 0)

if success_entries > 0:
    print(f"✅ Found {success_entries} successful LLM calls")
    print("   The agent is making VLM calls successfully")
elif error_entries > 0:
    print(f"⚠️ Found {error_entries} errors but no successful calls")
    print("   The VLM may be failing or not configured correctly")
    print("   Check the error messages in the logs")
elif session_entries == total_entries:
    print(f"ℹ️ Only found session_start entries ({session_entries} total)")
    print("   The agent may have been started but not run long enough")
    print("   Or VLM logging may not be working")
else:
    print("❓ Unclear log state")
    print(f"   Total entries: {total_entries}")
    print(f"   Session starts: {session_entries}")

print()
print("Recommendation:")
if success_entries == 0:
    print("   Run the agent with: python run.py --agent-auto")
    print("   Let it run for at least 100 steps to collect VLM call data")
else:
    print("   Re-run quick_vlm_check.py - it should find the VLM calls now")

print("="*80)

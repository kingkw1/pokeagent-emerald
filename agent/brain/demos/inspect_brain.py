# agent/brain/demos/inspect_brain.py
"""
Utility: Dumps the contents of the live ChromaDB memory database.

Usage (from project root):
    python -m agent.brain.demos.inspect_brain
    python -m agent.brain.demos.inspect_brain --spatial   # only show memories with coordinates
    python -m agent.brain.demos.inspect_brain --location "ROUTE 102"  # filter by location
"""
import sys
import argparse
from pathlib import Path

_PROJECT_ROOT = str(Path(__file__).resolve().parents[3])
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from agent.brain.memory import EpisodicMemory
import pandas as pd

def inspect(spatial_only: bool = False, location_filter: str | None = None):
    print("==================================================")
    print("🧠 BRAIN INSPECTOR")
    print("==================================================")
    
    # Load the real DB
    mem = EpisodicMemory(db_path="./memory_db")
    
    # Get all data
    data = mem.collection.get()
    
    if not data['documents']:
        print("❌ Memory is empty.")
        return

    print(f"📚 Total Memories: {len(data['documents'])}\n")
    
    # Build DataFrame with spatial columns
    df = pd.DataFrame({
        'Memory (Snippet)': [d[:80] + "..." if len(d) > 80 else d for d in data['documents']],
        'Type': [m.get('type', 'unknown') for m in data['metadatas']],
        'X': [m.get('pos_x', '') for m in data['metadatas']],
        'Y': [m.get('pos_y', '') for m in data['metadatas']],
        'Location': [m.get('location', '') for m in data['metadatas']],
        'Timestamp': [m.get('timestamp', 0) for m in data['metadatas']]
    })
    
    # Sort by time (newest last)
    df = df.sort_values('Timestamp')

    # Optional filters
    if spatial_only:
        df = df[df['X'] != '']
        print(f"🗺️  Showing {len(df)} memories with spatial data\n")
    if location_filter:
        df = df[df['Location'].str.contains(location_filter, case=False, na=False)]
        print(f"📍 Filtered to location: {location_filter} ({len(df)} memories)\n")

    # Count spatial coverage
    has_coords = (df['X'] != '').sum()
    print(f"🗺️  Spatial coverage: {has_coords}/{len(df)} memories have coordinates\n")
    
    print(df[['Type', 'X', 'Y', 'Location', 'Memory (Snippet)']].to_string(index=False))
    print("==================================================")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inspect ChromaDB episodic memory")
    parser.add_argument("--spatial", action="store_true", help="Only show memories with coordinates")
    parser.add_argument("--location", type=str, default=None, help="Filter by location name (substring match)")
    args = parser.parse_args()

    try:
        inspect(spatial_only=args.spatial, location_filter=args.location)
    except ImportError:
        print("Pandas not installed? Just printing raw list:")
        mem = EpisodicMemory(db_path="./memory_db")
        print(mem.collection.get())
import json
from pathlib import Path
from datetime import datetime, timedelta

# json operation in unix to find nongsep event: ❯ jq '[ .[] | select([.source] | flatten | all(. | ascii_downcase | contains("gsep") | not)) ]' merge/all_events_merged.json > merge/no_gsep.json
# json operation to find nongsep before gsep and after gsep: ❯ jq '[.[] | select(.start_time >= "1987-01-01" and .start_time < "2018-01-01")]' merge/no_gsep.json > merge/no_gsep_events_1987-2017.json
# Parameters
INPUT_DIR = Path('.')
OUTPUT_DIR = Path('merge')
CONSOLIDATED_OUTPUT = 'all_events_merged.json'
DELTA_MINUTES = 30

OUTPUT_DIR.mkdir(exist_ok=True)

def merge_events(group):
    """Combines multiple event objects into a single rebinned event."""
    if len(group) == 1:
        return group[0]
    
    # Use the event with the highest peak flux as the template for metadata
    top_event = max(group, key=lambda e: e.get('peak_flux', 0))
    merged = dict(top_event)
    
    # Calculate merged time bounds
    merged['start_time'] = min(e['start_time'] for e in group)
    merged['end_time'] = max(e['end_time'] for e in group)
    
    # Peak flux and time should come from the strongest detection
    merged['peak_time'] = top_event['peak_time']
    merged['peak_flux'] = top_event['peak_flux']
    merged['fluence_pfu_s'] = sum(e.get('fluence_pfu_s', 0) for e in group)
    
    all_sources = []
    all_slices = []
    
    for e in group:
        # Flatten and collect sources
        src = e.get('source')
        if src:
            if isinstance(src, list): all_sources.extend(src)
            else: all_sources.append(src)
        
        # Flatten and collect slice paths
        path = e.get('slice_path')
        if path:
            if isinstance(path, list): all_slices.extend(path)
            else: all_slices.append(path)
            
    # Unique, sorted lists
    unique_sources = sorted(list(set(all_sources)))
    unique_slices = sorted(list(set(all_slices)))
    
    # Handle Source assignment
    merged['source'] = unique_sources[0] if len(unique_sources) == 1 else unique_sources
    
    # Handle Slice Path assignment (FIXED logic)
    if not unique_slices:
        merged['slice_path'] = None
    elif len(unique_slices) == 1:
        merged['slice_path'] = unique_slices[0]
    else:
        merged['slice_path'] = unique_slices
    
    # Recalculate duration
    try:
        s_dt = datetime.fromisoformat(merged['start_time'].replace(' ', 'T'))
        e_dt = datetime.fromisoformat(merged['end_time'].replace(' ', 'T'))
        merged['duration_sec'] = (e_dt - s_dt).total_seconds()
    except Exception:
        merged['duration_sec'] = 0
    
    return merged

# 1. Collect ALL events from all subdirectories
all_events = []
files = [
    p for p in INPUT_DIR.rglob('*.json') 
    if p.name != CONSOLIDATED_OUTPUT and OUTPUT_DIR not in p.parents
]

for file in files:
    try:
        with open(file, 'r') as f:
            data = json.load(f)
            events_in_file = data if isinstance(data, list) else [data]
            all_events.extend(events_in_file)
            print(f"Loaded {len(events_in_file)} events from {file.name}")
    except Exception as e:
        print(f"Skipping {file.name}: {e}")

if not all_events:
    print("No events found.")
    exit()

# 2. Sort by start_time (parsed as datetime for safety)
all_events.sort(key=lambda e: datetime.fromisoformat(e['start_time'].replace(' ', 'T')))

# 3. Rebinning Logic with GSEP protection
final_events = []
current_group = [all_events[0]]
# Track the absolute furthest reach of the current group
max_group_end = datetime.fromisoformat(all_events[0]['end_time'].replace(' ', 'T'))

def has_gsep(event_or_group):
    """Checks if 'gsep' is in the source of an event or any event in a group."""
    if isinstance(event_or_group, list):
        return any(has_gsep(e) for e in event_or_group)
    src = str(event_or_group.get('source', '')).lower()
    return "gsep" in src

for i in range(1, len(all_events)):
    curr_ev = all_events[i]
    curr_start = datetime.fromisoformat(curr_ev['start_time'].replace(' ', 'T'))
    curr_end = datetime.fromisoformat(curr_ev['end_time'].replace(' ', 'T'))

    # If start is within 30m of max_group_end 
    # AND we aren't trying to merge two GSEP catalog entries
    time_overlap = curr_start <= (max_group_end + timedelta(minutes=DELTA_MINUTES))
    both_gsep = has_gsep(current_group) and has_gsep(curr_ev)

    if time_overlap and not both_gsep:
        current_group.append(curr_ev)
        if curr_end > max_group_end:
            max_group_end = curr_end
    else:
        final_events.append(merge_events(current_group))
        current_group = [curr_ev]
        max_group_end = curr_end
    
final_events.append(merge_events(current_group))

# 4. Final Output
out_path = OUTPUT_DIR / CONSOLIDATED_OUTPUT
with open(out_path, 'w') as f:
    json.dump(final_events, f, indent=2)

print("\n--- Summary ---")
print(f"Total raw events:   {len(all_events)}")
print(f"Merged events:      {len(final_events)}")
print(f"Saved to:           {out_path}")
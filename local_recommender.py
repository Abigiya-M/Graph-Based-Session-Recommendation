# ==============================================================================
# Graph-Based Session Recommendation (Synthetic Dataset) - Cloud Execution Script
# ==============================================================================

# ------------------------------------------------------------------------------
# 1. Import Libraries
# ------------------------------------------------------------------------------
import pandas as pd
import numpy as np
from neo4j import GraphDatabase
import getpass
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import datetime
import math
import os

# ------------------------------------------------------------------------------
# 2. Data Acquisition & Setup
# ------------------------------------------------------------------------------
# Generate synthetic data
print("Generating synthetic data...")
np.random.seed(42)  # For reproducibility

# Parameters for synthetic data
num_sessions = 1000
max_events_per_session = 10
unique_items = 100
unique_categories = 5

# Generate session data
session_ids = np.arange(num_sessions)
item_ids = np.random.randint(1, unique_items + 1, size=(num_sessions, max_events_per_session))
timestamps = [pd.date_range(start="2023-01-01", periods=max_events_per_session, freq='min') for _ in range(num_sessions)]
categories = np.random.randint(1, unique_categories + 1, size=(num_sessions, max_events_per_session))

# Create DataFrame with variable-length sequences
data = []
for sid, items, ts, cats in zip(session_ids, item_ids, timestamps, categories):
    num_events = np.random.randint(2, max_events_per_session + 1)
    data.append({
        'session_id': sid,
        'item_id': items[:num_events].tolist(),
        'timestamp': ts[:num_events].tolist(),
        'category': [str(cat) for cat in cats[:num_events].tolist()]
    })

df = pd.DataFrame(data)
print(f"Generated {len(df)} synthetic sessions.")

# Split into train/test (80/20)
train_sessions, test_sessions = train_test_split(df, test_size=0.2, random_state=42)

print(f"Number of training sessions: {len(train_sessions)}")
print(f"Number of test sessions: {len(test_sessions)}")

# Helper function for fast flattening
def flatten_sessions_fast(df):
    sids, eids, cats, orders, ts_list = [], [], [], [], []
    for _, row in df.iterrows():
        n = len(row['item_id'])
        sids.extend([row['session_id']] * n)
        eids.extend(row['item_id'])
        cats.extend(row['category'])
        orders.extend(list(range(1, n + 1)))
        ts_list.extend([ts.isoformat() for ts in row['timestamp']])
    return pd.DataFrame({
        'sid': sids,
        'eid': eids,
        'cat': cats,
        'order': orders,
        'ts': ts_list
    })

# --- Flatten the training dataset ---
flat_data = flatten_sessions_fast(train_sessions)

# ------------------------------------------------------------------------------
# 3. Neo4j Connection Setup
# ------------------------------------------------------------------------------

# Connect to cloud Neo4j (AuraDB)
neo4j_uri = "neo4j+s://63c356b1.databases.neo4j.io"
neo4j_user = "neo4j"
# Prompt for password with fallback to hardcoded value
try:
    # NOTE: Replace the fallback password with your actual Neo4j AuraDB password
    neo4j_password = getpass.getpass(f"Enter password for {neo4j_user}@{neo4j_uri} (press Enter for default): ") or "T09WvjN-00q_0sjK5B9qj_tjPWUky3Afwu946zvVZl8"
except AttributeError:
    print("Error: 'getpass' function not available. Using hardcoded password as fallback.")
    neo4j_password = "T09WvjN-00q_0sjK5B9qj_tjPWUky3Afwu946zvVZl8"
try:
    driver = GraphDatabase.driver(neo4j_uri, auth=(neo4j_user, neo4j_password))
    driver.verify_connectivity()
    print("Connection to AuraDB successful!")
except Exception as e:
    print(f"Connection failed: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

# Function to run Cypher queries
def run_query(tx, query, params=None):
    params = params or {}
    return tx.run(query, **params)

# Clear existing graph and create constraints
with driver.session() as session:
    print("Clearing graph and creating constraints...")
    session.execute_write(run_query, "MATCH (n) DETACH DELETE n")
    session.execute_write(run_query, "CREATE CONSTRAINT IF NOT EXISTS FOR (s:Session) REQUIRE s.id IS UNIQUE")
    session.execute_write(run_query, "CREATE CONSTRAINT IF NOT EXISTS FOR (e:Event) REQUIRE e.id IS UNIQUE")
    session.execute_write(run_query, "CREATE CONSTRAINT IF NOT EXISTS FOR (c:Category) REQUIRE c.id IS UNIQUE")

# ------------------------------------------------------------------------------
# 4. Build Graph (Nodes and :HAS_EVENT)
# ------------------------------------------------------------------------------

query = """
UNWIND $rows AS row
MERGE (s:Session {id: row.sid})
MERGE (e:Event {id: row.eid})
ON CREATE SET e.category = row.cat
WITH s, e, row
FOREACH (_ IN CASE WHEN row.cat IS NOT NULL AND row.cat <> 'None' THEN [1] ELSE [] END |
    MERGE (c:Category {id: row.cat})
    MERGE (e)-[:BELONGS_TO]->(c)
)
MERGE (s)-[:HAS_EVENT {order: row.order, timestamp: row.ts}]->(e)
"""

# Send data in batches
BATCH_SIZE = 5000
with driver.session() as session:
    print("Building Graph (Nodes and :HAS_EVENT relationships)...")
    for i in tqdm(range(0, len(flat_data), BATCH_SIZE), desc="Building Nodes"):
        batch = flat_data.iloc[i:i+BATCH_SIZE].to_dict("records")
        session.execute_write(lambda tx: tx.run(query, rows=batch))

print("Initial graph construction complete!")

# ------------------------------------------------------------------------------
# 4.5. CRITICAL STEP: Create Session Flow Relationships (:NEXT) 🚀
# ------------------------------------------------------------------------------
session_flow_query = """
MATCH (s:Session)
// Get all events in order for the session, collecting event details
MATCH (s)-[r:HAS_EVENT]->(e:Event)
WITH s, collect({event: e, order: r.order, timestamp: r.timestamp}) AS session_events
ORDER BY s.id

// Unwind the events to pair current event (e1) with next event (e2)
UNWIND session_events AS current_event
WITH session_events, current_event
WHERE current_event.order < size(session_events) // Exclude the last item
WITH current_event.event AS e1, current_event.timestamp AS ts1, 
     session_events[current_event.order].event AS e2, 
     session_events[current_event.order].timestamp AS ts2

// Calculate time difference in minutes
WITH e1, e2, duration.between(datetime(ts1), datetime(ts2)).minutes AS time_diff_minutes

// Calculate weight (simple inverse linear decay capped at 60 minutes)
WITH e1, e2, CASE 
    WHEN time_diff_minutes IS NULL OR time_diff_minutes > 60 THEN 0.0
    ELSE 1.0 - (time_diff_minutes / 60.0) // Weight is 1.0 for immediate, 0.0 for 60 min gap
END AS weight

// Create or update the :NEXT relationship
MERGE (e1)-[rel:NEXT]->(e2)
ON CREATE SET rel.frequency = 1, rel.weight = weight
ON MATCH SET rel.frequency = rel.frequency + 1,
             rel.weight = rel.weight + weight // Accumulate weights
"""

# Block 1: WRITE Operation - Creates the :NEXT relationships
with driver.session() as session:
    print("Creating :NEXT sequential relationships (session flow)...")
    session.execute_write(run_query, session_flow_query)
    
# Block 2: READ Operation - Sanity Check (MUST be in a NEW session context to avoid ResultConsumedError)
# THIS IS THE FIXED BLOCK
with driver.session() as session:
    next_rels = session.execute_read(run_query, "MATCH ()-[r:NEXT]->() RETURN count(r) as count").single()['count']
    print(f"Created/Updated {next_rels} :NEXT relationships.")

# ------------------------------------------------------------------------------
# 5. Graph Metrics
# ------------------------------------------------------------------------------

# Block 3: READ Operations - Graph Metrics
with driver.session() as session:
    num_sessions = session.execute_read(run_query, "MATCH (s:Session) RETURN count(s) as count").single()['count']
    num_events = session.execute_read(run_query, "MATCH (e:Event) RETURN count(e) as count").single()['count']
    num_categories = session.execute_read(run_query, "MATCH (c:Category) RETURN count(c) as count").single()['count']

print(f"Graph Metrics: Sessions: {num_sessions}, Events: {num_events}, Categories: {num_categories}")

# ------------------------------------------------------------------------------
# 6. Define Recommendation Function
# ------------------------------------------------------------------------------

def recommend_next(driver, partial_sequence, category_filter=None):
    if not partial_sequence:
        return []

    last_event = partial_sequence[-1]
    
    query = """
    MATCH (e:Event {id: $eid})-[r:NEXT]->(next:Event)
    """
    if category_filter:
        query += """
        WHERE next.category = $category
        """
    query += """
    // Score combines frequency and accumulated temporal weight
    RETURN next.id as next_id, coalesce(r.frequency, 1) as freq, coalesce(r.weight, 1.0) as weight,
           (coalesce(r.weight, 1.0) * coalesce(r.frequency, 1)) AS score
    ORDER BY score DESC
    LIMIT 5
    """

    with driver.session() as session:
        result = session.execute_read(run_query, query, {'eid': last_event, 'category': str(category_filter)})
        recommendations = [(rec['next_id'], rec['freq'], rec['weight']) for rec in result]
    return recommendations

# ------------------------------------------------------------------------------
# 7. Evaluation (MRR)
# ------------------------------------------------------------------------------

def compute_mrr(driver, test_sessions, use_category_filter=False):
    mrr_scores = []
    # Sample a fixed number of sessions for fast evaluation
    sample_sessions = test_sessions.sample(min(100, len(test_sessions)), random_state=42)
    print(f"Evaluating on a sample of {len(sample_sessions)} test sessions...")

    for _, row in tqdm(sample_sessions.iterrows(), total=len(sample_sessions), desc="Evaluating"):
        sequence = row['item_id']
        categories = row['category']
        if len(sequence) < 2:
            continue
            
        partial = sequence[:-1]
        true_next = sequence[-1]
        true_category = categories[-1]
        
        category_filter_str = str(true_category) if use_category_filter and true_category is not None else None

        recs = recommend_next(driver, partial, category_filter=category_filter_str)
        recommended_ids = [r[0] for r in recs]
        
        if true_next in recommended_ids:
            rank = recommended_ids.index(true_next) + 1
            mrr_scores.append(1 / rank)
        else:
            mrr_scores.append(0)
            
    return np.mean(mrr_scores) if mrr_scores else 0

# --- MRR calculation ---
mrr_no_filter = compute_mrr(driver, test_sessions, use_category_filter=False)
mrr_with_filter = compute_mrr(driver, test_sessions, use_category_filter=True)
print(f"\nMRR (No Category Filter): {mrr_no_filter}")
print(f"MRR (With Category Filter): {mrr_with_filter}")

# ------------------------------------------------------------------------------
# 8. Close Driver
# ------------------------------------------------------------------------------
driver.close()
print("Neo4j driver closed.")
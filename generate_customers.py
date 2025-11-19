#!/usr/bin/env python3
"""
Generate customer personas and customers using Gaussian Mixture Model
"""

import numpy as np
import json
from pathlib import Path
from sklearn.mixture import GaussianMixture

# Set random seed for reproducibility
np.random.seed(42)

# Configuration
NUM_PERSONAS = 20
NUM_CUSTOMERS = 250000
REGIONS = ['Northwest', 'West', 'South', 'Midwest', 'East', 'Northeast']
GENDERS = ['Male', 'Female']

def generate_personas():
    """Generate 20 customer personas as centroids"""
    personas = []

    for i in range(NUM_PERSONAS):
        persona = {
            'id': i + 1,
            'name': f'Persona {i + 1}',
            'gender': GENDERS[i % 2],
            'age_mean': float(np.random.randint(25, 70)),
            'age_std': float(np.random.uniform(3, 8)),
            'num_orders_mean': float(np.random.uniform(1, 15)),
            'num_orders_std': float(np.random.uniform(0.5, 3)),
            'revenue_per_customer_mean': float(np.random.uniform(100, 1200)),
            'revenue_per_customer_std': float(np.random.uniform(50, 300)),
            'region': REGIONS[i % len(REGIONS)],
            'weight': float(np.random.dirichlet(np.ones(1))[0])  # Mixing coefficient
        }
        personas.append(persona)

    # Normalize weights to sum to 1
    total_weight = sum(p['weight'] for p in personas)
    for persona in personas:
        persona['weight'] = persona['weight'] / total_weight

    return personas

def generate_customers_from_personas(personas):
    """Generate customers based on persona distributions using GMM"""
    customers = []

    # Calculate how many customers per persona based on weights
    customers_per_persona = []
    remaining = NUM_CUSTOMERS

    for i, persona in enumerate(personas):
        if i == len(personas) - 1:
            # Last persona gets remaining customers
            count = remaining
        else:
            count = int(NUM_CUSTOMERS * persona['weight'])
            remaining -= count
        customers_per_persona.append(count)

    customer_id = 1

    for persona_idx, (persona, count) in enumerate(zip(personas, customers_per_persona)):
        # Generate customers for this persona using normal distributions
        ages = np.random.normal(
            persona['age_mean'],
            persona['age_std'],
            count
        )
        ages = np.clip(ages, 18, 85).astype(int)

        num_orders = np.random.normal(
            persona['num_orders_mean'],
            persona['num_orders_std'],
            count
        )
        num_orders = np.clip(num_orders, 0, 50).astype(int)

        revenues = np.random.normal(
            persona['revenue_per_customer_mean'],
            persona['revenue_per_customer_std'],
            count
        )
        revenues = np.clip(revenues, 0, 5000)

        for i in range(count):
            customer = {
                'id': customer_id,
                'persona_id': persona['id'],
                'gender': persona['gender'],
                'age': int(ages[i]),
                'num_past_orders': int(num_orders[i]),
                'total_revenue': round(float(revenues[i]), 2),
                'region': persona['region']
            }
            customers.append(customer)
            customer_id += 1

    # Shuffle customers to mix personas
    np.random.shuffle(customers)

    # Reassign sequential IDs after shuffle
    for idx, customer in enumerate(customers):
        customer['id'] = idx + 1

    return customers

def save_data(personas, customers, output_dir):
    """Save personas and customers to JSON files"""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Save personas
    personas_file = output_path / 'personas.json'
    with open(personas_file, 'w') as f:
        json.dump(personas, f, indent=2)
    print(f"Saved {len(personas)} personas to {personas_file}")

    # Save customers in chunks of 100 for pagination (2,500 chunks)
    chunk_size = 100
    num_chunks = (len(customers) + chunk_size - 1) // chunk_size

    for i in range(num_chunks):
        start_idx = i * chunk_size
        end_idx = min((i + 1) * chunk_size, len(customers))
        chunk = customers[start_idx:end_idx]

        chunk_file = output_path / f'customers_chunk_{i+1}.json'
        with open(chunk_file, 'w') as f:
            json.dump(chunk, f, indent=2)
        if (i + 1) % 100 == 0 or i == num_chunks - 1:
            print(f"Saved chunk {i+1}/{num_chunks} ({len(chunk)} customers)")

    # Save summary statistics
    summary = {
        'total_customers': len(customers),
        'total_personas': len(personas),
        'num_chunks': num_chunks,
        'chunk_size': chunk_size,
        'statistics': {
            'avg_age': round(float(np.mean([c['age'] for c in customers])), 2),
            'avg_orders': round(float(np.mean([c['num_past_orders'] for c in customers])), 2),
            'avg_revenue': round(float(np.mean([c['total_revenue'] for c in customers])), 2),
            'total_revenue': round(float(sum(c['total_revenue'] for c in customers)), 2),
        },
        'region_distribution': {
            region: len([c for c in customers if c['region'] == region])
            for region in REGIONS
        },
        'gender_distribution': {
            gender: len([c for c in customers if c['gender'] == gender])
            for gender in GENDERS
        }
    }

    summary_file = output_path / 'summary.json'
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"Saved summary to {summary_file}")

def main():
    print("Generating customer personas and customers...")
    print(f"Configuration: {NUM_PERSONAS} personas, {NUM_CUSTOMERS} customers")
    print()

    # Generate personas
    print("Step 1: Generating personas...")
    personas = generate_personas()
    print(f"Generated {len(personas)} personas")
    print()

    # Generate customers
    print("Step 2: Generating customers from persona distributions...")
    customers = generate_customers_from_personas(personas)
    print(f"Generated {len(customers)} customers")
    print()

    # Save to files
    print("Step 3: Saving data to JSON files...")
    output_dir = Path(__file__).parent.parent / 'frontend' / 'public' / 'data' / 'customers'
    save_data(personas, customers, output_dir)
    print()
    print("✓ Customer generation complete!")

if __name__ == '__main__':
    main()

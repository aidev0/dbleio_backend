#!/usr/bin/env python3
"""
Run all AI model evaluations for all personas and store results in a single JSON file
"""

import os
import json
from pathlib import Path
from ai_agent import load_video_analyses, load_personas, evaluate_persona, MODELS

def run_all_evaluations():
    """Run evaluations for all personas with all available models"""

    print("Loading data...")
    videos = load_video_analyses()
    personas = load_personas()

    print(f"Loaded {len(videos)} videos and {len(personas)} personas")
    print(f"Starting evaluations with all available models...\n")

    all_results = []

    # Track progress
    total_evaluations = 0
    for provider in MODELS:
        total_evaluations += len(MODELS[provider]) * len(personas)

    current = 0

    # Run evaluations for each provider and model
    for provider in MODELS:
        for model_name in MODELS[provider]:
            print(f"\n{'='*80}")
            print(f"Provider: {provider.upper()} | Model: {model_name}")
            print(f"{'='*80}\n")

            for persona in personas:
                current += 1
                print(f"[{current}/{total_evaluations}] Evaluating Persona {persona['id']}: {persona['name']}...")

                try:
                    evaluation = evaluate_persona(persona, videos, provider, model_name)

                    if evaluation:
                        result = {
                            'persona_id': persona['id'],
                            'persona_name': persona['name'],
                            'provider': provider,
                            'model': model_name,
                            'model_full_name': MODELS[provider][model_name],
                            'evaluation': evaluation
                        }
                        all_results.append(result)

                        print(f"  ✓ Most preferred: Video {evaluation.get('most_preferred_video')}")
                        print(f"  ✓ Confidence: {evaluation.get('confidence_score')}%")
                    else:
                        print(f"  ✗ Evaluation failed")

                except Exception as e:
                    print(f"  ✗ Error: {str(e)}")

    # Save all results to a single JSON file
    output_dir = Path(__file__).parent / 'results'
    output_dir.mkdir(exist_ok=True)

    output_file = output_dir / 'all_persona_evaluations.json'

    with open(output_file, 'w') as f:
        json.dump(all_results, f, indent=2)

    print(f"\n{'='*80}")
    print(f"✓ ALL EVALUATIONS COMPLETE!")
    print(f"{'='*80}")
    print(f"Total evaluations: {len(all_results)}/{total_evaluations}")
    print(f"Results saved to: {output_file}")
    print(f"\nSummary by provider:")

    # Print summary
    for provider in MODELS:
        provider_results = [r for r in all_results if r['provider'] == provider]
        print(f"  {provider.upper()}: {len(provider_results)} evaluations")

    return all_results

if __name__ == '__main__':
    run_all_evaluations()

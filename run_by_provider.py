#!/usr/bin/env python3
"""
Run evaluations for a specific provider
"""

import sys
import json
from pathlib import Path
from ai_agent import load_video_analyses, load_personas, evaluate_persona, MODELS

def run_provider_evaluations(provider_name):
    """Run evaluations for all personas with all models from a specific provider"""

    print(f"Loading data...")
    videos = load_video_analyses()
    personas = load_personas()

    print(f"Loaded {len(videos)} videos and {len(personas)} personas")
    print(f"\nRunning evaluations for provider: {provider_name.upper()}")
    print(f"Models to test: {list(MODELS[provider_name].keys())}")
    print(f"Total evaluations: {len(MODELS[provider_name]) * len(personas)}\n")

    results = []
    total = len(MODELS[provider_name]) * len(personas)
    current = 0

    for model_name in MODELS[provider_name]:
        print(f"\n{'='*80}")
        print(f"Model: {model_name}")
        print(f"{'='*80}\n")

        for persona in personas:
            current += 1
            print(f"[{current}/{total}] Evaluating Persona {persona['id']}: {persona['name']}...", end=" ")

            try:
                evaluation = evaluate_persona(persona, videos, provider_name, model_name)

                if evaluation:
                    result = {
                        'persona_id': persona['id'],
                        'persona_name': persona['name'],
                        'provider': provider_name,
                        'model': model_name,
                        'model_full_name': MODELS[provider_name][model_name],
                        'evaluation': evaluation
                    }
                    results.append(result)

                    print(f"✓ Video {evaluation.get('most_preferred_video')} (Confidence: {evaluation.get('confidence_score')}%)")
                else:
                    print(f"✗ Failed")

            except Exception as e:
                print(f"✗ Error: {str(e)}")

    # Save results
    output_dir = Path(__file__).parent / 'results'
    output_dir.mkdir(exist_ok=True)

    output_file = output_dir / f'{provider_name}_evaluations.json'

    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n{'='*80}")
    print(f"✓ {provider_name.upper()} EVALUATIONS COMPLETE!")
    print(f"{'='*80}")
    print(f"Completed: {len(results)}/{total} evaluations")
    print(f"Results saved to: {output_file}")

    return results

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python3 run_by_provider.py <provider>")
        print("Available providers: openai, anthropic, google")
        sys.exit(1)

    provider = sys.argv[1].lower()

    if provider not in MODELS:
        print(f"Error: Unknown provider '{provider}'")
        print(f"Available providers: {', '.join(MODELS.keys())}")
        sys.exit(1)

    run_provider_evaluations(provider)

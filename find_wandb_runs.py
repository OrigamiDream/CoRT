import json
import wandb
import logging
import argparse

from utils import formatting_utils

formatting_utils.setup_formatter(logging.INFO)


def parse_adaptive_args(arg):
    if not arg:
        return None

    if ',' in arg:
        return arg.split(',')
    return [arg]


def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('--project', default='',
                        help='A name of W&B project')
    parser.add_argument('--model_name', default='',
                        help='A model name to query in runs')
    parser.add_argument('--sweep', default='',
                        help='A W&B sweep name to query in runs')
    parser.add_argument('--state', default='',
                        help='The W&B run state to query in runs')

    args = parser.parse_args()

    print()
    print('***** W&B Run Query *****')
    print(' - Project name: {}'.format(args.project))
    print(' - Model name(s): {}'.format(', '.join(parse_adaptive_args(args.model_name))))
    print(' - Sweep name(s): {}'.format(', '.join(parse_adaptive_args(args.sweep))))
    print(' - Run state(s): {}'.format(', '.join(parse_adaptive_args(args.state))))
    print()

    api = wandb.Api()
    runs = api.runs('{}/{}'.format(api.default_entity, args.project), per_page=1000)
    results = []
    for run in runs:
        cfg = json.loads(run.json_config)
        model_name = cfg['model_name']['value']

        # conditional searches
        if args.model_name and model_name not in parse_adaptive_args(args.model_name):
            continue
        if args.sweep and run.sweep.name not in parse_adaptive_args(args.sweep):
            continue
        if args.state and run.state not in parse_adaptive_args(args.state):
            continue

        results.append(run)

    run_ids = []
    for run in results:
        print('Run ({}) `{}` ({}), Sweep ID: {}'.format(run.id, run.name, run.state, run.sweep.name))
        run_ids.append(run.id)

    print('Copy and paste the arranged W&B Run IDs:')
    print('[' + ', '.join(run_ids) + ']')


if __name__ == '__main__':
    main()

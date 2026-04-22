import json
import os

BASE_SUSP = 'gdc_suspicious_results/results.json'
BASE_RISK = 'gdc_risk_results/results.json'
RATIO_SUSP = 'gdc_suspicious_results_ratio_1_1/results.json'
RATIO_RISK = 'gdc_risk_results_ratio_1_1/results.json'


def load_json(path):
    if not os.path.exists(path):
        return None
    with open(path, 'r') as f:
        return json.load(f)


def summarize(task_name, base, ratio):
    print('\n' + '=' * 80)
    print(task_name)
    print('=' * 80)

    if base is None:
        print(f'Base result missing: {task_name}')
        return
    if ratio is None:
        print(f'Ratio-1:1 result missing: {task_name}')
        return

    base_auc = base.get('roc_auc')
    base_pr = base.get('pr_auc')
    base_acc = base.get('test_accuracy')

    ratio_auc = ratio.get('roc_auc')
    ratio_pr = ratio.get('pr_auc')
    ratio_acc = ratio.get('test_accuracy')

    print('Base model:')
    print(f'  accuracy={base_acc:.4f}  auc={base_auc:.4f}  pr_auc={base_pr:.4f}')

    print('Ratio 1:1 model:')
    print(f'  accuracy={ratio_acc:.4f}  auc={ratio_auc:.4f}  pr_auc={ratio_pr:.4f}')

    print('Delta (1:1 - base):')
    print(f'  accuracy={ratio_acc - base_acc:+.4f}')
    print(f'  auc={ratio_auc - base_auc:+.4f}')
    print(f'  pr_auc={ratio_pr - base_pr:+.4f}')


if __name__ == '__main__':
    base_susp = load_json(BASE_SUSP)
    base_risk = load_json(BASE_RISK)
    ratio_susp = load_json(RATIO_SUSP)
    ratio_risk = load_json(RATIO_RISK)

    summarize('SUSPICIOUS vs NON-SUSPICIOUS', base_susp, ratio_susp)
    summarize('HIGH-RISK vs LOW-RISK', base_risk, ratio_risk)

    print('\nDone.')

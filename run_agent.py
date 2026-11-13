#!/usr/bin/env python3
import argparse
import copy
import asyncio
import json
import math
import os
import sys
import time
from pathlib import Path

from baseline.experience import Experience
from baseline.orchestration import Pacer, route
from baseline.portfolio import Portfolio, settings
from baseline.task_features import task_features, screen_task
from baseline.input_estimator import InputEstimator
from baseline.core import Budget, MODELS, StopRun, Trace, atomic_write, load_tasks, priority

# A single proxy hiccup (HTTP 429/5xx, timeout, malformed body) or an unexpected
# per-task error must not forfeit every remaining task. Such failures skip only
# the current task; the run stops after this many consecutive ones. Budget and
# wall-time stops remain immediate.
TRANSIENT_STOP_PREFIXES = ('proxy_http_', 'proxy_transport_or_response_error', 'unsupported_proxy_response_schema')
MAX_CONSECUTIVE_FAILURES = 3
FAILURE_BACKOFF_SECONDS = 5


def arguments():
    parser = argparse.ArgumentParser(description='openJiuwen competition baseline')
    parser.add_argument('--tasks-dir', type=Path, required=True)
    parser.add_argument('--out', type=Path, help='Optional aggregate JSON, not a replacement for answer.txt')
    parser.add_argument('--logs-dir', type=Path, help='Default: TASKS/.baseline')
    parser.add_argument('--orchestration', choices=('off','boundary'), default='off')
    parser.add_argument('--experience-file', type=Path)
    parser.add_argument('--pacing', choices=('off','weighted','advisory'), default='off')
    parser.add_argument('--reasoning-budget', type=float, default=0, help='Run-wide cap for ALL tokens of reasoning-enabled model calls; default disables them')
    parser.add_argument('--router-model', choices=MODELS, default='qwen3.6-35b-a3b-noreason')
    parser.add_argument('--light-model', choices=MODELS, default='qwen3.6-35b-a3b-noreason')
    parser.add_argument('--light-steps', type=int, default=6)
    parser.add_argument('--light-budget', type=float, default=1200)
    parser.add_argument('--router-budget', type=float, default=400)
    parser.add_argument('--loop-policy', choices=('off','finish'), default='off')
    parser.add_argument('--selection-policy', choices=('fixed','shadow','learned'), default='fixed')
    parser.add_argument('--selector-artifact', type=Path)
    parser.add_argument('--candidate-models', nargs='+', choices=MODELS, default=list(MODELS))
    parser.add_argument('--candidate-modes', nargs='+', choices=('history','state','hybrid'), default=['history','state','hybrid'])
    parser.add_argument('--screening', choices=('off','shadow'), default='shadow')
    parser.add_argument('--response-protocol', choices=('legacy','unified'), default='unified')
    parser.add_argument('--output-format', choices=('text','json_object','json_schema'), default='text')
    parser.add_argument('--schema-accounting', choices=('conservative','messages'), default='conservative')
    parser.add_argument('--edit-tool', action='store_true')
    parser.add_argument('--edit-domains',nargs='+',choices=('coding','data_analysis','math','search','generic'))
    parser.add_argument('--progress-state', action='store_true')
    parser.add_argument('--progress-domains',nargs='+',choices=('coding','data_analysis','math','search','generic'))
    parser.add_argument('--evidence-policy', choices=('off','precise'), default='off')
    parser.add_argument('--input-estimator', choices=('legacy','calibrated'), default='calibrated')
    parser.add_argument('--budget-policy', choices=('off','reserve'), default='reserve')
    parser.add_argument('--finish-output-tokens', type=int, default=256)
    parser.add_argument('--finish-seconds', type=float, default=20)
    parser.add_argument('--context-mode', choices=('history','state','hybrid'), default='state')
    parser.add_argument('--state-chars', type=int, default=6000)
    parser.add_argument('--observation-memory', choices=('off','bounded'), default='off')
    parser.add_argument('--observation-memory-chars', type=int, default=3000)
    parser.add_argument('--hybrid-turns', type=int, default=1)
    parser.add_argument('--model', choices=MODELS, default=os.getenv('LLM_MODEL', 'qwen3.8-27b-noreason'))
    parser.add_argument('--budget', type=float, default=os.getenv('TOKEN_BUDGET'), help='Output-equivalent tokens; unset = proxy enforces total')
    parser.add_argument('--task-budget', type=float, default=8000)
    parser.add_argument('--max-steps', type=int, default=12)
    parser.add_argument('--max-output-tokens', type=int, default=1500)
    parser.add_argument('--time-limit', type=float, default=3500)
    parser.add_argument('--task-seconds', type=float, default=300)
    parser.add_argument('--request-seconds', type=float, default=90)
    parser.add_argument('--tool-seconds', type=float, default=20)
    parser.add_argument('--tool-output-chars', type=int, default=12000)
    parser.add_argument('--history-chars', type=int, default=24000)
    parser.add_argument('--reasoning-usage', choices=('included', 'separate'), default='included')
    parser.add_argument('--max-tokens-field', choices=('max_tokens', 'max_completion_tokens'), default='max_tokens')
    parser.add_argument('--task-id', action='append', help='Restrict to these IDs (repeatable)')
    parser.add_argument('--resume', action='store_true', help='Skip nonempty existing root answer.txt files')
    parser.add_argument('--dry-run', action='store_true', help='Show plan without imports, writes or API calls')
    args = parser.parse_args()
    for name in ('budget', 'task_budget', 'max_steps', 'max_output_tokens', 'time_limit', 'task_seconds', 'request_seconds', 'tool_seconds', 'tool_output_chars', 'history_chars', 'state_chars', 'finish_output_tokens', 'finish_seconds', 'light_steps', 'light_budget', 'router_budget'):
        value = getattr(args, name)
        if value is not None and (not math.isfinite(value) or value <= 0):
            parser.error(f'--{name.replace("_", "-")} must be positive and finite')
    if args.hybrid_turns < 0 or args.hybrid_turns > 2:
        parser.error('--hybrid-turns must be 0..2')
    if args.observation_memory_chars < 1024:
        parser.error('--observation-memory-chars must be >=1024')
    if (args.output_format!='text' or args.edit_tool or args.progress_state) and (args.response_protocol!='unified' or args.orchestration!='off' or args.selection_policy!='fixed'):
        parser.error('State-v2 features require unified protocol, fixed selection and no routing')
    if args.progress_state and (args.context_mode=='history' or args.observation_memory!='off'):
        parser.error('Progress state requires state/hybrid without the older observation buffer')
    if args.evidence_policy!='off' and not args.progress_state:
        parser.error('Precise evidence requires --progress-state')
    if args.progress_domains is not None and not args.progress_state:
        parser.error('--progress-domains requires --progress-state')
    if args.edit_domains is not None and not args.edit_tool:
        parser.error('--edit-domains requires --edit-tool')
    if args.observation_memory == 'bounded' and (args.response_protocol != 'unified' or args.context_mode == 'history' or args.orchestration != 'off' or args.selection_policy != 'fixed'):
        parser.error('Bounded observation memory currently requires unified state/hybrid and fixed selection without routing')
    if not math.isfinite(args.reasoning_budget) or args.reasoning_budget < 0:
        parser.error('--reasoning-budget must be nonnegative and finite')
    used_models=[args.model]+([args.router_model,args.light_model] if args.orchestration=='boundary' else [])
    if args.reasoning_budget == 0 and any(not model.endswith('-noreason') for model in used_models):
        parser.error('Reasoning models require an explicit positive --reasoning-budget')
    if args.model not in MODELS:
        parser.error('LLM_MODEL must be one of the four permitted models')
    if args.selection_policy != 'fixed' and args.selector_artifact is None:
        parser.error('--selector-artifact is required for shadow/learned selection')
    if args.orchestration != 'off' and args.selection_policy != 'fixed':
        parser.error('Boundary routing and empirical selection must be separate experiments')
    return args


async def main(args):
    tasks = sorted(load_tasks(args.tasks_dir.resolve()), key=priority)
    if args.task_id:
        missing = set(args.task_id) - {t.task_id for t in tasks}
        if missing:
            raise ValueError(f'Unknown task IDs: {sorted(missing)}')
        tasks = [t for t in tasks if t.task_id in args.task_id]
    if args.dry_run:
        print(json.dumps([{'task_id': t.task_id, 'type': t.task_type, 'reward': t.reward} for t in tasks], indent=2))
        return 0
    if not args.resume and any((t.directory/'answer.txt').exists() or (t.workspace/'answer.txt').exists() for t in tasks):
        raise ValueError('Existing answers found. Use a fresh task copy or --resume.')
    base_url, key = os.getenv('LLM_BASE_URL'), os.getenv('LLM_API_KEY')
    if not base_url or not key:
        raise ValueError('Set LLM_BASE_URL and LLM_API_KEY in the environment')
    try:
        from baseline.workflow import run_task
    except ImportError as exc:
        raise RuntimeError('openJiuwen unavailable. Install requirements.txt using Python 3.11–3.13. ' + str(exc)) from None
    portfolio = Portfolio.load(args.selector_artifact) if args.selector_artifact else None
    experience = Experience.load(args.experience_file) if args.experience_file else None
    pacer=Pacer(args.pacing)
    from baseline.client import Client
    logs = args.logs_dir or args.tasks_dir/'.baseline'
    run_id = time.strftime('%Y%m%d-%H%M%S') + f'-{os.getpid()}'
    trace = Trace(logs/f'{run_id}.jsonl')
    budget = Budget(args.budget, time.monotonic() + args.time_limit, estimator=InputEstimator(args.input_estimator))
    client = Client(base_url, key, args.model, budget, trace, args.request_seconds,
                    args.reasoning_usage, args.max_tokens_field, reasoning_budget=args.reasoning_budget)
    summary = {'reasoning_budget':args.reasoning_budget,'orchestration':args.orchestration,'pacing':args.pacing,'experience_file':str(args.experience_file) if args.experience_file else None,'loop_policy':args.loop_policy,'settings':settings(args), 'run_id': run_id, 'model': args.model, 'tasks': [], 'spent': 0,
               'budget': args.budget, 'selection_policy': args.selection_policy, 'screening': args.screening, 'input_estimator': args.input_estimator, 'budget_policy': args.budget_policy, 'finish_output_tokens': args.finish_output_tokens, 'finish_seconds': args.finish_seconds, 'context_mode': args.context_mode, 'response_protocol': args.response_protocol, 'status': 'running'}
    output = args.out or logs/f'{run_id}.json'
    atomic_write(output, json.dumps(summary, indent=2))
    features_by_id={task.task_id:task_features(task) for task in tasks}
    if portfolio and args.selection_policy=='learned':
        def rank(task):
            prediction=portfolio.recommend(features_by_id[task.task_id],settings(args),[m for m in args.candidate_models if args.reasoning_budget>0 or m.endswith('-noreason')],args.candidate_modes,
                args.task_budget if budget.limit is None else min(args.task_budget,budget.limit),task.reward)
            best=prediction['recommendation']
            score=best['utility'] if best else task.reward/({'math':100,'search':200,'data_analysis':200,'coding':500}.get(task.task_type,500))
            return (-score,task.task_id)
        tasks.sort(key=rank)
    failures = 0
    for task_index, task in enumerate(tasks):
        if args.resume and (task.directory/'answer.txt').is_file() and (task.directory/'answer.txt').read_text().strip():
            summary['tasks'].append({'task_id': task.task_id, 'status': 'existing_answer'})
            continue
        task_config=copy.copy(args)
        features=features_by_id[task.task_id]
        screening=screen_task(task) if args.screening=='shadow' else None
        trace.emit('task_inspect', task_id=task.task_id, features=features, settings=settings(args))
        if screening is not None:
            trace.emit('screening',task_id=task.task_id,**screening)
        decision=None
        if portfolio and args.selection_policy!='fixed':
            remaining=args.task_budget if budget.limit is None else min(args.task_budget,max(0,budget.limit-budget.spent))
            decision=portfolio.recommend(features,settings(args),[m for m in args.candidate_models if args.reasoning_budget>0 or m.endswith('-noreason')],args.candidate_modes,remaining,task.reward)
            trace.emit('selection',task_id=task.task_id,policy=args.selection_policy,**decision)
            if args.selection_policy=='learned' and decision['recommendation']:
                task_config.model=decision['recommendation']['model']
                task_config.context_mode=decision['recommendation']['mode']
        client.model=task_config.model
        task_config._portfolio=portfolio
        task_config._task_features=features
        before = budget.spent
        start = time.monotonic()
        print(f'[{task.task_id}] start; spent={budget.spent:.1f}', flush=True)
        stop = False
        transient = False
        try:
            budget.remaining_seconds()
            remaining=None if budget.limit is None else max(0,budget.limit-budget.spent)
            pending=[t.reward for t in tasks[task_index:] if not (args.resume and (t.directory/'answer.txt').is_file())]
            task_config.task_budget=pacer.allocate(remaining,args.task_budget,task.reward,pending)
            routing=None
            reference=Pacer('weighted').allocate(remaining,args.task_budget,task.reward,pending)
            if args.pacing=='advisory':task_config._pacing_reference=reference
            trace.emit('pacing',task_id=task.task_id,policy=args.pacing,allowance=task_config.task_budget,reference=reference,pending=len(pending))
            if args.orchestration=='boundary':
                routing=route(task,features,client,budget,trace,task_config,experience)
                task_config.task_budget=max(0,task_config.task_budget-routing['cost'])
                if routing['strategy']=='light':
                    task_config.model=args.light_model;task_config.context_mode='history'
                    task_config.max_steps=min(args.light_steps,args.max_steps)
                    task_config.task_budget=min(task_config.task_budget,args.light_budget)
                else:
                    task_config.context_mode='state'
                trace.emit('routing',task_id=task.task_id,router_model=args.router_model,selected_model=task_config.model,selected_mode=task_config.context_mode,**routing)
            client.model=task_config.model
            lesson=experience.lesson(task) if experience else None
            task_config._experience_lesson=lesson
            trace.emit('experience',task_id=task.task_id,lesson=lesson)
            trace.emit('task_start',task_id=task.task_id,features=features,settings=settings(task_config),model=task_config.model,context_mode=task_config.context_mode)
            result = await run_task(task, client, budget, trace, task_config)
        except StopRun as exc:
            result = {'status': 'stopped', 'reason': str(exc)}
            transient = str(exc).startswith(TRANSIENT_STOP_PREFIXES)
            stop = not transient
        except Exception as exc:
            result = {'status': 'error', 'reason': f'{type(exc).__name__}: {exc}'}
            # Unexpected infrastructure/framework errors should not repeat for every
            # task: tolerated per task, but bounded by MAX_CONSECUTIVE_FAILURES.
            transient = True
        if transient:
            failures += 1
            if failures >= MAX_CONSECUTIVE_FAILURES:
                stop = True
        else:
            failures = 0
        record = {'settings':settings(task_config),'task_id': task.task_id, 'model':task_config.model, 'context_mode':task_config.context_mode, 'selection_fallback':decision['fallback'] if decision else None, 'reward': task.reward, **result,
                  'cost': budget.spent-before, 'seconds': time.monotonic()-start}
        trace.emit('task_end', **record)
        summary['tasks'].append(record)
        summary.update(spent=budget.spent, reasoning_spent=client.reasoning_spent, usage_estimated=budget.estimated)
        atomic_write(output, json.dumps(summary, ensure_ascii=False, indent=2))
        print(f'[{task.task_id}] {result["status"]}; cost={record["cost"]:.1f}', flush=True)
        if stop:
            summary['status'] = 'stopped'
            break
        if transient:
            trace.emit('task_failure_continue', task_id=task.task_id, consecutive=failures, reason=result['reason'])
            delay = min(FAILURE_BACKOFF_SECONDS * failures, budget.deadline - time.monotonic() - 1)
            if delay > 0:
                await asyncio.sleep(delay)
    if summary['status'] == 'running':
        summary['status'] = 'finished'
    atomic_write(output, json.dumps(summary, ensure_ascii=False, indent=2))
    print(f'Summary: {output}', flush=True)
    return 0 if summary['status'] == 'finished' else 2


if __name__ == '__main__':
    try:
        sys.exit(asyncio.run(main(arguments())))
    except (ValueError, RuntimeError) as exc:
        print(f'Error: {exc}', file=sys.stderr)
        sys.exit(2)

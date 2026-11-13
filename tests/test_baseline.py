import asyncio
import importlib.util
import json
import os
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from baseline.client import Client
from baseline.core import Budget, StopRun, Task, Trace, load_tasks
from baseline.solver import solve
from baseline.tools import execute, run_command, safe_path

ROOT = Path(__file__).resolve().parents[1]


def config(**overrides):
    values = dict(task_seconds=30, task_budget=8000, max_steps=8,
                  history_chars=24000, max_output_tokens=1500,
                  tool_seconds=2, tool_output_chars=12000)
    values.update(overrides)
    return SimpleNamespace(**values)


class ScriptedClient:
    timeout = 10

    def __init__(self, actions, budget):
        self.actions = iter(actions)
        self.budget = budget
        self.messages = []

    def complete(self, messages, task_id, max_output, task_remaining):
        self.messages.append(messages)
        self.budget.spent += 10
        action = next(self.actions)
        if 'Response protocol: unified-v1.' in messages[0]['content']:
            return json.dumps({'action': action})
        return json.dumps(action)


class BudgetTests(unittest.TestCase):
    def test_reasoning_included_and_cache_not_discounted(self):
        usage = {'prompt_tokens': 100, 'completion_tokens': 40,
                 'prompt_tokens_details': {'cached_tokens': 80},
                 'completion_tokens_details': {'reasoning_tokens': 30}}
        budget = Budget(1000, time.monotonic()+10)
        self.assertEqual(budget.charge(usage, 1000, 500, 'included')['cost'], 50)
        self.assertEqual(budget.charge(usage, 1000, 500, 'separate')['cost'], 80)

    def test_missing_usage_is_reserved(self):
        budget = Budget(None, time.monotonic()+10)
        result = budget.charge(None, 100, 500, 'included')
        self.assertEqual(result['cost'], 510)
        self.assertTrue(result['estimated'])

    def test_allowance_and_deadline(self):
        budget = Budget(100, time.monotonic()+10)
        cap, estimate = budget.allowance([], 1500, 1000)
        self.assertLessEqual(cap+estimate/10, 100)
        with self.assertRaises(StopRun):
            Budget(1, time.monotonic()+10).allowance([], 1500, 1000)
        with self.assertRaises(StopRun):
            Budget(None, time.monotonic()-1).remaining_seconds()


class ToolTests(unittest.TestCase):
    def test_timeout_and_output_limit(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            result = run_command("python3 -c 'print(\"x\"*100000)'", root, 3, 100)
            self.assertEqual(result['returncode'], 0)
            self.assertEqual(len(result['output']), 100)
            self.assertTrue(result['truncated'])
            start = time.monotonic()
            result = run_command('sleep 20', root, 0.15)
            self.assertTrue(result['timeout'])
            self.assertLess(time.monotonic()-start, 3)

    def test_paths_and_no_secret_environment(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with self.assertRaises(ValueError):
                safe_path(root, '../secret')
            execute({'action': 'write', 'path': 'a.txt', 'content': 'hello'}, root, 1, 100)
            self.assertEqual(execute({'action': 'read', 'path': 'a.txt'}, root, 1, 100)['content'], 'hello')
            with patch.dict(os.environ, {'LLM_API_KEY': 'test-secret'}):
                result = run_command('printf "%s" "${LLM_API_KEY-unset}"', root, 2)
            self.assertEqual(result['output'], 'unset')


class ClientTests(unittest.TestCase):
    def test_url_usage_and_payload(self):
        with tempfile.TemporaryDirectory() as directory:
            budget = Budget(10000, time.monotonic()+10)
            client = Client('https://example.invalid/v1', 'private', 'qwen3.8-27b-noreason', budget, Trace(Path(directory)/'trace'))
            class Response:
                def __enter__(self): return self
                def __exit__(self, *args): pass
                def read(self, limit):
                    return json.dumps({'choices': [{'message': {'content': '{"action":"final","answer":"42"}'}, 'finish_reason': 'stop'}],
                                       'usage': {'prompt_tokens': 100, 'completion_tokens': 20}}).encode()
            with patch('urllib.request.urlopen', return_value=Response()) as request:
                result = client.complete([{'role': 'user', 'content': 'hi'}], 'id', 500, 5000)
            self.assertEqual(json.loads(result)['answer'], '42')
            self.assertEqual(budget.spent, 30)
            payload = json.loads(request.call_args.args[0].data)
            self.assertEqual(payload['max_tokens'], 500)
            self.assertEqual(client.url, 'https://example.invalid/v1/chat/completions')
            self.assertNotIn('private', (Path(directory)/'trace').read_text())

    def test_bad_response_stops_and_charges_estimate(self):
        with tempfile.TemporaryDirectory() as directory:
            budget = Budget(None, time.monotonic()+10)
            client = Client('https://example.invalid/v1', 'key', 'model', budget, Trace(Path(directory)/'trace'))
            with patch('urllib.request.urlopen', side_effect=TimeoutError):
                with self.assertRaises(StopRun):
                    client.complete([], 'id', 500, 5000)
            self.assertGreaterEqual(budget.spent, 500)
            self.assertTrue(budget.estimated)


class SolverTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.directory = self.root/'tasks'/'synthetic'
        self.directory.mkdir(parents=True)
        self.task = Task(self.directory, 'synthetic', 'math', 'Compute 6*7', 5)
        self.budget = Budget(10000, time.monotonic()+30)
        self.trace = Trace(self.root/'trace.jsonl')

    def run_solver(self, actions, **options):
        client = ScriptedClient(actions, self.budget)
        result = solve(self.task, client, self.budget, self.trace, config(**options))
        return result, client

    def test_calculate_then_submit(self):
        result, client = self.run_solver([
            {'action': 'bash', 'command': 'python3 -c "print(6*7)"'},
            {'action': 'final', 'answer': '42'}])
        self.assertEqual(result['status'], 'submitted')
        self.assertEqual((self.directory/'answer.txt').read_text(), '42\n')
        self.assertIn('42', client.messages[1][-2]['content'])

    def test_no_submit_when_global_budget_exhausted(self):
        self.budget.limit = 10
        with self.assertRaises(StopRun):
            self.run_solver([{'action': 'final', 'answer': '42'}])
        self.assertFalse((self.directory/'answer.txt').exists())

    def test_skip_and_step_limit(self):
        result, _ = self.run_solver([{'action': 'skip', 'reason': 'missing data'}])
        self.assertEqual(result['status'], 'skipped')
        result, _ = self.run_solver([{'action': 'read', 'path': 'absent'}], max_steps=1)
        self.assertEqual(result['reason'], 'max_steps')
        self.assertFalse((self.directory/'answer.txt').exists())

    def coding_fixture(self):
        self.task.task_type = 'coding'
        self.task.workspace.mkdir()
        (self.task.workspace/'mylib.py').write_text('def total(xs):\n    return -sum(xs)\n')
        (self.task.workspace/'test_mylib.py').write_text('from mylib import total\nassert total([1, 2]) == 3\n')

    def test_coding_fix_validates_and_persists(self):
        self.coding_fixture()
        result, _ = self.run_solver([
            {'action': 'write', 'path': 'mylib.py', 'content': 'def total(xs):\n    return sum(xs)\n'},
            {'action': 'final', 'answer': 'DONE'}])
        self.assertEqual(result['status'], 'submitted')
        self.assertIn('return sum', (self.task.workspace/'mylib.py').read_text())
        self.assertEqual((self.task.workspace/'answer.txt').read_text(), 'DONE\n')

    def test_failed_tests_do_not_submit(self):
        self.coding_fixture()
        result, _ = self.run_solver([{'action': 'final', 'answer': 'DONE'}], max_steps=1)
        self.assertEqual(result['status'], 'skipped')
        self.assertFalse((self.directory/'answer.txt').exists())
        self.assertIn('-sum', (self.task.workspace/'mylib.py').read_text())

    def test_modified_tests_rejected(self):
        self.coding_fixture()
        original = (self.task.workspace/'test_mylib.py').read_text()
        result, _ = self.run_solver([
            {'action': 'write', 'path': 'test_mylib.py', 'content': 'pass\n'},
            {'action': 'final', 'answer': 'DONE'}], max_steps=2)
        self.assertEqual(result['status'], 'skipped')
        self.assertFalse((self.directory/'answer.txt').exists())
        self.assertEqual((self.task.workspace/'test_mylib.py').read_text(), original)

    @unittest.skipUnless(importlib.util.find_spec('openjiuwen'), 'openJiuwen not installed')
    def test_real_workflow_with_scripted_model(self):
        from baseline.workflow import run_task
        client = ScriptedClient([{'action': 'final', 'answer': '42'}], self.budget)
        result = asyncio.run(run_task(self.task, client, self.budget, self.trace, config()))
        self.assertEqual(result['status'], 'submitted')
        self.assertEqual((self.directory/'answer.txt').read_text(), '42\n')


class ScriptTests(unittest.TestCase):
    @unittest.skipUnless(importlib.util.find_spec('openjiuwen'), 'openJiuwen not installed')
    def test_full_cli_workflow_http_and_summary(self):
        import run_agent
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            task = root/'tasks'/'synthetic'
            task.mkdir(parents=True)
            (task/'task.json').write_text(json.dumps({'task_id': 'synthetic', 'task_type': 'math',
                'instruction': 'Compute 6*7', 'reward': 5}))
            class Response:
                def __enter__(self): return self
                def __exit__(self, *args): pass
                def read(self, limit):
                    return json.dumps({'choices': [{'message': {'content': '{"action":{"action":"final","answer":"42"}}'}}],
                        'usage': {'prompt_tokens': 100, 'completion_tokens': 20}}).encode()
            with patch.object(sys, 'argv', ['run_agent.py', '--tasks-dir', str(root/'tasks'), '--out', str(root/'summary.json'), '--budget', '1000']):
                args = run_agent.arguments()
            with patch.dict(os.environ, {'LLM_BASE_URL': 'https://example.invalid/v1', 'LLM_API_KEY': 'test-private-key'}):
                with patch('urllib.request.urlopen', return_value=Response()):
                    code = asyncio.run(run_agent.main(args))
            self.assertEqual(code, 0)
            self.assertEqual((task/'answer.txt').read_text(), '42\n')
            summary = json.loads((root/'summary.json').read_text())
            self.assertEqual(summary['spent'], 30)
            self.assertEqual(summary['tasks'][0]['status'], 'submitted')
            self.assertNotIn('test-private-key', ''.join(p.read_text() for p in (root/'tasks'/'.baseline').glob('*.jsonl')))

    def _proxy_failure_run(self, task_rewards, failing_calls):
        import run_agent
        import urllib.error
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        root = Path(directory.name)
        for task_id, reward in task_rewards.items():
            task = root/'tasks'/task_id
            task.mkdir(parents=True)
            (task/'task.json').write_text(json.dumps({'task_id': task_id, 'task_type': 'math',
                'instruction': 'Compute 6*7', 'reward': reward}))
        class Response:
            def __enter__(self): return self
            def __exit__(self, *args): pass
            def read(self, limit):
                return json.dumps({'choices': [{'message': {'content': '{"action":{"action":"final","answer":"42"}}'}}],
                    'usage': {'prompt_tokens': 100, 'completion_tokens': 20}}).encode()
        calls = []
        def urlopen(request, timeout=None):
            calls.append(1)
            if len(calls) <= failing_calls:
                raise urllib.error.HTTPError('https://example.invalid', 502, 'Bad Gateway', {}, None)
            return Response()
        with patch.object(sys, 'argv', ['run_agent.py', '--tasks-dir', str(root/'tasks'), '--out', str(root/'summary.json')]):
            args = run_agent.arguments()
        with patch.dict(os.environ, {'LLM_BASE_URL': 'https://example.invalid/v1', 'LLM_API_KEY': 'k'}), \
             patch.object(run_agent, 'FAILURE_BACKOFF_SECONDS', 0), \
             patch('urllib.request.urlopen', side_effect=urlopen):
            code = asyncio.run(run_agent.main(args))
        return code, root, json.loads((root/'summary.json').read_text())

    @unittest.skipUnless(importlib.util.find_spec('openjiuwen'), 'openJiuwen not installed')
    def test_single_proxy_error_skips_only_that_task(self):
        code, root, summary = self._proxy_failure_run({'first': 5, 'second': 1}, failing_calls=1)
        self.assertEqual(code, 0)
        self.assertEqual(summary['status'], 'finished')
        self.assertEqual([t['status'] for t in summary['tasks']], ['stopped', 'submitted'])
        self.assertFalse((root/'tasks'/'first'/'answer.txt').exists())
        self.assertEqual((root/'tasks'/'second'/'answer.txt').read_text(), '42\n')

    @unittest.skipUnless(importlib.util.find_spec('openjiuwen'), 'openJiuwen not installed')
    def test_consecutive_proxy_errors_stop_run(self):
        import run_agent
        rewards = {f't{i}': 10-i for i in range(run_agent.MAX_CONSECUTIVE_FAILURES+2)}
        code, root, summary = self._proxy_failure_run(rewards, failing_calls=100)
        self.assertEqual(code, 2)
        self.assertEqual(summary['status'], 'stopped')
        self.assertEqual(len(summary['tasks']), run_agent.MAX_CONSECUTIVE_FAILURES)
        self.assertTrue(all(t['status'] == 'stopped' for t in summary['tasks']))

    def test_prepare_hides_answers_and_cli_reads_tasks(self):
        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory)/'tasks'
            result = subprocess.run([sys.executable, str(ROOT/'scripts/prepare_run.py'), '--source', str(ROOT/'task-dir'), '--dest', str(destination)], capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(len(load_tasks(destination)), 8)
            for file in destination.glob('*/task.json'):
                raw = json.loads(file.read_text())
                self.assertNotIn('expected', raw)
                self.assertNotIn('check', raw)
            result = subprocess.run([sys.executable, str(ROOT/'run_agent.py'), '--tasks-dir', str(destination), '--dry-run'], capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(len(json.loads(result.stdout)), 8)
            self.assertFalse((destination/'.baseline').exists())

    def test_evaluator_scores_only_saved_answers(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            reference, tasks = root/'reference'/'math', root/'tasks'/'math'
            reference.mkdir(parents=True)
            tasks.mkdir(parents=True)
            (reference/'task.json').write_text(json.dumps({'task_id': 'math', 'reward': 5, 'expected': '42', 'check': 'exact_int'}))
            command = [sys.executable, str(ROOT/'scripts/evaluate.py'), '--reference', str(root/'reference'), '--tasks-dir', str(root/'tasks')]
            result = subprocess.run(command, capture_output=True, text=True)
            self.assertEqual(json.loads(result.stdout)['score'], 0)
            self.assertEqual(result.returncode, 1)
            (tasks/'answer.txt').write_text('42\n')
            result = subprocess.run(command, capture_output=True, text=True)
            self.assertEqual(json.loads(result.stdout)['score'], 5)
            self.assertEqual(result.returncode, 0)

class UnifiedSolverTests(unittest.TestCase):
    def test_invalid_envelope_never_executes_then_solver_recovers_in_all_modes(self):
        for mode in ('history','state','hybrid'):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as directory:
                root=Path(directory);task_dir=root/'task';task_dir.mkdir()
                task=Task(task_dir,'recovery','math','Compute 6*7',1)
                budget=Budget(10000,time.monotonic()+30)
                responses=iter([
                    {'action':{'action':'bash','command':'touch marker','unexpected':True},'state_patch':{'facts':{'bad':'never commit'}}},
                    {'action':{'action':'bash','command':'test ! -e marker && echo clean'}},
                    {'action':{'action':'final','answer':'42'}},
                ])
                class RawClient:
                    timeout=10
                    def complete(self,*args):
                        budget.spent+=10
                        return json.dumps(next(responses))
                trace=Trace(root/'trace.jsonl')
                result=solve(task,RawClient(),budget,trace,config(context_mode=mode,response_protocol='unified'))
                self.assertEqual(result['status'],'submitted')
                events=[json.loads(line) for line in trace.path.read_text().splitlines()]
                self.assertEqual(sum(e['event']=='response_error' for e in events),1)
                successful_tools=[e for e in events if e['event']=='tool' and e['result'].get('returncode')==0]
                self.assertEqual(successful_tools[0]['result']['output'],'clean\n')
                for e in events:
                    if e['event']=='state': self.assertNotIn('bad',e['state']['facts'])

    def test_solver_legacy_and_unified_controls_submit_identical_answer(self):
        for protocol in ('legacy','unified'):
            for mode in ('history','state','hybrid'):
                with self.subTest(protocol=protocol,mode=mode), tempfile.TemporaryDirectory() as directory:
                    root=Path(directory);task=Task(root,'control','math','Compute 6*7',1)
                    budget=Budget(1000,time.monotonic()+30)
                    class RawClient:
                        timeout=10
                        def complete(self,*args):
                            budget.spent+=10
                            action={'action':'final','answer':'42'}
                            if protocol=='legacy' and mode=='history': return json.dumps(action)
                            return json.dumps({'action':action,'state_patch':{}})
                    result=solve(task,RawClient(),budget,Trace(root/'trace'),config(context_mode=mode,response_protocol=protocol))
                    self.assertEqual(result['status'],'submitted')
                    self.assertEqual((root/'answer.txt').read_text(),'42\n')

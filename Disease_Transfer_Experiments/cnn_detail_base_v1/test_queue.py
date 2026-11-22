import json,sys,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
import supervisor


class QueueTests(unittest.TestCase):
    def exercise(self,pause_first=False,already_done_first=False):
        with tempfile.TemporaryDirectory(dir=supervisor.HERE) as tmp:
            here=Path(tmp);cfg={'run_name':'toy','arms':['resolution384','global_local'],'progress_interval_seconds':900}
            config=here/'config.json';config.write_text(json.dumps(cfg));root=here/'runs/toy';seen=[]
            def prepared(*args):
                if already_done_first:
                    supervisor.atomic_json(root/'resolution384/status.json',{'phase':'complete'})
                    supervisor.atomic_json(root/'resolution384/verification.json',{'passed':True})
                return {'fingerprint':'toy'}
            def worker(command,run,config,**kwargs):
                arm=Path(run).name;seen.append(arm)
                self.assertEqual(command[command.index('--arm')+1],arm)
                supervisor.atomic_json(Path(run)/'status.json',{'phase':'paused' if pause_first else 'complete'})
                callback=kwargs.get('progress_callback')
                if callback:callback()
            with patch.object(supervisor,'HERE',here),patch.object(supervisor,'run_controller',side_effect=worker),\
                 patch('protocol.load_config',return_value=cfg),patch('protocol.prepare',side_effect=prepared),\
                 patch('evaluation.finalize_study',return_value=42),patch.object(sys,'argv',['supervisor','--config',str(config)]):
                supervisor.main()
            return seen,json.loads((root/'status.json').read_text())
    def test_runs_two_arms_in_order_and_reports_complete(self):
        seen,state=self.exercise();self.assertEqual(seen,['resolution384','global_local'])
        self.assertEqual(state['phase'],'complete');self.assertEqual(state['completed_arms'],seen)
    def test_pausing_first_arm_does_not_start_second(self):
        seen,state=self.exercise(pause_first=True);self.assertEqual(seen,['resolution384']);self.assertEqual(state['phase'],'paused')
    def test_resume_skips_verified_completed_first_arm(self):
        seen,state=self.exercise(already_done_first=True);self.assertEqual(seen,['global_local'])
        self.assertEqual(state['completed_arms'],['resolution384','global_local'])

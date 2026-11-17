import hashlib,json
from environment import BASE
from run_experiments import run
p=BASE/'results/runs/PSO_revised_42.csv';before=hashlib.sha256(p.read_bytes()).hexdigest();run('PSO','revised',42);after=hashlib.sha256(p.read_bytes()).hexdigest();assert before==after
(BASE/'validation/deterministic_repeat.json').write_text(json.dumps({'algorithm':'PSO','arm':'revised','seed':42,'steps':200,'learning_updates':73,'csv_sha256_before':before,'csv_sha256_repeat':after,'identical':before==after},indent=2),'utf8')

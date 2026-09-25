Detail on #2 — bayes_opt.py:298-312.
 
The loop:
 
while submitted_jobs < max_trials or jobs:
    def run_trials():
        trial_index_to_param = client.get_next_trials(
            min(parallel_limit - len(jobs), max_trials - submitted_jobs)
        )
        for trial_index, parameters in trial_index_to_param.items():
            job = executor.submit(objective_function, parameters, trial_index)
            submitted_jobs += 1
            jobs.append(...)
            time.sleep(1)
 
The min(...) is meant to say "ask for zero trials when the queue is full or the budget is spent". Ax does not honor that. In generation_strategy.py:
 
# Only check trial limit when requesting multiple trials; when num_trials <= 1,
if num_trials > 1:
    ...
    num_trials = min(num_trials, new_trials_limit)
num_trials = max(num_trials, 1)  # Ensure at least 1 trial
for _ in range(num_trials):
 
Two things go wrong at once. max(num_trials, 1) clamps 0 and every negative to 1, so a request for zero always yields one trial. And the if num_trials > 1 guard means Ax's own parallelism backstop (new_trials_limit, from the generation node's MaxTrialsAwaitingData(threshold=5)) is skipped precisely in the case you're relying on it.
 
Verified live:
 
>>> c.get_next_trials(0)
{0: {'discrete_learning_rate': 3.16e-05, 'update_ratio': 0.5}}
 
Two consequences, very different in severity.
 
Local (cluster="debug", what's running now): DebugJob executes synchronously inside job.result(), so jobs is drained to empty every pass and len(jobs) is 0 whenever run_trials is called. The only leak is at the end — once submitted_jobs == max_trials, min(...) is 0, Ax hands back one more trial, it runs, then jobs empties and the while exits. You get 501 trials instead of 500. Cosmetic.
 
SLURM (cluster="slurm", PARALLEL_LIMIT=5) is the real bug, and it is self-amplifying. Jobs are removed from jobs only when job.done(). While five real jobs are running, each pass through the while calls get_next_trials, gets a trial anyway, submits it, and appends — so len(jobs) becomes 6, then parallel_limit - len(jobs) is -1, clamped to 1 again, then 7, -2, and so on. Nothing in the loop pushes back. The only brake is time.sleep(1) per submission, so it queues roughly one job per second for as long as the first batch takes to finish. A 3-minute trial means ~180 surplus jobs before anything drains, each one burning a real SLURM allocation and an Ax trial index.
 
Fix is a guard at the call site, since Ax won't provide one:
 
def run_trials():
    global submitted_jobs
    n = min(parallel_limit - len(jobs), max_trials - submitted_jobs)
    if n <= 0:
        time.sleep(5)   # nothing to submit: wait for a job to finish (also fixes #4)
        return
    trial_index_to_param = client.get_next_trials(n)
    ...
 
The time.sleep there does double duty — it's also the fix for the busy-wait in #4, since n <= 0 is exactly the "queue full, wait" state.

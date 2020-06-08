# against baseline

#python eval_agents.py --right ppo > result_baseline_ppo.txt &
#python eval_agents.py --right cma > result_baseline_cma.txt &
#python eval_agents.py --right ga > result_baseline_ga.txt &
python eval_agents.py --right ppo --rightpath zoo/ppo_sp/history_00000144.zip > result_baseline_ppo_sp.txt &
python eval_agents.py --right cma --rightpath zoo/cmaes_sp/slimevolley.cma.16.384.best.json > result_baseline_cma_sp.txt &

# against ppo

#python eval_agents.py --left ppo --right cma > result_ppo_cma.txt &
#python eval_agents.py --left ppo --right ga > result_ppo_ga.txt &
python eval_agents.py --left ppo --right ppo --rightpath zoo/ppo_sp/history_00000144.zip > result_ppo_ppo_sp.txt &
python eval_agents.py --left ppo --right cma --rightpath zoo/cmaes_sp/slimevolley.cma.16.384.best.json > result_ppo_cma_sp.txt &

# against cma

#python eval_agents.py --left cma --right ga > result_cma_ga.txt &
python eval_agents.py --left cma --right ppo --rightpath zoo/ppo_sp/history_00000144.zip > result_cma_ppo_sp.txt &
python eval_agents.py --left cma --right cma --rightpath zoo/cmaes_sp/slimevolley.cma.16.384.best.json > result_cma_cma_sp.txt &

# against ga

python eval_agents.py --left ga --right ppo --rightpath zoo/ppo_sp/history_00000144.zip > result_ga_ppo_sp.txt &
python eval_agents.py --left ga --right cma --rightpath zoo/cmaes_sp/slimevolley.cma.16.384.best.json > result_ga_cma_sp.txt &

# against ppo_sp

python eval_agents.py --left ppo --leftpath zoo/ppo_sp/history_00000144.zip --right cma --rightpath zoo/cmaes_sp/slimevolley.cma.16.384.best.json > result_ppo_sp_cma_sp.txt &


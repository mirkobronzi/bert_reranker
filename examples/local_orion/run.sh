export ORION_DB_ADDRESS='orion_db.pkl'
export ORION_DB_TYPE='pickleddb'

orion hunt --non-monitored-arguments gpu --config orion_config.yaml main --config config.yaml --output '{exp.working_dir}/{exp.name}_{trial.id}/' --train --log '{exp.working_dir}/{exp.name}_{trial.id}/exp.log' 

# luxai
Kaggle Lux AI Competition submission

Competition link: https://www.kaggle.com/c/lux-ai-2021

Highlights:
1. Training commenced in stages. In a single stage, training is done on a selection of seed models based on different reward policies. As a stage is completed (all models trained), the best models in the stage are selected to be the new seed models in the next stage. 
2. 4 phases of training: phase 1 was more exploratory (wide range of parameters reward policies used), then as the phases progressed, the training parameters and reward policies became narrower and less version of models were produced in each stage
3. Training and evaluation in each stage was done on 2 devices. 2 roles: master and worker. Master was responsible for syncing the jobs and files while worker performed the training and evaluation jobs and reports and gets jobs from master.

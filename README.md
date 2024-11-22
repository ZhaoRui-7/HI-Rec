# HI-Rec
#### Quick start for downstream tasks with generated augment data:  
```bash
cd HI-Rec/JDR
```
download dataset:   
[dataset](https://drive.google.com/file/d/1eZ8riGuqifk-_kRBsKDxPp64kPFfqC1i/view?usp=sharing)
```bash
cd HI-Rec/JDR/RS_JDR   
python run_ctr.py   
```
#### The total process for HI_Rec:
##### HID
train the collaborate model:    
```bash
cd RS
python main.py --data_path dataset/ml-1m/feature --model_path checkpoint --whether_train train --epochs 10 --batch_size 2048 --dropout 0.0 --lr 0.001 --optimizer Adam --device gpu --encode_min 0.0 --neg_sampling 5 --use_senet False 
```
infer user/item embeddings    
```bash
python main.py --data_path dataset/ml-1m/feature --model_path checkpoint --whether_train test --epochs 10 --batch_size 2048 --dropout 0.0 --lr 0.001 --optimizer Adam --device gpu --encode_min 0.0 --neg_sampling 5 --use_senet False 
```
generate HID    
```bash
cd ID_encoding
python cluster_user.py
python cluster_item.py
python filter.py
python gen_user_id.py
python gen_item_id.py
```

##### JDR
gen_prompt     
```bash
python ./JDR/preprocess/amz_prompt/amz_prompt_prepare.py
python ./JDR/preprocess/amz_prompt/amz_prompt_usertask.py
```
same for other task and other dataset, and fine-tuning or inference with any LLM backbone   

downstream task    
```bash
python ./JDR/RS_JDR/run_ctr.py
```

Our implementation code is based on : 
[KAR](https://github.com/YunjiaXi/Open-World-Knowledge-Augmented-Recommendation/tree/main)
[DSSM](https://github.com/HeartFu/DSSM)

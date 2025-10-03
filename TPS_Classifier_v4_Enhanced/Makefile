.PHONY: knn predict calibrate eval

knn:
	@python scripts/build_index.py --train_fasta data/train.fasta --labels data/train_labels.csv --class_list data/classes.txt --out_index models/knn/index.npy --out_meta models/knn/index_meta.json

predict:
	@python scripts/predict.py --input data/val.fasta --class_list data/classes.txt --out preds/val_base.jsonl

calibrate:
	@python scripts/calibrate_thresholds.py --preds preds/val_base.jsonl --labels data/val_labels_binary.csv --class_list data/classes.txt --f1beta --beta 0.7 --out_dir models/calibration/

eval:
	@python scripts/predict.py --input data/val.fasta --class_list data/classes.txt --out preds/val_final.jsonl --use_knn --alpha 0.7 --calibration_dir models/calibration/
	@python scripts/evaluate.py --preds preds/val_base.jsonl preds/val_final.jsonl --labels data/val_labels_binary.csv --bootstrap 1000 --report reports/val_compare.json


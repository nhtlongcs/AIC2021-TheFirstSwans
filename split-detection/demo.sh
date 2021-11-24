# ./demo.sh ../data/labels/ ./folds/ 
mkdir -p $2
python data2csv.py --inp $1 --out $2/data.csv
python scripts/split_kfold.py --csv $2/data.csv --out $2/fold.csv --k 5
python scripts/split.py --csv $2/fold.csv --out $2/

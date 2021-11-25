# ./demo.sh ../data/labels/ ./folds/ 
mkdir -p $2
python data2csv.py --inp $1 --out $2/data.csv
python scripts/split_kfold.py --csv $2/data.csv --out $2/fold.csv --k 5
python scripts/split.py --csv $2/fold.csv --out $2/csv/
python csv2data.py --inp $2/csv/0/ --out $2
python csv2data.py --inp $2/csv/1/ --out $2
python csv2data.py --inp $2/csv/2/ --out $2
python csv2data.py --inp $2/csv/3/ --out $2
python csv2data.py --inp $2/csv/4/ --out $2

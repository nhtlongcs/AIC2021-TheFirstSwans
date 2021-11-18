# Evaluate

```bash
cd this-repo
python convert/label2pred.py <submission-folder-path>
python evaluation_det_e2e_offline/main.py <mode> text_results.json
```

# Ensemble

Update by dictionary

```bash
cd this-repo
python ensemble/update_dict.py <submission-folder-path>/* --threshold 0.5 --output_dir outputs/
```

Tuning hyperparameters

```bash
cd this-repo
python ensemble/params_search.py <submission-folder-path> --study <save-study-name>
```

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


# Docker

Pull already built docker:
```
docker pull vinhloiit/aic2021:latest
```

Or build using command (see build preparation below):
```
DOCKER_BUILDKIT=1 docker build --build-arg USERNAME=aic -t aic:latest .
```

Run command (final):
```
docker run --rm -it --gpus device=0 -v "$(pwd)"/TestB1:/data/test_data:ro -v "$(pwd)"/submission_output:/data/submission_output aic:latest /bin/bash run.sh
```

Go to inside docker container:
```
docker run --rm --gpus device=0 -it -v "$(pwd)"/test_data:/data/test_data -v "$(pwd)"/submission_output:/data/submission_output aic:latest /bin/bash
```

## Build preparation

- Put images that we want to test (e.g. TestB1) to root folder, e.g `./TestB1`. Or you can modify the path in the running command above (`-v "$(pwd)"/TestB1:/data/test_data:ro`. Format: `<path outside docker>:<path inside docker>:ro`, `ro` for `read-only`)
- Download weights to `weights` folder
- Update weight paths and run command in `run.sh`

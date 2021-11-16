import sys
# eval_mode = 'BASE'
eval_mode = sys.argv[1]

if eval_mode.upper() == 'BASE':
    from text_evaluation_raw import TextEvaluator
elif eval_mode.upper() == 'VIN':
    from text_evaluation_vin import TextEvaluator
else:
    ValueError('eval_mode not found')
print(f'using eval mode: {eval_mode}')
eval_dataset = 'vintext'
if eval_dataset == 'ctw1500':
    dataset_name = ['ctw1500']
    outdir = 'ctw1500_res'
elif eval_dataset == 'totaltext':
    dataset_name = ['totaltext']
    outdir = 'totaltext_res'
elif eval_dataset == 'vintext':
    dataset_name = ['vintext']
    outdir = 'vintext_res'
cfg = {}
cfg['INFERENCE_TH_TEST'] = 0.4  # tune this parameter to achieve best result
e = TextEvaluator(dataset_name, cfg, False,
                  output_dir=outdir, filepath=sys.argv[2])
res = e.evaluate()

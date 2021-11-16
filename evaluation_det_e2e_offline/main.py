eval_mode = 'BASE'

if eval_mode == 'BASE':
    from text_evaluation_raw import TextEvaluator
elif eval_mode == 'VIN':
    from text_evaluation_raw import TextEvaluator
else:
    ValueError('eval_mode not found')

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
e = TextEvaluator(dataset_name, cfg, False, output_dir=outdir)
res = e.evaluate()
print(res)

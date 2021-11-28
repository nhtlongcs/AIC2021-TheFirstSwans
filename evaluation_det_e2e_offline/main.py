import sys
# eval_mode = 'BASE'
eval_dataset = sys.argv[2]
eval_mode = 'base'

if eval_mode.upper() == 'BASE':
    from text_evaluation_raw import TextEvaluator
elif eval_mode.upper() == 'VIN':
    aaaaaa
    # from text_evaluation_vin import TextEvaluator
else:
    ValueError('eval_mode not found')
print(f'using eval mode: {eval_mode}')
if eval_dataset == '0':
    dataset_name = ['0']
    outdir = '0_res'
elif eval_dataset == '1':
    dataset_name = ['1']
    outdir = '1_res'
elif eval_dataset == '2':
    dataset_name = ['2']
    outdir = '2_res'
elif eval_dataset == '3':
    dataset_name = ['3']
    outdir = '3_res'
elif eval_dataset == '4':
    dataset_name = ['4']
    outdir = '4_res'
elif eval_dataset == 'vintext':
    dataset_name = ['vintext']
    outdir = 'vintext_res'
elif eval_dataset == 'TESTA':
    dataset_name = ['TESTA']
    outdir = 'TESTA_res'
cfg = {}
cfg['INFERENCE_TH_TEST'] = 0.4  # tune this parameter to achieve best result
e = TextEvaluator(dataset_name, cfg, False,
                  output_dir=outdir, filepath=sys.argv[1])
res = e.evaluate()

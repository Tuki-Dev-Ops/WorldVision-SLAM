import sys; sys.path.insert(0,'.')
from wme.eval.tum import load_sequence, load_trajectory
from wme.eval.trajectory import evaluate_ate
rows = [('freiburg3_sitting_halfsphere', 13052.56, 12.54),
        ('freiburg3_sitting_xyz',            1.63,  1.03),
        ('freiburg3_walking_xyz',            5.19,  6.48)]
print('%-32s %8s %10s %10s %10s' % ('sequence','OFF','was','now','agent'))
for s, was, said in rows:
    gt = load_sequence('../data/rgbd_dataset_'+s).trajectory()
    off = evaluate_ate(load_trajectory('../results/vr_%s_off.txt'%s), gt, align=True).rmse*100
    tok = evaluate_ate(load_trajectory('../results/vr_%s_token.txt'%s), gt, align=True).rmse*100
    print('%-32s %8.2f %10.2f %10.2f %10.2f' % (s, off, was, tok, said))

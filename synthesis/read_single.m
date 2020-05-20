clear
clc

fname = 'kamchatka_shot_45.output';
direct = '.\outputs\kamchatka\';
outdir = '.\color_info\';
get_shot_color(direct, fname, outdir)
get_shot_tss(direct, fname, outdir)
get_shot_lp(direct, fname, outdir)
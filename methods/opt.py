import argparse


parser = argparse.ArgumentParser(description='Video Analysis')

"----------------------------- General options -----------------------------"
parser.add_argument('--video_path', type=str, required=True)
parser.add_argument('--batch_size', type=int, default=2)
parser.add_argument('--win_len', type=int, default=100)
parser.add_argument('--processing_shape', type=tuple, default=(360, 640))

parser.add_argument('--sample_interval', type=int, default=1,
                    help='number of frames to process in one second')
parser.add_argument('--outputpath', type=str, default='./output')

"----------------------------- AlphaPose options -----------------------------"
parser.add_argument('--sp', default=True, action='store_true',
                    help='Use single process for pytorch')
parser.add_argument('--num_classes', type=int, default=80)
parser.add_argument('--write_json', default=False, action='store_true')

# ########## Model options ########## #
parser.add_argument('--nClasses', default=33, type=int,
                    help='Number of output channel')

# ########## Data options ########## #
parser.add_argument('--inputResH', default=320, type=int,
                    help='Input image height')
parser.add_argument('--inputResW', default=256, type=int,
                    help='Input image width')
parser.add_argument('--outputResH', default=80, type=int,
                    help='Output heatmap height')
parser.add_argument('--outputResW', default=64, type=int,
                    help='Output heatmap width')

# ########## Detection options ########## #
parser.add_argument('--inp_dim', dest='inp_dim', type=int, default=608,
                    help='inpdim')
parser.add_argument('--conf', dest='confidence', type=float, default=0.05,
                    help='bounding box confidence threshold')
parser.add_argument('--nms', dest='nms_thesh', type=float, default=0.6,
                    help='bounding box nms threshold')
parser.add_argument('--save_img', default=False, action='store_true',
                    help='save result as image')
parser.add_argument('--vis', default=False, action='store_true',
                    help='visualize image')
parser.add_argument('--matching', default=False, action='store_true',
                    help='use best matching')
parser.add_argument('--format', type=str,
                    help='save in the format of cmu or coco or openpose, option: coco/cmu/open')
parser.add_argument('--detbatch', type=int, default=1,
                    help='detection batch size')
parser.add_argument('--posebatch', type=int, default=80,
                    help='pose estimation maximum batch size')

# ########## Video options ########## #
parser.add_argument('--save_video', dest='save_video',
                    help='whether to save rendered video', default=False, action='store_true')

"----------------------------- Tone options -----------------------------"
parser.add_argument('--gray_bins', type=int, default=128)

"----------------------------- Symmetry options -----------------------------"
parser.add_argument('--cell_size', type=tuple, default=(30, 40), help='cell size')
parser.add_argument('--rgbbins', type=int, default=32)


opt = parser.parse_args()

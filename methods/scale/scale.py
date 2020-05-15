import torch

from ..opt import opt
from .pose_utils import (DataWriter, DetectionLoader, DetectionProcessor,
                         ImageLoader, InferenNet_fast, getTime)
from .pPose_nms import write_json
from .read_pose import PoseReader

args = opt
if not args.sp:
    torch.multiprocessing.set_start_method('forkserver', force=True)
    torch.multiprocessing.set_sharing_strategy('file_system')


class PoseRecog(object):
    def __init__(self):
        self.data_loader = ImageLoader(batchSize=args.detbatch)
        self.det_loader = DetectionLoader(self.data_loader, batchSize=args.detbatch)
        self.det_processor = DetectionProcessor(self.det_loader)
        self.pose_model = InferenNet_fast()
        self.pose_model.cuda()
        self.pose_model.eval()
        self.writer = DataWriter()
        self.pose_reader = PoseReader()

    def _reset(self):
        self.data_loader.reset()

    def get_pose(self, imgs, raw=False):
        datalen = imgs.shape[0]
        height = imgs.shape[1]
        width = imgs.shape[2]

        self.data_loader.feed(imgs, (width, height))
        self.data_loader.start()

        self.det_loader.init()
        self.det_loader.start()
        self.det_processor.init()
        self.det_processor.start()

        self.writer.init()
        self.writer.start()

        runtime_profile = {
            'dt': [],
            'pt': [],
            'pn': []
        }

        batchSize = args.posebatch
        for i in range(datalen):
            start_time = getTime()
            with torch.no_grad():
                (inps, orig_img, boxes, scores, pt1, pt2) = self.det_processor.read()
                if boxes is None or boxes.nelement() == 0:
                    self.writer.save(None, None, None, None, None, orig_img)
                    continue

                ckpt_time, det_time = getTime(start_time)
                runtime_profile['dt'].append(det_time)
                # Pose Estimation

                datalen = inps.size(0)
                leftover = 0
                if (datalen) % batchSize:
                    leftover = 1
                num_batches = datalen // batchSize + leftover
                hm = []
                for j in range(num_batches):
                    inps_j = inps[j * batchSize:min((j + 1) * batchSize, datalen)].cuda()
                    hm_j = self.pose_model(inps_j)
                    hm.append(hm_j)
                hm = torch.cat(hm)
                ckpt_time, pose_time = getTime(ckpt_time)
                runtime_profile['pt'].append(pose_time)
                hm = hm.cpu()
                self.writer.save(boxes, scores, hm, pt1, pt2, orig_img)

                ckpt_time, post_time = getTime(ckpt_time)
                runtime_profile['pn'].append(post_time)

        while(self.writer.running()):
            pass
        self.writer.stop()
        final_result = self.writer.results()
        if args.write_json:
            write_json(final_result, args.outputpath)
        if raw:
            return final_result, height
        return self.pose_reader.read_pose_mlp(final_result, height)

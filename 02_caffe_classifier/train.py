import argparse

#from ttictoc import TicToc
import sys

import google.protobuf as pb2
import google.protobuf.text_format

sys.path.insert(0, "../..")
sys.path.insert(0, "../detector_3D/lib")

from global_utils.sys_paths import SysPaths
from global_utils.logger_csv import Loss_logger
from train.configs.ModulesConfig import TrainConfig, MODULES_CONFIG


def parse_args():
    parser = argparse.ArgumentParser(description='Dataset loader server.')
    parser.add_argument("--config_path", dest="config_path", type=str, help='select configs file from directory configs/')

    args = parser.parse_args()
    return args


class TrainWrapper(object):

    def __init__(self, train_config):
        assert isinstance(train_config, TrainConfig)
        import caffe
        from caffe.proto import caffe_pb2
        caffe.set_mode_gpu()
        #caffe.set_device(train_config.gpu_id)

        self._solver = caffe.SGDSolver(train_config.solver_path)
        if train_config.caffemodel_path is not None:
            print('Loading pretrained model weights from {:s}'.format(train_config.caffemodel_path))
            self._solver.net.copy_from(train_config.caffemodel_path)

        self.solver_param = caffe_pb2.SolverParameter()
        with open(train_config.solver_path, 'rt') as f:
            pb2.text_format.Merge(f.read(), self.solver_param)

        self._train_config = train_config
        self._loss_logger = Loss_logger( self.solver_param.snapshot_prefix)

    def train_model(self):
        """Network training loop."""
        #timer = TicToc()
        while self._solver.iter < self.solver_param.max_iter:

            #timer.tic()
            self._solver.step(1)
            #timer.toc()

            #TODO: self._train_config.loss_blob_name_list[0] - for each loss blob
            loss = self._solver.net.blobs[self._train_config.loss_blob_name_list[0]].data.flat[0]
            self._loss_logger.add_loss(self._solver.iter, loss)

            # if self._solver.iter % (1 * self.solver_param.display) == 0:
            #     print('speed: {:.3f}s / iter'.format(timer.elapsed))


if __name__ == '__main__':

    args = parse_args()
    MODULES_CONFIG.setup(args.config_path)
    SysPaths.add_path(MODULES_CONFIG.init_config.caffe_path)

    sw = TrainWrapper(MODULES_CONFIG.train_config)
    print('Solving...')
    sw.train_model()
    print('done solving')

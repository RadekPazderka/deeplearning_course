import argparse

#from ttictoc import TicToc
import sys

# import google.protobuf as pb2
# import google.protobuf.text_format
#
# sys.path.insert(0, "../..")
# sys.path.insert(0, "../detector_3D/lib")
#
# from
# from global_utils.logger_csv import Loss_logger
# from train.configs.ModulesConfig import TrainConfig, MODULES_CONFIG
from typing import Optional

from .utils.sys_paths import SysPaths
from .utils.loss_logger import Loss_logger

def parse_args():
    parser = argparse.ArgumentParser(description='Dataset loader server.')
    parser.add_argument("--config_path", dest="config_path", type=str, help='select configs file from directory configs/')

    args = parser.parse_args()
    return args


class TrainWrapper(object):

    def __init__(self, caffemodel_path: Optional[str], solver_path: str):
        import caffe
        from caffe.proto import caffe_pb2
        caffe.set_mode_gpu()

        self._solver = caffe.SGDSolver(solver_path)
        if caffemodel_path is not None:
            print('Loading pretrained model weights from {:s}'.format(caffemodel_path))
            self._solver.net.copy_from(caffemodel_path)

        self.solver_param = caffe_pb2.SolverParameter()
        with open(solver_path, 'rt') as f:
            pb2.text_format.Merge(f.read(), self.solver_param)

        self._loss_logger = Loss_logger( self.solver_param.snapshot_prefix)

    def train_model(self):
        """Network training loop."""
        while self._solver.iter < self.solver_param.max_iter:

            self._solver.step(1)

            loss = self._solver.net.blobs["loss"].data.flat[0]
            self._loss_logger.add_loss(self._solver.iter, loss)


if __name__ == '__main__':
    PYTHON_CAFFE_PATH = ""

    SOLVER_PATH = ""
    CAFFE_MODEL_PATH = ""

    args = parse_args()
    SysPaths.add_path(PYTHON_CAFFE_PATH)

    sw = TrainWrapper(MODULES_CONFIG.train_config)
    print('Solving...')
    sw.train_model()
    print('done solving')

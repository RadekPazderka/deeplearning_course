import google.protobuf as pb2
import google.protobuf.text_format

from utils import sys_paths
from utils.loss_logger import Loss_logger

import caffe
from caffe.proto import caffe_pb2

class TrainWrapper(object):

    def __init__(self, solver_path, caffemodel_path):

        caffe.set_mode_gpu()

        self._solver = caffe.SGDSolver(solver_path)
        if caffemodel_path is not None:
            print('Loading pretrained models weights from {:s}'.format(caffemodel_path))
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

    SOLVER_PATH = "prototxt/animal_classifier/squeeze_net/solver.prototxt"
    CAFFEMODEL_PATH = "pretrained/squeeze_net/squeezenet_v1.0.caffemodel"


    sw = TrainWrapper(SOLVER_PATH, CAFFEMODEL_PATH)
    print('Solving...')
    sw.train_model()
    print('done solving')

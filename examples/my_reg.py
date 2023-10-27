import copy
import numpy as np
use_cuda = True
if use_cuda:
    import cupy as cp
    to_cpu = cp.asnumpy
    cp.cuda.set_allocator(cp.cuda.MemoryPool().malloc)
else:
    cp = np
    to_cpu = lambda x: x
import open3d as o3
from probreg import callbacks
import matplotlib.pyplot as plt
import os
import time

from probreg import transformation
from probreg import filterreg

PATH_SAVE = "/home/eric/github/probreg/result/"
PATH_DATA = "/home/eric/github/probreg/3dMerge/"


class savePly(object):
    """최적화 과정에서의 각 iteration 마다 데이터를 ply로 저장

    Args:
        source (numpy.ndarray): Source point cloud data.
        target (numpy.ndarray): Target point cloud data.
        save (bool, optional): If this flag is True,
            each iteration image is saved in a sequential number.
    """

    def __init__(self, source: np.ndarray, target: np.ndarray, save: bool = True, name: str = "tmp"):
        self._source = source
        self._target = target
        self._result = copy.deepcopy(self._source)
        self._save = save
        self._cnt = 0
        self._name = name

    def __call__(self, transformation: transformation) -> None:
        self._result = transformation.transform(self._source)

        pc1 = o3.geometry.PointCloud()
        pc1.points = o3.utility.Vector3dVector(self._result)
        pc1.paint_uniform_color([0, 1, 0])

        pc2 = o3.geometry.PointCloud()
        pc2.points = o3.utility.Vector3dVector(self._target)
        pc2.paint_uniform_color([1, 0, 0])


        combined_pc = o3.geometry.PointCloud()
        combined_pc += pc1
        combined_pc += pc2


        self.save_result(self._name, combined_pc)
        self._cnt += 1

    def save_result(self, name, res):
        save_path = PATH_SAVE + name
        
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        #result_filename = "{}/result_{}.ply".format(save_path, self._cnt)
        result_filename = "{}/source_{}.ply".format(save_path, name)
        o3.io.write_point_cloud(result_filename, res)
        #print(result_filename)

class reg(savePly):
    """지정된 폴더에서 source 클라우드와 target 클라우드를 open3d로 동작시킬 수 있게 불러옴 

    Args:
        data_name (str): file name.
    """
    def __init__(self, source, target, save = False):
        self.source = source
        self.target = target
        self.save = save
        

        self.cbs = [savePly(source, target, name = file)]

    
    def fillterreg(self, tf_param = {'rot': np.identity(3), 't': np.zeros(3)}): 
        tf_res, _, _ = filterreg.registration_filterreg(source, target, maxiter=32, update_sigma2 = True, w = 0.0000303556673416789, tol = 0.000486636460597193, tf_init_params = tf_param)
        #tf_param['rot'], tf_param['t'] = tf_res.rot, tf_res.t
        #tf_res, _, _ = filterreg.registration_filterreg(source, target, update_sigma2 = True, w = 0.02, tol = 0.0001, tf_init_params = tf_param) #callbacks=self.cbs)        
     
        return tf_res.rot, tf_res.t

    def display_inlier_outlier(cloud, ind):
        inlier_cloud = cloud.select_by_index(ind)
        outlier_cloud = cloud.select_by_index(ind, invert=True)
        sample_cloud = copy.deepcopy(inlier_cloud)
        print("Showing outliers (red) and inliers (gray): ")
        #outlier_cloud.paint_uniform_color([1, 0, 0])
        #inlier_cloud.paint_uniform_color([0.8, 0.8, 0.8])
        #o3.visualization.draw_geometries([inlier_cloud, outlier_cloud])
        return sample_cloud



    def pcl(self, tf_param = {'rot': np.identity(3), 't': np.zeros(3)}):
        threshold = 1000
        
        ##cl, ind = self.source.remove_statistical_outlier(nb_neighbors=100,std_ratio=1.0)
        #cl, ind = self.source.remove_radius_outlier(nb_points=32, radius=5)
        #
        #self.source = self.display_inlier_outlier(self.source, ind)
        ##evaluation = o3.pipelines.registration.evaluate_registration(self.source, self.target, threshold, tf_param)

        #reg_p2p = o3.pipelines.registration.registration_icp(
        #                            self.source, self.target, threshold, tf_param,
        #                            o3.pipelines.registration.TransformationEstimationPointToPoint(),
        #                            o3.pipelines.registration.ICPConvergenceCriteria(max_iteration=1000))

        #print("reg_p2p.transformation", reg_p2p.transformation)


        #return self.tf_res.rot, self.tf_res.t

###############################################################################

class getCmesData:
    """지정된 폴더에서 source 클라우드와 target 클라우드를 open3d로 동작시킬 수 있게 불러옴 

    Args:
        data_name (str): file name.
    """
    def __init__(self, data_name):
        print("### Geting CMES Logging Data",)

        source = PATH_DATA + '20231011_scan/015.ply'
        source = PATH_DATA + '20231011_scan/' + data_name + '.ply'
        target = PATH_DATA + 'data/square_points.ply'

        pcd = o3.io.read_point_cloud(source)
        pcd = pcd.voxel_down_sample(voxel_size=3)
        self._source = np.asarray(pcd.points)

        pcd = o3.io.read_point_cloud(target)    
        pcd = pcd.voxel_down_sample(voxel_size=3)
        self._target = np.asarray(pcd.points)

        print("## source: ", source)
        print("## target: ", target)

    def run(self):
        return self._source, self._target

file_list = ["001", "002", "003", "004", "005", "006", "007", "008", "009", "010", "011", "012", "013", "014", "015"]
rot, t = np.identity(3), np.zeros(3)


for file in file_list:
    getData = getCmesData(file)
    source, target = getData.run()

    my_reg = reg(source, target)

    start = time.time()
    rot, t = my_reg.fillterreg()     # run fillterreg
    #rot, t = my_reg.pcl()            # run pcl
    elapsed = time.time() - start

    print("time: ", elapsed)
    print("rot: ", rot)
    print("t: ", t)
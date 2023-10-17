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
        #combined_pc += pc2


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

for file in file_list:
    getData = getCmesData(file)
    source, target = getData.run()
    cbs = [savePly(source, target, name = file)]


    tf_params = {'rot': np.identity(3), 't': np.zeros(3)}
    sigma = 250080.42669642388


    start = time.time()

    tf_param, sigma, q = filterreg.registration_filterreg(source, target, update_sigma2 = True, w = 0.3, tol = 0.1)
    tf_params['rot'], tf_params['t'] = tf_param.rot, tf_param.t
    tf_param, sigma, q = filterreg.registration_filterreg(source, target, update_sigma2 = True, w = 0.02, tol = 0.0001, tf_init_params = tf_params, callbacks=cbs) #save

    #tf_param, sigma, q = filterreg.registration_filterreg(source, target, \
    #                                                        #target_normals = target, \
    #                                                        #sigma2 = sigma, \
    #                                                        update_sigma2 = True, \
    #                                                        w = 0.3, \
    #                                                        maxiter = 50, \
    #                                                        tol = 0.1, \
    #                                                        #min_sigma2 = 100, \
    #                                                        callbacks=cbs)

    elapsed = time.time() - start

    print("rot: ", tf_param.rot)
    print("t: ", tf_param.t)
    print("time: ", elapsed)


"""FilterReg registration

    Args:
        source (numpy.ndarray): Source point cloud data.
        target (numpy.ndarray): Target point cloud data.
        target_normals (numpy.ndarray, optional): Normal vectors of target point cloud.
        sigma2 (float, optional): Variance of GMM. If `sigma2` is `None`, `sigma2` is automatically updated.
        w (float, optional): Weight of the uniform distribution, 0 < `w` < 1.
        objective_type (str, optional): The type of objective function selected by 'pt2pt' or 'pt2pl'.
        maxitr (int, optional): Maximum number of iterations to EM algorithm.
        tol (float, optional): Tolerance for termination.
        min_sigma2 (float, optional): Minimum variance of GMM.
        feature_fn (function, optional): Feature function. If you use FPFH feature, set `feature_fn=probreg.feature.FPFH()`.
        callback (:obj:`list` of :obj:`function`, optional): Called after each iteration.
            `callback(probreg.Transformation)`

    Keyword Args:
        tf_init_params (dict, optional): Parameters to initialize transformation (for rigid).

        Args:
            rot (numpy.ndarray, optional): Rotation matrix.
            t (numpy.ndarray, optional): Translation vector.
            scale (Float, optional): Scale factor.
            xp (module, optional): Numpy or Cupy.

    Returns:
        MstepResult: Result of the registration (transformation, sigma2, q)
"""


source_raw = o3d.io.read_point_cloud(source_path)
target = o3d.io.read_point_cloud(target_path)
# source_raw = source_raw.voxel_down_sample(voxel_size=1.5)



cl, ind = source_raw.remove_statistical_outlier(nb_neighbors=100,std_ratio=1.0)
# cl, ind = source_raw.remove_radius_outlier(nb_points=32, radius=5)


def display_inlier_outlier(cloud, ind):
    inlier_cloud = cloud.select_by_index(ind)
    outlier_cloud = cloud.select_by_index(ind, invert=True)
    sample_cloud = copy.deepcopy(inlier_cloud)
    print("Showing outliers (red) and inliers (gray): ")
    outlier_cloud.paint_uniform_color([1, 0, 0])
    inlier_cloud.paint_uniform_color([0.8, 0.8, 0.8])
    o3.visualization.draw_geometries([inlier_cloud, outlier_cloud])
    return sample_cloud

source = display_inlier_outlier(source_raw, ind)

# source=copy.deepcopy(source_raw)
# source = source.select_by_index(ind)
# target = 'V:/emmett_ubuntu_backup/emmett/data/NIKE/SIPING/master/master_sampled.ply'
                         

threshold = 1000
trans_init = np.asarray([[  1.  ,         0.     ,      0.   ,     0],
                         [  0.  ,         1.     ,      0.   ,     0],
                         [  0.  ,         0.     ,      1.   ,     0],
                         [  0.  ,         0.     ,      0.   ,     1.]])

def draw_registration_result(source, target, transformation):
    source_temp = copy.deepcopy(source)
    target_temp = copy.deepcopy(target)
                                     
    # source_temp.paint_uniform_color([1, 0.706, 0])
    # target_temp.paint_uniform_color([0, 0.651, 0.929])
                                      
    source_temp.transform(transformation)
    o3.io.write_point_cloud("./cube_data/scan_trans/001_trans_outlier_remove.ply", source_temp)
    o3.visualization.draw_geometries([source_temp, target_temp],
                                      zoom=0.4459,
                                      front=[0.9288, -0.2951, -0.2242],
                                      lookat=[1.6784, 2.0612, 1.4451],
                                      up=[-0.3402, -0.9189, -0.1996])
                                      
# draw_registration_result(source, target, trans_init)
print("Initial alignment")
evaluation = o3.pipelines.registration.evaluate_registration(source, target, threshold, trans_init)
print(evaluation)
# print("Apply point-to-point ICP")
# reg_p2p = o3.pipelines.registration.registration_icp(
#     source, target, threshold, trans_init,
#     o3.pipelines.registration.TransformationEstimationPointToPoint())
# print("reg_p2p : ", reg_p2p)
# print("Transformation is:")
# print(reg_p2p.transformation)
# draw_registration_result(source, target, reg_p2p.transformation)
start = time.time()
reg_p2p = o3.pipelines.registration.registration_icp(
    source, target, threshold, trans_init,
    o3.pipelines.registration.TransformationEstimationPointToPoint(),
    o3.pipelines.registration.ICPConvergenceCriteria(max_iteration=1000))
end = time.time()
print(f"{end - start:.5f} sec")
print(reg_p2p)
print("Transformation is:")
print(reg_p2p.transformation)
draw_registration_result(source, target, reg_p2p.transformation)
# start = time.time()
# reg_p2l = o3.pipelines.registration.registration_icp(
#     source, target, threshold, trans_init,
#     o3.pipelines.registration.TransformationEstimationPointToPlane())
# end = time.time()
# print(f"{end - start:.5f} sec")
# print(reg_p2l)
# print("Transformation is:")
# print(reg_p2l.transformation)
# draw_registration_result(source, target, reg_p2l.transformation)
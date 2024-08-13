import copy
import numpy as np
use_cuda = False
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
import open3d as o3d

from probreg import transformation
from probreg import filterreg




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
        print(result_filename)

class reg(savePly):
    """지정된 폴더에서 source 클라우드와 target 클라우드를 open3d로 동작시킬 수 있게 불러옴 

    Args:
        data_name (str): file name.
    """
    def __init__(self, source, target, save = False):
        self.source = source
        self.target = target
        self.save = save
        

        self.cbs = [savePly(source, target)]

    
    def fillterreg(self, tf_param = {'rot': np.identity(3), 't': np.zeros(3)}): 
        tf_res, _, _ = filterreg.registration_filterreg(self.source, self.target, maxiter=40, update_sigma2 = True, w = 0.0000203556673416789, tol = 0.486636460597193, tf_init_params = tf_param)
        #tf_param['rot'], tf_param['t'] = tf_res.rot, tf_res.t
        #tf_res, _, _ = filterreg.registration_filterreg(self.source, self.target, update_sigma2 = True, w = 0.0000001, tol = 0.0000001, tf_init_params = tf_param) #callbacks=self.cbs)        
     
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
    def __init__(self, path_data):
        self.path_data = path_data
        self.sources = []
        self.targets = []
        
        self.load_data(self.path_data) 

    def load_data(self, current_path):
        """지정된 폴더 경로에서 재귀적으로 source.ply와 Master.ply를 탐색하여 리스트에 저장."""
        for folder_name in os.listdir(current_path):
            folder_path = os.path.join(current_path, folder_name)
            
            if os.path.isdir(folder_path):
                self.load_data(folder_path)
            else:
                source_path = os.path.join(current_path, 'source.ply')
                target_path = os.path.join(current_path, 'Master.ply')

                if os.path.exists(source_path) and os.path.exists(target_path):
                    # Source point cloud
                    pcd_source = o3d.io.read_point_cloud(source_path)
                    pcd_source = pcd_source.voxel_down_sample(voxel_size=3)

                    source_points = np.asarray(pcd_source.points)
                    self.sources.append(source_points)

                    # Target point cloud
                    pcd_target = o3d.io.read_point_cloud(target_path)
                    pcd_target = pcd_target.voxel_down_sample(voxel_size=3)
                    target_points = np.asarray(pcd_target.points)
                    self.targets.append(target_points)

                    #print(f"## Loaded source: {source_path}")
                    #print(f"## Loaded target: {target_path}")

                    break
                else:
                    pass
                    #print(f"## Missing source or target in folder: {current_path}")


    def get(self):
        """로딩된 소스와 타겟 리스트를 반환."""
        return self.sources, self.targets


PATH_SAVE = "/home/eric/github/probreg/result/"
PATH_DATA = "/home/eric/github/probreg/data/"

def random_rotation_matrix(max_angle_deg=30):
    """랜덤 회전 행렬 생성, 회전 각도는 -max_angle_deg에서 +max_angle_deg 사이."""
    angle_rad = np.deg2rad(np.random.uniform(-max_angle_deg, max_angle_deg))
    
    # 임의의 축으로 회전
    axis = np.random.normal(size=3)
    axis /= np.linalg.norm(axis)  # 정규화

    # 회전 행렬 생성 (Rodrigues' rotation formula)
    K = np.array([
        [0, -axis[2], axis[1]],
        [axis[2], 0, -axis[0]],
        [-axis[1], axis[0], 0]
    ])
    I = np.eye(3)
    rot_matrix = I + np.sin(angle_rad) * K + (1 - np.cos(angle_rad)) * np.dot(K, K)
    
    return rot_matrix

def random_translation_vector(max_translation=100):
    return np.random.uniform(-max_translation, max_translation, size=3)


def transform_point_cloud(points, rot, t):
    return np.dot(points, rot.T) + t

def save_point_cloud(source, target, file_path):
    pcd_source = o3d.geometry.PointCloud()
    pcd_source.points = o3d.utility.Vector3dVector(np.array(source) / 1000.0)
    
    source_colors = np.array([[1, 0, 0]] * len(source), dtype=np.float64)  # Ensure float64 type
    pcd_source.colors = o3d.utility.Vector3dVector(source_colors)


    pcd_target = o3d.geometry.PointCloud()
    pcd_target.points = o3d.utility.Vector3dVector(np.array(target) / 1000.0)
    
    target_colors = np.array([[0, 1, 0]] * len(target), dtype=np.float64)  # Ensure float64 type
    pcd_target.colors = o3d.utility.Vector3dVector(target_colors)


    pcd_combined = pcd_source + pcd_target

    o3d.io.write_point_cloud(file_path, pcd_combined)

    print(f"Saved transformed point cloud to {file_path}")

def main():
    rot, t = np.identity(3), np.zeros(3)

    getData = getCmesData(PATH_DATA)
    source, target = getData.get()

    for i in range(len(source)):
        rot = random_rotation_matrix(max_angle_deg=30)
        t = random_translation_vector(max_translation=100)
        source[i] = transform_point_cloud(source[i], rot, t)

        my_reg = reg(source[i], target[i])

        start = time.time()
        rot, t = my_reg.fillterreg()     # run fillterreg
        # rot, t = my_reg.pcl()          # run pcl
        elapsed = time.time() - start

        source_transformed = transform_point_cloud(source[i], rot, t)
        save_point_cloud(source_transformed, target[i], f"{PATH_SAVE}{i}_{elapsed}.ply")

        print(f"Result for pair {i+1}:")
        print("time: ", elapsed)
        print("rot: ", rot)
        print("t: ", t)

if __name__ == "__main__":
    main()
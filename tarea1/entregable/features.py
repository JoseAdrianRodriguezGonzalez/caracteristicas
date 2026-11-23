#Segundo archivo a revisar, proceso de calculo de caracteristicas
import numpy as np
from scipy.spatial import ConvexHull
import open3d as o3d
import pandas as pd
class features:
    def __init__(self, name):
        df=pd.read_csv(name)
        if {"x","y","z"}.issubset(df.columns):
            self.points=df[['x','y','z']].values
        else:
            df=pd.read_csv(name,header=None)
            df.columns = ['x','y','z']
            self.points=df[['x','y','z']].values
    def subsample(self, fraction):
        N = self.points.shape[0]
        n_sample = max(1, int(N * fraction))
        idx = np.random.choice(N, n_sample, replace=False)
        return self.points[idx, :]
    def feature_compacity(self,point=None):
        dimensions=self.feature_dimensions(point)
        volume=self.volume(point)
        return volume/(dimensions[0]*dimensions[1]*dimensions[2])
    # Modificar los métodos para recibir puntos opcionales
    def volume(self, points=None):
        pts = self.points if points is None else points
        hull = ConvexHull(pts)
        return hull.volume

    def feature_dimensions(self, points=None):
        pts = self.points if points is None else points
        return pts.max(axis=0) - pts.min(axis=0)

    def centroid(self, points=None):
        pts = self.points if points is None else points
        return pts.mean(axis=0)

    def feature_curvature(self, points=None, radius=0.05):
        pts = self.points if points is None else points
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pts)
        pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=radius, max_nn=30))
        kdtree = o3d.geometry.KDTreeFlann(pcd)
        curvatures = []
        for i, p in enumerate(pcd.points):
            _, idx, _ = kdtree.search_knn_vector_3d(p, 20)
            neigh = np.asarray(pcd.points)[idx, :]
            cov = np.cov(neigh.T)
            eigvals, _ = np.linalg.eigh(cov)
            eigvals = np.sort(eigvals)
            curvature = eigvals[0] / eigvals.sum()
            curvatures.append(curvature)
        return np.mean(curvatures)


    def feature_border(self, points=None):
        pts = self.points if points is None else points
        hull = ConvexHull(pts)
        border_points = np.unique(hull.vertices)
        return len(border_points) / len(pts)

    def feature_fpfh(self, points=None, radius=0.05):
        pts = self.points if points is None else points
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pts)
        pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=radius, max_nn=30))
        fpfh = o3d.pipelines.registration.compute_fpfh_feature(
            pcd,
            o3d.geometry.KDTreeSearchParamHybrid(radius=radius, max_nn=100)
        )
        desc = np.asarray(fpfh.data)  
        mean_hist = desc.mean(axis=1)
        p = mean_hist / mean_hist.sum()
        return -np.sum(p * np.log(p + 1e-10))

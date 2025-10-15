# encoding: utf-8
"""
AIHub KAIST Person Re-identification Dataset
"""

import glob
import re
import os.path as osp

from .bases import BaseImageDataset


class AIHubKAIST(BaseImageDataset):
    """
    AIHub KAIST Person Re-identification Dataset
    
    Dataset statistics:
    # identities: varies by split
    # images: varies by split
    """
    dataset_dir = 'aihub_kaist'

    def __init__(self, root='./data', train_folder='bounding_box_train_1', 
                 query_folder='query', gallery_folder='bounding_box_test', 
                 verbose=True, **kwargs):
        """
        Args:
            root (str): root directory path
            train_folder (str): name of training folder (bounding_box_train_1/2/3/sample_train)
            query_folder (str): name of query folder (query/sample_query/sample_query_reduced)
            gallery_folder (str): name of gallery folder (bounding_box_test/sample_test/sample_test_reduced)
            verbose (bool): print dataset statistics
        """
        super(AIHubKAIST, self).__init__()
        self.dataset_dir = osp.join(root, self.dataset_dir)
        self.train_dir = osp.join(self.dataset_dir, train_folder)
        self.query_dir = osp.join(self.dataset_dir, query_folder)
        self.gallery_dir = osp.join(self.dataset_dir, gallery_folder)

        self._check_before_run()

        train = self._process_dir(self.train_dir, relabel=True)
        query = self._process_dir(self.query_dir, relabel=False)
        gallery = self._process_dir(self.gallery_dir, relabel=False)

        if verbose:
            print("=> AIHub KAIST loaded")
            print(f"   Train folder: {train_folder}")
            print(f"   Query folder: {query_folder}")
            print(f"   Gallery folder: {gallery_folder}")
            self.print_dataset_statistics(train, query, gallery)

        self.train = train
        self.query = query
        self.gallery = gallery

        self.num_train_pids, self.num_train_imgs, self.num_train_cams = self.get_imagedata_info(self.train)
        self.num_query_pids, self.num_query_imgs, self.num_query_cams = self.get_imagedata_info(self.query)
        self.num_gallery_pids, self.num_gallery_imgs, self.num_gallery_cams = self.get_imagedata_info(self.gallery)

    def _check_before_run(self):
        """Check if all files are available before going deeper"""
        if not osp.exists(self.dataset_dir):
            raise RuntimeError("'{}' is not available".format(self.dataset_dir))
        if not osp.exists(self.train_dir):
            raise RuntimeError("'{}' is not available".format(self.train_dir))
        if not osp.exists(self.query_dir):
            raise RuntimeError("'{}' is not available. Please extract query.zip first.".format(self.query_dir))
        if not osp.exists(self.gallery_dir):
            raise RuntimeError("'{}' is not available".format(self.gallery_dir))

    def _process_dir(self, dir_path, relabel=False):
        """
        Process directory to extract person ID and camera ID from filename
        Filename format: {pid}_c{camid}s{seq}_{frame}.jpg
        Example: 0000_c07s32_005875.jpg
        """
        img_paths = glob.glob(osp.join(dir_path, '*.jpg'))
        # Pattern to match: {pid}_c{camid}s{seq}_{frame}.jpg
        pattern = re.compile(r'(\d+)_c(\d+)s')

        pid_container = set()
        for img_path in img_paths:
            match = pattern.search(img_path)
            if match:
                pid, _ = map(int, match.groups())
                pid_container.add(pid)
        
        pid2label = {pid: label for label, pid in enumerate(sorted(pid_container))}

        dataset = []
        for img_path in img_paths:
            match = pattern.search(img_path)
            if match:
                pid, camid = map(int, match.groups())
                if relabel:
                    pid = pid2label[pid]
                dataset.append((img_path, pid, camid))

        return dataset


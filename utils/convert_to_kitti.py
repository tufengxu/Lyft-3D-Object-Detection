from export_kitti import KittiConverter

scene_num = 0
converter = KittiConverter(store_dir='kitti/scene0/train/')
data_path = '3d-object-detection-for-autonomous-vehicles/'

train_path = data_path + 'train/'
test_path = data_path + 'test/'

converter.nuscenes_gt_to_kitti(lyft_dataroot=train_path, table_folder=train_path+'data/', scene_num=scene_num)
converter.nuscenes_gt_to_kitti(lyft_dataroot=test_path, table_folder=test_path+'data/', scene_num=scene_num)

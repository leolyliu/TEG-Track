import os


def get_framelist(datalist_path, test_set_only=False, stride=1):
    frame_list_dict = {}
    idx = 0
    with open(datalist_path, "r") as f:
        for line in f:
            content = line.strip()
            if len(content) == 0:
                break
            if not test_set_only:
                path, category, instance, video_idx, frame_idx = content.split(",")
                path = path.replace("/nas/Tactile-tracking_Data", "/nas/datasets/Tactile-tracking_Data")
                if video_idx == "video_idx":  # title line
                    continue
            else:
                path, frame_idx = content.split(",")
                path = path.replace("/nas/Tactile-tracking_Data", "/nas/datasets/Tactile-tracking_Data")
                video_dir, video_idx = os.path.split(path)
                instance_dir, instance = os.path.split(video_dir)
                category_dir, category = os.path.split(instance_dir)

            video_name = path
            if not video_name in frame_list_dict:
                frame_list_dict[video_name] = []
                idx = 0
            content = {'path': path, 'category': category, 'instance': instance, 'video_idx': video_idx, 'frame_idx': frame_idx}
            if idx % stride == 0:
                frame_list_dict[video_name].append(content)
            idx += 1
    
    frame_lists = []
    for video_name in frame_list_dict:
        frame_lists.append(frame_list_dict[video_name])
    return frame_lists
